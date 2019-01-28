import sys
import random
import os
from pathlib import Path
import json
import shutil

import argparse
import tqdm
import spacy
from spacy.gold import minibatch
from spacy import util


sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from scispacy.data_util import read_med_mentions, read_full_med_mentions
from scispacy.per_class_scorer import PerClassScorer
from scispacy.umls_semantic_type_tree import construct_umls_tree_from_tsv
from scispacy.train_utils import evaluate_ner


def train_ner(output_dir: str,
              data_path: str,
              run_test: bool = None,
              model: str = None,
              n_iter: int = 100,
              label_granularity: int = None):

    if label_granularity is not None:
        umls_tree = construct_umls_tree_from_tsv("data/umls_semantic_type_tree.tsv")
        label_mapping = umls_tree.get_collapsed_type_id_map_at_level(label_granularity)
    else:
        label_mapping = None
    train_data, dev_data, test_data = read_full_med_mentions(data_path, label_mapping)
    os.makedirs(output_dir, exist_ok=True)
    if run_test:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
        evaluate_ner(nlp, dev_data, dump_path=os.path.join(output_dir, "dev_metrics.json"))
        evaluate_ner(nlp, test_data, dump_path=os.path.join(output_dir, "test_metrics.json"))
    else:
        train(model, train_data, dev_data, output_dir, n_iter)


def train(model, train_data, dev_data, output_dir, n_iter):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names and "tagger" in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, after="tagger")
    elif 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    dropout_rates = util.decaying(util.env_opt('dropout_from', 0.2),
                                  util.env_opt('dropout_to', 0.2),
                                  util.env_opt('dropout_decay', 0.005))
    batch_sizes = util.compounding(util.env_opt('batch_from', 1),
                                   util.env_opt('batch_to', 32),
                                   util.env_opt('batch_compound', 1.001))

    optimizer = nlp.begin_training()
    best_epoch = 0
    best_f1 = 0
    for i in range(n_iter):

        random.shuffle(train_data)
        count = 0
        losses = {}
        total = len(train_data)

        with nlp.disable_pipes(*other_pipes):  # only train NER
            with tqdm.tqdm(total=total, leave=True) as pbar:
                for batch in minibatch(train_data, size=batch_sizes):
                    docs, golds = zip(*batch)
                    nlp.update(docs, golds, sgd=optimizer,
                               losses=losses, drop=next(dropout_rates))
                    pbar.update(len(batch))
                    if count % 100 == 0 and count > 0:
                        print('sum loss: %s' % losses['ner'])
                    count += 1

        # save model to output directory
        output_dir_path = Path(output_dir + "/" + str(i))
        if not output_dir_path.exists():
            output_dir_path.mkdir()

        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir_path)
            print("Saved model to", output_dir_path)

        # test the saved model
        print("Loading from", output_dir_path)
        nlp2 = util.load_model_from_path(output_dir_path)

        metrics = evaluate_ner(nlp2, dev_data)
        if metrics["f1-measure-untyped"] > best_f1:
            best_f1 = metrics["f1-measure-untyped"]
            best_epoch = i
    # save model to output directory
    best_model_path = Path(output_dir + "/" + "best")
    shutil.copytree(os.path.join(output_dir, str(best_epoch)),
                    best_model_path)

    # test the saved model
    print("Loading from", best_model_path)
    nlp2 = util.load_model_from_path(best_model_path)

    evaluate_ner(nlp2, dev_data, dump_path=os.path.join(output_dir, "metrics.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--model_output_dir',
            help="Path to the directory to output the trained models to"
    )

    parser.add_argument(
            '--data_path',
            help="Path to the data directory."
    )

    parser.add_argument(
            '--run_test',
            help="Whether to run evaluation on the test dataset."
    )

    parser.add_argument(
            '--model_path',
            default=None,
            help="Path to the spacy model to load"
    )
    parser.add_argument(
            '--iterations',
            type=int,
            help="Number of iterations to run."
    )
    parser.add_argument(
            '--label_granularity',
            type=int,
            help="granularity of the labels, between 1-7."
    )

    args = parser.parse_args()
    train_ner(args.model_output_dir,
              args.data_path,
              args.run_test,
              args.model_path,
              args.iterations,
              args.label_granularity)
