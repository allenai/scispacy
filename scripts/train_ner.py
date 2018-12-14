import sys
import random
import os
from pathlib import Path

import plac
import tqdm
import spacy
from spacy.gold import minibatch
from spacy import util


sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from SciSpaCy.data_util import read_med_mentions, read_full_med_mentions
from SciSpaCy.per_class_scorer import PerClassScorer
from SciSpaCy.umls_semantic_type_tree import construct_umls_tree_from_tsv

@plac.annotations(
        output_dir=("output directory to store model in", "positional", None, str),
        data_path=("location of training data", "positional", None, str),
        run_test=("Whether or not to run on the test data.", "option", "test", bool),
        model=("location of base model", "option", "model", str),
        n_iter=("number of iterations", "option", "n", int),
        label_granularity=("granularity of the labels, between 1-7", "option", "granularity", int))
def main(output_dir: str,
         data_path: str,
         run_test: bool=None,
         model: str=None,
         n_iter: int=100,
         label_granularity: int=None):

    if label_granularity is not None:
        umls_tree = construct_umls_tree_from_tsv("data/umls_semantic_type_tree.tsv")
        label_mapping = umls_tree.get_collapsed_type_id_map_at_level(label_granularity)
    else:
        label_mapping = None
    train_data, dev_data, test_data = read_full_med_mentions(data_path, label_mapping)
    os.makedirs(output_dir, exist_ok=True)
    if run_test:
        test(model, test_data)
    else:
        train(model, train_data, dev_data, output_dir, n_iter)

def test(model, test_data):
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
    evaluate(nlp, test_data)

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
    if 'ner' not in nlp.pipe_names:
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

    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for i in range(n_iter):

            random.shuffle(train_data)
            count = 0
            losses = {}
            total = len(train_data)

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
            if output_dir is not None:
                output_dir_path = Path(output_dir + "/" + str(i))
                if not output_dir_path.exists():
                    output_dir_path.mkdir()

                with nlp.use_params(optimizer.averages):
                    nlp.to_disk(output_dir_path)
                    print("Saved model to", output_dir_path)

                # test the saved model
                print("Loading from", output_dir_path)
                nlp2 = spacy.load(output_dir_path)

            evaluate(nlp2, dev_data)


def evaluate(nlp, eval_data):

    scorer = PerClassScorer()
    print("Evaluating %d rows" % len(eval_data))
    for i, (text, gold_spans) in enumerate(tqdm.tqdm(eval_data)):

        # parse dev data with trained model
        doc = nlp(text)
        predicted_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        scorer(predicted_spans, gold_spans["entities"])

        if i % 1000 == 0 and i > 0:
            for name, metric in scorer.get_metric().items():
                print(f"{name}: {metric}")

    for name, metric in scorer.get_metric().items():
        print(f"{name}: \t\t {metric}")

plac.call(main, sys.argv[1:])
