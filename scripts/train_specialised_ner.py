import sys
import random
import os
from pathlib import Path
import shutil
import json

import argparse
import tqdm
import spacy
from spacy.gold import minibatch
from spacy.language import Language
from spacy import util


sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from scispacy.data_util import read_full_med_mentions, read_ner_from_tsv
from scispacy.util import WhitespaceTokenizer
from scispacy.per_class_scorer import PerClassScorer
from scispacy.train_utils import evaluate_ner


def train_ner(
    output_dir: str,
    train_data_path: str,
    dev_data_path: str,
    test_data_path: str,
    run_test: bool = None,
    model: str = None,
    n_iter: int = 10,
    meta_overrides: str = None,
):

    util.fix_random_seed(util.env_opt("seed", 0))
    train_data = read_ner_from_tsv(train_data_path)
    dev_data = read_ner_from_tsv(dev_data_path)
    test_data = read_ner_from_tsv(test_data_path)
    os.makedirs(output_dir, exist_ok=True)
    if run_test:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
        evaluate_ner(
            nlp, dev_data, dump_path=os.path.join(output_dir, "dev_metrics.json")
        )
        evaluate_ner(
            nlp, test_data, dump_path=os.path.join(output_dir, "test_metrics.json")
        )
    else:
        train(
            model, train_data, dev_data, test_data, output_dir, n_iter, meta_overrides
        )


def train(model, train_data, dev_data, test_data, output_dir, n_iter, meta_overrides):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    if meta_overrides is not None:
        metadata = json.load(open(meta_overrides))
        nlp.meta.update(metadata)

    original_tokenizer = nlp.tokenizer

    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names and "parser" in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, after="parser")
    elif "ner" not in nlp.pipe_names and "tagger" in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, after="tagger")
    elif "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    dropout_rates = util.decaying(
        util.env_opt("dropout_from", 0.2),
        util.env_opt("dropout_to", 0.2),
        util.env_opt("dropout_decay", 0.005),
    )
    batch_sizes = util.compounding(
        util.env_opt("batch_from", 1),
        util.env_opt("batch_to", 32),
        util.env_opt("batch_compound", 1.001),
    )

    with nlp.disable_pipes(*other_pipes):
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
                    nlp.update(
                        docs,
                        golds,
                        sgd=optimizer,
                        losses=losses,
                        drop=next(dropout_rates),
                    )
                    pbar.update(len(batch))
                    if count % 100 == 0 and count > 0:
                        print("sum loss: %s" % losses["ner"])
                    count += 1

        # save model to output directory
        output_dir_path = Path(output_dir + "/" + str(i))
        if not output_dir_path.exists():
            output_dir_path.mkdir()

        with nlp.use_params(optimizer.averages):
            nlp.tokenizer = original_tokenizer
            nlp.to_disk(output_dir_path)
            print("Saved model to", output_dir_path)

        # test the saved model
        print("Loading from", output_dir_path)
        nlp2 = util.load_model_from_path(output_dir_path)
        nlp2.tokenizer = WhitespaceTokenizer(nlp.vocab)

        metrics = evaluate_ner(nlp2, dev_data)
        if metrics["f1-measure-overall"] > best_f1:
            best_f1 = metrics["f1-measure-overall"]
            best_epoch = i
    # save model to output directory
    best_model_path = Path(output_dir + "/" + "best")
    print(f"Best Epoch: {best_epoch} of {n_iter}")
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    shutil.copytree(os.path.join(output_dir, str(best_epoch)), best_model_path)

    # test the saved model
    print("Loading from", best_model_path)
    nlp2 = util.load_model_from_path(best_model_path)
    nlp2.tokenizer = WhitespaceTokenizer(nlp.vocab)

    evaluate_ner(nlp2, dev_data, dump_path=os.path.join(output_dir, "dev_metrics.json"))
    evaluate_ner(
        nlp2, test_data, dump_path=os.path.join(output_dir, "test_metrics.json")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_output_dir",
        help="Path to the directory to output the trained models to",
    )

    parser.add_argument("--train_data_path", help="Path to the training data.")

    parser.add_argument("--dev_data_path", help="Path to the development data.")

    parser.add_argument("--test_data_path", help="Path to the test data.")

    parser.add_argument(
        "--run_test", help="Whether to run evaluation on the test dataset."
    )

    parser.add_argument(
        "--model_path", default=None, help="Path to the spacy model to load"
    )
    parser.add_argument("--iterations", type=int, help="Number of iterations to run.")
    parser.add_argument("--meta_overrides", type=str, help="Metadata to override.")

    args = parser.parse_args()
    train_ner(
        args.model_output_dir,
        args.train_data_path,
        args.dev_data_path,
        args.test_data_path,
        args.run_test,
        args.model_path,
        args.iterations,
        args.meta_overrides,
    )

