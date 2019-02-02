import os
import spacy
import sys
import shutil

from tqdm import tqdm
from spacy import util
from timeit import default_timer as timer
from spacy.cli.train import _get_progress
from spacy.vocab import Vocab
from spacy.gold import GoldCorpus
from wasabi import Printer
import argparse
import json
import random
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from scispacy.file_cache import cached_path
from scispacy import spacy_convert

def train_parser_and_tagger(train_json_path: str,
                            dev_json_path: str,
                            test_json_path: str,
                            model_output_dir: str,
                            model_path: str = None,
                            ontonotes_path: str = None,
                            ontonotes_train_percent: float = 0.0):
    """Function to train the spacy parser and tagger from a blank model, with the default, en_core_web_sm vocab.
       Training setup is mostly copied from the spacy cli train command.

       @param train_json_path: path to the conll formatted training data
       @param dev_json_path: path to the conll formatted dev data
       @param test_json_path: path to the conll formatted test data
       @param model_output_dir: path to the output directory for the trained models
       @param model_path: path to the model to load
       @param ontonotes_path: path to the directory containnig ontonotes in spacy format (optional)
       @param ontonotes_train_percent: percentage of the ontonotes training data to use (optional)
    """
    msg = Printer()

    train_json_path = cached_path(train_json_path)
    dev_json_path = cached_path(dev_json_path)
    test_json_path = cached_path(test_json_path)

    if model_path is not None:
        nlp = spacy.load(model_path)
    else:
        lang_class = util.get_lang_class('en')
        nlp = lang_class()

    if 'tagger' not in nlp.pipe_names:
        tagger = nlp.create_pipe('tagger')
        nlp.add_pipe(tagger, first=True)
    else:
        tagger = nlp.get_pipe('tagger')

    if 'parser' not in nlp.pipe_names:
        parser = nlp.create_pipe('parser')
        nlp.add_pipe(parser)
    else:
        parser = nlp.get_pipe('parser')

    train_corpus = GoldCorpus(train_json_path, dev_json_path)
    test_corpus = GoldCorpus(train_json_path, test_json_path)

    if ontonotes_path:
        onto_train_path = os.path.join(ontonotes_path, "train")
        onto_dev_path = os.path.join(ontonotes_path, "dev")
        onto_test_path = os.path.join(ontonotes_path, "test")
        onto_train_corpus = GoldCorpus(onto_train_path, onto_dev_path)
        onto_test_corpus = GoldCorpus(onto_train_path, onto_test_path)

    dropout_rates = util.decaying(0.2, 0.2, 0.0)
    batch_sizes = util.compounding(1., 16., 1.001)

    if model_path is not None:
        meta = nlp.meta
    else:
        meta = {}
        meta["lang"] = "en"
        meta["pipeline"] = ["tagger", "parser"]
        meta["name"] = "scispacy_core_web_sm"
        meta["license"] = "CC BY-SA 3.0"
        meta["author"] = "Allen Institute for Artificial Intelligence"
        meta["url"] = "allenai.org"
        meta["sources"] = ["OntoNotes 5", "Common Crawl", "GENIA 1.0"]
        meta["version"] = "1.0.0"
        meta["spacy_version"] = ">=2.0.18"
        meta["parent_package"] = "spacy"
        meta["email"] = "ai2-info@allenai.org"

    n_train_words = train_corpus.count_train()

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in  ['tagger', 'parser']]
    if ontonotes_path:
        optimizer = nlp.begin_training(lambda: itertools.chain(train_corpus.train_tuples, onto_train_corpus.train_tuples))
    else:
        optimizer = nlp.begin_training(lambda: train_corpus.train_tuples)
    nlp._optimizer = None

    train_docs = train_corpus.train_docs(nlp)
    train_docs = list(train_docs)

    train_mixture = train_docs
    if ontonotes_path:
        onto_train_docs = onto_train_corpus.train_docs(nlp)
        onto_train_docs = list(onto_train_docs)
        num_onto_docs = int(float(ontonotes_train_percent)*len(onto_train_docs))
        randomly_sampled_onto = random.sample(onto_train_docs, num_onto_docs)
        train_mixture += randomly_sampled_onto

    row_head = ("Itn", "Dep Loss", "NER Loss", "UAS", "NER P", "NER R", "NER F", "Tag %", "Token %", "CPU WPS", "GPU WPS")
    row_settings = {
        "widths": (3, 10, 10, 7, 7, 7, 7, 7, 7, 7, 7),
        "aligns": tuple(["r" for i in row_head]),
        "spacing": 2
    }

    print("")
    msg.row(row_head, **row_settings)
    msg.row(["-" * width for width in row_settings["widths"]], **row_settings)
    best_epoch = 0
    best_epoch_uas = 0.0
    for i in range(10):
        random.shuffle(train_mixture)
        with nlp.disable_pipes(*other_pipes):
            with tqdm(total=n_train_words, leave=False) as pbar:
                losses = {}
                minibatches = list(util.minibatch(train_docs, size=batch_sizes))
                for batch in minibatches:
                    docs, golds = zip(*batch)
                    nlp.update(docs, golds, sgd=optimizer,
                               drop=next(dropout_rates), losses=losses)
                    pbar.update(sum(len(doc) for doc in docs))

        # save intermediate model and output results on the dev set
        with nlp.use_params(optimizer.averages):
            epoch_model_path = os.path.join(model_output_dir, "model"+str(i))
            os.makedirs(epoch_model_path, exist_ok=True)
            nlp.to_disk(epoch_model_path)

            with open(os.path.join(model_output_dir, "model"+str(i), "meta.json"), "w") as meta_fp:
                meta_fp.write(json.dumps(meta))

            nlp_loaded = util.load_model_from_path(epoch_model_path)
            dev_docs = train_corpus.dev_docs(nlp_loaded)
            dev_docs = list(dev_docs)
            nwords = sum(len(doc_gold[0]) for doc_gold in dev_docs)
            start_time = timer()
            scorer = nlp_loaded.evaluate(dev_docs)
            end_time = timer()
            gpu_wps = None
            cpu_wps = nwords/(end_time-start_time)

            if ontonotes_path:
                onto_dev_docs = list(onto_train_corpus.dev_docs(nlp_loaded))
                onto_scorer = nlp_loaded.evaluate(onto_dev_docs)


        if scorer.scores["uas"] > best_epoch_uas:
            best_epoch_uas = scorer.scores["uas"]
            best_epoch = i
        progress = _get_progress(
            i, losses, scorer.scores, cpu_wps=cpu_wps, gpu_wps=gpu_wps
        )
        msg.row(progress, **row_settings)
        if ontonotes_path:
            progress = _get_progress(
                i, losses, onto_scorer.scores, cpu_wps=cpu_wps, gpu_wps=gpu_wps
            )
            msg.row(progress, **row_settings)

    # save final model and output results on the test set
    final_model_path = os.path.join(model_output_dir, "best")
    if os.path.exists(final_model_path):
        shutil.rmtree(final_model_path)
    shutil.copytree(os.path.join(model_output_dir, "model" + str(best_epoch)),
                    final_model_path)

    nlp_loaded = util.load_model_from_path(final_model_path)
    start_time = timer()
    test_docs = test_corpus.dev_docs(nlp_loaded)
    test_docs = list(test_docs)
    nwords = sum(len(doc_gold[0]) for doc_gold in test_docs)
    scorer = nlp_loaded.evaluate(test_docs)
    end_time = timer()
    gpu_wps = None
    cpu_wps = nwords/(end_time-start_time)
    meta["speed"] = {"gpu": None, "nwords": nwords, "cpu": cpu_wps}

    print("Retrained genia evaluation")
    print("Test results:")
    print("UAS:", scorer.uas)
    print("LAS:", scorer.las)
    print("Tag %:", scorer.tags_acc)
    print("Token acc:", scorer.token_acc)
    with open(os.path.join(model_output_dir, "genia_test.json"), "w+") as metric_file:
        json.dump(scorer.scores, metric_file)
    with open(os.path.join(model_output_dir, "best", "meta.json"), "w") as meta_fp:
        meta_fp.write(json.dumps(meta))

    if ontonotes_path:
        onto_test_docs = list(onto_test_corpus.dev_docs(nlp_loaded))
        print("Retrained ontonotes evaluation")
        scorer_onto_retrained = nlp_loaded.evaluate(onto_test_docs)
        print("Test results:")
        print("UAS:", scorer_onto_retrained.uas)
        print("LAS:", scorer_onto_retrained.las)
        print("Tag %:", scorer_onto_retrained.tags_acc)
        print("Token acc:", scorer_onto_retrained.token_acc)

        with open(os.path.join(model_output_dir, "ontonotes_test.json"), "w+") as metric_file:
            json.dump(scorer_onto_retrained.scores, metric_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_json_path',
        help="Path to the json formatted training data"
    )

    parser.add_argument(
        '--dev_json_path',
        help="Path to the json formatted dev data"
    )

    parser.add_argument(
        '--test_json_path',
        help="Path to the json formatted test data"
    )

    parser.add_argument(
        '--model_path',
        default=None,
        help="Path to the spacy model to load"
    )


    parser.add_argument(
        '--model_output_dir',
        help="Path to the directory to output the trained models to"
    )

    parser.add_argument(
        '--ontonotes_path',
        default=None,
        help="Path to the ontonotes folder in spacy format"
    )

    parser.add_argument(
        '--ontonotes_train_percent',
        default=0.0,
        help="Percentage of ontonotes training data to mix in with the genia data")

    args = parser.parse_args()
    train_parser_and_tagger(args.train_json_path,
                            args.dev_json_path,
                            args.test_json_path,
                            args.model_output_dir,
                            args.model_path,
                            args.ontonotes_path,
                            args.ontonotes_train_percent)
