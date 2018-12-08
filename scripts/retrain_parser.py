import os
import spacy
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from tqdm import tqdm
from spacy import util
from timeit import default_timer as timer
from spacy.cli.train import print_progress
from spacy.vocab import Vocab
import argparse
import json
import spacy_convert

def train_parser(train_conll_path: str,
                 dev_conll_path: str,
                 test_conll_path: str,
                 train_pmids_path: str,
                 dev_pmids_path: str,
                 test_pmids_path: str,
                 vocab_path: str,
                 model_output_dir: str):
    """Function to train the spacy parser from a blank model, with the default, en_core_web_sm vocab.
       Training setup is mostly copied from the spacy cli train command.

       @param train_conll_path: path to the conll formatted training data
       @param dev_conll_path: path to the conll formatted dev data
       @param test_conll_path: path to the conll formatted test data
       @param train_pmids_path: path to the training pmids, one per conll sentence
       @param dev_pmids_path: path to the dev pmids, one per conll sentence
       @param test_pmids_path: path to the test pmids, one per conll sentence
       @param vocab_path: path to the vocab to load
       @param model_output_dir: path to the output directory for the trained models
    """
    nlp = spacy.blank('en')
    nlp.vocab = Vocab().from_disk(vocab_path)

    if 'parser' not in nlp.pipe_names:
        parser = nlp.create_pipe('parser')
        nlp.add_pipe(parser)
    else:
        parser = nlp.get_pipe('parser')

    train_corpus = spacy_convert.convert_abstracts_to_docs(train_conll_path, train_pmids_path, vocab_path)
    dev_corpus = spacy_convert.convert_abstracts_to_docs(dev_conll_path, dev_pmids_path, vocab_path)
    test_corpus = spacy_convert.convert_abstracts_to_docs(test_conll_path, test_pmids_path, vocab_path)
    n_train_words = sum(len(doc_gold[0]) for doc_gold in train_corpus)

    dropout_rates = util.decaying(util.env_opt('dropout_from', 0.2),
                                  util.env_opt('dropout_to', 0.2),
                                  util.env_opt('dropout_decay', 0.0))
    batch_sizes = util.compounding(util.env_opt('batch_from', 1),
                                   util.env_opt('batch_to', 0.2),
                                   util.env_opt('batch_compound', 1.001))

    meta = {}
    meta["lang"] = "en"
    meta["pipeline"] = ["parser"]
    meta["name"] = "scispacy_core_web_sm"
    meta["license"] = "CC BY-SA 3.0"
    meta["author"] = "Allen Institute for Artificial Intelligence"
    meta["url"] = "allenai.org"
    meta["sources"] = ["OntoNotes 5", "Common Crawl", "GENIA 1.0"]
    meta["version"] = "1.0.0"
    meta["spacy_version"] = ">=2.0.18"
    meta["parent_package"] = "spacy"
    meta["email"] = "ai2-info@allenai.org"

    for doc, gold in train_corpus:
        for label in gold.labels:
            parser.add_label(label)

    optimizer = nlp.begin_training()
    nlp._optimizer = None
    print("Itn.  Dep Loss  NER Loss  UAS     NER P.  NER R.  NER F.  Tag %   Token %  CPU WPS  GPU WPS")
    for i in range(10):
        with tqdm(total=n_train_words, leave=False) as pbar:
            losses = {}
            minibatches = list(util.minibatch(train_corpus, size=batch_sizes))
            for batch in minibatches:
                docs, golds = zip(*batch)
                nlp.update(docs, golds, sgd=optimizer,
                           drop=next(dropout_rates), losses=losses)
                pbar.update(sum(len(doc) for doc in docs))

        # save intermediate model and output results on the dev set
        with nlp.use_params(optimizer.averages):
            epoch_model_path = os.path.join(model_output_dir, "model"+str(i))
            nlp.to_disk(epoch_model_path)

            with open(os.path.join(model_output_dir, "model"+str(i), "meta.json"), "w") as meta_fp:
                meta_fp.write(json.dumps(meta))

            nlp_loaded = util.load_model_from_path(epoch_model_path)
            nwords = sum(len(doc_gold[0]) for doc_gold in dev_corpus)
            start_time = timer()
            scorer = nlp_loaded.evaluate(dev_corpus)
            end_time = timer()
            gpu_wps = None
            cpu_wps = nwords/(end_time-start_time)

        print_progress(i, losses, scorer.scores, cpu_wps=cpu_wps, gpu_wps=gpu_wps)

    # save final model and output results on the test set
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(os.path.join(model_output_dir, "genia_trained_parser"))
    with open(os.path.join(model_output_dir, "genia_trained_parser", "meta.json"), "w") as meta_fp:
        meta_fp.write(json.dumps(meta))
    nlp_loaded = util.load_model_from_path(os.path.join(model_output_dir, "genia_trained_parser"))
    nwords = sum(len(doc_gold[0]) for doc_gold in test_corpus)
    start_time = timer()
    scorer = nlp_loaded.evaluate(test_corpus)
    end_time = timer()
    gpu_wps = None
    cpu_wps = nwords/(end_time-start_time)
    meta["speed"] = {"gpu": None, "nwords": nwords, "cpu": cpu_wps}

    print("Test results:")
    print("UAS:", scorer.unlabelled.fscore)
    print("LAS:", scorer.labelled.fscore)
    with open(os.path.join(model_output_dir, "genia_trained_parser", "meta.json"), "w") as meta_fp:
        meta_fp.write(json.dumps(meta))

def main(train_conll_path,
         dev_conll_path,
         test_conll_path,
         train_pmids_path,
         dev_pmids_path,
         test_pmids_path,
         vocab_path,
         model_output_dir):

    train_parser(train_conll_path,
                 dev_conll_path,
                 test_conll_path,
                 train_pmids_path,
                 dev_pmids_path,
                 test_pmids_path,
                 vocab_path,
                 model_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_conll_path',
        help="Path to the conll formatted training data"
    )

    parser.add_argument(
        '--dev_conll_path',
        help="Path to the conll formatted dev data"
    )

    parser.add_argument(
        '--test_conll_path',
        help="Path to the conll formatted test data"
    )

    parser.add_argument(
        '--train_pmids_path',
        help="Path to the training pmids, one per conll sentence"
    )

    parser.add_argument(
        '--dev_pmids_path',
        help="Path to the dev pmids, one per conll sentence"
    )

    parser.add_argument(
        '--test_pmids_path',
        help="Path to the test pmids, one per conll sentence"
    )

    parser.add_argument(
        '--vocab_path',
        help="Path to the spacy vocabulary to load"
    )

    parser.add_argument(
        '--model_output_dir',
        help="Path to the directory to output the trained models to"
    )

    args = parser.parse_args()
    main(args.train_conll_path,
         args.dev_conll_path,
         args.test_conll_path,
         args.train_pmids_path,
         args.dev_pmids_path,
         args.test_pmids_path,
         args.vocab_path,
         args.model_output_dir)
