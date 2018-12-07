import os
import spacy

from tqdm import tqdm
import random
from spacy import util
from timeit import default_timer as timer
from spacy.cli.train import print_progress
import argparse
import json
import spacy_convert

def train_parser(train_json_path: str,
                 dev_json_path: str,
                 test_json_path: str,
                 output_dir: str):
    """Function to train the spacy parser from a blank model, with the default, en_core_web_sm vocab.
       Training setup is mostly copied from the spacy cli train command.

       @param train_json_path: path to the training data in spacy json format
       @param dev_json_path: path to the development data in spacy json format
       @param test_json_path: path to the test data in spacy json format
       @param output_dir: path to the directory to output the trained models to
    """
    nlp_en = spacy.load('en_core_web_sm')
    nlp = spacy.blank('en')
    nlp.vocab = nlp_en.vocab

    if 'parser' not in nlp.pipe_names:
        parser = nlp.create_pipe('parser')
        nlp.add_pipe(parser)
    else:
        parser = nlp.get_pipe('parser')

    train_corpus = spacy_convert.convert_abstracts_to_docs("../data/daniel_data/train.conllu", "../data/daniel_data/train_new.pmids", "../SciSpaCy/models/combined_all_model/vocab")
    dev_corpus = spacy_convert.convert_abstracts_to_docs("../data/daniel_data/dev.conllu", "../data/daniel_data/dev_new.pmids", "../SciSpaCy/models/combined_all_model/vocab")
    test_corpus = spacy_convert.convert_abstracts_to_docs("../data/daniel_data/test.conllu", "../data/daniel_data/test_new.pmids", "../SciSpaCy/models/combined_all_model/vocab")
    n_train_words = sum(len(doc_gold[0]) for doc_gold in train_corpus)

    dropout_rates = util.decaying(util.env_opt('dropout_from', 0.2),
                                  util.env_opt('dropout_to', 0.2),
                                  util.env_opt('dropout_decay', 0.0))
    batch_sizes = util.compounding(util.env_opt('batch_from', 1),
                                   util.env_opt('batch_to', 0.2),
                                   util.env_opt('batch_compound', 1.001))

    for doc, gold in train_corpus:
        for label in gold.labels:
            parser.add_label(label)

    optimizer = nlp.begin_training()
    nlp._optimizer = None
    print("Itn.  Dep Loss  NER Loss  UAS     NER P.  NER R.  NER F.  Tag %   Token %  CPU WPS  GPU WPS")
    for i in range(8):
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
            epoch_model_path = '../SciSpaCy/models/manual-model'+str(i)
            nlp.to_disk(epoch_model_path)
            nlp_loaded = util.load_model_from_path(epoch_model_path)
            nwords = sum(len(doc_gold[0]) for doc_gold in dev_corpus)
            start_time = timer()
            scorer = nlp_loaded.evaluate(dev_corpus)
            end_time = timer()
            gpu_wps = None
            cpu_wps = nwords/(end_time-start_time)

        print_progress(i, losses, scorer.scores, cpu_wps=cpu_wps, gpu_wps=gpu_wps)

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
    meta["speed"] = {"gpu": None, "nwords": nwords, "cpu": cpu_wps}
    meta["email"] = "ai2-info@allenai.org"

    # save final model and output results on the test set
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(os.path.join("../SciSpaCy/models/", "genia_trained_parser"))
    nlp_loaded = util.load_model_from_path(os.path.join("../SciSpaCy/models/", "genia_trained_parser"))
    nwords = sum(len(doc_gold[0]) for doc_gold in test_corpus)
    start_time = timer()
    scorer = nlp_loaded.evaluate(test_corpus)
    end_time = timer()
    gpu_wps = None
    cpu_wps = nwords/(end_time-start_time)

    print("Test results:")
    print("UAS:", scorer.unlabelled.fscore)
    print("LAS:", scorer.labelled.fscore)
    with open(os.path.join(output_dir, "genia_trained_parser", "meta.json"), "w") as meta_fp:
        meta_fp.write(json.dumps(meta))

def main(train_json_path,
         dev_json_path,
         test_json_path,
         output_dir):

    train_parser(train_json_path, dev_json_path, test_json_path, output_dir)

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
        '--output_dir',
        help="Path to the directory to output the trained models to"
    )

    args = parser.parse_args()
    main(args.train_json_path,
         args.dev_json_path,
         args.test_json_path,
         args.output_dir)
