import sys
import random
import os

import plac
import tqdm
import spacy
from spacy.gold import minibatch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from SciSpaCy.data_util import read_med_mentions

@plac.annotations(
        output_dir=("output directory to store model in", "positional", None, str),
        train_path=("location of training data", "positional", None, str),
        dev_path=("location of development data", "option", None, str),
        test_path=("location of test data", "option", "test", str),
        model=("location of base model", "option", "model", str),
        n_iter=("number of iterations", "option", "n", int),
        batch_size=("batch size", "option", "b", int))
def main(output_dir, train_path, dev_path, test_path=None, model=None, n_iter=100, batch_size=32):

    os.makedirs(output_dir, exist_ok=True)
    if test_path is not None:
        test_data = read_med_mentions(test_path)
        test(model, test_data)
    else:
        train_data = read_med_mentions(train_path)
        if dev_path is None:
            train_examples = len(train_data)
            split = int(train_examples * 0.2)
            dev_data = train_data[:split]
            train_data = train_data[split:]
        else:
            dev_data = read_med_mentions(dev_path)

        train(model, train_data, dev_data, output_dir, batch_size, n_iter)

def test(model, test_data):
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
    evaluate(nlp, test_data)

def train(model, train_data, dev_data, output_dir, batch_size, n_iter):
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


    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for i in range(n_iter):

            random.shuffle(train_data)
            count = 0
            losses = {}
            total = len(train_data)

            with tqdm.tqdm(total=total, leave=True) as pbar:
                for batch in minibatch(train_data, size=batch_size):
                    docs, golds = zip(*batch)
                    nlp.update(docs, golds, sgd=optimizer, losses=losses, drop=0.01)
                    pbar.update(len(batch))
                    if count % 100 == 0 and count > 0:
                        print('sum loss: %s' % losses['ner'])
                    count += 1

            # save model to output directory
            if output_dir is not None:
                output_dir_path = Path(output_dir + "/" + str(i))
                if not output_dir_path.exists():
                    output_dir_path.mkdir()

                nlp.to_disk(output_dir_path)
                print("Saved model to", output_dir_path)

                # test the saved model
                print("Loading from", output_dir_path)
                nlp2 = spacy.load(output_dir_path)

            evaluate(nlp, dev_data)


def evaluate(nlp, eval_data):
    print("eval %d rows" % len(eval_data))
    num_correct = 0
    num_predicted = 0  # for precision
    num_relevant = 0  # for recall
    total = 0

    count = 0
    for text, gold in tqdm.tqdm(eval_data):
        # gold entity mentions
        # gold_ents = [(text[start_ind:end_ind], label) for start_ind, end_ind, label in gold['entities']]
        gold_ents = [text[start_ind:end_ind] for start_ind, end_ind, label in gold['entities']]

        # parse dev data with trained model
        doc = nlp(text)
        # identified_ents = [(ent.text, ent.label_) for ent in doc.ents]
        identified_ents = [ent.text for ent in doc.ents]

        # compute correct matches
        num_correct += len(set(gold_ents).intersection(set(identified_ents)))
        num_predicted += len(set(identified_ents))
        num_relevant += len(set(gold_ents))
        total += len(set(gold_ents).union(set(identified_ents)))

        if count % 1000 == 0 and count > 0:

            p = num_correct/num_predicted if num_predicted else 0
            r = num_correct/num_relevant if num_relevant else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            print('Acc: %0.4f (%d/%d), p: %0.4f (%d/%d), r: %0.4f (%d/%d), f1: %0.4f' % (num_correct/total, num_correct, total,
                   p, num_correct, num_predicted, r, num_correct, num_relevant, f1))

        count += 1
    p = num_correct/num_predicted if num_predicted else 0
    r = num_correct/num_relevant if num_relevant else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print('Acc: %0.4f (%d/%d), p: %0.4f (%d/%d), r: %0.4f (%d/%d), f1: %0.4f' % (num_correct/total, num_correct, total,
           p, num_correct, num_predicted, r, num_correct, num_relevant, f1))

plac.call(main, sys.argv[1:])
