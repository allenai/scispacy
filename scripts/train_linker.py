from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import datetime
from scispacy import umls_semantic_type_tree
from linking import Linker
import argparse
from tqdm import tqdm
import json

def read_file(filename, limit):
        x = []
        y = []
        with open(filename) as f:
            for line in tqdm(f, total=limit):
                d = json.loads(line)
                x.append(Linker.featurizer(d))
                y.append(d['label'])
                if len(x) >= limit:
                    break
        return x, y

def main(data_path: str):
    start_time = datetime.datetime.now()

    x_train, y_train = read_file(f'{data_path}/train.jsonl', 5000000)  # the full set is unnecessarily large
    x_dev, y_dev = read_file(f'{data_path}/dev.jsonl', 1)  # the full set is unnecessarily large
    x_test, y_test = read_file(f'{data_path}/test.jsonl', 5000000)

    # sklearn classifier already splits the training set into train and dev, so we don't need separate sets
    x_train.extend(x_dev)
    y_train.extend(y_dev)

    classifier = GradientBoostingClassifier(verbose=1)

    classifier.fit(x_train, y_train)
    linking_classifier_path = f'{data_path}/linking_classifier.joblib'
    dump(classifier, linking_classifier_path)
    classifier = load(linking_classifier_path)
    pred = classifier.predict(x_train)
    accuracy = accuracy_score(y_train, pred)
    report = classification_report(y_train, pred)

    print('Train+Dev results:')
    print(accuracy)
    print(report)

    pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    print('Test results:')
    print(accuracy)
    print(report)

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f'Time: {total_time.total_seconds()} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data_path',
            help='Path to a directory with training set.'
    )
    args = parser.parse_args()
    main(args.data_path)
