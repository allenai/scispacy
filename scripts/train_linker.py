from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import jsonlines
from statistics import mean
import datetime
from scispacy import umls_semantic_type_tree
from linking import featurizer
import argparse

def main(data_path: str):
    train = []
    dev = []
    test = []
    root = ''
    files = [f'{data_path}/train.jsonl', f'{data_path}/dev.jsonl', f'{data_path}/test.jsonl']
    lists = [train, dev, test]
    # limits = [40000000, 4000000, 4000000]
    limits = [4000000, 0, 4000000]

    start_time = datetime.datetime.now()

    # load data from files
    for filename, data, limit in zip(files, lists, limits):
        with jsonlines.open(filename) as f:
            for line in f:
                data.append(line)
                if len(data) % 100000 == 0:
                    print(len(data))
                if len(data) >= limit:
                    break

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f'Time: {total_time.total_seconds()} seconds')

    x = []
    y = []
    for data in lists:
        features  = [featurizer(d) for d in data]
        x.append(features)
        y.append([d['label'] for d in data])

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f'Time: {total_time.total_seconds()} seconds')

    cls = GradientBoostingClassifier(verbose=1)

    cls.fit(x[0] + x[1], y[0] + y[1])
    linking_cls_path = f'{data_path}/linking_classifier.joblib'
    dump(cls, linking_cls_path)
    cls = load(linking_cls_path)

    pred = cls.predict(x[0] + x[1])
    accuracy = accuracy_score(y[0] + y[1], pred)
    cls_report = classification_report(y[0] + y[1], pred)

    print('Train+Dev results:')
    print(accuracy)
    print(cls_report)

    pred = cls.predict(x[2])
    accuracy = accuracy_score(y[2], pred)
    cls_report = classification_report(y[2], pred)
    print('Test results:')
    print(accuracy)
    print(cls_report)

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
