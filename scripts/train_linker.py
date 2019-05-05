from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import datetime
from scispacy import umls_semantic_type_tree
from linking import Linker
import argparse
from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict

def read_file(filename, limit):
        data = []
        x = []
        y = []
        with open(filename) as f:
            for line in tqdm(f, total=limit):
                d = json.loads(line)
                data.append(d)
                x.append(Linker.featurizer(d))
                y.append(d['label'])
                if len(x) >= limit:
                    break
        return data, x, y

def entity_level_eval(data, all_features, classifier):
    k_list = [1, 3, 5, 10]
    classifier_correct_predictions = defaultdict(int)
    classifier_total_predictions = defaultdict(int)
    candidate_features = []
    gold_index = -1
    for i, d in tqdm(enumerate(data)):
        candidate_index = d['candidate_index']
        if candidate_index == 0:
            if len(candidate_features) > 0:
                if gold_index != -1:  # compute "normalized" recall
                    scores = classifier.predict_proba(candidate_features)
                    sorted_ids = np.argsort(-scores[:, 1], kind='mergesort')
                    for k in k_list:
                        if gold_index in sorted_ids[:k]:
                            classifier_correct_predictions[k] += 1
                        classifier_total_predictions[k] += 1
            candidate_features = []
            gold_index = -1

        assert candidate_index == len(candidate_features)
        if d['label'] == 1:
            assert gold_index == -1
            gold_index = candidate_index
        candidate_features.append(all_features[i])

    for k in k_list:
        print(f'normalized recall@{k}: {100 * classifier_correct_predictions[k] / classifier_total_predictions[k]}')

def main(data_path: str, train:bool):
    start_time = datetime.datetime.now()
    linking_classifier_path = f'{data_path}/linking_classifier.joblib'

    if train:
        print('Reading training set')
        d_train, x_train, y_train = read_file(f'{data_path}/train.jsonl', 330000)  # the full set is unnecessarily large
        classifier = GradientBoostingClassifier(verbose=1)

        classifier.fit(x_train, y_train)
        print(f'Save classifier to: {linking_classifier_path}')
        dump(classifier, linking_classifier_path)
        for i, fimp in enumerate(classifier.feature_importances_):
            print('{0}: {1:.24f}'.format(i, fimp))

        pred = classifier.predict(x_train)
        accuracy = accuracy_score(y_train, pred)
        report = classification_report(y_train, pred)
        print('candidate-level training results:')
        print(f'accuracy: {accuracy}')
        print(report)

    print(f'Load classifier from: {linking_classifier_path}')
    classifier = load(linking_classifier_path)

    print('Reading dev set')
    d_test, x_test, y_test = read_file(f'{data_path}/dev.jsonl', 330000)  # 3300000 use dev set for testing

    pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    print('candidate-level dev results:')
    print(f'accuracy: {accuracy}')
    print(report)

    print('Entity-level dev results (normalized recall)')
    entity_level_eval(d_test, x_test, classifier)

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f'Time: {total_time.total_seconds()} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data_path',
            help='Path to a directory with training set.'
    )
    parser.add_argument(
            '--train',
            action="store_true",
            help='Train classifier.'
    )
    args = parser.parse_args()
    main(args.data_path, args.train)