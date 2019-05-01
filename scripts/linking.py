"""
Linking using char-n-gram with approximate nearest neighbors.
"""
import os
import os.path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from typing import List, Dict, Tuple, NamedTuple, Any, Set
import json
import argparse
import datetime
from collections import defaultdict
from tqdm import tqdm

import scipy
import numpy as np
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import ClassifierMixin
import nmslib
from nmslib.dist import FloatIndex
import spacy
from scispacy import data_util

def load_umls_kb(umls_path: str) -> List[Dict]:
    """
    Reads a UMLS json release and return it as a list of concepts.
    Each concept is a dictionary.
    """
    with open(umls_path) as f:
        print(f'Loading umls concepts from {umls_path}')
        umls_concept_list = json.load(f)
    print(f'Number of umls concepts: {len(umls_concept_list)}')
    return umls_concept_list


class MentionCandidate(NamedTuple):
    concept_id: str
    distances: List[float]
    aliases: List[str]

class CandidateGenerator:

    """
    A candidate generator for entity linking to the Unified Medical Language System (UMLS).

    It uses a sklearn.TfidfVectorizer to embed mention text into a sparse embedding of character 3-grams.
    These are then compared via cosine distance in a pre-indexed approximate nearest neighbours index of
    a subset of all entities and aliases in UMLS.

    Once the K nearest neighbours have been retrieved, they are canonicalized to their UMLS canonical ids.
    This step is required because the index also includes entity aliases, which map to a particular canonical
    entity. This point is important for two reasons:

    1. K nearest neighbours will return a list of Y possible neighbours, where Y < K, because the entity ids
    are canonicalized.

    2. A single string may be an alias for multiple canonical entities. For example, "Jefferson County" may be an
    alias for both the canonical ids "Jefferson County, Iowa" and "Jefferson County, Texas". These are completely
    valid and important aliases to include, but it means that using the candidate generator to implement a naive
    k-nn baseline linker results in very poor performance, because there are multiple entities for some strings
    which have an exact char3-gram match, as these entities contain the same alias string. This situation results
    in multiple entities returned with a distance of 0.0, because they exactly match an alias, making a k-nn baseline
    effectively a random choice between these candidates. However, this doesn't matter if you have a classifier
    on top of the candidate generator, as is intended!

    Parameters
    ----------
    ann_index: FloatIndex
        An nmslib approximate nearest neighbours index.
    tfidf_vectorizer: TfidfVectorizer
        The vectorizer used to encode mentions.
    ann_concept_aliases_list: List[str]
        A list of strings, mapping the indices used in the ann_index to canonical UMLS ids.
    mention_to_concept: Dict[str, Set[str]], required.
        A mapping from aliases to canonical ids that they are aliases of.
    verbose: bool
        Setting to true will print extra information about the generated candidates

    """
    def __init__(self,
                 ann_index: FloatIndex,
                 tfidf_vectorizer: TfidfVectorizer,
                 ann_concept_aliases_list: List[str],
                 mention_to_concept: Dict[str, Set[str]],
                 verbose: bool = True) -> None:

        self.ann_index = ann_index
        self.vectorizer = tfidf_vectorizer
        self.ann_concept_aliases_list = ann_concept_aliases_list
        self.mention_to_concept = mention_to_concept
        self.verbose = verbose


    def nmslib_knn_with_zero_vectors(self, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        ann_index.knnQueryBatch crashes if any of the vectors is all zeros.
        This function is a wrapper around `ann_index.knnQueryBatch` that solves this problem. It works as follows:
        - remove empty vectors from `vectors`.
        - call `ann_index.knnQueryBatch` with the non-empty vectors only. This returns `neighbors`,
        a list of list of neighbors. `len(neighbors)` equals the length of the non-empty vectors.
        - extend the list `neighbors` with `None`s in place of empty vectors.
        - return the extended list of neighbors and distances.
        """
        empty_vectors_boolean_flags = np.array(vectors.sum(axis=1) != 0).reshape(-1,)
        empty_vectors_count = vectors.shape[0] - sum(empty_vectors_boolean_flags)
        if self.verbose:
            print(f'Number of empty vectors: {empty_vectors_count}')

        # remove empty vectors before calling `ann_index.knnQueryBatch`
        vectors = vectors[empty_vectors_boolean_flags]

        # call `knnQueryBatch` to get neighbors
        original_neighbours = self.ann_index.knnQueryBatch(vectors, k=k)
        neighbors, distances = zip(*[(x[0].tolist(), x[1].tolist()) for x in original_neighbours])
        neighbors = list(neighbors)
        distances = list(distances)
        # all an empty list in place for each empty vector to make sure len(extended_neighbors) == len(vectors)

        # init extended_neighbors with a list of Nones
        extended_neighbors = np.empty((len(empty_vectors_boolean_flags),), dtype=object)
        extended_distances = np.empty((len(empty_vectors_boolean_flags),), dtype=object)

        # neighbors need to be convected to an np.array of objects instead of ndarray of dimensions len(vectors)xk
        # Solution: add a row to `neighbors` with any length other than k. This way, calling np.array(neighbors)
        # returns an np.array of objects
        neighbors.append([])
        distances.append([])
        # interleave `neighbors` and Nones in `extended_neighbors`
        extended_neighbors[empty_vectors_boolean_flags] = np.array(neighbors)[:-1]
        extended_distances[empty_vectors_boolean_flags] = np.array(distances)[:-1]

        return extended_neighbors, extended_distances

    def generate_candidates(self, mention_texts: List[str], k: int) -> List[Dict[str, List[int]]]:
        """
        Given a list of mention texts, returns a list of candidate neighbors.

        NOTE: Because we include canonical name aliases in the ann index, the list
        of candidates returned will not necessarily be of length k for each candidate,
        because we then map these to canonical ids only.
        # TODO Mark: We should be able to use this signal somehow, maybe a voting system?
        args:
            mention_texts: list of mention texts
            k: number of ann neighbors

        returns:
            A list of dictionaries, each containing the mapping from umls concept ids -> a list of
            the cosine distances between them. Note that these are lists for each concept id, because
            the index contains aliases which are canonicalized, so multiple values may map to the same
            canonical id.
        """
        if self.verbose:
            print(f'Generating candidates for {len(mention_texts)} mentions')
        tfidfs = self.vectorizer.transform(mention_texts)
        start_time = datetime.datetime.now()

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch` that addresses this issue.
        batch_neighbors, batch_distances = self.nmslib_knn_with_zero_vectors(tfidfs, k)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        if self.verbose:
            print(f'Finding neighbors took {total_time.total_seconds()} seconds')
        neighbors_by_concept_ids = []
        for neighbors, distances in zip(batch_neighbors, batch_distances):
            if neighbors is None:
                neighbors = []

            if distances is None:
                distances = []
            predicted_umls_concept_ids = defaultdict(list)
            for n, d in zip(neighbors, distances):
                mention = self.ann_concept_aliases_list[n]
                concepts_for_mention = self.mention_to_concept[mention]
                for concept_id in concepts_for_mention:
                    predicted_umls_concept_ids[concept_id].append((mention, d))

            neighbors_by_concept_ids.append({**predicted_umls_concept_ids})
        return neighbors_by_concept_ids

def create_tfidf_ann_index(model_path: str, text_to_concept: Dict[str, Set[str]]) -> None:
    """
    Build tfidf vectorizer and ann index.
    """
    tfidf_vectorizer_path = f'{model_path}/tfidf_vectorizer.joblib'
    ann_index_path = f'{model_path}/nmslib_index.bin'
    tfidf_vectors_path = f'{model_path}/tfidf_vectors_sparse.npz'
    uml_concept_aliases_path = f'{model_path}/concept_aliases.json'

    # nmslib hyperparameters (very important)
    # guide: https://github.com/nmslib/nmslib/blob/master/python_bindings/parameters.md
    # default values resulted in very low recall
    M = 100  # set to the maximum recommended value. Improves recall at the expense of longer indexing time
    efC = 2000  # `C` for Construction. Set to the maximum recommended value
                # Improves recall at the expense of longer indexing time
    efS = 1000  # `S` for Search. This controls performance at query time. Maximum recommended value is 2000.
                # It makes the query slow without significant gain in recall.
    num_threads = 60  # set based on the machine
    index_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 0}

    print(f'No tfidf vectorizer on {tfidf_vectorizer_path} or ann index on {ann_index_path}')
    uml_concept_aliases = list(text_to_concept.keys())

    uml_concept_aliases = np.array(uml_concept_aliases)

    print(f'Fitting tfidf vectorizer on {len(uml_concept_aliases)} aliases')
    tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=10, dtype=np.float32)
    start_time = datetime.datetime.now()
    uml_concept_alias_tfidfs = tfidf_vectorizer.fit_transform(uml_concept_aliases)
    print(f'Saving tfidf vectorizer to {tfidf_vectorizer_path}')
    dump(tfidf_vectorizer, tfidf_vectorizer_path)
    end_time = datetime.datetime.now()
    total_time = (end_time - start_time)
    print(f'Fitting and saving vectorizer took {total_time.total_seconds()} seconds')

    print(f'Finding empty (all zeros) tfidf vectors')
    empty_tfidfs_boolean_flags = np.array(uml_concept_alias_tfidfs.sum(axis=1) != 0).reshape(-1,)
    deleted_aliases = uml_concept_aliases[empty_tfidfs_boolean_flags == False]
    number_of_non_empty_tfidfs = len(deleted_aliases)
    total_number_of_tfidfs = uml_concept_alias_tfidfs.shape[0]

    print(f'Deleting {number_of_non_empty_tfidfs}/{total_number_of_tfidfs} aliases because their tfidf is empty')
    # remove empty tfidf vectors, otherwise nmslib will crash
    uml_concept_aliases = uml_concept_aliases[empty_tfidfs_boolean_flags]
    uml_concept_alias_tfidfs = uml_concept_alias_tfidfs[empty_tfidfs_boolean_flags]
    print(deleted_aliases)

    print(f'Saving list of concept ids and tfidfs vectors to {uml_concept_aliases_path} and {tfidf_vectors_path}')
    json.dump(uml_concept_aliases.tolist(), open(uml_concept_aliases_path, "w"))
    scipy.sparse.save_npz(tfidf_vectors_path, uml_concept_alias_tfidfs.astype(np.float16))

    print(f'Fitting ann index on {len(uml_concept_aliases)} aliases (takes 2 hours)')
    start_time = datetime.datetime.now()
    ann_index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
    ann_index.addDataPointBatch(uml_concept_alias_tfidfs)
    ann_index.createIndex(index_params, print_progress=True)
    ann_index.saveIndex(ann_index_path)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f'Fitting ann index took {elapsed_time.total_seconds()} seconds')

def load_tfidf_ann_index(model_path: str):
    # `S` for Search. This controls performance at query time. Maximum recommended value is 2000.
    # It makes the query slow without significant gain in recall.
    efS = 1000
    tfidf_vectorizer_path = f'{model_path}/tfidf_vectorizer.joblib'
    linking_classifier_path = f'{model_path}/linking_classifier.joblib'
    ann_index_path = f'{model_path}/nmslib_index.bin'
    tfidf_vectors_path = f'{model_path}/tfidf_vectors_sparse.npz'
    uml_concept_aliases_path = f'{model_path}/concept_aliases.json'

    start_time = datetime.datetime.now()
    print(f'Loading list of concepted ids from {uml_concept_aliases_path}')
    uml_concept_aliases = json.load(open(uml_concept_aliases_path))

    print(f'Loading tfidf vectorizer from {tfidf_vectorizer_path}')
    tfidf_vectorizer = load(tfidf_vectorizer_path)
    if isinstance(tfidf_vectorizer, TfidfVectorizer):
        print(f'Tfidf vocab size: {len(tfidf_vectorizer.vocabulary_)}')

    print(f'Loading linking classifier from {linking_classifier_path}')
    linking_classifier = load(linking_classifier_path)

    print(f'Loading tfidf vectors from {tfidf_vectors_path}')
    uml_concept_alias_tfidfs = scipy.sparse.load_npz(tfidf_vectors_path).astype(np.float32)

    print(f'Loading ann index from {ann_index_path}')
    ann_index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
    ann_index.addDataPointBatch(uml_concept_alias_tfidfs)
    ann_index.loadIndex(ann_index_path)
    query_time_params = {'efSearch': efS}
    ann_index.setQueryTimeParams(query_time_params)

    end_time = datetime.datetime.now()
    total_time = (end_time - start_time)

    print(f'Loading concept ids, vectorizer, linking classifier, tfidf vectors and ann index took {total_time.total_seconds()} seconds')
    return uml_concept_aliases, tfidf_vectorizer, linking_classifier, ann_index


def get_mention_text_and_ids(data: List[data_util.MedMentionExample],
                             umls: Dict[str, Any]):
    missing_entity_ids = []  # entities in MedMentions but not in UMLS

    # don't care about context for now. Just do the processing based on mention text only
    # collect all the data in one list to use ann.knnQueryBatch which is a lot faster than
    # calling ann.knnQuery for each individual example
    mention_texts = []
    gold_umls_ids = []

    for example in data:
        for entity in example.entities:
            if entity.umls_id not in umls:
                missing_entity_ids.append(entity)  # the UMLS release doesn't contan all UMLS concepts
                continue

            mention_texts.append(entity.mention_text)
            gold_umls_ids.append(entity.umls_id)
            continue

    return mention_texts, gold_umls_ids, missing_entity_ids

def get_mention_text_and_ids_by_doc(data: List[data_util.MedMentionExample],
                                    umls: Dict[str, Any]):
    """
    Returns a list of tuples containing a MedMentionExample and the texts and ids contianed in it

    Parameters
    ----------
    data: List[data_util.MedMentionExample]
        A list of MedMentionExamples being evaluated
    umls: Dict[str, Any]
        A dictionary of UMLS concepts
    """
    missing_entity_ids = []  # entities in MedMentions but not in UMLS

    examples_with_labels = []

    for example in data:
        mention_texts = []
        gold_umls_ids = []
        for entity in example.entities:
            if entity.umls_id not in umls:
                missing_entity_ids.append(entity)  # the UMLS release doesn't contan all UMLS concepts
                continue

            mention_texts.append(entity.mention_text)
            gold_umls_ids.append(entity.umls_id)
            continue
        examples_with_labels.append((example, mention_texts, gold_umls_ids))

    return examples_with_labels, missing_entity_ids

def featurizer(example: Dict):
    """Featurize a dictionary of values for the linking classifier."""
    features = []
    features.append(int(example['has_definition']))  # 0 if candidate doesn't have definition, 1 otherwise

    features.append(min(example['cosines']))
    features.append(max(example['cosines']))
    features.append(len(example['cosines']))
    features.append(np.mean(example['cosines']))

    gold_types = set(example['mention_types'])
    candidate_types = set(example['candidate_types'])

    features.append(len(gold_types))
    features.append(len(candidate_types))
    features.append(len(candidate_types.intersection(gold_types)))

    return features

def eval_candidate_generation_and_linking(examples: List[data_util.MedMentionExample],
                                          umls_concept_dict_by_id: Dict[str, Dict],
                                          candidate_generator: CandidateGenerator,
                                          linking_classifier: ClassifierMixin,
                                          k_list: List[int],
                                          thresholds: List[float],
                                          use_gold_mentions: bool,
                                          spacy_model: str):
    """
    Evaluate candidate generation and linking using either gold mentions or spacy mentions.
    The evaluation is done both at the mention level and at the document level. If the evaluation
    is done with spacy mentions at the mention level, a pair is only considered correct if
    both the mention and the entity are exactly correct. This could potentially be relaxed, but this 
    matches the evaluation setup from the MedMentions paper.

    Parameters
    ----------
    examples: List[data_util.MedMentionExample]
        An list of MedMentionExamples being evaluted
    umls_concept_dict_by_id: Dict[str, Dict]
        A dictionary of UMLS concepts
    candidate_generator: CandidateGenerator
        A CandidateGenerator instance for generating linking candidates for mentions
    linking_classifier: ClassifierMixin
        An sklearn classifier
    k_list: List[int]
        A list of k values determining how many candidates are generated
    thresholds: List[float]
        A list of threshold values determining the cutoff score for candidates
    """
    examples_with_text_and_ids, missing_entity_ids = get_mention_text_and_ids_by_doc(examples,
                                                                   umls_concept_dict_by_id)

    if not use_gold_mentions:
        nlp = spacy.load(spacy_model)
        docs = [nlp(example.text) for example in examples]

    for k in k_list:
        for threshold in thresholds:

            entity_correct_links_count = 0  # number of correctly linked entities
            entity_wrong_links_count = 0  # number of wrongly linked entities
            entity_no_links_count = 0  # number of entities that are not linked
            num_candidates = []
            num_filtered_candidates = []

            doc_entity_correct_links_count = 0  # number of correctly linked entities
            doc_entity_missed_count = 0  # number of gold entities missed
            doc_mention_no_links_count = 0  # number of ner mentions that did not have any linking candidates
            doc_num_candidates = []
            doc_num_filtered_candidates = []

            all_golds_per_doc_set = []
            all_golds = []
            all_mentions = []

            classifier_correct_predictions = 0
            classifier_wrong_predictions = 0

            for i, example in tqdm(enumerate(examples), desc="Iterating over examples", total=len(examples)):
                entities = [entity for entity in example.entities if entity.umls_id in umls_concept_dict_by_id]
                gold_umls_ids = [entity.umls_id for entity in entities]
                doc_golds = set(gold_umls_ids)
                doc_candidates = set()

                if use_gold_mentions:
                    mention_texts = [entity.mention_text for entity in entities]
                else:
                    doc = docs[i]
                    ner_entities = [ent for ent in doc.ents]
                    mention_texts = [ent.text for ent in doc.ents]

                batch_candidate_neighbor_ids = candidate_generator.generate_candidates(mention_texts, k)

                filtered_batch_candidate_neighbor_ids = []
                for candidate_neighbor_ids in batch_candidate_neighbor_ids:
                    # Keep only canonical entities for which at least one mention has a score less than the threshold.
                    filtered_ids = {k: v for k, v in candidate_neighbor_ids.items() if any([z[1] <= threshold for z in v])}
                    filtered_batch_candidate_neighbor_ids.append(filtered_ids)
                    num_candidates.append(len(candidate_neighbor_ids))
                    num_filtered_candidates.append(len(filtered_ids))
                    doc_candidates.update(filtered_ids)

                for i, gold_entity in enumerate(entities):
                    if use_gold_mentions:
                        candidates = filtered_batch_candidate_neighbor_ids[i]  # for gold mentions, len(entities) == len(filtered_batch_candidate_neighbor_ids)
                    else:
                        # for each gold entity, search for a corresponding predicted entity that has the same span
                        span_from_doc = doc.char_span(gold_entity.start, gold_entity.end)
                        candidates = {}
                        for j, predicted_entity in enumerate(ner_entities):
                            if predicted_entity == span_from_doc:
                                candidates = filtered_batch_candidate_neighbor_ids[j]
                                break

                    # Evaluating candidate generation
                    if len(candidates) == 0:
                        entity_no_links_count += 1
                    elif gold_entity.umls_id in candidates:
                        entity_correct_links_count += 1
                    else:
                        entity_wrong_links_count += 1

                    # Evaluating linking
                    features = []
                    candidate_ids = list(candidates.keys())
                    for candidate_id in candidate_ids:
                        has_definition = 'definition' in umls_concept_dict_by_id[candidate_id]
                        cosine_scores = [cosine for alias, cosine in candidates[candidate_id]]
                        classifier_example = ({'has_definition': has_definition, 'cosines': cosine_scores,
                                               'mention_types': umls_concept_dict_by_id[gold_entity.umls_id]['types'],
                                               'candidate_types': umls_concept_dict_by_id[candidate_id]['types'],})
                        features.append(featurizer(classifier_example))
                    if len(features) > 0:
                        scores = linking_classifier.predict(features)
                        pred_id = candidate_ids[np.argmax(scores)]
                    else:
                        pred_id = -1
                    if pred_id == gold_entity.umls_id:
                        classifier_correct_predictions += 1
                    else:
                        classifier_wrong_predictions += 1

                # the number of correct entities for a given document is the number of gold entities contained in the candidates
                # produced for that document
                doc_entity_correct_links_count += len(doc_candidates.intersection(doc_golds))
                # the number of incorrect entities for a given document is the number of gold entities not contained in the candidates
                # produced for that document
                doc_entity_missed_count += len(doc_golds - doc_candidates)

                all_golds_per_doc_set += list(doc_golds)
                all_golds += gold_umls_ids
                all_mentions += mention_texts

            print(f'MedMentions entities not in UMLS: {len(missing_entity_ids)}')
            print(f'MedMentions entities found in UMLS: {len(gold_umls_ids)}')
            print(f'K: {k}, Filtered threshold : {threshold}')
            print('Gold concept in candidates: {0:.2f}%'.format(100 * entity_correct_links_count / len(all_golds)))
            print('Gold concept not in candidates: {0:.2f}%'.format(100 * entity_wrong_links_count / len(all_golds)))
            print('Doc level gold concept in candidates: {0:.2f}%'.format(100 * doc_entity_correct_links_count / len(all_golds_per_doc_set)))
            print('Doc level gold concepts missed: {0:.2f}%'.format(100 * doc_entity_missed_count / len(all_golds_per_doc_set)))
            print('Candidate generation failed: {0:.2f}%'.format(100 * entity_no_links_count / len(all_golds)))
            print('Correct mention-level linking: {0:.2f}%'.format(100 * classifier_correct_predictions / (classifier_correct_predictions + classifier_wrong_predictions)))
            print('Mean, std, min, max candidate ids: {0:.2f}%, {1:.2f}%, {2}, {3}'.format(np.mean(num_candidates), np.std(num_candidates), np.min(num_candidates), np.max(num_candidates)))
            print('Mean, std, min, max filtered candidate ids: {0:.2f}%, {1:.2f}%, {2}, {3}'.format(np.mean(num_filtered_candidates), np.std(num_filtered_candidates), np.min(num_filtered_candidates), np.max(num_filtered_candidates)))

def main(medmentions_path: str,
         umls_path: str,
         model_path: str,
         ks: str,
         thresholds,
         use_gold_mentions: bool = False,
         train: bool = False,
         spacy_model: str = ""):

    umls_concept_list = load_umls_kb(umls_path)
    umls_concept_dict_by_id = {c['concept_id']: c for c in umls_concept_list}

    # We need to keep around a map from text to possible canonical ids that they map to.
    text_to_concept_id: Dict[str, Set[str]] = defaultdict(set)

    for concept in umls_concept_list:
        for alias in set(concept["aliases"]).union({concept["canonical_name"]}):
            text_to_concept_id[alias].add(concept["concept_id"])

    if train:
        create_tfidf_ann_index(model_path, text_to_concept_id)
    ann_concept_aliases_list, tfidf_vectorizer, linking_classifier, ann_index = load_tfidf_ann_index(model_path)

    candidate_generator = CandidateGenerator(ann_index, tfidf_vectorizer, ann_concept_aliases_list, text_to_concept_id, False)
    print('Reading MedMentions...')
    train_examples, dev_examples, test_examples = data_util.read_full_med_mentions(medmentions_path,
                                                                                   spacy_format=False)

    k_list = [int(k) for k in ks.split(',')]
    if thresholds is None:
        thresholds = [1.0]
    else:
        thresholds = [float(x) for x in thresholds.split(",")]

    # only evaluate on the dev examples for now because we don't have a trained model
    eval_candidate_generation_and_linking(dev_examples, umls_concept_dict_by_id, candidate_generator, linking_classifier, k_list, thresholds, use_gold_mentions, spacy_model)

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument(
             '--medmentions_path',
             help='Path to the MedMentions dataset.'
     )
     parser.add_argument(
             '--umls_path',
             help='Path to the json UMLS release.'
     )
     parser.add_argument(
             '--model_path',
             help='Path to a directory with tfidf vectorizer and nmslib ann index.'
     )
     parser.add_argument(
             '--ks',
             help='Comma separated list of number of candidates.',
     )
     parser.add_argument(
             '--thresholds',
             default=None,
             help='Comma separated list of threshold values.',
     )
     parser.add_argument(
             '--train',
             action="store_true",
             help='Fit the tfidf vectorizer and create the ANN index.',
     )
     parser.add_argument(
             '--use_gold_mentions',
             action="store_true",
             help="Use gold mentions for evaluation rather than a model's predicted mentions"
     )
     parser.add_argument(
             '--spacy_model',
             default="",
             help="The name of the spacy model to use for evaluation (when not using gold mentions)"
     )

     args = parser.parse_args()
     main(args.medmentions_path, args.umls_path, args.model_path, args.ks, args.thresholds, args.use_gold_mentions, args.train, args.spacy_model)
