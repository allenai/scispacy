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


import scipy
import numpy as np
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
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

    """
    def __init__(self,
                 ann_index: FloatIndex,
                 tfidf_vectorizer: TfidfVectorizer,
                 ann_concept_aliases_list: List[str],
                 mention_to_concept: Dict[str, Set[str]]) -> None:

        self.ann_index = ann_index
        self.vectorizer = tfidf_vectorizer
        self.ann_concept_aliases_list = ann_concept_aliases_list
        self.mention_to_concept = mention_to_concept


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
        # print(f'Number of empty vectors: {empty_vectors_count}')

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
        # print(f'Generating candidates for {len(mention_texts)} mentions')
        tfidfs = self.vectorizer.transform(mention_texts)
        start_time = datetime.datetime.now()

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch` that addresses this issue.
        batch_neighbors, batch_distances = self.nmslib_knn_with_zero_vectors(tfidfs, k)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        # print(f'Finding neighbors took {total_time.total_seconds()} seconds')
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

    print(f'Loading concept ids, vectorizer, tfidf vectors and ann index took {total_time.total_seconds()} seconds')
    return uml_concept_aliases, tfidf_vectorizer, ann_index


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

def eval_spacy_mentions(examples: List[data_util.MedMentionExample],
                        umls_concept_dict_by_id: Dict[str, Dict],
                        candidate_generator: CandidateGenerator,
                        k_list: List[int],
                        thresholds: List[float],
                        spacy_model: str):
    """
    Evaluates candidate generation using mentions produced by a spacy model. This means that an entity is considered
    correct if that entity appears anywhere in the abstract
    """
    nlp = spacy.load(spacy_model)

    # only loop over the dev examples for now because we don't have a trained model
    examples_with_labels, missing_entity_ids = get_mention_text_and_ids_by_doc(examples, umls_concept_dict_by_id)
    for k in k_list:
        for threshold in thresholds:
            entity_correct_links_count = 0  # number of correctly linked entities
            entity_missed_count = 0  # number of gold entities missed
            mention_no_links_count = 0  # number of ner mentions that did not have any linking candidates
            num_candidates = []
            num_filtered_candidates = []

            all_golds = []
            all_mentions = []
            for example, mention_texts, gold_umls_ids in examples_with_labels:
                doc = nlp(example.text)
                ner_mentions = [ent.text for ent in doc.ents]
                doc_candidates = set()
                doc_golds = set(gold_umls_ids)

                # it is possible that a spacy model does not find any entities in an abstract
                if ner_mentions == []:
                    entity_missed_count += len(doc_golds)
                    all_golds += list(doc_golds)
                    continue

                batch_candidate_neighbor_ids = candidate_generator.generate_candidates(ner_mentions, k)

                for ner_mention, candidate_neighbor_ids in zip(ner_mentions, batch_candidate_neighbor_ids):
                    # Keep only canonical entities for which at least one mention has a score less than the threshold.
                    filtered_ids = {k: v for k, v in candidate_neighbor_ids.items() if any([z[1] <= threshold for z in v])}
                    num_candidates.append(len(candidate_neighbor_ids))
                    num_filtered_candidates.append(len(filtered_ids))

                    doc_candidates.update(filtered_ids)

                    if len(filtered_ids) == 0:
                        mention_no_links_count += 1

                # the number of correct entities for a given document is the number of gold entities contained in the candidates
                # produced for that document
                entity_correct_links_count += len(doc_candidates.intersection(doc_golds))
                # the number of incorrect entities for a given document is the number of gold entities not contained in the candidates
                # produced for that document
                entity_missed_count += len(doc_golds - doc_candidates)
                
                all_golds += list(doc_golds)
                all_mentions += ner_mentions

            print(f'MedMentions entities not in UMLS: {len(missing_entity_ids)}')
            print(f'MedMentions entities found in UMLS: {len(all_golds)}')
            print(f'K: {k}, Filtered threshold : {threshold}')
            print('Gold concept in candidates: {0:.2f}%'.format(100 * entity_correct_links_count / len(all_golds)))
            print('Gold concepts missed: {0:.2f}%'.format(100 * entity_missed_count / len(all_golds)))
            print('Candidate generation failed: {0:.2f}%'.format(100 * mention_no_links_count / len(all_mentions)))
            print("Mean, std, min, max candidate ids: ", np.mean(num_candidates), np.std(num_candidates), np.min(num_candidates), np.max(num_candidates))
            print("Mean, std, min, max filtered candidate ids: ", np.mean(num_filtered_candidates), np.std(num_filtered_candidates), np.min(num_filtered_candidates), np.max(num_filtered_candidates))

def eval_gold_mentions(dev_examples: List[data_util.MedMentionExample],
                       umls_concept_dict_by_id: Dict[str, Dict],
                       candidate_generator: CandidateGenerator,
                       k_list: List[int],
                       thresholds: List[float]):
    """
    Evaluate candidate generation using gold mentions. This evaluation is at the mention level.
    """
    # only loop over the dev examples for now because we don't have a trained model
    mention_texts, gold_umls_ids, missing_entity_ids = get_mention_text_and_ids(dev_examples,
                                                                                umls_concept_dict_by_id)

    for k in k_list:
        batch_candidate_neighbor_ids = candidate_generator.generate_candidates(mention_texts, k)
        for threshold in thresholds:

            entity_correct_links_count = 0  # number of correctly linked entities
            entity_wrong_links_count = 0  # number of wrongly linked entities
            entity_no_links_count = 0  # number of entities that are not linked
            num_candidates = []
            num_filtered_candidates = []

            for mention_text, gold_umls_id, candidate_neighbor_ids in zip(mention_texts, gold_umls_ids, batch_candidate_neighbor_ids):
                gold_canonical_name = umls_concept_dict_by_id[gold_umls_id]['canonical_name']

                # Keep only canonical entities for which at least one mention has a score less than the threshold.
                filtered_ids = {k: v for k, v in candidate_neighbor_ids.items() if any([z[1] <= threshold for z in v])}
                num_candidates.append(len(candidate_neighbor_ids))
                num_filtered_candidates.append(len(filtered_ids))

                if len(filtered_ids) == 0:
                    entity_no_links_count += 1
                    # print(f'No candidates. Mention Text: {mention_text}, Canonical Name: {gold_canonical_name}')
                elif gold_umls_id in filtered_ids:
                    entity_correct_links_count += 1
                else:
                    entity_wrong_links_count += 1
                    # print(f'Wrong candidates. Mention Text: {mention_text}, Canonical Name: {gold_canonical_name}')

            print(f'MedMentions entities not in UMLS: {len(missing_entity_ids)}')
            print(f'MedMentions entities found in UMLS: {len(gold_umls_ids)}')
            print(f'K: {k}, Filtered threshold : {threshold}')
            print('Gold concept in candidates: {0:.2f}%'.format(100 * entity_correct_links_count / len(gold_umls_ids)))
            print('Gold concept not in candidates: {0:.2f}%'.format(100 * entity_wrong_links_count / len(gold_umls_ids)))
            print('Candidate generation failed: {0:.2f}%'.format(100 * entity_no_links_count / len(gold_umls_ids)))
            print("Mean, std, min, max candidate ids: ", np.mean(num_candidates), np.std(num_candidates), np.min(num_candidates), np.max(num_candidates))
            print("Mean, std, min, max filtered candidate ids: ", np.mean(num_filtered_candidates), np.std(num_filtered_candidates), np.min(num_filtered_candidates), np.max(num_filtered_candidates))

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
    ann_concept_aliases_list, tfidf_vectorizer, ann_index = load_tfidf_ann_index(model_path)

    candidate_generator = CandidateGenerator(ann_index, tfidf_vectorizer, ann_concept_aliases_list, text_to_concept_id)
    print('Reading MedMentions ... ')
    train_examples, dev_examples, test_examples = data_util.read_full_med_mentions(medmentions_path,
                                                                                   spacy_format=False)

    k_list = [int(k) for k in ks.split(',')]
    if thresholds is None:
        thresholds = [1.0]
    else:
        thresholds = [float(x) for x in thresholds.split(",")]

    if use_gold_mentions:
        eval_gold_mentions(dev_examples, umls_concept_dict_by_id, candidate_generator, k_list, thresholds)
    else:
        eval_spacy_mentions(dev_examples, umls_concept_dict_by_id, candidate_generator, k_list, thresholds, spacy_model)

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
