"""
Linking using char-n-gram with approximate nearest neighbors.
"""
import os
import os.path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from typing import List, Dict, Tuple, NamedTuple, Any
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
    ann_concept_id_list: List[str]
        A list of strings, mapping the indices used in the ann_index to canonical UMLS ids.

    """
    def __init__(self,
                 ann_index: FloatIndex,
                 tfidf_vectorizer: TfidfVectorizer,
                 ann_concept_id_list: List[str]) -> None:

        self.ann_index = ann_index
        self.vectorizer = tfidf_vectorizer
        self.ann_concept_id_list = ann_concept_id_list

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
        print(f'Generating candidates for {len(mention_texts)} mentions')
        tfidfs = self.vectorizer.transform(mention_texts)
        start_time = datetime.datetime.now()

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch` that addresses this issue.
        batch_neighbors, batch_distances = self.nmslib_knn_with_zero_vectors(tfidfs, k)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print(f'Finding neighbors took {total_time.total_seconds()} seconds')
        neighbors_by_concept_ids = []
        for neighbors, distances in zip(batch_neighbors, batch_distances):
            if neighbors is None:
                neighbors = []

            if distances is None:
                distances = []
            predicted_umls_concept_ids = defaultdict(list)
            for n, d in zip(neighbors, distances):
                predicted_umls_concept_ids[self.ann_concept_id_list[n]].append(d)
            neighbors_by_concept_ids.append({**predicted_umls_concept_ids})
        return neighbors_by_concept_ids

def create_tfidf_ann_index(model_path: str, umls_concept_list: List[str]) -> None:
    """
    Build tfidf vectorizer and ann index.
    """
    tfidf_vectorizer_path = f'{model_path}/tfidf_vectorizer.joblib'
    ann_index_path = f'{model_path}/nmslib_index.bin'
    tfidf_vectors_path = f'{model_path}/tfidf_vectors_sparse.npz'
    uml_concept_ids_path = f'{model_path}/concept_ids.json'

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
    uml_concept_ids = []
    uml_concept_aliases = []
    print('Collecting aliases ... ')
    for i, concept in enumerate(umls_concept_list):
        concept_id = concept['concept_id']

        # Alias lists for concepts are not unique, so calling set here means
        # we don't duplicate items in the index. In practice this reduces the size
        # of the index by 15%.
        concept_aliases = list(set(concept['aliases'])) + [concept['canonical_name']]

        uml_concept_ids.extend([concept_id] * len(concept_aliases))
        uml_concept_aliases.extend(concept_aliases)

        if i % 1000000 == 0 and i > 0:
            print(f'Processed {i} or {len(umls_concept_list)} concepts')

    uml_concept_ids = np.array(uml_concept_ids)
    uml_concept_aliases = np.array(uml_concept_aliases)
    assert len(uml_concept_ids) == len(uml_concept_aliases)

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
    uml_concept_ids = uml_concept_ids[empty_tfidfs_boolean_flags]
    uml_concept_aliases = uml_concept_aliases[empty_tfidfs_boolean_flags]
    uml_concept_alias_tfidfs = uml_concept_alias_tfidfs[empty_tfidfs_boolean_flags]
    print(deleted_aliases)

    print(f'Saving list of concept ids and tfidfs vectors to {uml_concept_ids_path} and {tfidf_vectors_path}')
    json.dump(uml_concept_ids.tolist(), open(uml_concept_ids_path, "w"))
    scipy.sparse.save_npz(tfidf_vectors_path, uml_concept_alias_tfidfs.astype(np.float16))
    assert len(uml_concept_ids) == len(uml_concept_aliases)
    assert len(uml_concept_ids) == uml_concept_alias_tfidfs.shape[0]

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
    efS = 100
    tfidf_vectorizer_path = f'{model_path}/tfidf_vectorizer.joblib'
    ann_index_path = f'{model_path}/nmslib_index.bin'
    tfidf_vectors_path = f'{model_path}/tfidf_vectors_sparse.npz'
    uml_concept_ids_path = f'{model_path}/concept_ids.json'

    start_time = datetime.datetime.now()
    print(f'Loading list of concepted ids from {uml_concept_ids_path}')
    uml_concept_ids = json.load(open(uml_concept_ids_path))

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
    return uml_concept_ids, tfidf_vectorizer, ann_index


def get_mention_text_and_ids(data: List[data_util.MedMentionsExample],
                             umls: Dict[str, Any]):
    missing_entity_ids = []  # entities in MedMentions but not in UMLS

    # don't care about context for now. Just do the processing based on mention text only
    # collect all the data in one list to use ann.knnQueryBatch which is a lot faster than
    # calling ann.knnQuery for each individual example
    mention_texts = []
    gold_umls_ids = []

    # only loop over the dev examples for now because we don't have a trained model
    for example in data:
        for entity in example.entities:
            if entity.umls_id not in umls:
                missing_entity_ids.append(entity)  # the UMLS release doesn't contan all UMLS concepts
                continue

            mention_texts.append(entity.mention_text)
            gold_umls_ids.append(entity.umls_id)
            continue

    return mention_texts, gold_umls_ids, missing_entity_ids


def main(medmentions_path: str, umls_path: str, model_path: str, ks: str, train: bool = False):

    umls_concept_list = load_umls_kb(umls_path)
    umls_concept_dict_by_id = dict((c['concept_id'], c) for c in umls_concept_list)

    if train:
        create_tfidf_ann_index(model_path, umls_concept_list)
    ann_concept_id_list, tfidf_vectorizer, ann_index = load_tfidf_ann_index(model_path)

    candidate_generator = CandidateGenerator(ann_index, tfidf_vectorizer, ann_concept_id_list)
    print('Reading MedMentions ... ')
    train_examples, dev_examples, test_examples = data_util.read_full_med_mentions(medmentions_path,
                                                                                   spacy_format=False)

    mention_texts, gold_umls_ids, missing_entity_ids = get_mention_text_and_ids(dev_examples,
                                                                                umls_concept_dict_by_id)

    k_list = [int(k) for k in ks.split(',')]
    for k in k_list:
        print(f'for k = {k}')
        entity_correct_links_count = 0  # number of correctly linked entities
        entity_wrong_links_count = 0  # number of wrongly linked entities
        entity_no_links_count = 0  # number of entities that are not linked

        candidate_neighbor_ids = candidate_generator.generate_candidates(mention_texts, k)

        for mention_text, gold_umls_id, candidate_neighbor_ids in zip(mention_texts, gold_umls_ids, candidate_neighbor_ids):
            gold_canonical_name = umls_concept_dict_by_id[gold_umls_id]['canonical_name']
            if len(candidate_neighbor_ids) == 0:
                entity_no_links_count += 1
                # print(f'No candidates. Mention Text: {mention_text}, Canonical Name: {gold_canonical_name}')
            elif gold_umls_id in candidate_neighbor_ids:
                entity_correct_links_count += 1
            else:
                entity_wrong_links_count += 1
                # print(f'Wrong candidates. Mention Text: {mention_text}, Canonical Name: {gold_canonical_name}')


        print(f'MedMentions entities not in UMLS: {len(missing_entity_ids)}')
        print(f'MedMentions entities found in UMLS: {len(gold_umls_ids)}')
        print(f'K: {k}')
        print('Gold concept in candidates: {0:.2f}%'.format(100 * entity_correct_links_count / len(gold_umls_ids)))
        print('Gold concept not in candidates: {0:.2f}%'.format(100 * entity_wrong_links_count / len(gold_umls_ids)))
        print('Candidate generation failed: {0:.2f}%'.format(100 * entity_no_links_count / len(gold_umls_ids)))

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
             '--train',
             action="store_true",
             help='Fit the tfidf vectorizer and create the ANN index.',
     )

     args = parser.parse_args()
     main(args.medmentions_path, args.umls_path, args.model_path, args.ks, args.train)
