"""
Linking using char-n-gram with approximate nearest neighbors.
"""
from typing import List, Dict, Tuple
import json
import argparse
import os.path
import datetime
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

def nmslis_knn_with_zero_vectors(vectors: np.ndarray, k: int, ann_index: FloatIndex) -> List[List]:
    """ ann_index.knnQueryBatch crashes if any of the vectors is all zeros.
    This function is a wrapper around `ann_index.knnQueryBatch` that solves this problem. It works as follows:
    - remove empty vectors from `vectors`
    - call `ann_index.knnQueryBatch` with the non-empty vectors only. This returns `neighbors`,
    a list of list of neighbors. `len(neighbors)` equals the length of the non-empty vectors.
    - extend the list `neighbors` with `None`s in place of empty vectors.
    - return the extended list of neighbors
    """
    empty_vectors_boolean_flags = np.array(vectors.sum(axis=1) != 0).reshape(-1,)
    empty_vectors_count = vectors.shape[0] - sum(empty_vectors_boolean_flags)
    print(f'Number of empty vectors: {empty_vectors_count}')

    # remove empty vectors before calling `ann_index.knnQueryBatch`
    vectors = vectors[empty_vectors_boolean_flags]

    # call `knnQueryBatch` to get neighbors
    neighbors = [x[0].tolist() for x in ann_index.knnQueryBatch(vectors, k=k)]

    # all an empty list in place for each empty vector to make sure len(extended_neighbors) == len(vectors)

    # init extended_neighbors with a list of Nones
    extended_neighbors = np.empty((len(empty_vectors_boolean_flags),), dtype=object)

    # neighbors need to be convected to an np.array of objects instead of ndarray of dimensions len(vectors)xk
    # Solution: add a row to `neighbors` with any length other than k. This way, calling np.array(neighbors)
    # returns an np.array of objects
    neighbors.append([])
    # interleave `neighbors` and Nones in `extended_neighbors`
    extended_neighbors[empty_vectors_boolean_flags] = np.array(neighbors)[:-1]

    return extended_neighbors

def generate_candidates(mention_texts: List[str], k: int, tfidf_vectorizer: TfidfVectorizer,
                        ann_index: FloatIndex, ann_concept_id_list: List[int]) -> List[List[int]]:
    """Given a list of mention texts, returns a list of candidate neighbors
    args:
        mention_texts: list of mention texts
        k: number of ann neighbors
        tfidf_vectorizer: text to vector
        ann_index: approximate nearest neighbor index
        ann_concept_id_list: a list of concept ids that the ann_index is referencing
    """
    print(f'Generating candidates for {len(mention_texts)} mentions')
    tfidfs = tfidf_vectorizer.transform(mention_texts)
    start_time = datetime.datetime.now()

    # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
    # `nmslis_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch` that addresses this issue.
    neighbors = nmslis_knn_with_zero_vectors(tfidfs, k, ann_index)
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f'Finding neighbors took {total_time.total_seconds()} seconds')
    neighbors_by_concept_ids = []
    for n in neighbors:
        if n is None:
            n = []
        predicted_umls_concept_ids = set([ann_concept_id_list[x] for x in n])
        neighbors_by_concept_ids.append(predicted_umls_concept_ids)
    return neighbors_by_concept_ids

def create_load_tfidf_ann_index(ann_index_path: str, tfidf_vectorizer_path: str,
                                umls_concept_list: List) -> Tuple[List[int], TfidfVectorizer, FloatIndex]:
    """
    Build or load tfidf vectorizer and ann index
    """
    uml_concept_ids = []
    uml_concept_aliases = []
    print('Collecting aliases ... ')
    for i, concept in enumerate(umls_concept_list):
        concept_id = concept['concept_id']
        concept_aliases = concept['aliases'] + [concept['canonical_name']]

        # TODO: remove the following line and rebuild index
        concept_aliases = [a if len(a) >= 2 else f'{a}00' for a in concept_aliases]

        uml_concept_ids.extend([concept_id] * len(concept_aliases))
        uml_concept_aliases.extend(concept_aliases)

        if i % 1000000 == 0 and i > 0:
            print(f'Processed {i} or {len(umls_concept_list)} concepts')

    uml_concept_ids = np.array(uml_concept_ids)
    uml_concept_aliases = np.array(uml_concept_aliases)
    assert len(uml_concept_ids) == len(uml_concept_aliases)

    if not os.path.isfile(tfidf_vectorizer_path):
        print(f'No tfidf vectorizer on {tfidf_vectorizer_path}')
        print(f'Fitting tfidf vectorizer on {len(uml_concept_aliases)} aliases')
        tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4))
        tfidf_vectorizer.fit(uml_concept_aliases)
        print(f'Saving tfidf vectorizer to {tfidf_vectorizer_path}')
        dump(tfidf_vectorizer, tfidf_vectorizer_path)
    print(f'Loading tfidf vectorizer from {tfidf_vectorizer_path}')
    tfidf_vectorizer = load(tfidf_vectorizer_path)
    print(f'Vectorizing aliases ... ')
    start_time = datetime.datetime.now()
    uml_concept_alias_tfidfs = tfidf_vectorizer.transform(uml_concept_aliases)
    end_time = datetime.datetime.now()
    total_time = (end_time - start_time)
    print(f'Vectorizing aliases took {total_time.total_seconds()} seconds')

    # find empty (all zeros) tfidf vectors
    empty_tfidfs_boolean_flags = np.array(uml_concept_alias_tfidfs.sum(axis=1) != 0).reshape(-1,)
    deleted_aliases = uml_concept_aliases[empty_tfidfs_boolean_flags == False]
    number_of_non_empty_tfidfs = len(deleted_aliases)
    total_number_of_tfidfs = uml_concept_alias_tfidfs.shape[0]
    print(f'Deleting {number_of_non_empty_tfidfs}/{total_number_of_tfidfs} aliases because their tfidf is empty')

    # remove empty tfidf vectors, otherwise nmslib will crashd
    uml_concept_ids = uml_concept_ids[empty_tfidfs_boolean_flags]
    uml_concept_aliases = uml_concept_aliases[empty_tfidfs_boolean_flags]
    uml_concept_alias_tfidfs = uml_concept_alias_tfidfs[empty_tfidfs_boolean_flags]
    print(deleted_aliases)
    assert len(uml_concept_ids) == len(uml_concept_aliases)
    assert len(uml_concept_ids) == uml_concept_alias_tfidfs.shape[0]

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

    if not os.path.isfile(ann_index_path):
        print(f'No ann index on {ann_index_path}')
        print(f'Fitting ann index on {len(uml_concept_aliases)} aliases (takes 2 hours)')

        start_time = datetime.datetime.now()
        ann_index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
        ann_index.addDataPointBatch(uml_concept_alias_tfidfs)
        ann_index.createIndex(index_params, print_progress=True)
        ann_index.saveIndex(ann_index_path)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f'Fitting ann index took {elapsed_time.total_seconds()} seconds')

    print(f'Loading ann index from {ann_index_path}')
    ann_index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
    ann_index.addDataPointBatch(uml_concept_alias_tfidfs)
    ann_index.loadIndex(ann_index_path)
    query_time_params = {'efSearch': efS}
    ann_index.setQueryTimeParams(query_time_params)

    return uml_concept_ids, tfidf_vectorizer, ann_index

def main(medmentions_path: str, umls_path: str, ann_index_path: str, tfidf_vectorizer_path: str, ks: str):

    umls_concept_list = load_umls_kb(umls_path)
    umls_concept_dict_by_id = dict((c['concept_id'], c) for c in umls_concept_list)

    ann_concept_id_list, tfidf_vectorizer, ann_index = \
            create_load_tfidf_ann_index(ann_index_path, tfidf_vectorizer_path, umls_concept_list)

    print('Reading MedMentions ... ')
    train_examples, dev_examples, test_examples = data_util.read_full_med_mentions(medmentions_path,
                                                                                   spacy_format=False)

    missing_entity_ids = []  # entities in MedMentions but not in UMLS
    found_entity_ids = []  # entities in MedMentions and in UMLS

    # don't care about context for now. Just do the processing based on mention text only
    # collect all the data in one list to use ann.knnQueryBatch which is a lot faster than
    # calling ann.knnQuery for each individual example
    mention_texts = []
    gold_umls_ids = []

    # only loop over the dev examples for now because we don't have a trained model
    for example in dev_examples:
        for entity in example.entities:
            if entity.umls_id not in umls_concept_dict_by_id:
                missing_entity_ids.append(entity)  # the UMLS release doesn't contan all UMLS concepts
                continue
            found_entity_ids.append(entity)

            mention_texts.append(entity.mention_text)
            gold_umls_ids.append(entity.umls_id)
            continue

    k_list = [int(k) for k in ks.split(',')]
    for k in k_list:
        print(f'for k = {k}')
        entity_correct_links_count = 0  # number of correctly linked entities
        entity_wrong_links_count = 0  # number of wrongly linked entities
        entity_no_links_count = 0  # number of entities that are not linked

        candidate_neighbor_ids = generate_candidates(mention_texts, k, tfidf_vectorizer, ann_index, ann_concept_id_list)

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
        print(f'MedMentions entities found in UMLS: {len(found_entity_ids)}')
        print(f'K: {k}')
        print('Gold concept in candidates: {0:.2f}%'.format(100 * entity_correct_links_count / len(found_entity_ids)))
        print('Gold concept not in candidates: {0:.2f}%'.format(100 * entity_wrong_links_count / len(found_entity_ids)))
        print('Candidate generation failed: {0:.2f}%'.format(100 * entity_no_links_count / len(found_entity_ids)))

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
            '--ann_index_path',
            help='Path to the nmslib ann index.'
    )
    parser.add_argument(
            '--tfidf_vectorizer_path',
            help='Path to sklearn tfidf char-ngram vectorizer.'
    )
    parser.add_argument(
            '--ks',
            help='Comma separated list of number of candidates.',
    )

    args = parser.parse_args()
    main(args.medmentions_path, args.umls_path, args.ann_index_path, args.tfidf_vectorizer_path, args.ks)
