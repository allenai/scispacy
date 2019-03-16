"""
Linking based on simple text normalization and exact matching
"""

import json
import argparse
from collections import defaultdict
from scispacy import data_util
import re
import string
import datetime
import spacy
# nlp = spacy.load('en_core_sci_sm', disable=['ner', 'tagger', 'parser'])
# regex_split = re.compile('[%s]' % re.escape(string.punctuation))
# regex_remove_suffix = re.compile('ed$|al$|ment$|ing$|s$|est$|er$|tion$')
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

def get_char_ngrams(text, n):
    return [text[i:i+n] for i in range(len(text)-n+1)]

def normalize(text):
    # return [t.lemma_ for t in nlp(text)]
    # return [regex_remove_suffix.sub('', t) for t in regex_split.sub(' ', text).lower().split()]
    text = text.lower()
    return get_char_ngrams(text, 4)

def linking(entity: data_util.MedMentionEntity, umls_concept_dict_by_name):
    """
    Links the entity mention and returns umls concept
    """
    words = normalize(entity.mention_text)
    concept_candidates  = []
    for word in words:
        some_candidates = umls_concept_dict_by_name.get(word)
        if some_candidates:
            concept_candidates.extend(some_candidates)
    return set(concept_candidates)

def prepare_umls_indices(umls_path: str):
    """
    Returns two indices, one by entity id, and one by concept name.
    """
    umls_concept_dict_by_id = {}
    umls_concept_dict_by_name = defaultdict(list)
    with open(umls_path) as f:
        print('Loading umls concepts ... ')
        concepts = json.load(f)
        print('Building indices ... ')
        for i, concept in enumerate(concepts):
            # concept id -> concept object  (UMLS calls them concepts, MedMentions calls them entities)
            umls_concept_dict_by_id[concept['concept_id']] = concept
            # normalized entity name -> entity object  (useful for exact matching)
            words = normalize(concept['canonical_name'])
            for word in words:
                umls_concept_dict_by_name[word].append(concept['concept_id'])
            if i % 100000 == 0:
                print(f'Processed {i} or {len(concepts)} concepts')
    print(f'Number of umls concepts: {len(umls_concept_dict_by_id)}')
    print(f'Number of unique names: {len(umls_concept_dict_by_name)}')
    return umls_concept_dict_by_id, umls_concept_dict_by_name

def prepare_tfidf_nn(umls_path: str, k: int):
    """
    Returns two indices, one by entity id, and one by concept name.
    """
    """
    Returns two indices, one by entity id, and one by concept name.
    """
    umls_concept_dict_by_id = {}
    kb = []
    canonical_names = []
    with open(umls_path) as f:
        print('Loading umls concepts ... ')
        concepts = json.load(f)
        print('Building indices ... ')
        for i, concept in enumerate(concepts):
            # concept id -> concept object  (UMLS calls them concepts, MedMentions calls them entities)
            umls_concept_dict_by_id[concept['concept_id']] = concept
            kb.append(concept)
            canonical_names.append(concept['canonical_name'])
            if i % 100000 == 0:
                print(f'Processed {i} or {len(concepts)} concepts')
    tfidf_vec = TfidfVectorizer(analyzer='char', ngram_range=(4,4))
    print('Fitting tfidf vectorizer')
    canonical_names_tfidf = tfidf_vec.fit_transform(canonical_names)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=10)
    print('Fitting nn')
    nn.fit(canonical_names_tfidf)
    print(f'Number of umls concepts: {len(umls_concept_dict_by_id)}')
    print(f'Shape of tfidf: {canonical_names_tfidf.shape}')
    return umls_concept_dict_by_id, kb, tfidf_vec, nn


def main(medmentions_path: str, umls_path: str, k: int):

    # umls_concept_dict_by_id, umls_concept_dict_by_name = prepare_umls_indices(umls_path)
    umls_concept_dict_by_id, kb, tfidf_vec, nn = prepare_tfidf_nn(umls_path, k)

    print('Reading corpus ... ')
    train_examples, dev_examples, test_examples = data_util.read_full_med_mentions(medmentions_path,
                                                                                   spacy_format=False)

    missing_entity_ids = []  # entities in MedMentions but not in UMLS
    found_entity_ids = []  # entities in MedMentions and in UMLS
    entity_correct_links_count = 0  # number of correctly linked entities
    entity_wrong_links_count = 0  # number of wrongly linked entities
    entity_no_links_count = 0  # number of entities that are not linked
    candidate_list_len = []
    print('Linking ... ')
    for i, example in enumerate(dev_examples):  # only loop over the dev examples for now because we don't have a trained model
        for entity in example.entities:
            if entity.umls_id not in umls_concept_dict_by_id:
                missing_entity_ids.append(entity)
                continue
            found_entity_ids.append(entity)

            # predicted_umls_concept_ids = linking(entity, umls_concept_dict_by_name)
            tfidf = tfidf_vec.transform([entity.mention_text])
            neighbors = nn.kneighbors(tfidf)
            predicted_umls_concept_ids = set([kb[x]['concept_id'] for x in neighbors[1][0]])
            # candidate_list_len.append(len(predicted_umls_concept_ids))
            if len(predicted_umls_concept_ids) == 0:
                entity_no_links_count += 1
                print(entity.mention_text, " ===> ", umls_concept_dict_by_id[entity.umls_id]['canonical_name'])
            elif entity.umls_id in predicted_umls_concept_ids:
                entity_correct_links_count += 1
            else:
                entity_wrong_links_count += 1
                print(entity.mention_text, " ===> ", umls_concept_dict_by_id[entity.umls_id]['canonical_name'])
        if i % 5 == 0:
            print(datetime.datetime.now())
            print(f' >>>>>>>>>>>>>>>>>>>>  Processed {i} of {len(dev_examples)} examples <<<<<<<<<<<<<<<<<<<<')
            print('Correct linking: {0:.2f}%'.format(100 * entity_correct_links_count / len(found_entity_ids)))
            print('Wrong linking: {0:.2f}%'.format(100 * entity_wrong_links_count / len(found_entity_ids)))
            print('No linking: {0:.2f}%'.format(100 * entity_no_links_count / len(found_entity_ids)))

    print(f'MedMentions entities not in UMLS: {len(missing_entity_ids)}')
    print(f'MedMentions entities found in UMLS: {len(found_entity_ids)}')
    print('Correct linking: {0:.2f}%'.format(100 * entity_correct_links_count / len(found_entity_ids)))
    print('Wrong linking: {0:.2f}%'.format(100 * entity_wrong_links_count / len(found_entity_ids)))
    print('No linking: {0:.2f}%'.format(100 * entity_no_links_count / len(found_entity_ids)))
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--medmentions_path',
            help="Path to the MedMentions dataset."
    )
    parser.add_argument(
            '--umls_path',
            help="Path to the json UMLS release."
    )
    parser.add_argument(
            '--k',
            help="Number of candidates.",
            type=int
    )
    args = parser.parse_args()
    main(args.medmentions_path, args.umls_path, args.k)
