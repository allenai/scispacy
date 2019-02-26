"""
Linking based on simple text normalization and exact matching
"""

import json
import argparse
from collections import defaultdict
from scispacy import data_util

def normalize(text):
    return " ".join(text.lower().split(" "))

def linking(entity: data_util.MedMentionEntity, umls_concept_dict_by_name):
    """
    Links the entity mention and returns umls concept
    """
    normalized_mention = normalize(entity.mention_text)
    concept_candidates = umls_concept_dict_by_name.get(normalized_mention)
    return None if concept_candidates is None else concept_candidates[0]

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
        for concept in concepts:
            # concept id -> concept object  (UMLS calls them concepts, MedMentions calls them entities)
            umls_concept_dict_by_id[concept['concept_id']] = concept
            # normalized entity name -> entity object  (useful for exact matching)
            umls_concept_dict_by_name[normalize(concept['canonical_name'])].append(concept)
    print(f'Number of umls concepts: {len(umls_concept_dict_by_id)}')
    print(f'Number of unique names: {len(umls_concept_dict_by_name)}')
    return umls_concept_dict_by_id, umls_concept_dict_by_name

def main(medmentions_path: str, umls_path: str):

    umls_concept_dict_by_id, umls_concept_dict_by_name = prepare_umls_indices(umls_path)

    print('Reading corpus ... ')
    train_examples, dev_examples, test_examples = data_util.read_full_med_mentions(medmentions_path,
                                                                                   spacy_format=False)

    missing_entity_ids = []  # entities in MedMentions but not in UMLS
    found_entity_ids = []  # entities in MedMentions and in UMLS
    entity_correct_links_count = 0  # number of correctly linked entities
    entity_wrong_links_count = 0  # number of wrongly linked entities
    entity_no_links_count = 0  # number of entities that are not linked
    print('Linking ... ')
    for example in dev_examples:  # only loop over the dev examples for now because we don't have a trained model
        for entity in example.entities:
            if entity.umls_id not in umls_concept_dict_by_id:
                missing_entity_ids.append(entity)
                continue
            found_entity_ids.append(entity)

            predicted_umls_concept = linking(entity, umls_concept_dict_by_name)
            if predicted_umls_concept is None:
                entity_no_links_count += 1
            elif predicted_umls_concept['concept_id'] == entity.umls_id:
                entity_correct_links_count += 1
            else:
                entity_wrong_links_count += 1

    print(f'MedMentions entities not in UMLS: {len(missing_entity_ids)}')
    print(f'MedMentions entities found in UMLS: {len(found_entity_ids)}')
    print('Correct linking: {0:.2f}%'.format(100 * entity_correct_links_count / len(found_entity_ids)))
    print('Wrong linking: {0:.2f}%'.format(100 * entity_wrong_links_count / len(found_entity_ids)))
    print('No linking: {0:.2f}%'.format(100 * entity_no_links_count / len(found_entity_ids)))

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
    args = parser.parse_args()
    main(args.medmentions_path, args.umls_path)
