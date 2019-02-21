"""
Linking based on simple text normalization and exact matching
"""

import json
import argparse
from collections import defaultdict

def normalize(text):
    normalized = " ".join(sorted(text.lower().split(" ")))
    return normalized

def main(medmentions_path, umls_path):

    # reading entity information from umls
    entities_dict_by_id = {}
    entities_dict_by_name = defaultdict(set)
    with open(umls_path) as f:
        print('loading entities')
        entities = json.load(f)
        print('building indices')
        for entity in entities:
            # entity id -> entity object  (UMLS calls them concepts, and MedMentions calls them entities)
            entities_dict_by_id[entity['concept_id']] = entity
            # normalized entity name -> entity object  (useful for exact matching)
            entities_dict_by_name[normalize(entity['canonical_name'])].add(entity['concept_id'])
    print("entities_dict_by_id", len(entities_dict_by_id))
    print("entities_dict_by_name", len(entities_dict_by_name))

    # abstract ids
    train_ids = set()
    dev_ids = set()
    test_ids = set()
    print('reading train/dev/test abstract splits of MedMentions')
    with open(f'{medmentions_path}/corpus_pubtator_pmids_trng.txt') as f:
        for line in f:
            train_ids.add(line.strip())
    with open(f'{medmentions_path}/corpus_pubtator_pmids_dev.txt') as f:
        for line in f:
            dev_ids.add(line.strip())
    with open(f'{medmentions_path}/corpus_pubtator_pmids_test.txt') as f:
        for line in f:
            test_ids.add(line.strip())

    missing_entity_ids = []  # entities in MedMentions but not in UMLS
    found_entity_ids = []  # entities in MedMentions and in UMLS
    entity_correct_links_count = 0  # number of correctly linked entities
    entity_wrong_links_count = 0  # number of wrongly linked entities
    entity_no_links_count = 0  # number of entities that are not linked
    print('reading corpus')

    # This file is mostly a tsv. Each line is the title, abstract or an entity mention. 
    # Entity mention lines have the following format:
    # abstract id, char start offset, char end offset, entity mention, entity type, entity id
    with open(f'{medmentions_path}/corpus_pubtator.txt') as f:
        for line in f:
            if line.strip() == "":
                continue  # empty line is the start of a new paper

            splits = line.strip().split('\t')
            if len(splits) == 1:
                continue  # title or abstract, not used for now

            assert len(splits) == 6
            paper_id, _, _, mention, gold_entity_type, gold_entity_id = splits

            if paper_id not in dev_ids:  # run only on dev for now 
                continue

            if gold_entity_id not in entities_dict_by_id:
                missing_entity_ids.append((mention, gold_entity_type, gold_entity_id))
                continue

            found_entity_ids.append((mention, gold_entity_type, gold_entity_id))

            # exact match linking
            normalized_mention = normalize(mention)
            if normalized_mention in entities_dict_by_name:
                if gold_entity_id in entities_dict_by_name[normalized_mention]:
                    entity_correct_links_count += 1
                else:
                    entity_wrong_links_count += 1
            else:
                entity_no_links_count += 1

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