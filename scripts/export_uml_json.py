"""

Convert a umls release to a json file of concepts.

"""
import json
import argparse

# preferred definitions (from S2)
DEF_SOURCES_PREFERRED = {'NCI_BRIDG', 'NCI_NCI-GLOSS', 'NCI', 'GO', 'MSH', 'NCI_FDA'}

def main(meta_path, output_path):

    concepts_filename = '{}/MRCONSO.RRF'.format(meta_path)
    types_filename = '{}/MRSTY.RRF'.format(meta_path)
    definitions_filename = '{}/MRDEF.RRF'.format(meta_path)
    file_descriptors = '{}/MRFILES.RRF'.format(meta_path)  # to get column names

    concepts_header = types_header = definitions_header = None
    with open(file_descriptors) as fin:
        for line in fin:
            splits = line.split('|')
            filename = splits[0]
            column_names = (splits[2] + ',').split(',')  # ugly hack because all files end with an empty column
            if filename == 'MRCONSO.RRF':
                concepts_header = column_names
                print('Headers for MRCONSO.RRF: {}'.format(concepts_header))
            elif filename == 'MRSTY.RRF':
                types_header = column_names
                print('Headers for MRSTY.RRF: {}'.format(types_header))
            elif filename == 'MRDEF.RRF':
                definitions_header = column_names
                print('Headers for MRDEF.RRF: {}'.format(definitions_header))

    if concepts_header is None or  types_header is None or definitions_header is None:
        print('ERROR .. column names are not found')
        exit()

    concept_details = {}  # dictionary of concept_id -> concept_dictionary

    print('Reading concepts ... ')
    with open(concepts_filename) as fin:
        headers = concepts_header
        for line in fin:
            splits = line.strip().split('|')
            assert len(headers) == len(splits)
            concept = dict(zip(headers, splits))
            if concept['LAT'] != 'ENG' or concept['SUPPRESS'] != 'N':
                continue  # Keep English non-suppressed concepts only

            concept_id = concept['CUI']
            if concept_id not in concept_details:  # a new concept
                # add it to the dictionary with an empty list of aliases and types
                concept_details[concept_id] = {'concept_id': concept_id, 'aliases': [], 'types': []}

            concept_name = concept['STR']
            # this condition is copied from S2. It checks if the concept name is canonical or not
            is_canonical = concept['ISPREF'] == 'Y' and concept['TS'] == 'P' and concept['STT'] == 'PF'

            if not is_canonical or 'canonical_name' in concept_details[concept_id]:
                # not a canonical name or a canonical name already found
                concept_details[concept_id]['aliases'].append(concept_name)  # add it as an alias
            else:
                concept_details[concept_id]['canonical_name'] = concept_name  # set as canonical name

    print('Reading types ... ')
    with open(types_filename) as fin:
        headers = types_header
        for line in fin:
            splits = line.strip().split('|')
            assert len(headers) == len(splits)
            concept_type = dict(zip(headers, splits))

            concept = concept_details.get(concept_type['CUI'])
            if concept is not None:
                concept['types'].append(concept_type['TUI'])

    print('Reading definitions ... ')
    with open(definitions_filename) as fin:
        headers = definitions_header
        for line in fin:
            splits = line.strip().split('|')
            assert len(headers) == len(splits)
            definition = dict(zip(headers, splits))

            if definition['SUPPRESS'] != 'N':
                continue
            is_from_preferred_source = definition['SAB'] in DEF_SOURCES_PREFERRED
            concept = concept_details.get(definition['CUI'])
            if concept is None:
                continue

            if 'definition' not in concept or  \
                is_from_preferred_source and concept['is_from_preferred_source'] == 'N':
                concept['definition'] = definition['DEF']
                concept['is_from_preferred_source'] = 'Y' if is_from_preferred_source else 'N'

    without_canonical_name_count = 0
    without_aliases_count = 0
    with_one_alias_count = 0
    with_more_than_one_alias_count = 0
    without_type_count = 0
    with_one_type_count = 0
    with_more_than_one_type_count = 0
    without_definition_count = 0
    with_definition_pref_source_count = 0
    with_definition_other_sources_count = 0
    for concept in concept_details.values():
        without_canonical_name_count += 1 if 'canonical_name' not in concept else 0
        without_aliases_count += 1 if len(concept['aliases']) == 0 else 0
        with_one_alias_count += 1 if len(concept['aliases']) == 1 else 0
        with_more_than_one_alias_count += 1 if len(concept['aliases']) > 1 else 0
        without_type_count += 1 if len(concept['types']) == 0 else 0
        with_one_type_count += 1 if len(concept['types']) == 1 else 0
        with_more_than_one_type_count += 1 if len(concept['types']) >= 1 else 0
        without_definition_count += 1 if 'definition' not in concept else 0
        with_definition_pref_source_count += 1 if concept.get('is_from_preferred_source') == 'Y' else 0
        with_definition_other_sources_count += 1 if concept.get('is_from_preferred_source') == 'N' else 0

    print(f'Number of concepts: {len(concept_details)}')
    print(f'Number of concepts without canonical name (one of the aliases will be used instead): '
          f'{without_canonical_name_count}')
    print(f'Number of concepts with no aliases: {without_aliases_count}')
    print(f'Number of concepts with 1 alias: {with_one_alias_count}')
    print(f'Number of concepts with > 1 alias: {with_more_than_one_alias_count}')
    print(f'Number of concepts with no type: {without_type_count}')
    print(f'Number of concepts with 1 type: {with_one_type_count}')
    print(f'Number of concepts with > 1 type: {with_more_than_one_type_count}')
    print(f'Number of concepts with no definition: {without_definition_count}')
    print(f'Number of concepts with definition from preferred sources: {with_definition_pref_source_count}')
    print(f'Number of concepts with definition from other sources: {with_definition_other_sources_count}')

    print('Deleting unused fields and choosing a canonical name from aliases ... ')
    for concept in concept_details.values():
        # if a concept doesn't have a canonical name, use the first alias instead
        if 'canonical_name' not in concept:
            aliases = concept['aliases']
            concept['canonical_name'] = aliases[0]
            del aliases[0]

        # deleting `is_from_preferred_source`
        if 'is_from_preferred_source' in concept:
            del concept['is_from_preferred_source']

    print('Exporting to the a json file {} ...'.format(output_path))
    with open(output_path, 'w') as fout:
        json.dump(list(concept_details.values()), fout)

    print('DONE.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--meta_path',
        help="Path to the META directory of an UMLS release."
    )
    parser.add_argument(
        '--output_path',
        help="Path to the output json file"
    )
    args = parser.parse_args()
    main(args.meta_path, args.output_path)
