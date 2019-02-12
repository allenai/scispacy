"""

Convert a umls release to a json file of concepts.

"""
import sys
import json
import plac

# preferred definitions (from S2)
DEF_SOURCES_PREFERRED = {'NCI_BRIDG', 'NCI_NCI-GLOSS', 'NCI', 'GO', 'MSH', 'NCI_FDA'}


@plac.annotations(
        output_filename=('name of the output json file', 'positional', None, str),
        umls_meta_directory=('path of the META directory', 'positional', None, str))
def main(output_filename, umls_meta_directory):

    concepts_filename = '{}/MRCONSO.RRF'.format(umls_meta_directory)
    types_filename = '{}/MRSTY.RRF'.format(umls_meta_directory)
    definitions_filename = '{}/MRDEF.RRF'.format(umls_meta_directory)
    file_descriptors = '{}/MRFILES.RRF'.format(umls_meta_directory)  # to get column names

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

            # this condition is copied from S2
            is_canonical = concept['ISPREF'] == 'Y' and concept['TS'] == 'P' and concept['STT'] == 'PF'

            concept_id = concept['CUI']
            if concept_id not in concept_details:
                concept_details[concept_id] = {'concept_id': concept_id, 'aliases': list(), 'types': list()}
            concept_name = concept['STR']

            if not is_canonical or 'canonical_name' in concept_details[concept_id]:
                # not a canonical name or a canonical name already found
                concept_details[concept_id]['aliases'].append(concept_name)  # add it as an alisase
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

    print('Number of concepts: {}'.format(len(concept_details)))
    print('Number of concepts without canonical name (one of the aliases will be used instead): {}'.format(len(
            [1 for c in concept_details.values() if 'canonical_name' not in c])))
    print('Number of concepts with no aliases: {}'.format(len(
            [1 for c in concept_details.values() if len(c['aliases']) == 0])))
    print('Number of concepts with 1 aliase: {}'.format(len(
            [1 for c in concept_details.values() if len(c['aliases']) == 1])))
    print('Number of concepts with > 1 aliase: {}'.format(len(
            [1 for c in concept_details.values() if len(c['aliases']) > 1])))
    print('Number of concepts with no type: {}'.format(len(
            [1 for c in concept_details.values() if len(c['types']) == 0])))
    print('Number of concepts with 1 type: {}'.format(len(
            [1 for c in concept_details.values() if len(c['types']) == 1])))
    print('Number of concepts with > 1 type: {}'.format(len(
            [1 for c in concept_details.values() if len(c['types']) > 1])))
    print('Number of concepts with no definition: {}'.format(len(
            [1 for c in concept_details.values() if 'definition' not in c])))
    print('Number of concepts with definition from preferred sources: {}'.format(len(
            [1 for c in concept_details.values()
             if 'is_from_preferred_source' in c and c['is_from_preferred_source'] == 'Y'])))
    print('Number of concepts with definition from other sources: {}'.format(len(
            [1 for c in concept_details.values()
             if 'is_from_preferred_source' in c and c['is_from_preferred_source'] == 'N'])))

    print('Deleting unused fields and choosing a canonical name from aliases ... ')
    for concept in concept_details.values():
        # if a concept doesn't have a canonical name, use the first aliase instead
        if 'canonical_name' not in concept:
            aliases = concept['aliases']
            concept['canonical_name'] = aliases[0]
            del aliases[0]

        # deleting `is_from_preferred_source`
        if 'is_from_preferred_source' in concept:
            del concept['is_from_preferred_source']

    print('Exporting to the a json file {} ...'.format(output_filename))
    with open(output_filename, 'w') as fout:
        json.dump(list(concept_details.values()), fout)

    print('DONE.')


plac.call(main, sys.argv[1:])
