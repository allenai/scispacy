"""

Convert a umls release to a json file of concepts.
This script assumes the first row of each RFF file has the column names (from MRFILES.RRF)

"""
import sys
import json
import plac

# preferred definitions (from S2)
DEF_SOURCES_PREFERRED = {'NCI_BRIDG', 'NCI_NCI-GLOSS', 'NCI', 'GO', 'MSH', 'NCI_FDA'}


@plac.annotations(
        output_filename=('name of the output json file', 'positional', None, str),
        concepts_filename=('path of MRCONSO.RRF file for concepts', 'positional', None, str),
        definitions_filename=('name of MRDEF.RRF file for concept definitions', 'positional', None, str),
        types_filename=('name of MRSTY.RRF file for concept types', 'positional', None, str))
def main(output_filename, concepts_filename, definitions_filename, types_filename):

    concept_details = {}  # dictionary of concept_id -> concept_dictionary

    print('Reading concepts ... ')
    with open(concepts_filename) as fin:
        headers = next(fin).strip().split('|')
        for line in fin:
            concept = dict(zip(headers, line.strip().split('|')))

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
        headers = next(fin).strip().split('|')
        for line in fin:
            concept_type = dict(zip(headers, line.strip().split('|')))

            concept = concept_details.get(concept_type['CUI'])
            if concept is not None:
                concept['types'].append(concept_type['TUI'])

    print('Reading definitions ... ')
    with open(definitions_filename) as fin:
        headers = next(fin).strip().split('|')
        for line in fin:
            definition = dict(zip(headers, line.strip().split('|')))

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
