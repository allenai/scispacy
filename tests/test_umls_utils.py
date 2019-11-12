import unittest

from scispacy import umls_utils

class TestUtil(unittest.TestCase):

    expected_concepts = [
            {'concept_id': 'C0000005', 'canonical_name': '(131)I-Macroaggregated Albumin',
             'types': ['T116'], 'aliases': ['(131)I-MAA']},
            {'concept_id': 'C0000039', 'aliases': ['1,2-Dipalmitoylphosphatidylcholine'],
             'types': ['T109', 'T121'], 'definition':
             'Synthetic phospholipid used in liposomes and lipid bilayers to study biological membranes.'}
    ]

    def test_read_umls_concepts(self):
        meta_path = 'tests/fixtures/umls_META'
        concept_details = {}
        umls_utils.read_umls_concepts(meta_path, concept_details)
        assert len(self.expected_concepts) == len(concept_details)

        for expected_concept in self.expected_concepts:
            assert expected_concept['concept_id'] in concept_details
            concept = concept_details[expected_concept['concept_id']]
            if 'canonical_name' in expected_concept:
                assert concept['canonical_name'] == expected_concept['canonical_name']
            assert concept['aliases'] == expected_concept['aliases']

    def test_read_umls_types(self):
        meta_path = 'tests/fixtures/umls_META'
        concept_details = {}
        umls_utils.read_umls_concepts(meta_path, concept_details)
        umls_utils.read_umls_types(meta_path, concept_details)
        for expected_concept in self.expected_concepts:
            concept = concept_details[expected_concept['concept_id']]
            assert concept['types'] == expected_concept['types']

    def test_read_umls_definitions(self):
        meta_path = 'tests/fixtures/umls_META'
        concept_details = {}
        umls_utils.read_umls_concepts(meta_path, concept_details)
        umls_utils.read_umls_definitions(meta_path, concept_details)
        for expected_concept in self.expected_concepts:
            concept = concept_details[expected_concept['concept_id']]
            if 'definition' in expected_concept:
                assert concept['definition'] == expected_concept['definition']
