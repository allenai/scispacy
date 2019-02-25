# pylint: disable=no-self-use,invalid-name
import unittest

from scispacy import umls_utils

class TestUtil(unittest.TestCase):

    def test_read_umls_concepts(self):
        meta_path = 'tests/fixtures/umls_META'
        concept_details = {}
        umls_utils.read_umls_concepts(meta_path, concept_details)
        assert len(concept_details) == 3
        for concept in concept_details.values():
            assert 'aliases' in concept
            assert concept['aliases'] or concept['canonical_name']
            assert 'types' in concept
            assert not concept['types']

    def test_read_umls_types(self):
        meta_path = 'tests/fixtures/umls_META'
        concept_details = {}
        umls_utils.read_umls_concepts(meta_path, concept_details)
        umls_utils.read_umls_types(meta_path, concept_details)
        for concept in concept_details.values():
            assert concept['types']

    def test_read_umls_definitions(self):
        meta_path = 'tests/fixtures/umls_META'
        concept_details = {}
        umls_utils.read_umls_concepts(meta_path, concept_details)
        umls_utils.read_umls_definitions(meta_path, concept_details)
        concepts_with_definitions_count = 0
        for concept in concept_details.values():
            if concept.get('definition'):
                concepts_with_definitions_count += 1
        assert concepts_with_definitions_count > 0
