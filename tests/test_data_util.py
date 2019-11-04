import os
import unittest
import shutil


from scispacy.data_util import read_full_med_mentions, med_mentions_example_iterator, remove_overlapping_entities
from scispacy.data_util import read_ner_from_tsv

class TestDataUtil(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.TEST_DIR = "/tmp/scispacy"
        os.makedirs(self.TEST_DIR, exist_ok=True)

        self.med_mentions = "tests/fixtures/med_mentions.txt"
        self.ner_tsv = "tests/fixtures/ner_test.tsv"

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)

    def test_example_iterator(self):
        iterator = med_mentions_example_iterator(self.med_mentions)
        for example in iterator:
            assert example.text == example.title + " " + example.abstract

            for entity in example.entities:
                assert entity.start < entity.end
                assert entity.start < len(example.text)
                assert entity.end < len(example.text)
                assert entity.mention_text == example.text[entity.start: entity.end]

    def test_remove_overlaps(self):
        test_entities = [(0, 5, 'ENTITY'), (6, 10, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(0, 5, 'ENTITY'), (6, 10, 'ENTITY')]

        test_entities = [(0, 5, 'ENTITY'), (5, 10, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(0, 5, 'ENTITY'), (5, 10, 'ENTITY')]

        test_entities = [(0, 5, 'ENTITY'), (4, 10, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(4, 10, 'ENTITY')]

        test_entities = [(0, 5, 'ENTITY'), (0, 5, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(0, 5, 'ENTITY')]

        test_entities = [(0, 5, 'ENTITY'), (4, 11, 'ENTITY'), (6, 20, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(0, 5, 'ENTITY'), (6, 20, 'ENTITY')]

        test_entities = [(0, 5, 'ENTITY'), (4, 7, 'ENTITY'), (10, 20, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(0, 5, 'ENTITY'), (10, 20, 'ENTITY')]

        test_entities = [(1368, 1374, 'ENTITY'), (1368, 1376, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(1368, 1376, 'ENTITY')]

        test_entities = [(12, 33, 'ENTITY'), (769, 779, 'ENTITY'), (769, 787, 'ENTITY'), (806, 811, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(12, 33, 'ENTITY'), (769, 787, 'ENTITY'), (806, 811, 'ENTITY')]

        test_entities = [(189, 209, 'ENTITY'),
                         (317, 362, 'ENTITY'),
                         (345, 354, 'ENTITY'),
                         (364, 368, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(189, 209, 'ENTITY'), (317, 362, 'ENTITY'), (364, 368, 'ENTITY')]

        test_entities = [(445, 502, 'ENTITY'),
                         (461, 473, 'ENTITY'),
                         (474, 489, 'ENTITY')]
        result = remove_overlapping_entities(test_entities)
        assert result == [(445, 502, 'ENTITY')]

    def test_read_ner_from_tsv(self):

        data = read_ner_from_tsv(self.ner_tsv)
        assert len(data) == 4       
        example = data[0]
        assert example[0] == 'Intraocular pressure in genetically distinct mice : an update and strain survey'
        assert example[1] ==  {'entities': [(24, 35, 'SO'), (45, 49, 'TAXON')]}
        example = data[1]
        assert example[0] == 'Abstract'
        assert example[1] ==  {'entities': []}
        example = data[2]
        assert example[0] == 'Background'
        assert example[1] ==  {'entities': []}
        example = data[3]
        assert example[0] == 'Little is known about genetic factors affecting intraocular pressure ( IOP ) in mice and other mammals .'
        assert example[1] ==  {'entities': [(22, 29, 'SO'), (80, 84, 'TAXON'), (95, 102, 'TAXON')]}
