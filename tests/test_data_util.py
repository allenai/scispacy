# pylint: disable=no-self-use,invalid-name
import os
import pathlib
import json
import unittest
import shutil

import pytest

from SciSpaCy.data_util import read_med_mentions, med_mentions_example_iterator

class TestDataUtil(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.TEST_DIR = "/tmp/scispacy"
        os.makedirs(self.TEST_DIR, exist_ok=True)

        self.med_mentions = "tests/fixtures/med_mentions.txt"

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

    def test_read_med_mentions(self):
        examples = read_med_mentions(self.med_mentions)
        assert len(examples) == 3
        