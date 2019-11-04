import unittest

import spacy

from scispacy.util import WhitespaceTokenizer

class TestUtil(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.nlp = spacy.load("en_core_web_sm")

    def test_whitespace_tokenizer(self):

        self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)
        text = "don't split this contraction."
        doc = self.nlp(text)

        assert [t.text for t in doc] == text.split(" ")
