# coding: utf-8
"""Test that tokens are created correctly for whitespace."""


from __future__ import unicode_literals

import pytest

import spacy
from spacy.language import Language as SpacyModelType

from scispacy.custom_sentence_segmenter import pysbd_sentencizer


class TestWhitespace:
    nlp = spacy.load("en_core_sci_sm")

    @pytest.mark.parametrize("text", ["lorem ipsum"])
    def test_tokenizer_splits_single_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 2

    @pytest.mark.parametrize("text", ["lorem  ipsum"])
    def test_tokenizer_splits_double_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
        assert tokens[1].text == " "

    @pytest.mark.parametrize("text", ["lorem ipsum  "])
    def test_tokenizer_handles_double_trainling_ws(self, text):
        tokens = self.nlp(text)
        assert repr(tokens.text_with_ws) == repr(text)

    @pytest.mark.parametrize("text", ["lorem\nipsum"])
    def test_tokenizer_splits_newline(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
        assert tokens[1].text == "\n"

    @pytest.mark.parametrize("text", ["lorem \nipsum"])
    def test_tokenizer_splits_newline_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3

    @pytest.mark.parametrize("text", ["lorem  \nipsum"])
    def test_tokenizer_splits_newline_double_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3

    @pytest.mark.parametrize("text", ["lorem \n ipsum"])
    def test_tokenizer_splits_newline_space_wrap(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
