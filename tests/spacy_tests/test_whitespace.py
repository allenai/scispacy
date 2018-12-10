# coding: utf-8
"""Test that tokens are created correctly for whitespace."""


from __future__ import unicode_literals

import pytest
import spacy

@pytest.mark.parametrize('text', ["lorem ipsum"])
def test_tokenizer_splits_single_space(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 2


@pytest.mark.parametrize('text', ["lorem  ipsum"])
def test_tokenizer_splits_double_space(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 3
    assert tokens[1].text == " "


@pytest.mark.parametrize('text', ["lorem ipsum  "])
def test_tokenizer_handles_double_trainling_ws(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert repr(tokens.text_with_ws) == repr(text)


@pytest.mark.parametrize('text', ["lorem\nipsum"])
def test_tokenizer_splits_newline(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 3
    assert tokens[1].text == "\n"


@pytest.mark.parametrize('text', ["lorem \nipsum"])
def test_tokenizer_splits_newline_space(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 3


@pytest.mark.parametrize('text', ["lorem  \nipsum"])
def test_tokenizer_splits_newline_double_space(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 3


@pytest.mark.parametrize('text', ["lorem \n ipsum"])
def test_tokenizer_splits_newline_space_wrap(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 3
