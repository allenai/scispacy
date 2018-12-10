# coding: utf-8
from __future__ import unicode_literals

import pytest
import spacy

def test_en_tokenizer_handles_basic_contraction(combined_all_model_fixture):
    text = "don't giggle"
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 3
    assert tokens[1].text == "n't"
    text = "i said don't!"
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 5
    assert tokens[4].text == "!"


@pytest.mark.parametrize('text', ["`ain't", '''"isn't''', "can't!"])
def test_en_tokenizer_handles_basic_contraction_punct(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 3


@pytest.mark.parametrize('text_poss,text', [("Robin's", "Robin"), ("Alexis's", "Alexis")])
def test_en_tokenizer_handles_poss_contraction(combined_all_model_fixture, text_poss, text):
    tokens = combined_all_model_fixture(text_poss)
    assert len(tokens) == 2
    assert tokens[0].text == text
    assert tokens[1].text == "'s"


@pytest.mark.parametrize('text', ["schools'", "Alexis'"])
def test_en_tokenizer_splits_trailing_apos(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 2
    assert tokens[0].text == text.split("'")[0]
    assert tokens[1].text == "'"


@pytest.mark.parametrize('text', ["'em", "nothin'", "ol'"])
def text_tokenizer_doesnt_split_apos_exc(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 1
    assert tokens[0].text == text


@pytest.mark.parametrize('text', ["we'll", "You'll", "there'll"])
def test_en_tokenizer_handles_ll_contraction(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 2
    assert tokens[0].text == text.split("'")[0]
    assert tokens[1].text == "'ll"
    assert tokens[1].lemma_ == "will"


@pytest.mark.parametrize('text_lower,text_title', [("can't", "Can't"), ("ain't", "Ain't")])
def test_en_tokenizer_handles_capitalization(combined_all_model_fixture, text_lower, text_title):
    tokens_lower = combined_all_model_fixture(text_lower)
    tokens_title = combined_all_model_fixture(text_title)
    assert tokens_title[0].text == tokens_lower[0].text.title()
    assert tokens_lower[0].text == tokens_title[0].text.lower()
    assert tokens_lower[1].text == tokens_title[1].text


@pytest.mark.parametrize('pron', ["I", "You", "He", "She", "It", "We", "They"])
@pytest.mark.parametrize('contraction', ["'ll", "'d"])
def test_en_tokenizer_keeps_title_case(combined_all_model_fixture, pron, contraction):
    tokens = combined_all_model_fixture(pron + contraction)
    assert tokens[0].text == pron
    assert tokens[1].text == contraction


@pytest.mark.parametrize('exc', ["Ill", "ill", "Hell", "hell", "Well", "well"])
def test_en_tokenizer_excludes_ambiguous(combined_all_model_fixture, exc):
    tokens = combined_all_model_fixture(exc)
    assert len(tokens) == 1


@pytest.mark.parametrize('wo_punct,w_punct', [("We've", "``We've"), ("couldn't", "couldn't)")])
def test_en_tokenizer_splits_defined_punct(combined_all_model_fixture, wo_punct, w_punct):
    tokens = combined_all_model_fixture(wo_punct)
    assert len(tokens) == 2
    tokens = combined_all_model_fixture(w_punct)
    assert len(tokens) == 3


@pytest.mark.parametrize('text', ["e.g.", "p.m.", "Jan.", "Dec.", "Inc."])
def test_en_tokenizer_handles_abbr(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 1


def test_en_tokenizer_handles_exc_in_text(combined_all_model_fixture):
    text = "It's mediocre i.e. bad."
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 6
    assert tokens[3].text == "i.e."


@pytest.mark.parametrize('text', ["1am", "12a.m.", "11p.m.", "4pm"])
def test_en_tokenizer_handles_times(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 2
    assert tokens[1].lemma_ in ["a.m.", "p.m."]


@pytest.mark.parametrize('text,norms', [("I'm", ["i", "am"]), ("shan't", ["shall", "not"])])
def test_en_tokenizer_norm_exceptions(combined_all_model_fixture, text, norms):
    tokens = combined_all_model_fixture(text)
    assert [token.norm_ for token in tokens] == norms


@pytest.mark.parametrize('text,norm', [("radicalised", "radicalized"), ("cuz", "because")])
def test_en_lex_attrs_norm_exceptions(combined_all_model_fixture, text, norm):
    tokens = combined_all_model_fixture(text)
    assert tokens[0].norm_ == norm
