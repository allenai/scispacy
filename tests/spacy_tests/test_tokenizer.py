# coding: utf-8
from __future__ import unicode_literals

import pytest
import spacy
from spacy import util

def test_tokenizer_handles_no_word(combined_all_model_fixture):
    tokens = combined_all_model_fixture("")
    assert len(tokens) == 0


@pytest.mark.parametrize('text', ["lorem"])
def test_tokenizer_handles_single_word(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert tokens[0].text == text


def test_tokenizer_handles_punct(combined_all_model_fixture):
    text = "Lorem, ipsum."
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 4
    assert tokens[0].text == "Lorem"
    assert tokens[1].text == ","
    assert tokens[2].text == "ipsum"
    assert tokens[1].text != "Lorem"


def test_tokenizer_handles_digits(combined_all_model_fixture):
    exceptions = ["hu", "bn"]
    text = "Lorem ipsum: 1984."
    tokens = combined_all_model_fixture(text)

    if tokens[0].lang_ not in exceptions:
        assert len(tokens) == 5
        assert tokens[0].text == "Lorem"
        assert tokens[3].text == "1984"


@pytest.mark.parametrize('text', ["google.com", "python.org", "spacy.io", "explosion.ai", "http://www.google.com"])
def test_tokenizer_keep_urls(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 1


@pytest.mark.parametrize('text', ["NASDAQ:GOOG"])
def test_tokenizer_colons(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 3


@pytest.mark.parametrize('text', ["hello123@example.com", "hi+there@gmail.it", "matt@explosion.ai"])
def test_tokenizer_keeps_email(combined_all_model_fixture, text):
    tokens = combined_all_model_fixture(text)
    assert len(tokens) == 1


def test_tokenizer_handles_long_text(combined_all_model_fixture):
    text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit

Cras egestas orci non porttitor maximus.
Maecenas quis odio id dolor rhoncus dignissim. Curabitur sed velit at orci ultrices sagittis. Nulla commodo euismod arcu eget vulputate.

Phasellus tincidunt, augue quis porta finibus, massa sapien consectetur augue, non lacinia enim nibh eget ipsum. Vestibulum in bibendum mauris.

"Nullam porta fringilla enim, a dictum orci consequat in." Mauris nec malesuada justo."""

    tokens = combined_all_model_fixture(text)
    assert len(tokens) > 5


@pytest.mark.parametrize('file_name', ["sun.txt"])
def test_tokenizer_handle_text_from_file(combined_all_model_fixture, file_name):
    loc = util.ensure_path(__file__).parent / file_name
    text = loc.open('r', encoding='utf8').read()
    assert len(text) != 0
    tokens = combined_all_model_fixture(text)
    assert len(tokens) > 100


def test_tokenizer_suspected_freeing_strings(combined_all_model_fixture):
    text1 = "Lorem dolor sit amet, consectetur adipiscing elit."
    text2 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    tokens1 = combined_all_model_fixture(text1)
    tokens2 = combined_all_model_fixture(text2)
    assert tokens1[0].text == "Lorem"
    assert tokens2[0].text == "Lorem"


@pytest.mark.parametrize('text,tokens', [
    ("lorem", [{'orth': 'lo'}, {'orth': 'rem'}])])
def test_tokenizer_add_special_case(combined_all_model_fixture, text, tokens):
    combined_all_model_fixture.tokenizer.add_special_case(text, tokens)
    doc = combined_all_model_fixture(text)
    assert doc[0].text == tokens[0]['orth']
    assert doc[1].text == tokens[1]['orth']
