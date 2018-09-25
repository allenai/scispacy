import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../SciSpaCy/")

import pytest
import spacy

from custom_sentence_segmenter import combined_rule_sentence_segmenter
from custom_tokenizer import combined_rule_tokenizer, combined_rule_prefixes, remove_new_lines

@pytest.fixture()
def combined_rule_tokenizer_fixture():
    nlp = spacy.load('en_core_web_sm')
    tokenizer = combined_rule_tokenizer(nlp)
    return tokenizer

@pytest.fixture()
def en_with_combined_rule_tokenizer_fixture():
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = combined_rule_tokenizer(nlp)
    return nlp

@pytest.fixture()
def en_with_combined_tokenizer_and_segmenter_fixture():
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = combined_rule_tokenizer(nlp)
    nlp.add_pipe(combined_rule_sentence_segmenter, first=True)
    return nlp

@pytest.fixture()
def combined_rule_prefixes_fixture():
    return combined_rule_prefixes()

@pytest.fixture()
def remove_new_lines_fixture():
    return remove_new_lines

@pytest.fixture()
def default_en_tokenizer_fixture():
    nlp = spacy.load('en_core_web_sm')
    return nlp.tokenizer

@pytest.fixture()
def default_en_model_fixture():
    nlp = spacy.load('en_core_web_sm')
    return nlp