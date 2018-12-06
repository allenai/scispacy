import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../SciSpaCy/")

import pytest
import spacy
from spacy.language import Language

from custom_sentence_segmenter import combined_rule_sentence_segmenter
from custom_tokenizer import combined_rule_tokenizer, combined_rule_prefixes, remove_new_lines

@pytest.fixture()
def combined_all_model_fixture():
    Language.factories['combined_rule_sentence_segmenter'] = lambda nlp, **cfg: combined_rule_sentence_segmenter
    nlp = spacy.load('SciSpaCy/models/combined_all_model')
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