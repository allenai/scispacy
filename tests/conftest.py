import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../SciSpaCy/")

import pytest
import spacy

from custom_sentence_segmenter import combined_rule_sentence_segmenter
from custom_tokenizer import combined_rule_tokenizer

@pytest.fixture()
def combined_rule_tokenizer():
    nlp = spacy.load('en_core_web_sm')
    tokenizer = combined_rule_tokenizer(nlp)
    return tokenizer

@pytest.fixture()
def en_with_combined_tokenizer_and_segmenter():
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = combined_rule_tokenizer(nlp)
    nlp.add_pipe(combined_rule_splitter, first=True)
    return nlp

@pytest.fixture()
def default_en_tokenizer():
    nlp = spacy.load('en_core_web_sm')
    return nlp.tokenizer

@pytest.fixture()
def default_en_model():
    nlp = spacy.load('en_core_web_sm')
    return nlp