import os
import sys
import spacy
from spacy.vocab import Vocab
import shutil


def test_custom_segmentation(combined_all_model_fixture):
    text = "Induction of cytokine expression in leukocytes by binding of thrombin-stimulated platelets. BACKGROUND: Activated platelets tether and activate myeloid leukocytes."
    doc = combined_all_model_fixture(text)
    sents = list(doc.sents)
    assert len(sents) == 2
    expected_tokens = [
        "Induction",
        "of",
        "cytokine",
        "expression",
        "in",
        "leukocytes",
        "by",
        "binding",
        "of",
        "thrombin-stimulated",
        "platelets",
        ".",
        "BACKGROUND",
        ":",
        "Activated",
        "platelets",
        "tether",
        "and",
        "activate",
        "myeloid",
        "leukocytes",
        ".",
    ]
    actual_tokens = [t.text for t in doc]
    assert expected_tokens == actual_tokens
    assert doc.is_parsed
    assert doc[0].dep_ == "ROOT"
    assert doc[0].tag_ == "NN"
