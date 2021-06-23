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
    assert doc.has_annotation("DEP")
    assert doc[0].dep_ == "ROOT"
    assert doc[0].tag_ == "NN"

def test_full_pipe_serializable(combined_all_model_fixture):
    text = "Induction of cytokine expression in leukocytes (CEIL) by binding of thrombin-stimulated platelets. BACKGROUND: Activated platelets tether and activate myeloid leukocytes."
    doc = [doc for doc in combined_all_model_fixture.pipe([text, text], n_process = 2)][0]
    # If we got here this means that both model is serializable and there is an abbreviation that would break if it wasn't
    assert len(doc._.abbreviations) > 0