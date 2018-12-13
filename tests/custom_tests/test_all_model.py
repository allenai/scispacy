import os
import sys
from scripts import retrain_parser_and_tagger
import spacy
from spacy.vocab import Vocab
import shutil

def test_retraining(test_model_dir, test_conll_path, test_pmids_path, test_vocab_dir):
    if os.path.isdir(test_model_dir):
        shutil.rmtree(test_model_dir)
    if not os.path.isdir(test_model_dir):
        os.mkdir(test_model_dir)
    retrain_parser_and_tagger.train_parser_and_tagger(test_conll_path,
                                                      test_conll_path,
                                                      test_conll_path,
                                                      test_pmids_path,
                                                      test_pmids_path,
                                                      test_pmids_path,
                                                      test_vocab_dir,
                                                      test_model_dir)
    nlp = spacy.load(os.path.join(test_model_dir, "genia_trained_parser"))
    text = "Induction of cytokine expression in leukocytes by binding of thrombin-stimulated platelets. BACKGROUND: Activated platelets tether and activate myeloid leukocytes."
    doc = nlp(text)
    assert doc.is_parsed
    assert doc[0].text == "Induction"
    shutil.rmtree(test_model_dir)

def test_custom_segmentation(combined_all_model_fixture):
    text = "Induction of cytokine expression in leukocytes by binding of thrombin-stimulated platelets. BACKGROUND: Activated platelets tether and activate myeloid leukocytes."
    doc = combined_all_model_fixture(text)
    sents = list(doc.sents)
    assert len(sents) == 2
    expected_tokens = ["Induction", "of", "cytokine", "expression", "in", "leukocytes", "by", "binding", "of", "thrombin-stimulated", "platelets", ".", "BACKGROUND", ":", "Activated", "platelets", "tether", "and", "activate", "myeloid", "leukocytes", "."]
    actual_tokens = [t.text for t in doc]
    assert expected_tokens == actual_tokens
    assert doc.is_parsed
    assert doc[0].dep_ == "ROOT"
    assert doc[0].tag_ == "NN"