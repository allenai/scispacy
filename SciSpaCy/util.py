import spacy

from custom_sentence_segmenter import combined_rule_sentence_segmenter # pylint: disable-msg=E0611,E0401
from custom_tokenizer import combined_rule_tokenizer # pylint: disable-msg=E0611,E0401

def save_model(nlp, outputPath):
    nlp.to_disk(outputPath)

def create_combined_rule_model():
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = combined_rule_tokenizer(nlp)
    nlp.add_pipe(combined_rule_sentence_segmenter, first=True)
    return nlp