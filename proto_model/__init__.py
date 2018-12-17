from __future__ import unicode_literals

from pathlib import Path
from spacy.language import Language
from spacy.util import load_model_from_init_py, get_model_meta

from scispacy.custom_sentence_segmenter import combined_rule_sentence_segmenter

__version__ = get_model_meta(Path(__file__).parent)['version']


def load(**overrides):
    Language.factories['combined_rule_sentence_segmenter'] = lambda nlp, **cfg: combined_rule_sentence_segmenter
    nlp = load_model_from_init_py(__file__, **overrides)
    return nlp
