from __future__ import unicode_literals

from pathlib import Path
from spacy.language import Language
from spacy.util import load_model_from_init_py, get_model_meta

__version__ = get_model_meta(Path(__file__).parent)['version']

def load(**overrides):
    nlp = load_model_from_init_py(__file__, **overrides)
    return nlp
