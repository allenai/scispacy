
from typing import List

import pysbd

from spacy.tokens import Doc

from scispacy.consts import ABBREVIATIONS  # pylint: disable-msg=E0611,E0401


def combined_rule_sentence_segmenter(doc: Doc) -> Doc:
    """Adds sentence boundaries to a Doc. Intended to be used as a pipe in a spaCy pipeline.
       New lines cannot be end of sentence tokens. New lines that separate sentences will be
       added to the beginning of the next sentence.

    @param doc: the spaCy document to be annotated with sentence boundaries
    """
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    sents_char_spans = segmenter.segment(doc.text)

    char_spans = [
        doc.char_span(sent_span.start, sent_span.end)
        for sent_span in sents_char_spans
    ]
    start_token_ids = [span[0].idx for span in char_spans if span is not None]
    for token in doc:
        if token.idx in start_token_ids:
            prev_token = token.nbor(-1).text if token.idx != 0 else None
            if prev_token in ABBREVIATIONS:
                token.is_sent_start = False
            else:
                token.is_sent_start = True
        else:
            token.is_sent_start = False
    return doc
