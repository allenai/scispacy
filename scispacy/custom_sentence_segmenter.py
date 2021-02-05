from typing import List

import pysbd

from spacy.tokens import Doc
from spacy.language import Language
from pysbd.utils import TextSpan

from scispacy.consts import ABBREVIATIONS


@Language.component("pysbd_sentencizer")
def pysbd_sentencizer(doc: Doc) -> Doc:
    """Adds sentence boundaries to a Doc.
    Intended to be used as a pipe in a spaCy pipeline.
    Uses https://github.com/nipunsadvilkar/pySBD to get proper sentence and
    respective char_spans

    Handle special cases:
    New lines cannot be end of sentence tokens.
    New lines that separate sentences will be added to the
    beginning of the next sentence.

    @param doc: the spaCy document to be annotated with sentence boundaries
    """
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    sents_char_spans: List[TextSpan] = segmenter.segment(doc.text)

    char_spans = [
        doc.char_span(
            sent_span.start,
            # strip off trailing spaces when creating spans to accomodate spacy
            sent_span.end - (len(sent_span.sent) - len(sent_span.sent.rstrip(" "))),
        )
        for sent_span in sents_char_spans
    ]
    start_token_char_offsets = [span[0].idx for span in char_spans if span is not None]
    for token in doc:
        prev_token = token.nbor(-1) if token.i != 0 else None
        if token.idx in start_token_char_offsets:
            if prev_token and (
                prev_token.text in ABBREVIATIONS
                # Glom new lines at the beginning of the text onto the following sentence
                or (prev_token.i == 0 and all(c == "\n" for c in prev_token.text))
            ):
                token.is_sent_start = False
            else:
                token.is_sent_start = True
        # check if previous token contains more than 2 newline chars
        elif prev_token and prev_token.i != 0 and prev_token.text.count("\n") >= 2:
            token.is_sent_start = True
        else:
            token.is_sent_start = False
    return doc
