
from typing import List

import pysbd

from spacy.tokens import Doc
from pysbd.utils import TextSpan

from scispacy.consts import ABBREVIATIONS  # pylint: disable-msg=E0611,E0401


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
        doc.char_span(sent_span.start, sent_span.end)
        for sent_span in sents_char_spans
    ]
    start_token_char_offsets = [span[0].idx for span in char_spans if span is not None]
    for token in doc:
        prev_token = token.nbor(-1) if token.i != 0 else None
        if token.idx in start_token_char_offsets:
            if prev_token and prev_token.text in ABBREVIATIONS:
                token.is_sent_start = False
            else:
                token.is_sent_start = True
        # check if previous token contains more than 2 newline chars
        elif prev_token and prev_token.i != 0 and prev_token.text.count('\n') >= 2:
            token.is_sent_start = True
        else:
            token.is_sent_start = False
    return doc


if __name__ == "__main__":
    import spacy
    from scispacy.custom_tokenizer import combined_rule_tokenizer, combined_rule_prefixes

    nlp = spacy.blank('en')
    nlp.tokenizer = combined_rule_tokenizer(nlp)
    nlp.add_pipe(combined_rule_sentence_segmenter, first=True)
    # text = "When the tree is simply a chain, both Eqs. 2–8 and Eqs. 9–14 reduce to the standard LSTM transitions, Eqs. 1."
    # text = "First sentence with char in the middle.adjacent sentence to it."
    text = 'How about tomorrow?We can meet at eden garden.'
    doc = nlp(text)
    for sent_id, sent in enumerate(doc.sents, start=1):
        print(sent_id, repr(sent.text), sep='\t')
    # expected_sents = ['\n\n2 Long Short-Term Memory Networks\n\n\n\n', '2.1 Overview\n\n', 'Recurrent neural networks (RNNs) are able to pr...put sequences of arbitrary length via the recursive application of a transition function on a hidden state vector ht.']
