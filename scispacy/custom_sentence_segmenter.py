
from typing import List

import pysbd

from spacy.tokens import Doc

from scispacy.consts import ABBREVIATIONS # pylint: disable-msg=E0611,E0401

def merge_segments(segments: List[str]) -> List[str]:
    adjusted_segments = []
    temp_segment = ""
    for segment in segments:
        if temp_segment != "":
            temp_segment += " "
        temp_segment += segment
        if not segment.endswith(tuple(ABBREVIATIONS)):
            adjusted_segments.append(temp_segment)
            temp_segment = ""
    return adjusted_segments

def combined_rule_sentence_segmenter(doc: Doc) -> Doc:
    """Adds sentence boundaries to a Doc. Intended to be used as a pipe in a spaCy pipeline.

    @param doc: the spaCy document to be annotated with sentence boundaries
    """
    segmenter = pysbd.Segmenter(language="en", clean=False)
    segments = merge_segments(segmenter.segment(doc.text))

    # pysbd splits raw text into sentences, so we have to do our best to align those
    # segments with spacy tokens
    segment_index = 0
    current_segment = segments[segment_index]
    built_up_sentence = ""
    for token in enumerate(doc):
        if token.text.replace('\n', '') == '':
            token.is_sent_start = False
        elif len(built_up_sentence) >= len(current_segment):
            token.is_sent_start = True

            # handle the rare (impossible?) case where spacy tokenizes over a sentence boundary that
            # pysbd finds
            built_up_sentence = ' '*int(len(built_up_sentence) - len(current_segment))
            built_up_sentence = token.string
            segment_index += 1
            current_segment = segments[segment_index]
        else:
            built_up_sentence += token.string
            token.is_sent_start = False

    return doc
