
from typing import Tuple, List, Optional
from collections import defaultdict
import spacy
from spacy.tokens import Span, Doc
from spacy.matcher import Matcher

def find_abbreviation(long_form_candidate: Span,
                      short_form_candidate: Span) -> Tuple[Span, Optional[Span]]:
    """
    Implements the abbreviation detection algorithm in "A simple algorithm
    for identifying abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

    The algorithm works by enumerating the characters in the short form of the abbreviation,
    checking that they can be matched against characters in a candidate text for the long form
    in order, as well as requiring that the first letter of the abbreviated form matches the
    _beginning_ letter of a word.

    Parameters
    ----------
    long_form_candidate: Span, required.
        The spaCy span for the long form candidate of the definition.
    short_form_candidate: Span, required.
        The spaCy span for the abbreviation candidate.

    Returns
    -------
    A Tuple[Span, Optional[Span]], representing the short form abbreviation and the
    span corresponding to the long form expansion, or None if a match is not found.
    """
    long_form = " ".join([x.text for x in long_form_candidate])
    short_form = " ".join([x.text for x in short_form_candidate])

    long_index = len(long_form) - 1
    short_index = len(short_form) - 1

    while short_index >=0:
        current_char = short_form[short_index].lower()
        # We don't check non alpha-numeric characters.
        if not current_char.isalnum():
            short_index -= 1
            continue

                # Does the character match at this position? ...
        while ((long_index >= 0 and long_form[long_index].lower() != current_char) or
                # .... or if we are checking the first character of the abbreviation, we enforce
                # to be the _starting_ character of a span.
                (short_index == 0 and long_index > 0 and long_form[long_index -1].isalnum())):
            long_index -= 1
            if long_index < 0:
                return short_form_candidate, None
        long_index -= 1
        short_index -= 1

    # If we complete the string, we end up with -1 here,
    # but really we want all of the text.
    long_index = max(long_index, 0)

    # Now we know the character index of the start of the character span,
    # here we just translate that to the first token beginning after that
    # value, so we can return a spaCy span instead.
    word_lengths = 0
    starting_index = None
    for i, word in enumerate(long_form_candidate):
        word_lengths += len(word)
        if word_lengths > long_index:
            starting_index = i
            break

    return short_form_candidate, long_form_candidate[starting_index:]

def filter_matches(matcher_output, doc: Doc) -> List[Tuple[Span, Span]]:
    # Filter into two cases:
    # 1. <Short Form> ( <Long Form> )
    # 2. <Long Form> (<Short Form>) [this case is most common].
    candidates = []
    for match in matcher_output:
        start = match[1]
        end = match[2]
        # Ignore spans with more than 8 words in (+ 2 for parens).
        if end - start > 10:
            continue
        if end - start > 5:
            # Long form is inside the parens.
            # Take two words before.
            short_form_candidate = doc[start - 2: start]
            if short_form_filter(short_form_candidate):
                candidates.append((doc[start +1: end - 1], short_form_candidate))
        else:
            # Normal case.
            # Short form is inside the parens.
            # Sum character lengths of contents of parens.
            abbreviation_length = sum([len(x) for x in  doc[start + 1: end -1]])
            max_words = min(abbreviation_length + 5, abbreviation_length * 2)
            # Look up to max_words backwards
            long_form_candidate = doc[max(start - max_words, 0): start]
            candidates.append((long_form_candidate, doc[start + 1: end - 1]))
    return candidates


def short_form_filter(span: Span) -> bool:
    # All words are between length 2 and 10
    if not all([2 < len(x) < 10 for x in span]):
        return False
    # At least one word is alpha numeric
    if not any([x.is_alpha for x in span]):
        return False
    return True

class AbbreviationDetector:
    """
    Detects abbreviations using the algorithm in "A simple algorithm for identifying
    abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

    This class sets the `._.abbreviations` attribute on spaCy Doc.

    The abbreviations attribute is a `List[Tuple[Span, Set[Span]]]` mapping long forms
    of abbreviations to all occurences of that abbreviation within a document.

    Note that this class does not replace the spans, or merge them.
    """
    def __init__(self, nlp):
        Doc.set_extension("abbreviations", default=[], force=True)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("parenthesis", None, [{'ORTH': '('}, {'OP': '+'}, {'ORTH': ')'}])

        self.global_matcher = Matcher(nlp.vocab)

    def __call__(self, doc: Doc):
        matches = self.matcher(doc)
        filtered = filter_matches(matches, doc)

        rules = {}
        all_occurences = defaultdict(set)
        already_seen_long = set()
        already_seen_short = set()
        for (long_candidate, short_candidate) in filtered:
            short, long = find_abbreviation(long_candidate, short_candidate)
            # We need the long and short form definitions to be unique, because we need
            # to store them so we can look them up later. This is a bit of a
            # pathalogical case also, as it would mean an abbreviation had been
            # defined twice in a document. There's not much we can do about this,
            # but at least the case which is discarded will be picked up below by
            # the global matcher. So it's likely that things will work out ok most of the time.
            if (long is not None and
                long.string not in already_seen_long and
                short.string not in already_seen_short):

                already_seen_long.add(long.string)
                already_seen_short.add(short.string)
                all_occurences[long].add(short)
                rules[long.string] = long
                # Add a rule to a matcher to find exactly this substring.
                self.global_matcher.add(long.string, None, [{"ORTH": x.text} for x in short])
        to_remove = set()
        global_matches = self.global_matcher(doc)
        for match, start, end in global_matches:
            string_key = self.global_matcher.vocab.strings[match]
            to_remove.add(string_key)
            all_occurences[rules[string_key]].add(doc[start:end])
        for key in to_remove:
            # Clean up the global matcher.
            self.global_matcher.remove(key)

        doc._.abbreviations = [(k,v) for k,v in all_occurences.items()]
        return doc


if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")

    nlp.add_pipe(AbbreviationDetector(nlp), last=True)
    text = "Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron disease caused by the expansion of a polyglutamine tract within the androgen receptor (AR). The pathologic features of SBMA are motor neuron loss in the spinal cord and brainstem and diffuse nuclear accumulation and nuclear inclusions of the mutant AR in the residual motor neurons and certain visceral organs. Many components of the ubiquitin-proteasome and molecular chaperones are also sequestered in the inclusions, suggesting that they may be actively engaged in an attempt to degrade or refold the mutant AR. C terminus of Hsc70 (heat shock cognate protein 70)-interacting protein (CHIP), a U-box type E3 ubiquitin ligase, has been shown to interact with heat shock protein 90 (Hsp90) or Hsp70 and ubiquitylates unfolded proteins trapped by molecular chaperones and degrades them. Here, we demonstrate that transient overexpression of CHIP in a neuronal cell model reduces the monomeric mutant AR more effectively than it does the wild type, suggesting that the mutant AR is more sensitive to CHIP than is the wild type. High expression of CHIP in an SBMA transgenic mouse model also ameliorated motor symptoms and inhibited neuronal nuclear accumulation of the mutant AR. When CHIP was overexpressed in transgenic SBMA mice, mutant AR was also preferentially degraded over wild-type AR. These findings suggest that CHIP overexpression ameliorates SBMA phenotypes in mice by reducing nuclear-localized mutant AR via enhanced mutant AR degradation. Thus, CHIP overexpression would provide a potential therapeutic avenue for SBMA."

    doc  = nlp(text)
    for long, shorts in doc._.abbreviations.items():
        print(long, shorts)




