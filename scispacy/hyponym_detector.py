from spacy.matcher import Matcher
from spacy.tokens import Token, Doc

from scispacy.hearst_patterns import BASE_PATTERNS, EXTENDED_PATTERNS


class HyponymDetector:
    name = "HyponymDetector"

    """
    A spaCy pipe for detecting hyponyms using Hearst patterns.
    This class sets the following attributes:

    - `Doc._.hearst_patterns`: A List[Tuple[str, Span, Span]] corresonding to
       the matching predicate, extracted general term and specific term
       that matched a Hearst pattern.
    - `Token._.is_hypernym`: Whether the token is a hypernym.
    - `Token._.hyponyms`: The hyponyms for this token.
    - `Token._.is_hyponym`: Whether the token is a hyponym.
    - `Token._.hypernym`: The hypernym for this token.

    Parts of the implementation taken from
    https://github.com/mmichelsonIF/hearst_patterns_python/blob/master/hearstPatterns/hearstPatterns.py
    and
    https://github.com/Fourthought/CNDPipeline/blob/master/cndlib/hpspacy.py

    The pipe can be used with an instantiated spacy model like so:
    ```
    hyponym_pipe = HyponymDetector(extended=True)
    nlp.add_pipe(hyponym_pipe, last=True)
    """

    def __init__(self, nlp, extended=False):

        self.nlp = nlp

        self.patterns = BASE_PATTERNS
        if extended:
            self.patterns.extend(EXTENDED_PATTERNS)

        self.matcher = Matcher(self.nlp.vocab)

        Doc.set_extension("hearst_patterns", default=[], force=True)
        Token.set_extension("is_hypernym", default=False, force=True)
        Token.set_extension("hyponyms", default=[], force=True)
        Token.set_extension("is_hyponym", default=False, force=True)
        Token.set_extension("hypernym", default=None, force=True)

        self.predicates = []
        self.first = set()
        self.last = set()

        # add patterns to matcher
        for pattern in self.patterns:
            self.matcher.add(pattern["label"], None, pattern["pattern"])

            # gather list of predicates where the hypernym appears first
            if pattern["position"] == "first":
                self.first.add(pattern["label"])

            # gather list of predicates where the hypernym appears last
            if pattern["position"] == "last":
                self.last.add(pattern["label"])

    def expand_to_noun_compound(self, token: Token, doc: Doc):
        """
        Expand a token to it's noun phrase based
        on a simple POS tag heuristic.
        """

        start = token.i
        while True:
            previous = doc[start - 1]
            if previous.pos_ in {"PROPN", "NOUN", "PRON"}:
                start -= 1
            else:
                break

        end = token.i + 1
        while True:
            previous = doc[end]
            if previous.pos_ in {"PROPN", "NOUN", "PRON"}:
                end += 1
            else:
                break

        return doc[start:end]

    def find_noun_compound_head(self, token: Token):

        while token.head.pos_ in {"PROPN", "NOUN", "PRON"} and token.dep_ == "compound":
            token = token.head
        return token

    def __call__(self, doc: Doc):
        """
        Runs the matcher on the Doc object and sets token and
        doc level attributes for hypernym and hyponym relations.
        """
        # Find matches in doc
        matches = self.matcher(doc)

        # If none are found then return None
        if not matches:
            return doc

        for match_id, start, end in matches:
            predicate = self.nlp.vocab.strings[match_id]

            # if the predicate is in the list where the hypernym is last, else hypernym is first
            if predicate in self.last:
                hypernym = doc[end - 1]
                hyponym = doc[start]
            else:
                # An inelegent way to deal with the "such_NOUN_as pattern"
                # since the first token is not the hypernym.
                if doc[start].lemma_ == "such":
                    start += 1
                hypernym = doc[start]
                hyponym = doc[end - 1]

            hypernym = self.find_noun_compound_head(hypernym)
            hyponym = self.find_noun_compound_head(hyponym)

            # hypernym recorded as True and list of hyponyms created
            hypernym._.is_hypernym = True
            hypernym._.hyponyms.append(hyponym)

            # hyponym recorded as True and its hypernym is recorded
            hyponym._.is_hyponym = True
            hyponym._.hypernym = hypernym

            # iterate over conjunct list attached to hyponym
            for token in hyponym.conjuncts:
                if token != hypernym and token is not None:
                    hypernym._.hyponyms.append(token)
                    token._.is_hyponym = True
                    token._.hypernym = hypernym

            # For the document level, we expand to contain noun phrases.
            hypernym_extended = self.expand_to_noun_compound(hypernym, doc)
            hyponym_extended = self.expand_to_noun_compound(hyponym, doc)
            doc._.hearst_patterns.append(
                (predicate, hypernym_extended, hyponym_extended)
            )

            for token in hyponym.conjuncts:

                token_extended = self.expand_to_noun_compound(token, doc)
                if token != hypernym and token is not None:
                    doc._.hearst_patterns.append(
                        (predicate, hypernym_extended, token_extended)
                    )

        return doc
