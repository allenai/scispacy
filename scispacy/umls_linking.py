from spacy.tokens import Doc
from spacy.tokens import Span

from scispacy.candidate_generation import CandidateGenerator


class UmlsEntityLinker:

    def __init__(self,
                 candidate_generator: CandidateGenerator = None,
                 resolve_abbreviations: bool = True,
                 k: int = 30):

        Span.set_extension("cui", default=[], force=True)
        Span.set_extension("tui", default=[], force=True)

        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.resolve_abbreviations = resolve_abbreviations
        self.k = k

    def __call__(self, doc: Doc) -> Doc:
        mentions = doc.ents
        #if self.resolve_abbreviations:
        #    mentions = self.replace_abbreviations(mentions)

        mention_strings = [x.text for x in mentions]

        candidates = self.candidate_generator(mention_strings, self.k)

        print(candidates)
        return doc
