from spacy.tokens import Doc
from spacy.tokens import Span

from scispacy.candidate_generation import CandidateGenerator


class UmlsEntityLinker:

    def __init__(self,
                 candidate_generator: CandidateGenerator = None,
                 resolve_abbreviations: bool = True,
                 k: int = 30,
                 threshold: float = 0.7):

        Span.set_extension("umls_ent", default=[], force=True)

        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.resolve_abbreviations = resolve_abbreviations
        self.k = k
        self.threshold = threshold
        self.umls = self.candidate_generator.umls

    def __call__(self, doc: Doc) -> Doc:
        mentions = doc.ents
        #if self.resolve_abbreviations:
        #    mentions = self.replace_abbreviations(mentions)
        mention_strings = [x.text for x in mentions]
        batch_candidates = self.candidate_generator(mention_strings, self.k)

        for mention, candidates in zip(mentions, batch_candidates):
            for cand in candidates:
                score = max([1.0 - s for s in cand.distances])
                if score > self.threshold:
                    mention._.umls_ent.append((cand.concept_id, score))
        return doc
