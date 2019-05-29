from spacy.tokens import Doc
from spacy.tokens import Span

from scispacy.candidate_generation import CandidateGenerator


class UmlsEntityLinker:

    """
    A spacy pipeline component which identifies entities in text which appear
    in the Unified Medical Language System (UMLS).

    Currently this implementation just compares string similarity, returning
    entities above a given threshold.


    This class sets the `._.umls_ent` attribute on spacy Spans, which consists of a
    List[Tuple[str, float]] corresponding to the UMLS concept_id and the associated score.

    You can look up more information for a given id using the umls attribute of this class:

    print(linker.umls.cui_to_entity[concept_id])

    Parameters
    ----------

    candidate_generator : `CandidateGenerator`, optional, (default = None)
        A CandidateGenerator to generate entity candidates for mentions.
        If no candidate generator is passed, the default pretrained one is used.
    resolve_abbreviations : bool = True, optional (default = False)
        Whether to resolve abbreviations identified in the Doc before performing linking.
        This parameter has no effect if there is no `AbbreviationDetector` in the spacy
        pipeline.
    k : int, optional, (default = 30)
        The number of nearest neighbours to look up from the candidate generator per mention.
    threshold : float, optional, (default = 0.7)
        The threshold that a mention candidate must reach to be added to the mention in the Doc
        as a mention candidate.
    """
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
        mentions = []
        if self.resolve_abbreviations and Doc.has_extension("abbreviations"):

            for ent in doc.ents:
                if ent._.long_form is not None:
                    mentions.append(ent._.long_form)
                else:
                    mentions.append(ent)
        else:
            mentions = doc.ents

        mention_strings = [x.text for x in mentions]
        batch_candidates = self.candidate_generator(mention_strings, self.k)

        for mention, candidates in zip(doc.ents, batch_candidates):
            for cand in candidates:
                score = max([1.0 - s for s in cand.distances])
                if score > self.threshold:
                    mention._.umls_ent.append((cand.concept_id, score))
        return doc
