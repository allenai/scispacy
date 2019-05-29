from spacy.tokens import Doc
from spacy.tokens import Span

from scispacy.candidate_generation import CandidateGenerator


class UmlsEntityLinker:
    """
    A spacy pipeline component which identifies entities in text which appear
    in the Unified Medical Language System (UMLS).

    Currently this implementation just compares string similarity, returning
    entities above a given threshold.


    This class sets the `._.umls_ents` attribute on spacy Spans, which consists of a
    List[Tuple[str, float]] corresponding to the UMLS concept_id and the associated score
    for a list of `max_entities_per_mention` number of entities.

    You can look up more information for a given id using the umls attribute of this class:

    print(linker.umls.cui_to_entity[concept_id])

    A Note on Definitions:
    Only 187767 entities, or 6.74% of the UMLS KB we are matching against, have definitions. However,
    the MedMentions dataset links to entities which have definitions 82.9% of the time. So by
    default, we only link to entities which have definitions (typically they are more salient / cleaner),
    but this might not suit your use case. YMMV.


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
    filter_for_definitions: bool, default = True
        Whether to filter entities that can be returned to only include those with definitions
        in the knowledge base.
    max_entities_per_mention : int, optional, default = 5
        The maximum number of entities which will be returned for a given mention, regardless of
        how many are nearest neighbours are found.

    """
    def __init__(self,
                 candidate_generator: CandidateGenerator = None,
                 resolve_abbreviations: bool = True,
                 k: int = 30,
                 threshold: float = 0.7,
                 filter_for_definitions: bool = True,
                 max_entities_per_mention: int = 5):

        Span.set_extension("umls_ents", default=[], force=True)

        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.resolve_abbreviations = resolve_abbreviations
        self.k = k
        self.threshold = threshold
        self.umls = self.candidate_generator.umls
        self.filter_for_definitions = filter_for_definitions
        self.max_entities_per_mention = max_entities_per_mention

    def __call__(self, doc: Doc) -> Doc:
        mentions = []
        if self.resolve_abbreviations and Doc.has_extension("abbreviations"):

            for ent in doc.ents:
                # TODO: This is possibly sub-optimal - we might
                # prefer to look up both the long and short forms.
                if ent._.long_form is not None:
                    mentions.append(ent._.long_form)
                else:
                    mentions.append(ent)
        else:
            mentions = doc.ents

        mention_strings = [x.text for x in mentions]
        batch_candidates = self.candidate_generator(mention_strings, self.k)

        for mention, candidates in zip(doc.ents, batch_candidates):
            predicted = []
            for cand in candidates:
                score = max(cand.similarities)
                if self.filter_for_definitions and self.umls.cui_to_entity[cand.concept_id].definition is None:
                    continue
                if score > self.threshold:
                    predicted.append((cand.concept_id, score))
            sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[1])
            mention._.umls_ents = sorted_predicted[:self.max_entities_per_mention]

        return doc
