from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.language import Language

from scispacy.candidate_generation import CandidateGenerator


@Language.factory("scispacy_linker")
class EntityLinker:
    """
    A spacy pipeline component which identifies entities in text which appear
    in a knowledge base.

    Currently, there are two defaults: the Unified Medical Language System (UMLS) and
    the Medical Subject Headings (MESH) dictionary.

    To use these configured default KBs, pass the `name` parameter, either 'umls' or 'mesh'.

    Currently this implementation just compares string similarity, returning
    entities above a given threshold.

    This class sets the `._.kb_ents` attribute on spacy Spans, which consists of a
    List[Tuple[str, float]] corresponding to the KB concept_id and the associated score
    for a list of `max_entities_per_mention` number of entities.

    You can look up more information for a given id using the kb attribute of this class:

    print(linker.kb.cui_to_entity[concept_id])

    A Note on Definitions:
    Only 187767 entities, or 6.74% of the UMLS KB have definitions. However,
    the MedMentions dataset links to entities which have definitions 82.9% of the time. So by
    default, we only link to entities which have definitions (typically they are more salient / cleaner),
    but this might not suit your use case. YMMV.


    Parameters
    ----------

    nlp: `Language`, a required argument for spacy to use this as a factory
    name: `str`, a required argument for spacy to use this as a factory
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
        The threshold that a entity candidate must reach to be added to the mention in the Doc
        as a mention candidate.
    no_definition_threshold : float, optional, (default = 0.95)
        The threshold that a entity candidate must reach to be added to the mention in the Doc
        as a mention candidate if the entity candidate does not have a definition.
    filter_for_definitions: bool, default = True
        Whether to filter entities that can be returned to only include those with definitions
        in the knowledge base.
    max_entities_per_mention : int, optional, default = 5
        The maximum number of entities which will be returned for a given mention, regardless of
        how many are nearest neighbours are found.
    linker_name: str, optional (default = None)
        The name of the pretrained entity linker to load.
    """

    def __init__(
        self,
        nlp: Language = None,
        name: str = "scispacy_linker",
        candidate_generator: CandidateGenerator = None,
        resolve_abbreviations: bool = True,
        k: int = 30,
        threshold: float = 0.7,
        no_definition_threshold: float = 0.95,
        filter_for_definitions: bool = True,
        max_entities_per_mention: int = 5,
        linker_name: str = None,
    ):
        # TODO(Mark): Remove in scispacy v1.0.
        Span.set_extension("umls_ents", default=[], force=True)
        Span.set_extension("kb_ents", default=[], force=True)

        self.candidate_generator = candidate_generator or CandidateGenerator(
            name=linker_name
        )
        self.resolve_abbreviations = resolve_abbreviations
        self.k = k
        self.threshold = threshold
        self.no_definition_threshold = no_definition_threshold
        self.kb = self.candidate_generator.kb
        self.filter_for_definitions = filter_for_definitions
        self.max_entities_per_mention = max_entities_per_mention

        # TODO(Mark): Remove in scispacy v1.0. This is for backward compatability only.
        self.umls = self.kb

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
                if (
                    self.filter_for_definitions
                    and self.kb.cui_to_entity[cand.concept_id].definition is None
                    and score < self.no_definition_threshold
                ):
                    continue
                if score > self.threshold:
                    predicted.append((cand.concept_id, score))
            sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[1])
            mention._.umls_ents = sorted_predicted[: self.max_entities_per_mention]
            mention._.kb_ents = sorted_predicted[: self.max_entities_per_mention]

        return doc
