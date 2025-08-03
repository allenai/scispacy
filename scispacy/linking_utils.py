"""
This submodule contains a data structure for storing lexical
indexes over various biomedical vocabularies.

There are several built-in vocabularies, which can be imported
and instantiated like in:

.. code-block:: python

    from scispacy.linking_utils import UmlsKnowledgeBase

    kb = UmlsKnowledgeBase()

In general, new :class:`KnowledgeBase` objects can be constructed
from a list of :class:`Entity` objects, or a path to a JSON or JSONL
file containing dictionaries shaped the same way:

.. code-block:: python

    from scispacy.linking_utils import KnowledgeBase

    # UMLS
    kb = KnowledgeBase("https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/kbs/2023-04-23/umls_mesh_2022.jsonl")

New, :class:`KnowledgeBase` subclasses can be defined based on
:mod:`pyobo` (after running ``pip install pyobo``) like in the following:

.. code-block:: python

    from scispacy.linking_utils import KnowledgeBase, entities_from_pyobo

    class PlantTraitOntology(KnowledgeBase):
        def __init__(self) -> None:
            # see https://bioregistry.io/registry/to
            super().__init__(entities_from_pyobo("to"))

    kb = PlantTraitOntology()

New _ad-hoc_ :class:`KnowledgeBase` instances can be constructed
using :mod:`pyobo` like in the following, using the
`Plant Trait Ontology <https://bioregistry.io/to>`_ as an example:

.. code-block:: python

    from scispacy.linking_utils import KnowledgeBase

    kb = KnowledgeBase.from_pyobo("to")


Here's a recipe for using a SPARQL query on Wikidata to generate
a KnowledgeBase object (see discussion at https://github.com/allenai/scispacy/issues/346):

.. code-block:: python

    from scispacy.linking_utils import KnowledgeBase, entities_from_wikidata

    # this SPARQL query gets named cats
    sparql = '''\
        SELECT ?item ?itemLabel ?itemDescription ?itemAltLabel
        WHERE
        {
          ?item wdt:P31 wd:Q146. # Must be a cat
          SERVICE wikibase:label {
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en".
          }
        }
    '''
    entities = entities_from_wikidata(sparql, "item")
    kb = KnowledgeBase(entities)

You can go further and create a linker using the following:

.. code-block:: python

    import tempfile
    import spacy
    from scispacy.candidate_generation import CandidateGenerator, create_tfidf_ann_index
    from scispacy.linking_utils import KnowledgeBase
    from scispacy.linking import EntityLinker

    kb = KnowledgeBase.from_pyobo("to")

    with tempfile.TemporaryDirectory() as directory:
        concept_aliases, tfidf_vectorizer, ann_index = create_tfidf_ann_index(directory, kb)

    # Create the generator and linker
    candidate_generator = CandidateGenerator(ann_index, tfidf_vectorizer, concept_aliases, kb)
    linker = EntityLinker(candidate_generator=candidate_generator, filter_for_definitions=False)

    # now, put it all together with a NER model
    nlp = spacy.load("en_core_web_sm")
    doc = linker(nlp(text))
"""

from typing import (
    List,
    Dict,
    NamedTuple,
    Optional,
    Set,
    Union,
    Iterable,
    TYPE_CHECKING,
    Tuple,
    DefaultDict,
    Generator,
    Any,
)
import json
from collections import defaultdict
from contextlib import contextmanager
from typing_extensions import Self

from scispacy.file_cache import cached_path
from scispacy.umls_semantic_type_tree import (
    UmlsSemanticTypeTree,
    construct_umls_tree_from_tsv,
)

if TYPE_CHECKING:
    import pyobo

__all__ = [
    "Entity",
    "entities_from_wikidata",
    "entity_from_wikidata",
    "KnowledgeBase",
    "entities_from_pyobo",
    "UmlsKnowledgeBase",
    "Mesh",
    "GeneOntology",
    "HumanPhenotypeOntology",
    "RxNorm",
    "PlantTraitOntology",
]


class Entity(NamedTuple):
    concept_id: str
    canonical_name: str
    aliases: List[str]
    types: List[str] = []
    definition: Optional[str] = None

    def __repr__(self):
        rep = ""
        num_aliases = len(self.aliases)
        rep = rep + f"CUI: {self.concept_id}, Name: {self.canonical_name}\n"
        rep = rep + f"Definition: {self.definition}\n"
        rep = rep + f"TUI(s): {', '.join(self.types)}\n"
        if num_aliases > 10:
            rep = (
                rep
                + f"Aliases (abbreviated, total: {num_aliases}): \n\t {', '.join(self.aliases[:10])}"
            )
        else:
            rep = (
                rep + f"Aliases: (total: {num_aliases}): \n\t {', '.join(self.aliases)}"
            )
        return rep


DEFAULT_UMLS_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/kbs/2023-04-23/umls_2022_ab_cat0129.jsonl"  # noqa
DEFAULT_UMLS_TYPES_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/umls_semantic_type_tree.tsv"


class KnowledgeBase:
    """
    A class representing two commonly needed views of a Knowledge Base:
    1. A mapping from concept_id to an Entity NamedTuple with more information.
    2. A mapping from aliases to the sets of concept ids for which they are aliases.

    Parameters
    ----------
    file_path: str, required.
        The file path to the json/jsonl representation of the KB to load.
    """

    cui_to_entity: Dict[str, Entity]
    alias_to_cuis: Dict[str, Set[str]]

    def __init__(
        self,
        file_path: Union[None, str, Iterable[Entity]] = None,
    ):
        if file_path is None:
            raise ValueError(
                "Do not use the default arguments to KnowledgeBase. "
                "Instead, use a subclass (e.g UmlsKnowledgeBase) or pass a path to a kb."
            )
        with _iter_entities(file_path) as entities:
            self.cui_to_entity, self.alias_to_cuis = _index_entities(entities)

    @classmethod
    def from_pyobo(cls, prefix: str) -> Self:
        """Construct a biomedical knowledgebase from PyOBO."""
        return cls(entities_from_pyobo(prefix))


@contextmanager
def _iter_entities(
    path_or_entities: str | Iterable[Entity],
) -> Generator[Iterable[Entity], None, None]:
    if isinstance(path_or_entities, str):
        # do the following inside a context manager to
        # make sure the file gets closed properly
        with open(cached_path(path_or_entities)) as file:
            if path_or_entities.endswith("jsonl"):
                yield (Entity(**json.loads(line)) for line in file)
            else:
                yield (Entity(**record) for record in json.load(file))
    else:
        yield path_or_entities


def _index_entities(
    entities: Iterable[Entity],
) -> Tuple[Dict[str, Entity], Dict[str, Set[str]]]:
    """Create indexes over entities for use in a :class:`KnowledgeBase`.

    :param entities: An iterable (e.g., a list) of entity objects
    :returns: A pair of indexes for:

        1. A mapping from local unique identifiers (e.g., CUIs for UMLS) to entity objects
        2. A mapping from aliases (e.g., canonical names, aliases) to local unique identifiers
    """
    cui_to_entity: Dict[str, Entity] = {}
    alias_to_cuis: DefaultDict[str, Set[str]] = defaultdict(set)
    for entity in entities:
        alias_to_cuis[entity.canonical_name].add(entity.concept_id)
        for alias in entity.aliases:
            alias_to_cuis[alias].add(entity.concept_id)
        cui_to_entity[entity.concept_id] = entity
    return cui_to_entity, dict(alias_to_cuis)


def entities_from_pyobo(prefix: str, **kwargs: Any) -> Iterable[Entity]:
    """Iterate over entities in a given ontology via :mod:`pyobo`.

    :param prefix:
        The ontology's prefix, such as ``go` for Gene Ontology,
        ``doid`` for the Disease Ontology, or more.
    :param kwargs:
        keyword arguments to pass to :func:`pyobo.get_ontology`, such as ``version``.
    :yields:
        Entity objects for all terms in the ontology

    .. note:: several :class:`KnowledgeBase` subclasses could be re-implemented using this
    """
    import pyobo

    for term in pyobo.get_ontology(prefix, **kwargs):
        yield Entity(
            concept_id=term.curie,
            canonical_name=term.name,
            aliases=[s.name for s in term.synonyms],
            definition=term.definition,
        )


WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"


def entities_from_wikidata(sparql: str, key: str) -> List[Entity]:
    """Query Wikidata with SPARQL and return entities.

    :param sparql: A SPARQL query over Wikidata
    :param key:
        If the select line in your SPARQL looks like ``SELECT ?item ?itemLabel ?itemDescription ?itemAltLabel``,
        then give ``item`` for this
    :return: A list of entities extracted from Wikidata
    """
    import requests

    res = requests.get(
        WIKIDATA_SPARQL_ENDPOINT,
        params={"query": sparql, "format": "json"},
    )
    res.raise_for_status()
    entities = [
        entity
        for record in res.json()["results"]["bindings"]
        if (entity := entity_from_wikidata(record, key))
    ]
    return entities


def entity_from_wikidata(record: Dict[str, Any], key: str) -> Optional[Entity]:
    """Construct an entity from a Wikidata record.

    :param record: The wikidata record
    :param key: The name of the variable that is autofilled with a label, description, and alts
    :return: An entity, or none if there is no label available
    """
    label = record[f"{key}Label"]["value"]
    if not label:
        return None
    return Entity(
        concept_id=record[key]["value"].removeprefix("http://www.wikidata.org/entity/"),
        canonical_name=label,
        aliases=record[f"{key}AltLabel"]["value"].split(", ")
        if f"{key}AltLabel" in record
        else [],
        definition=record[f"{key}Description"]["value"] or None,
    )


class UmlsKnowledgeBase(KnowledgeBase):
    def __init__(
        self,
        file_path: str = DEFAULT_UMLS_PATH,
        types_file_path: str = DEFAULT_UMLS_TYPES_PATH,
    ):
        super().__init__(file_path)

        self.semantic_type_tree: UmlsSemanticTypeTree = construct_umls_tree_from_tsv(
            types_file_path
        )


class Mesh(KnowledgeBase):
    def __init__(
        self,
        file_path: str = "https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/kbs/2023-04-23/umls_mesh_2022.jsonl",  # noqa
    ):
        super().__init__(file_path)


class GeneOntology(KnowledgeBase):
    def __init__(
        self,
        file_path: str = "https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/kbs/2023-04-23/umls_go_2022.jsonl",  # noqa
    ):
        super().__init__(file_path)


class HumanPhenotypeOntology(KnowledgeBase):
    def __init__(
        self,
        file_path: str = "https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/kbs/2023-04-23/umls_hpo_2022.jsonl",  # noqa
    ):
        super().__init__(file_path)


class RxNorm(KnowledgeBase):
    def __init__(
        self,
        file_path: str = "https://ai2-s2-scispacy.s3-us-west-2.amazonaws.com/data/kbs/2023-04-23/umls_rxnorm_2022.jsonl",  # noqa
    ):
        super().__init__(file_path)


class PlantTraitOntology(KnowledgeBase):
    def __init__(self) -> None:
        # see https://bioregistry.io/registry/to
        super().__init__(entities_from_pyobo("to"))
