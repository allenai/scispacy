import unittest
import tempfile

import spacy

from scispacy.candidate_generation import CandidateGenerator, create_tfidf_ann_index
from scispacy.linking import EntityLinker
from scispacy.linking_utils import Entity, KnowledgeBase, _iter_entities
from scispacy.umls_utils import UmlsKnowledgeBase
from scispacy.abbreviation import AbbreviationDetector
from scispacy.util import scipy_supports_sparse_float16


class TestLinker(unittest.TestCase):
    def setUp(self):
        super().setUp()
        if not scipy_supports_sparse_float16():
            # https://github.com/allenai/scispacy/issues/519#issuecomment-2229915999
            self.skipTest("Candidate generation isn't supported for scipy>=1.11")

        self.nlp = spacy.load("en_core_web_sm")

        umls_fixture = UmlsKnowledgeBase("tests/fixtures/umls_test_fixture.json", "tests/fixtures/test_umls_tree.tsv")
        self.linker = EntityLinker.from_kb(umls_fixture, filter_for_definitions=False)

    def test_naive_entity_linking(self):
        self._test_linker(self.linker)

    def test_custom_loading(self):
        with _iter_entities("tests/fixtures/umls_test_fixture.json") as entities:
            kb = UmlsKnowledgeBase(entities,"tests/fixtures/test_umls_tree.tsv")
        linker = EntityLinker.from_kb(kb, filter_for_definitions=False)
        self._test_linker(linker)

    def _test_linker(self, linker: EntityLinker) -> None:
        text = "There was a lot of Dipalmitoylphosphatidylcholine."
        doc = self.nlp(text)

        # Check that the linker returns nothing if we set the filter_for_definitions flag
        # and set the threshold very high for entities without definitions.
        linker.filter_for_definitions = True
        linker.no_definition_threshold = 3.0
        doc = linker(doc)
        assert doc.ents[0]._.kb_ents == []

        # Check that the linker returns only high confidence entities if we
        # set the threshold to something more reasonable.
        linker.no_definition_threshold = 0.95
        doc = linker(doc)
        assert doc.ents[0]._.kb_ents == [("C0000039", 1.0)]

        linker.filter_for_definitions = False
        linker.threshold = 0.45
        doc = linker(doc)
        # Without the filter_for_definitions filter, we get 2 entities for
        # the first mention.
        assert len(doc.ents[0]._.kb_ents) == 2

        id_with_score = doc.ents[0]._.kb_ents[0]
        assert id_with_score == ("C0000039", 1.0)
        umls_entity = linker.kb.cui_to_entity[id_with_score[0]]
        assert umls_entity.concept_id == "C0000039"
        assert umls_entity.types == ["T109", "T121"]

    def test_linker_resolves_abbreviations(self):

        self.nlp.add_pipe("abbreviation_detector")
        # replace abbreivation with "CNN" so spacy recognizes at as en entity
        # and also prefix the term with "CNN" so that abbreviation detector passes
        text = "CNN1-Methyl-4-phenylpyridinium (CNN) is an abbreviation which doesn't exist in the baby index."
        doc = self.nlp(text)
        doc = self.linker(doc)

        id_with_score = doc.ents[0]._.kb_ents[0]
        assert id_with_score == ("C0000098", 0.9819725155830383)
        umls_entity = self.linker.kb.cui_to_entity[id_with_score[0]]
        assert umls_entity.concept_id == "C0000098"

    def test_linker_has_types(self):
        # Just checking that the type tree is accessible from the linker
        assert len(self.linker.kb.semantic_type_tree.flat_nodes) == 6
