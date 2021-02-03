import unittest
import tempfile

import spacy

from scispacy.candidate_generation import CandidateGenerator, create_tfidf_ann_index
from scispacy.linking import EntityLinker
from scispacy.umls_utils import UmlsKnowledgeBase
from scispacy.abbreviation import AbbreviationDetector


class TestLinker(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.nlp = spacy.load("en_core_web_sm")

        umls_fixture = UmlsKnowledgeBase("tests/fixtures/umls_test_fixture.json", "tests/fixtures/test_umls_tree.tsv")
        with tempfile.TemporaryDirectory() as dir_name:
            umls_concept_aliases, tfidf_vectorizer, ann_index = create_tfidf_ann_index(dir_name, umls_fixture)
        candidate_generator = CandidateGenerator(ann_index, tfidf_vectorizer, umls_concept_aliases, umls_fixture)

        self.linker = EntityLinker(candidate_generator=candidate_generator, filter_for_definitions=False)

    def test_naive_entity_linking(self):
        text = "There was a lot of Dipalmitoylphosphatidylcholine."
        doc = self.nlp(text)

        # Check that the linker returns nothing if we set the filter_for_definitions flag
        # and set the threshold very high for entities without definitions.
        self.linker.filter_for_definitions = True
        self.linker.no_definition_threshold = 3.0
        doc = self.linker(doc)
        assert doc.ents[0]._.kb_ents == []

        # Check that the linker returns only high confidence entities if we
        # set the threshold to something more reasonable.
        self.linker.no_definition_threshold = 0.95
        doc = self.linker(doc)
        assert doc.ents[0]._.kb_ents == [("C0000039", 1.0)]

        self.linker.filter_for_definitions = False
        self.linker.threshold = 0.45
        doc = self.linker(doc)
        # Without the filter_for_definitions filter, we get 2 entities for
        # the first mention.
        assert len(doc.ents[0]._.kb_ents) == 2

        id_with_score = doc.ents[0]._.kb_ents[0]
        assert id_with_score == ("C0000039", 1.0)
        umls_entity = self.linker.kb.cui_to_entity[id_with_score[0]]
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
