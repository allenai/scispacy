# pylint: disable=no-self-use,invalid-name
import unittest
import json
import tempfile

import spacy

from scispacy.candidate_generation import CandidateGenerator, create_tfidf_ann_index, MentionCandidate
from scispacy.umls_linking import UmlsEntityLinker

class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.nlp = spacy.load("en_core_web_sm")

        umls_fixture = json.load(open("tests/fixtures/umls_test_fixture.json"))
        with tempfile.TemporaryDirectory() as dir_name:
            umls_concept_aliases, tfidf_vectorizer, ann_index = create_tfidf_ann_index(dir_name, umls_fixture)

        candidate_generator = CandidateGenerator(ann_index, tfidf_vectorizer, umls_concept_aliases, umls_fixture)

        self.linker = UmlsEntityLinker(candidate_generator)

    def test_candidate_generation(self):

        #results = self.candidate_generator.generate_candidates(['(131)I-Macroaggregated Albumin'], 10)


        doc = self.nlp("There was a lot of (131)I-Macroaggregated Albumin.")

        # Ents are completely wrong from the web spacy model, correct them manually.
        doc.ents = [doc[5:10]]

        linked = self.linker(doc)


        # canonical_ids = [x.concept_id for x in results[0]]
        # assert canonical_ids == ['C0000005', 'C0000102', 'C0000084']

        # # The mention was an exact match, so should have a distance of zero to a concept:
        # assert results[0][0] == MentionCandidate(concept_id='C0000005',
        #                                          aliases=['(131)I-Macroaggregated Albumin'],
        #                                          distances=[0.0])

