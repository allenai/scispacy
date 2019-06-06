# pylint: disable=no-self-use,invalid-name
import unittest
import spacy

from scispacy.hearst_patterns import HyponymDetector

class TestHyponymDetector(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.nlp = spacy.load("en_core_sci_sm")
        self.detector = HyponymDetector(extended=True)

    def test_hyponym_cleanup(self):
        text = "NP_the_sole_carbon_source"
        cleaned_text = self.detector.clean_hyponym_text(text)

        assert cleaned_text == "carbon source"

    def test_get_chunks(self):
        text = ("This strain was named CCA53, and its lignin-degrading "
                "capability was assessed by observing its growth on medium "
                "containing alkali lignin or lignin -associated aromatic "
                "monomers as the sole carbon source.")

        expected_chunks = [
                            "CCA53",
                            "This strain",
                            "alkali lignin or lignin -associated aromatic monomers",
                            "its growth",
                            "its lignin-degrading capability",
                            "medium",
                            "the sole carbon source"
                          ]

        chunks = sorted(list(self.detector.get_chunks(self.nlp(text))), key=lambda x: x.text)
        assert all([chunk.text == expected_chunk for chunk, expected_chunk in zip(chunks, expected_chunks)])

    def test_apply_hearst_patterns(self):
        text = ("Recognizing that the preferred habitats for the species "
                "are in the valleys, systematic planting of keystone plant "
                "species such as fig trees (Ficus) creates the best microhabitats.")
        doc = self.nlp(text)
        chunks = self.detector.get_chunks(doc)
        doc_text_replaced = self.detector.replace_text_for_regex(doc, chunks)

        match_string = ("NP_keystone_plant_species such as NP_fig_trees")
        general = "NP_keystone_plant_species"
        specifics = ["NP_fig_trees"]

        expected_extractions = [(general, specifics, match_string)]
        matches = self.detector.apply_hearst_patterns(doc_text_replaced)

        assert matches == expected_extractions

    def test_pipe(self):
        hyponym_pipe = HyponymDetector(extended=True)
        self.nlp.add_pipe(hyponym_pipe, last=True)

        text = ("Recognizing that the preferred habitats for the species "
                "are in the valleys, systematic planting of keystone plant "
                "species such as fig trees (Ficus) creates the best microhabitats.")

        match_string = ("NP_keystone_plant_species such as NP_fig_trees")
        general = "keystone plant species"
        specifics = ["fig trees"]

        expected_extractions = [(general, specifics, match_string)]

        doc = self.nlp(text)
        assert doc._.hyponyms == expected_extractions
