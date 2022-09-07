# pylint: disable=no-self-use,invalid-name
import unittest
import spacy

from scispacy.hyponym_detector import HyponymDetector


class TestHyponymDetector(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.nlp = spacy.load("en_core_sci_sm")
        self.detector = HyponymDetector(self.nlp, extended=True)
        self.nlp.add_pipe("hyponym_detector", config={"extended": True}, last=True)

    def test_sentences(self):
        text = (
            "Recognizing that the preferred habitats for the species "
            "are in the valleys, systematic planting of keystone plant "
            "species such as fig trees (Ficus) creates the best microhabitats."
        )
        doc = self.nlp(text)
        fig_trees = doc[21:23]
        plant_species = doc[16:19]
        assert doc._.hearst_patterns == [("such_as", plant_species, fig_trees)]

        doc = self.nlp("SARS, or other coronaviruses, are bad.")
        assert doc._.hearst_patterns == [("other", doc[4:5], doc[0:1])]
        doc = self.nlp("Coronaviruses, including SARS and MERS, are bad.")
        assert doc._.hearst_patterns == [
            ("include", doc[0:1], doc[3:4]),
            ("include", doc[0:1], doc[5:6]),
        ]

    def test_find_noun_compound_head(self):

        doc = self.nlp("The potassium channel is good.")

        head = self.detector.find_noun_compound_head(doc[1])
        assert head == doc[2]

        doc = self.nlp("Planting of large plants.")
        head = self.detector.find_noun_compound_head(doc[3])
        # Planting is a noun, but not a compound with 'plants'.
        assert head != doc[0]
        assert head == doc[3]

    def test_expand_noun_phrase(self):
        doc = self.nlp("Keystone plant habitats are good.")
        chunk = self.detector.expand_to_noun_compound(doc[1], doc)
        assert chunk == doc[0:3]
