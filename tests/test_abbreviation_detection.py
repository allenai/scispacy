# pylint: disable=no-self-use,invalid-name
import unittest
import spacy

from scispacy.abbreviation import AbbreviationDetector, find_abbreviation, filter_matches

class TestAbbreviationDetector(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.nlp = spacy.load("en_core_web_sm")
        self.detector = AbbreviationDetector(self.nlp)
        self.text = "Spinal and bulbar muscular atrophy (SBMA) is an \
                inherited motor neuron disease caused by the expansion \
                of a polyglutamine tract within the androgen receptor (AR). \
                SBMA can be caused by this easily."

    def test_find_abbreviation(self):
        # Basic case
        doc = self.nlp("abbreviation (abbrn)")
        long = doc[0:1]
        short = doc[2:3]
        _, long_form = find_abbreviation(long, short)
        assert long_form.text == "abbreviation"

        # Hypenation and numbers within abbreviation
        doc = self.nlp("abbreviation (ab-b9rn)")
        long = doc[0:1]
        short = doc[2:3]
        _, long_form = find_abbreviation(long, short)
        assert long_form.text == "abbreviation"

        # No match
        doc = self.nlp("abbreviation (aeb-b9rn)")
        long = doc[0:1]
        short = doc[2:3]
        _, long_form = find_abbreviation(long, short)
        assert long_form is None

        # First letter must match start of word.
        doc = self.nlp("aaaabbreviation (ab-b9rn)")
        long = doc[0:1]
        short = doc[2:3]
        _, long_form = find_abbreviation(long, short)
        assert long_form.text == "aaaabbreviation"

        # Matching is greedy for first letter (are is not included).
        doc = self.nlp("more words are considered aaaabbreviation (ab-b9rn)")
        long = doc[0:5]
        short = doc[6:7]
        _, long_form = find_abbreviation(long, short)
        assert long_form.text == "aaaabbreviation"

    def test_filter_matches(self):
        doc = self.nlp(self.text)
        matches = self.detector.matcher(doc)
        matches_no_brackets = [(x[0], x[1] + 1, x[2] -1) for x in matches]
        filtered = filter_matches(matches_no_brackets, doc)

        assert len(filtered) == 2
        long, short  = filtered[0]
        assert long.string == "Spinal and bulbar muscular atrophy "
        assert short.string == "SBMA"
        long, short = filtered[1]
        assert long.string == "within the androgen receptor "
        assert short.string == "AR"

    def test_abbreviation_detection(self):
        # Attribute should be registered.
        doc = self.nlp(self.text)
        assert doc._.abbreviations == []
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 2

        long, shorts = doc2._.abbreviations[0]
        assert long.string == "Spinal and bulbar muscular atrophy "
        assert len(shorts) == 2
        assert {x.string for x in shorts} == {"SBMA", "SBMA "}

        long, shorts = doc2._.abbreviations[1]
        assert long.string == "androgen receptor "
        assert len(shorts) == 1
        assert shorts.pop().string == "AR"

    def test_find(self):
        doc = self.nlp(self.text)
        long, shorts = self.detector.find(doc[6:7], doc)
        assert long.string == "Spinal and bulbar muscular atrophy "
        assert len(shorts) == 2
        assert {x.string for x in shorts} == {"SBMA", "SBMA "}

        long, shorts = self.detector.find(doc[7:13], doc)
        assert shorts == set()
