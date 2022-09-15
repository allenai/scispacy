import unittest
import spacy
import pytest

from scispacy.abbreviation import (
    AbbreviationDetector,
    find_abbreviation,
    filter_matches,
)


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
        matches_no_brackets = [(x[0], x[1] + 1, x[2] - 1) for x in matches]
        filtered = filter_matches(matches_no_brackets, doc)

        assert len(filtered) == 2
        long, short = filtered[0]
        assert long.text_with_ws == "Spinal and bulbar muscular atrophy "
        assert short.text == "SBMA"
        long, short = filtered[1]
        assert long.text_with_ws == "within the androgen receptor "
        assert short.text == "AR"

    def test_abbreviation_detection(self):
        # Attribute should be registered.
        doc = self.nlp(self.text)
        assert doc._.abbreviations == []
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 3

        correct = set()
        span = doc[33:34]
        span._.long_form = doc[0:5]
        correct.add(span)
        span = doc[6:7]
        span._.long_form = doc[0:5]
        correct.add(span)
        span = doc[29:30]
        span._.long_form = doc[26:28]
        correct.add(span)
        correct_long = {x._.long_form for x in correct}

        assert set(doc2._.abbreviations) == correct
        assert {x._.long_form for x in doc2._.abbreviations} == correct_long

    def test_find(self):
        doc = self.nlp(self.text)
        long, shorts = self.detector.find(doc[6:7], doc)
        assert long.text_with_ws == "Spinal and bulbar muscular atrophy "
        assert len(shorts) == 2
        assert {x.text_with_ws for x in shorts} == {"SBMA", "SBMA "}

        long, shorts = self.detector.find(doc[7:13], doc)
        assert shorts == set()

    def test_issue_158(self):
        text = (
            "The PVO observations showed that the total transterminator flux "
            "was 23% of that at solar maximum and that the largest reductions in the "
            "number of ions transported antisunward occurred at the highest altitudes "
            "(Spenner et al., 1995)."
        )
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

    def test_issue_192(self):
        # test for <short> (<long>) pattern
        text = "blah SBMA (Spinal and bulbar muscular atrophy)"
        doc = self.nlp(text)
        doc2 = self.detector(doc)

        assert len(doc2._.abbreviations) == 1
        assert doc2._.abbreviations[0] == doc[1:2]
        assert doc2._.abbreviations[0]._.long_form == doc[3:8]

    def test_issue_161(self):
        # test some troublesome cases in the abbreviation detector
        text = "H2)]+(14)s.t. (1), (4).Similarly"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

        text = ".(21)In (21), λ"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

        text = "map expX (·) : R"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

        text = "0,(3)with the following data: (3-i) (q̄"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

        text = "Φg(h),ThΦg(v) ) , (h, v)"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

        text = "dimension;(S-iii) The optimal control problem obtained in (S-ii) is con-verted"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

        text = "z), πut (z)) )"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

        text = "repositories he/she already worked with or from previous collaborators. Nevertheless, 88% of the first action of users to a repository (repository discovery) is"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

    def test_empty_span(self):
        text = "(19, 9, 4) Hadamard Designs and Their Residual Designs"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

    def test_space_issue(self):
        text = "by designing A Lite BERT (ALBERT) architecture that has significantly fewer parameters than a traditional BERT architecture."
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 1
        assert doc2._.abbreviations[0]._.long_form.text == "A Lite BERT"

    def test_multiple_spaces(self):
        text = "by      designing A     Lite BERT (ALBERT) architecture that has significantly fewer parameters than a traditional BERT architecture."
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 1
        assert doc2._.abbreviations[0]._.long_form.text == "A     Lite BERT"

    def test_issue_441(self):
        text = "The thyroid hormone receptor (TR) inhibiting retinoic malate receptor (RMR) isoforms mediate ligand-independent repression."
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 2

    @pytest.mark.xfail
    def test_difficult_cases(self):
        # Don't see an obvious way of solving these. They require something more semantic to distinguish
        text = "is equivalent to (iv) of Theorem"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0

        text = "or to fork.Users work more on their repositories (owners) than on"
        doc = self.nlp(text)
        doc2 = self.detector(doc)
        assert len(doc2._.abbreviations) == 0
