# pylint: disable=no-self-use,invalid-name
import unittest
import spacy

from scispacy.hyponym_detector import HyponymDetector


class TestHyponymDetector(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.nlp = spacy.load("en_core_sci_sm")
        self.detector = HyponymDetector(self.nlp, extended=True)
        self.nlp.add_pipe(self.detector, last=True)

    def test_apply_hearst_patterns(self):
        text = ("Recognizing that the preferred habitats for the species "
                "are in the valleys, systematic planting of keystone plant "
                "species such as fig trees (Ficus) creates the best microhabitats.")
        doc = self.nlp(text)

        assert doc._.hearst_patterns == [('such_as', 'species', 'fig')]
        print(doc._.hearst_patterns)

    def test_pipe_sentence(self):

        text = ("Recognizing that the preferred habitats for the species "
                "are in the valleys, systematic planting of keystone plant "
                "species such as fig trees (Ficus) creates the best microhabitats.")

        doc = self.nlp(text)
        print(doc._.hearst_patterns)
        # assert doc._.hyponyms == expected_extractions

    def test_pipe_abstract(self):

        text = ("HERG1 potassium channel plays a critical role in the cell proliferation. "
                "HERG1 protein expression was analyzed by immunohistochemistry (IHC) in 62 "
                "patients with oral leukoplakias and 100 patients with oral squamous cell "
                "carcinomas (OSCC). HERG1 mRNA levels were assessed by real-time reverse "
                "transcriptase-polymerase chain reaction (RT-PCR) in 22 patients with primary "
                "head and neck squamous cell carcinoma (HNSCC). Statistically significant "
                "associations were found between HERG1 expression and tobacco consumption, disease stage, "
                "tumor differentiation, tumor recurrence, and reduced survival. There was no association "
                "between HERG1 expression and the risk of progression from oral leukoplakia to OSCC. "
                "In addition, a high proportion of tumors (80%) showed increased HERG1 mRNA levels "
                "compared to normal mucosa from nononcologic patients. Aberrant HERG1 expression "
                "increases as oral tumorigenesis progresses from oral hyperplasia to OSCC. "
                "Increased HERG1 mRNA levels were also frequently detected in OSCC and other "
                "HNSCC subsites. HERG1 expression emerges as a clinically relevant feature "
                "during tumor progression and a potential poor prognostic biomarker for OSCC. "
                "© 2016 Wiley Periodicals, Inc. Head Neck 38: 1708-1716, 2016.")

        doc = self.nlp(text)
        print(doc._.hearst_patterns)
        # assert doc._.hyponyms == expected_extractions
