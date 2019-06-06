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
        doc_text_replaced, _ = self.detector.replace_text_for_regex(doc, chunks)

        match_string = ("NP_keystone_plant_specie such as NP_fig_tree")
        general = "NP_keystone_plant_specie"
        specifics = ["NP_fig_tree"]

        expected_extractions = [(general, specifics, match_string)]
        matches = self.detector.apply_hearst_patterns(doc_text_replaced)

        assert matches == expected_extractions

    def test_pipe_sentence(self):
        hyponym_pipe = HyponymDetector(extended=True)
        self.nlp.add_pipe(hyponym_pipe, last=True)

        text = ("Recognizing that the preferred habitats for the species "
                "are in the valleys, systematic planting of keystone plant "
                "species such as fig trees (Ficus) creates the best microhabitats.")

        match_string = ("NP_keystone_plant_specie such as NP_fig_tree")
        general = "keystone plant species"
        specifics = ["fig trees"]

        expected_extractions = [(general, specifics, match_string)]

        doc = self.nlp(text)
        assert doc._.hyponyms == expected_extractions

    def test_pipe_abstract(self):
        hyponym_pipe = HyponymDetector(extended=True)
        self.nlp.add_pipe(hyponym_pipe, last=True)

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

        match_string = ("NP_oscc and other NP_hnscc_subsite")
        general = "HNSCC subsites"
        specifics = ["OSCC"]

        expected_extractions = [(general, specifics, match_string)]

        doc = self.nlp(text)
        assert doc._.hyponyms == expected_extractions
