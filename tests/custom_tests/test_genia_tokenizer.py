# pylint: disable=invalid-name,line-too-long
import pytest
import spacy

from scispacy.genia_tokenizer import GeniaTokenizer
from scispacy.custom_tokenizer import remove_new_lines

TEST_CASES = [("using a bag-of-words model", ["using", "a", "bag-of-words", "model"]),
              ("activators of cAMP- and cGMP-dependent protein", ["activators", "of", "cAMP-", "and", "cGMP-dependent", "protein"]),
              ("phorbol 12-myristate 13-acetate, caused almost", ["phorbol", "12-myristate", "13-acetate", ",", "caused", "almost"]),
              ("let C(j) denote", ["let", "C(j)", "denote"]),
              ("let (C(j)) denote", ["let", "(", "C(j)", ")", "denote"]),
              ("let C{j} denote", ["let", "C{j}", "denote"]),
              ("for the camera(s) and manipulator(s)", ["for", "the", "camera(s)", "and", "manipulator(s)"]),
              ("the (TRAP)-positive genes", ["the", "(TRAP)-positive", "genes"]),
              ("the {TRAP}-positive genes", ["the", "{TRAP}-positive", "genes"]),
              ("for [Ca2+]i protein", ["for", "[Ca2+]i", "protein"]),
              ("for pyrilamine[3H] protein", ["for", "pyrilamine[3H]", "protein"]),
              ("this is (normal) parens", ["this", "is", "(", "normal", ")", "parens"]),
              ("this is [normal] brackets", ["this", "is", "[", "normal", "]", "brackets"]),
              ("this is {normal} braces", ["this", "is", "{", "normal", "}", "braces"]),
              ("in the lan-\nguage of the", ["in", "the", "language", "of", "the"]),
              ("in the lan-\n\nguage of the", ["in", "the", "language", "of", "the"]),
              ("in the lan- \nguage of the", ["in", "the", "language", "of", "the"]),
              ("in the lan- \n\nguage of the", ["in", "the", "language", "of", "the"]),
              ("a 28× 28 image", ["a", "28", "×", "28", "image"]),
              ("a 28×28 image", ["a", "28", "×", "28", "image"]),
              ("a 28 × 28 image", ["a", "28", "×", "28", "image"]),
              ("the neurons’ activation", ["the", "neurons", "’", "activation"]),
              ("the neurons' activation", ["the", "neurons", "'", "activation"]),
              ("H3G 1Y6", ["H3G", "1Y6"]),
              ("HFG 1Y6", ["HFG", "1Y6"]),
              ("H3g 1Y6", ["H3g", "1Y6"]),
              ("h3g 1Y6", ["h3g", "1Y6"]),
              ("h36g 1Y6", ["h36g", "1Y6"]),
              ("h3gh 1Y6", ["h3gh", "1Y6"]),
              ("h3g3 1Y6", ["h3g3", "1Y6"]),
              ("interleukin (IL)-mediated", ["interleukin", "(IL)-mediated"]),
              ("This is a sentence.", ["This", "is", "a", "sentence", "."]),
              ("result of 1.345 is good", ["result", "of", "1.345", "is", "good"]),
              ("This sentence ends with a single 1.", ["This", "sentence", "ends", "with", "a", "single", "1", "."]),
              ("sec. secs. Sec. Secs. fig. figs. Fig. Figs. eq. eqs. Eq. Eqs. no. nos. No. Nos. al. .", ["sec.", "secs.", "Sec.", "Secs.", "fig.", "figs.", "Fig.", "Figs.", "eq.", "eqs.", "Eq.", "Eqs.", "no.", "nos.", "No.", "Nos.", "al.", "."]),
              ("in the Gq/G11 protein", ["in", "the", "Gq/G11", "protein"]),
              ("in the G1/G11 protein", ["in", "the", "G1/G11", "protein"]),
              ("in the G1/11 protein", ["in", "the", "G1/11", "protein"]),
              ("in the Gq/11 protein", ["in", "the", "Gq/11", "protein"]),
             ]


class TestGeniaTokenizer:

    nlp = spacy.load("en_core_web_sm")

    genia_tokenizer = GeniaTokenizer(nlp.vocab)
    nlp.tokenizer = genia_tokenizer

    def test_concrete_sentences(self):
        for text, expected_tokens in TEST_CASES:
            text = remove_new_lines(text)
            assert [t.text for t in self.nlp(text)] == expected_tokens

    @pytest.mark.parametrize('text', ["lorem ipsum"])
    def test_tokenizer_splits_single_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 2
        assert [t.whitespace_ == " " for t in tokens] == [True, False]


    @pytest.mark.parametrize('text', ["lorem  ipsum"])
    def test_tokenizer_splits_double_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
        assert tokens[1].text == " "
        assert [t.whitespace_ == " " for t in tokens] == [True, False, False]


    @pytest.mark.parametrize('text', ["lorem ipsum  "])
    def test_tokenizer_handles_double_trainling_ws(self, text):
        tokens = self.nlp(text)
        assert repr(tokens.text_with_ws) == repr(text)


    @pytest.mark.parametrize('text', ["lorem\nipsum"])
    def test_tokenizer_splits_newline(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
        assert tokens[1].text == "\n"


    @pytest.mark.parametrize('text', ["lorem \nipsum"])
    def test_tokenizer_splits_newline_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3


    @pytest.mark.parametrize('text', ["lorem  \nipsum"])
    def test_tokenizer_splits_newline_double_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
        assert [t.whitespace_ == " " for t in tokens] == [True, False, False]

    @pytest.mark.parametrize('text', ["lorem \n ipsum"])
    def test_tokenizer_splits_newline_space_wrap(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
        assert tokens[1].text == "\n "
        assert [t.whitespace_ == " " for t in tokens] == [True, False, False]