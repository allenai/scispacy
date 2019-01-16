"""Test that tokens are created correctly for whitespace."""

import spacy

from scispacy.genia_tokenizer import GeniaTokenizer, create_text_with_whitespace, split_whitespace_tokenized_string


class TestGeniaWhitespace:

    nlp = spacy.load("en_core_web_sm")
    genia_tokenizer = GeniaTokenizer(nlp.vocab)
    
    def test_splitting_with_multiple_spaces(self):

        sentence = "Hello    ,Mark"
        sentence_with_whitespace = create_text_with_whitespace(sentence)

        print(sentence)
        print(sentence_with_whitespace)
        split_string = split_whitespace_tokenized_string(sentence, sentence_with_whitespace)

        print(split_string)
    def test2(self):

        sentence = "Here, we Mark"
        sentence_with_whitespace = create_text_with_whitespace(sentence)

        print(sentence)
        print(sentence_with_whitespace)
        split_string = split_whitespace_tokenized_string(sentence, sentence_with_whitespace)

        print(split_string)

    def test3(self):

        sentence = "This is a sentence."
        sentence_with_whitespace = create_text_with_whitespace(sentence)

        print(sentence)
        print(sentence_with_whitespace)
        split_string = split_whitespace_tokenized_string(sentence, sentence_with_whitespace)

        print(split_string)

    def test4(self):

        sentence = "This is \n\n  \ta sentence."
        sentence_with_whitespace = create_text_with_whitespace(sentence)

        print(sentence)
        print(sentence_with_whitespace)
        split_string = split_whitespace_tokenized_string(sentence, sentence_with_whitespace)

        print(split_string)

    def test5(self):

        sentence = "This is a sentence.  "
        sentence_with_whitespace = create_text_with_whitespace(sentence)

        print(sentence)
        print(sentence_with_whitespace)
        split_string = split_whitespace_tokenized_string(sentence, sentence_with_whitespace)

        print(split_string)