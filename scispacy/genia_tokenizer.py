# Implements a GENIA Treebank - like tokenization.

# This is a python 3.6 translation of Patrick Verga's GTB tokenizer
# found here: https://github.com/patverga/bran/blob/master/src/processing/utils/genia_tokenizer.py

# NOTE: The tokenizer is also modified to _not_ replace brackets with -LRB-
# and be completely non-destructive, meaning that some regexes have been removed.

# This is a python translation of my GTB-tokenize.pl, which in turn
# draws in part on Robert MacIntyre's 1995 PTB tokenizer,
# (http://www.cis.upenn.edu/~treebank/tokenizer.sed) and Yoshimasa
# Tsuruoka's GENIA tagger tokenization (tokenize.cpp;
# www-tsujii.is.s.u-tokyo.ac.jp/GENIA/tagger)

# by Sampo Pyysalo, 2011. Licensed under the MIT license.
# http://www.opensource.org/licenses/mit-license.php

# NOTE: intended differences to GTB tokenization:
# - Does not break "protein(s)" -> "protein ( s )"

import re
from collections import deque
from spacy.tokens import Doc

INPUT_ENCODING = "UTF-8"
OUTPUT_ENCODING = "UTF-8"
DEBUG_GTB_TOKENIZATION = False

# Penn treebank bracket escapes (others excluded)
PTB_ESCAPES = [('(', '-LRB-'),
               (')', '-RRB-'),
               ('[', '-LSB-'),
               (']', '-RSB-'),
               ('{', '-LCB-'),
               ('}', '-RCB-'),
              ]

def ptb_escape(sentence: str):
    for original, ptb_symbol in PTB_ESCAPES:
        sentence = sentence.replace(original, ptb_symbol)
    return sentence

def ptb_unescape(sentence: str):
    for original, ptb_symbol in PTB_ESCAPES:
        sentence = sentence.replace(ptb_symbol, original)
    return sentence

# processing in three stages: "initial" regexs run first, then
# "repeated" run as long as there are changes, and then "final"
# run. As the tokenize() function itself is trivial, comments relating
# to regexes given with the re.compiles.

__initial = []
__repeated = []
__final = []

# separate but do not break ellipsis
__initial.append((re.compile(r'\.\.\.'), r' ... '))
#__initial.append((re.compile(r'(\n+)'), r' \1 '))

# To avoid breaking names of chemicals, protein complexes and similar,
# only add space to related special chars if there's already space on
# at least one side.
__initial.append((re.compile(r'([,;:@#]) '), r' \1 '))
__initial.append((re.compile(r' ([,;:@#])'), r' \1 '))

# always separated
__initial.append((re.compile(r'\$'), r' $ '))
__initial.append((re.compile(r'\%'), r' % '))
__initial.append((re.compile(r'\&'), r' & '))

# separate punctuation followed by space even if there's closing
# brackets or quotes in between, but only sentence-final for
# periods (don't break e.g. "E. coli").
# \u2019 is a alternate apostrophe that crops up in some papers.
__initial.append((re.compile(r'([\u2019,:;])([\[\]\)\}\>\"\']* +)', re.UNICODE), r' \1\2'))
__initial.append((re.compile(r'(\.+)([\[\]\)\}\>\"\']* +)$'), r' \1\2'))

# these always
__initial.append((re.compile(r'\?'), ' ? '))
__initial.append((re.compile(r'\!'), ' ! '))

# separate greater than and less than signs, avoiding breaking
# "arrows" (e.g. "-->", ">>") and compound operators (e.g. "</=")
__initial.append((re.compile(r'((?:=\/)?<+(?:\/=|--+>?)?)'), r' \1 '))
__initial.append((re.compile(r'((?:<?--+|=\/)?>+(?:\/=)?)'), r' \1 '))

# separate dashes, not breaking up "arrows"
__initial.append((re.compile(r'(<?--+\>?)'), r' \1 '))

# Parens only separated when there's space around a balanced
# bracketing. This aims to avoid splitting e.g. beta-(1,3)-glucan,
# CD34(+), CD8(-)CD3(-).

# Previously had a proper recursive implementation for this, but it
# was much too slow for large-scale use. The following is
# comparatively fast but a bit of a hack:

# First "protect" token-internal brackets by replacing them with
# their PTB escapes. "Token-internal" brackets are defined as
# matching brackets of which at least one has no space on either
# side. To match GTB tokenization for cases like "interleukin
# (IL)-mediated", and "p65(RelA)/p50", treat following dashes and
# slashes as space.  Nested brackets are resolved inside-out;
# to get this right, add a heuristic considering boundary
# brackets as "space".

# (First a special case (rareish): "protect" cases with dashes after
# paranthesized expressions that cannot be abbreviations to avoid
# breaking up e.g. "(+)-pentazocine". Here, "cannot be abbreviations"
# is taken as "contains no uppercase charater".)
#__initial.append((re.compile(r'\(([^ A-Z()\[\]{}]+)\)-'), r'-LRB-\1-RRB--'))
__initial.append((re.compile(r'\(([^ ()\[\]{}]+)\)-'), r'-LRB-\1-RRB--'))
__initial.append((re.compile(r'\{([^ ()\[\]{}]+)\}-'), r'-LCB-\1-RCB--'))
__initial.append((re.compile(r'\[([^ ()\[\]{}]+)\]-'), r'-LSB-\1-RSB--'))

# These are repeated until there's no more change (per above comment)
__repeated.append((re.compile(r'(?<![ (\[{])\(([^ ()\[\]{}]*)\)'), r'-LRB-\1-RRB-'))
__repeated.append((re.compile(r'\(([^ ()\[\]{}]*)\)(?![ )\]}\/-])'), r'-LRB-\1-RRB-'))
__repeated.append((re.compile(r'(?<![ (\[{])\[([^ ()\[\]{}]*)\]'), r'-LSB-\1-RSB-'))
__repeated.append((re.compile(r'\[([^ ()\[\]{}]*)\](?![ )\]}\/-])'), r'-LSB-\1-RSB-'))
__repeated.append((re.compile(r'(?<![ (\[{])\{([^ ()\[\]{}]*)\}'), r'-LCB-\1-RCB-'))
__repeated.append((re.compile(r'\{([^ ()\[\]{}]*)\}(?![ )\]}\/-])'), r'-LCB-\1-RCB-'))

# Remaining brackets are not token-internal and should be
# separated.
__final.append((re.compile(r'\('), r' -LRB- '))
__final.append((re.compile(r'\)'), r' -RRB- '))
__final.append((re.compile(r'\['), r' -LSB- '))
__final.append((re.compile(r'\]'), r' -RSB- '))
__final.append((re.compile(r'\{'), r' -LCB- '))
__final.append((re.compile(r'\}'), r' -RCB- '))

# initial single quotes always separated
__final.append((re.compile(r' (\'+)'), r' \1 '))
# final with the exception of 3' and 5' (rough heuristic)
__final.append((re.compile(r'(?<![35\'])(\'+) '), r' \1 '))

# This more frequently disagreed than agreed with GTB
#     # Separate slashes preceded by space (can arise from
#     # e.g. splitting "p65(RelA)/p50"
#     __final.append((re.compile(r' \/'), r' \/ '))

# Standard from PTB (TODO: pack)
__final.append((re.compile(r'\'s '), ' \'s '))
__final.append((re.compile(r'\'S '), ' \'S '))
__final.append((re.compile(r'\'m '), ' \'m '))
__final.append((re.compile(r'\'M '), ' \'M '))
__final.append((re.compile(r'\'d '), ' \'d '))
__final.append((re.compile(r'\'D '), ' \'D '))
__final.append((re.compile(r'\'ll '), ' \'ll '))
__final.append((re.compile(r'\'re '), ' \'re '))
__final.append((re.compile(r'\'ve '), ' \'ve '))
__final.append((re.compile(r'n\'t '), ' n\'t '))
__final.append((re.compile(r'\'LL '), ' \'LL '))
__final.append((re.compile(r'\'RE '), ' \'RE '))
__final.append((re.compile(r'\'VE '), ' \'VE '))
__final.append((re.compile(r'N\'T '), ' N\'T '))


# New custom additions
__final.append((re.compile(r'×'), ' × '))

# clean up possible extra space
#__final.append((re.compile(r'  +'), r' '))

def _tokenize(sentence: str):
    """
    Tokenizer core. Performs GTP-like tokenization, using PTB escapes
    for brackets (but not quotes). Assumes given string has initial
    and terminating space. You probably want to use tokenize() instead
    of this function.
    """

    # see re.complies for comments
    for regex, text in __initial:
        sentence = regex.sub(text, sentence)

    while True:
        original = sentence
        for regex, text in __repeated:
            sentence = regex.sub(text, sentence)
        if original == sentence:
            break

    for regex, text in __final:
        sentence = regex.sub(text, sentence)

    return sentence

def tokenize(sentence: str):
    """
    Tokenizes the given string with a GTB-like tokenization. Input
    will adjusted by removing surrounding space, if any.
    """

    if DEBUG_GTB_TOKENIZATION:
        orig = sentence

    # Core tokenization needs starting and ending space and no newline;
    # store to return string ending similarly
    # TODO: this isn't this difficult ... rewrite nicely
    sentence = re.sub(r'^', ' ', sentence)
    match = re.match(r'^((?:.+|\n)*?) *(\n*)$', sentence)
    assert match, "INTERNAL ERROR on '%s'" % sentence # should always match
    sentence, s_end = match.groups()
    sentence = re.sub(r'$', ' ', sentence)

    # no escaping, just separate
    sentence = re.sub(r'([ \(\[\{\<])\"', r'\1 " ', sentence)

    sentence = _tokenize(sentence)

    # as above (not quite sure why this is after primary tokenization...)
    sentence = sentence.replace('"', ' " ')

    # standard unescape for PTB escapes introduced in core tokenization
    sentence = ptb_unescape(sentence)

    # Clean up added space (well, maybe other also)
    #sentence = re.sub(r'  +', ' ', sentence)
    sentence = re.sub(r'^ ', '', sentence)
    sentence = re.sub(r' $', '', sentence)

    # Only do final comparison in debug mode.
    if DEBUG_GTB_TOKENIZATION:
        # revised must match original when whitespace, quotes (etc.)
        # and escapes are ignored
        # TODO: clean this up
        escaped_original = ptb_unescape(orig
                                        .replace(' ', '')
                                        .replace('\n', '')
                                        .replace("'", '')
                                        .replace('"', '')
                                        .replace('``', ''))
        escaped_sentence = ptb_unescape(sentence
                                        .replace(' ', '')
                                        .replace('\n', '')
                                        .replace("'", '')
                                        .replace('"', '')
                                        .replace('``', ''))

        if escaped_original != escaped_sentence:
            print(f"tokenize(): error: text mismatch (returning original):\nORIG: {orig}\nNEW: {sentence}")
            sentence = orig

    return sentence + s_end



def split_whitespace_tokenized_string(original_sentence: str, sentence_with_spaces: str):

    """
    This function is designed to mirror the SpaCy non-destructive tokenisation.
    It respects the following rules:

    1. All tokens will either 'own' or 'not own' whitespace such that the original
       string can be reconstructed.
    2. Tokens consisting of whitespace-only tokens are permitted.
    2. Empty string tokens are permitted.
    3. Whitespace-only tokens never 'own' whitespace.
    """

    original = deque(original_sentence)
    tokenized = deque(sentence_with_spaces)

    tokens = []
    whitespace_ownership = []
    current_word = []

    while True:

        if not original:
            if all([t.isspace() for t in current_word]):
                # We started a new token and the previous token was completely
                # whitespace, so it doesn't 'own' whitespace.
                tokens.append("".join(current_word))
                whitespace_ownership.append(False)
                break
            elif current_word[-1].isspace():
                # We started a new token and the previous token owned whitespace.
                tokens.append("".join(current_word[:-1]))
                whitespace_ownership.append(True)
                break
            else:
                tokens.append("".join(current_word))
                whitespace_ownership.append(False)
                break

        original_char = original.popleft()
        tokenized_char = tokenized.popleft()

        if original_char == tokenized_char:
            empty_or_last_char_space = False if not current_word else current_word[-1].isspace()
            if not tokenized_char.isspace() and not empty_or_last_char_space:
                # Just another character of current word.
                current_word.append(tokenized_char)

            elif not tokenized_char.isspace() and empty_or_last_char_space:
                if all([t.isspace() for t in current_word]):
                    # We started a new token and the previous token was completely
                    # whitespace, so it doesn't 'own' whitespace.
                    tokens.append("".join(current_word))
                    whitespace_ownership.append(False)
                else:
                    # We started a new token and the previous token owned whitespace.
                    tokens.append("".join(current_word[:-1]))
                    whitespace_ownership.append(True)
                # reset stack with current character
                current_word = [tokenized_char]

            elif tokenized_char.isspace() and not empty_or_last_char_space:

                if tokenized_char == " ":
                    # We handle this case in the next loop iteration.
                    current_word.append(tokenized_char)
                else:
                    # Otherwise we have a non-actual space char,
                    # which we always split on.
                    tokens.append("".join(current_word))
                    whitespace_ownership.append(False)
                    current_word = [tokenized_char]

            elif tokenized_char.isspace() and empty_or_last_char_space:
                # This is one of the weird cases.

                # whitespace is greedy - keep extending it.
                if all([t.isspace() for t in current_word]):
                    current_word.append(tokenized_char)
                else:
                    # We have a word followed by multiple spaces.
                    # The previous word owns a whitespace and we start
                    # a new only whitespace word.
                    tokens.append("".join(current_word[:-1]))
                    whitespace_ownership.append(True)
                    current_word = [tokenized_char]

        elif tokenized_char.isspace():
            # This happens if the tokenization has _inserted_ a space.
            # E.g "word." "word ."
            tokens.append("".join(current_word))
            whitespace_ownership.append(False)

            if not original:
                break
            current_word = [tokenized.popleft()]

    return tokens, whitespace_ownership


class GeniaTokenizer:
    """
    A custom tokenizer which uses heuristics to split biomedical text.
    nlp = spacy.load("en_core_web_md")
    # hack to replace tokenizer with a whitespace tokenizer
    nlp.tokenizer = GeniaTokenizer(nlp.vocab)
    ... use nlp("here is some text") as normal.
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        text_with_ws = tokenize(text)
        words, spaces = split_whitespace_tokenized_string(text, text_with_ws)
        return Doc(self.vocab, words=words, spaces=spaces)
