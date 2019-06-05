from typing import List, Tuple

import copy
import re

from spacy.tokens import Span, Doc

class HyponymDetector:
    """
    A spaCy pipe for detecting hyponyms using Hearst patterns.

    This class sets the `._.hyponyms` attribute on a spaCy Doc, which consists of
    a List[Tuple[str, List[str], str]] corresonding to the extracted general term, specific terms, and
    the string that matched a Hearst pattern.

    Parts of the implementation taken from https://github.com/mmichelsonIF/hearst_patterns_python/blob/master/hearstPatterns/hearstPatterns.py

    Parameters
    ----------
    extended: bool, optional (default=False).
        If True, use the extended set of hearst patterns.
    """
    def __init__(self, extended: bool = False):
        Doc.set_extension("hyponyms", default=[], force=True)

        self._adj_stopwords = ['sole',
                               'able',
                               'available',
                               'brief',
                               'certain',
                               'different',
                               'due',
                               'enough',
                               'especially',
                               'few',
                               'fifth',
                               'former',
                               'his',
                               'howbeit',
                               'immediate',
                               'important',
                               'inc',
                               'its',
                               'last',
                               'latter',
                               'least',
                               'less',
                               'likely',
                               'little',
                               'many',
                               # 'ml',
                               'more',
                               'most',
                               'much',
                               'my',
                               'necessary',
                               'new',
                               'next',
                               'non',
                               'old',
                               'other',
                               'our',
                               'ours',
                               'own',
                               'particular',
                               'past',
                               'possible',
                               'present',
                               'proud',
                               'recent',
                               'same',
                               'several',
                               'significant',
                               'similar',
                               'sole',
                               'such',
                               'sup',
                               'sure']
        self._prefix_stopwords = ['the', 'a']

        self._hearst_patterns = [
            ('(NP_[\\w\\-]+ (, )?such as (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
            ('(such NP_[\\w\\-]+ (, )?as (NP_[\\w\\-]+ ?(, )?(and |or )?)+)', 'first'),
            ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?other NP_[\\w\\-]+)', 'last'),
            ('(NP_[\\w\\-]+ (, )?include (NP_[\\w\\-]+ ?(, )?(and |or )?)+)', 'first'),
            ('(NP_[\\w\\-]+ (, )?especially (NP_[\\w\\-]+ ?(, )?(and |or )?)+)', 'first'),
        ]

        if extended:
            self._hearst_patterns.extend([
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?any other NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?some other NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?be a NP_[\\w\\-]+)', 'last'),
                ('(NP_[\\w\\-]+ (, )?like (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('such (NP_[\\w\\-]+ (, )?as (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?like other NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?one of the NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?one of these NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?one of those NP_[\\w\\-]+)', 'last'),
                ('example of (NP_[\\w\\-]+ (, )?be (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?be example of NP_[\\w\\-]+)', 'last'),
                ('(NP_[\\w\\-]+ (, )?for example (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?wich be call NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?which be name NP_[\\w\\-]+)', 'last'),
                ('(NP_[\\w\\-]+ (, )?mainly (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?mostly (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?notably (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?particularly (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?principally (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?in particular (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?except (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?other than (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?e.g. (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?i.e. (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?a kind of NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?kind of NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?form of NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?which look like NP_[\\w\\-]+)', 'last'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?which sound like NP_[\\w\\-]+)', 'last'),
                ('(NP_[\\w\\-]+ (, )?which be similar to (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?example of this be (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?type (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )? NP_[\\w\\-]+ type)', 'last'),
                ('(NP_[\\w\\-]+ (, )?whether (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(compare (NP_[\\w\\-]+ ?(, )?)+(and |or )?with NP_[\\w\\-]+)', 'last'),
                ('(NP_[\\w\\-]+ (, )?compare to (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_[\\w\\-]+ (, )?among -PRON- (NP_[\\w\\-]+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?as NP_[\\w\\-]+)', 'last'),
                ('(NP_[\\w\\-]+ (, )? (NP_[\\w\\-]+ ? (, )?(and |or )?)+ for instance)', 'first'),
                ('((NP_[\\w\\-]+ ?(, )?)+(and |or )?sort of NP_[\\w\\-]+)', 'last')
            ])

    def get_chunks(self, doc: Doc) -> List[Span]:
        """
        Returns noun chunks from a spacy Doc, augmenting the noun chunks that spaCy returns by default
        """

        tokens_in_chunks = set()
        # TODO: consider writing custom noun chunker
        chunks = set(doc.noun_chunks)
        for chunk in chunks:
            for token in chunk:
                tokens_in_chunks.add(token)

        span_start = None
        for i, token in enumerate(doc):
            if token.pos_ == "NOUN" and not token in tokens_in_chunks:
                if span_start is None:
                    span_start = token.i
            else:
                if span_start is not None:
                    new_chunk = doc[span_start:i]
                    for token in new_chunk:
                        tokens_in_chunks.add(token)
                    chunks.add(new_chunk)
                    span_start = None
        return chunks

    def clean_hyponym_text(self, input_text: str) -> str:
        """
        Cleans the matched hyponym text
        """

        # remove artifacts from regex creation
        input_text = input_text.replace("NP_", "").replace("_", " ").strip()

        # remove prefix words
        input_text = re.sub(r'\A{}\b'.format("(" + "|".join(self._prefix_stopwords) + ")"), "", input_text).strip()

        # remove uninteresting stopwords
        input_text = re.sub(r'\A{}\b'.format("(" + "|".join(self._adj_stopwords) + ")"), "", input_text).strip()

        return input_text

    def replace_text_for_regex(self, doc: Doc, chunks: List[Span]) -> str:
        """
        Joins noun chunks in the text with underscores, and adds a `NP_` prefix, in preparation
        for regex matching Hearst patterns
        """

        doc_text_replaced = doc.text
        for chunk in chunks:
            chunk_replacement_text = "NP_" + "_".join([token.text for token in chunk])
            chunk_original_text = chunk.text
            doc_text_replaced = re.sub(r'\b%s\b' % re.escape(chunk_original_text),
                                       r'%s' % chunk_replacement_text,
                                       doc_text_replaced)
        return doc_text_replaced

    def apply_hearst_patterns(self, text_replaced_for_regex: str) -> List[Tuple[str, str, str]]:
        """
        Applies hearst patterns to a piece of text
        """

        matches_to_return = []
        for (hearst_pattern, order) in self._hearst_patterns:
            matches = re.finditer(hearst_pattern, text_replaced_for_regex)
            for match in matches:
                match_string = match.group(0)
                nps = [a for a in match_string.split() if a.startswith("NP_")]

                if order == "first":
                    general = nps[0]
                    specifics = nps[1:]
                elif order == "last":
                    general = nps[-1]
                    specifics = nps[:-1]
                else:
                    raise Exception("Unknown order {} for hearst pattern {}".format(order, hearst_pattern))
                
                matches_to_return.append((general, specifics, match_string))
        return matches_to_return

    def __call__(self, doc: Doc) -> Doc:
        chunks = self.get_chunks(doc)
        doc_text_replaced = self.replace_text_for_regex(doc, chunks)

        matches = self.apply_hearst_patterns(doc_text_replaced)
        
        matches = self.apply_hearst_patterns(doc_text_replaced)
        for (general, specifics, match_string) in matches:
            doc._.hyponyms.append((self.clean_hyponym_text(general), [self.clean_hyponym_text(specific) for specific in specifics], match_string))

        return doc