from typing import List, Tuple

from spacy.tokens import Doc
from spacy.gold import GoldParse # pylint: disable=no-name-in-module
from spacy.vocab import Vocab # pylint: disable=no-name-in-module
from conllu.parser import parse_line, DEFAULT_FIELDS

def _lazy_parse(text: str, fields=DEFAULT_FIELDS):
    """
    Reads conllu annotations, yielding unwieldy OrderedDict-like
    objects per sentence.
    """
    for sentence in text.split("\n\n"):
        if sentence:
            yield [parse_line(line, fields)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")]

def get_dependency_annotations(path: str):
    """
    Reads data from a file formatted in the conllu format.
    Parameters
    ----------
    path : str
        The path to the file.
    Returns
    -------
    words : List[str]
        The words in the sentence.
    pos_tags : List[str]
        The part of speech tags for each word.
    heads : List[int]
        The integer head of each word. These are _one indexed_
        because in a dependency tree, there is a word which is the
        root, which has the value 0. Therefore, a word which has
        a head of 1 attaches to the first word in the sentence.
    tags : List[str]
        The string dependency label of the arc.
    """
    for annotation in _lazy_parse(open(path).read()):
        # This check for None is to filter for Elipsis, which is
        # when a word is implicitly referenced in a sentence.
        # We don't care about this (no-one does) so we ignore the
        # annotations, because they do confusing stuff like not
        # attaching to anything.
        annotation = [x for x in annotation if x["head"] is not None]
        ids = [x["id"] for x in annotation]
        heads = [x["head"] for x in annotation]
        tags = [x["deprel"] for x in annotation]
        words = [x["form"] for x in annotation]
        pos_tags = [x["upostag"] for x in annotation]

        yield (ids, words, pos_tags, heads, tags)

def convert_abstracts_to_docs(conll_path: str, pmids_path: str, vocab_path: str) -> List[Tuple[Doc, GoldParse]]:
    """Converts a conll file of abstracts to a doc and gold parse for
       each abstract

       @param conll_path: path to the conll formatted abstracts
       @param pmids_path: path to the pmids of the abstracts (one per conll sentence)
       @param vocab_path: path to the spacy vocabulary to load
    """
    vocab = Vocab().from_disk(vocab_path)
    pmids = []
    with open(pmids_path) as pmids_fp:
        line = pmids_fp.readline()
        while line:
            pmids.append(line.rstrip())
            line = pmids_fp.readline()

    corpus = []
    curr_pmid = None
    curr_words: List[str] = []
    curr_heads: List[int] = []
    curr_pos_tags: List[str] = []
    curr_deps: List[str] = []
    curr_offset = 0
    for sentence_parse, pmid in zip(get_dependency_annotations(conll_path), pmids):
        words = sentence_parse[1]
        pos_tags = sentence_parse[2]
        heads = sentence_parse[3]
        deps = [head if head != "root" else "ROOT" for head in sentence_parse[4]]
        if curr_pmid not in (None, pmid):
            doc = Doc(vocab, words=curr_words)
            gold = GoldParse(doc, heads=curr_heads, tags=curr_pos_tags, deps=curr_deps)
            corpus.append((doc, gold))

            curr_pmid = pmid
            curr_words = words
            curr_heads = [head - 1 if head != 0 else i for i, head in enumerate(heads)]
            curr_pos_tags = pos_tags
            curr_deps = deps
            curr_offset = len(words)
        else:
            curr_words += words
            curr_heads += [head + curr_offset - 1
                           if head != 0
                           else i + curr_offset for i, head in enumerate(heads)]
            curr_pos_tags += pos_tags
            curr_deps += deps
            curr_offset += len(words)
            curr_pmid = pmid

    return corpus
