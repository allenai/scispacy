
import os
import sys
from collections import defaultdict
import argparse
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import spacy_convert


def generate_sentence(sentence):
    (id_, word, tag, head, dep) = sentence
    sentence = {}
    tokens = []
    for i, token_id in enumerate(id_):
        # Note that this is not the same as the index
        # due to elided words which were ignored.
        token_id = token_id - 1
        token = {}
        token["id"] = token_id
        token["orth"] = word[i]
        token["tag"] = tag[i]
        if head[i] == 0:
            # The relative offset of the head to itself
            # is zero
            token["head"] = head[i]
        else:
            # What we are doing here is making the head
            # of the token a relative offset.
            token["head"] = (head[i] - 1) - token_id
        token["dep"] = dep[i] if dep[i] != "root" else "ROOT"
        tokens.append(token)
    sentence["tokens"] = tokens
    return sentence


def create_doc(sentences, sentence_id: str):
    doc = {}
    paragraph = {}
    doc["id"] = sentence_id
    doc["paragraphs"] = []
    paragraph["sentences"] = sentences
    doc["paragraphs"].append(paragraph)
    return doc


def main(input_path: str, pmids_path: str, output_path: str):

    pmids = []
    with open(pmids_path) as pmids_fp:
        line = pmids_fp.readline()
        while line:
            pmids.append(line.rstrip())
            line = pmids_fp.readline()

    # Create a dictionary mapping pubmed id to fully annotated docs.
    docs = defaultdict(list)
    for pubmed_id, sentence in zip(pmids, spacy_convert.get_dependency_annotations(input_path)):
        docs[pubmed_id].append(sentence)
    
    formatted_docs = []
    for pubmed_id, sentences in docs.items():
        formatted_sentences = [generate_sentence(sent) for sent in sentences]
        formatted_docs.append(create_doc(formatted_sentences, pubmed_id))

    input_path = Path(input_path)
    output_filename = input_path.parts[-1].replace(".conll", ".json")
    output_filename = input_path.parts[-1].replace(".conllu", ".json")
    output_file = Path(output_path) / output_filename
    with output_file.open('w', encoding='utf-8') as f:
        f.write(json.dumps(formatted_docs, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conll_path',
        help="Path to the conll formatted data"
    )
    parser.add_argument(
        '--output_path',
        help="Path to the output directory"
    )
    parser.add_argument(
        '--pubmed_ids',
        help="Path to the pubmed_ids"
    )
    args = parser.parse_args()
    main(args.conll_path, args.pubmed_ids, args.output_path)