import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

import argparse

from tqdm import tqdm

import spacy
from spacy.matcher import Matcher

from scispacy.data_util import read_full_med_mentions
from scispacy.umls_utils import UmlsKnowledgeBase
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.candidate_generation import CandidateGenerator

def main(medmentions_path: str):
    print("Loading spacy model...")
    nlp = spacy.load('en_core_sci_sm')
    candidate_generator = CandidateGenerator()
    linker = UmlsEntityLinker(candidate_generator, filter_for_definitions=False)

    # print("Loading UMLS...")
    # umls = UmlsKnowledgeBase()

    print("Loading MedMentions...")
    train_examples, dev_examples, test_examples = read_full_med_mentions(medmentions_path,
                                                                         spacy_format=False)

    texts = []
    for example in dev_examples:
        abstract_text = example.abstract
        texts.append(abstract_text)

    matcher = Matcher(nlp.vocab)
    such_as_base_pattern = [{"POS": "NOUN"},
                         {"TEXT": ",", "OP": "?"},
                         {"TEXT": "such"},
                         {"TEXT": "as"},
                         {"TEXT": ",", "OP": "?"},
                         {"POS": "NOUN"}]
    such_as_extension_pattern = [{"TEXT": {"IN": [",", "and", "or"]}, "OP": "+"},
                              {"POS": "NOUN"}]
    such_as_2_pattern = such_as_base_pattern + such_as_extension_pattern
    such_as_3_pattern = such_as_base_pattern + such_as_extension_pattern + such_as_extension_pattern
    such_as_4_pattern = such_as_base_pattern + such_as_extension_pattern + such_as_extension_pattern + such_as_extension_pattern
    matcher.add("SuchAs1", None, such_as_base_pattern)
    matcher.add("SuchAs2", None, such_as_2_pattern)
    matcher.add("SuchAs3", None, such_as_3_pattern)
    matcher.add("SuchAs4", None, such_as_4_pattern)

    sets = []
    for text_idx, text in enumerate(texts):
        # import pdb; pdb.set_trace()
        original_doc = nlp(text)
        doc = nlp(text)
        doc = linker(doc)

        # merge noun chunks
        with doc.retokenize() as retokenizer:
            for span in doc.noun_chunks:
                attrs = {"POS": "NOUN"}
                retokenizer.merge(span, attrs=attrs)

        # merge contiguous nouns
        with doc.retokenize() as retokenizer:
            span_start = None
            for i, token in enumerate(doc):
                if token.pos_ == "NOUN":
                    if span_start is None:
                        span_start = token.i
                else:
                    if span_start is not None:
                        retokenizer.merge(doc[span_start:i])
                        span_start = None

        matches = matcher(doc)
        if matches != []:
            print()
            for match in matches:
                print(nlp.vocab.strings[match[0]], match[1], match[2])
            import pdb; pdb.set_trace()





        # print(text_idx)
        # print(doc)
        # chunks = list(doc.noun_chunks)
        # token_to_chunk = {}
        # for i, chunk in enumerate(chunks):
        #     for token in chunk:
        #         token_to_chunk[token] = chunk
        
        # chunk_idx = 0
        # token_idx = 0
        # if "extraction and determination of seven parabens" in text:
        #     import pdb; pdb.set_trace()
        # while chunk_idx < len(chunks):
        #     # print(chunk_idx)
        #     chunk = chunks[chunk_idx]
        #     chunk_start = chunk.start
        #     chunk_end = chunk.end
        #     token_idx = chunk.end

        #     if token_idx + 2 >= len(doc):
        #         break

        #     if ((doc[token_idx].text == "such" and doc[token_idx+1].text == "as") or 
        #         (doc[token_idx].text == "," and doc[token_idx+1].text == "such" and doc[token_idx+2].text == "as")):
        #         token_idx += 3
        #         root = chunk
        #         following = []
        #         halt = False

        #         while not halt:
        #             if doc[token_idx].text == "," or doc[token_idx].text == "and" or doc[token_idx].text == "or":
        #                 token_idx += 1
        #                 continue
        #             else:
        #                 if doc[token_idx] in token_to_chunk:
        #                     following.append(token_to_chunk[doc[token_idx]])
        #                     chunk_idx += 1
        #                     token_idx = token_to_chunk[doc[token_idx]].end
        #                     continue
        #                 else:
        #                     halt = True
        #                     sets.append((root, following))
        #                     continue
        #     else:
        #         chunk_idx += 1

    # import pdb; pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--medmentions_path',
            help='Path to the MedMentions dataset.'
    )

    args = parser.parse_args()
    main(args.medmentions_path)