import argparse
import os

from scispacy.candidate_generation import create_tfidf_ann_index
from scispacy.linking_utils import KnowledgeBase


def main(kb_path: str, output_path: str, n_grams: int):

    os.makedirs(output_path, exist_ok=True)
    kb = KnowledgeBase(kb_path)
    create_tfidf_ann_index(output_path, kb, n_grams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kb_path',
        help="Path to the KB file."
    )
    parser.add_argument(
        '--output_path',
        help="Path to the output directory."
    )
    parser.add_argument(
        '--n_grams',
        type=int,
        help="Use n grams to build the index",
        default=(3,3),
    )

    args = parser.parse_args()
    main(args.kb_path, args.output_path, args.n_grams)
