import argparse
import os

def sentence_count(conll_path):
    with open(conll_path) as fp:
        line = fp.readline()
        sents = 0
        while line:
            if line.startswith("\n"):
                sents += 1
            line = fp.readline()
    return sents

def main(conll_path, output_dir):
    num_sentences = sentence_count(conll_path)
    convert_command = "python -m spacy convert {input_path} {output_dir} --n-sents {num_sents}".format(input_path = conll_path,
                                                                                                       output_dir = output_dir,
                                                                                                       num_sents = num_sentences)
    os.system(convert_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--conll_path',
        help="Path to the conll data to count the sentences of"
    )

    parser.add_argument(
        '--output_dir',
        help="Path to the directory to output the spacy formatted file to"
    )

    args = parser.parse_args()
    main(args.conll_path, args.output_dir)