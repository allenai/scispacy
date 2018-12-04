import argparse

def main(conll_path):
    with open(conll_path) as fp:
        line = fp.readline()
        sents = 0
        while line:
            if line.startswith("\n"):
                sents += 1
            line = fp.readline()    print("{} sentences in {}".format(sents, conll_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--conll_path',
        help="Path to the conll data to count the sentences of"
    )

    args = parser.parse_args()
    main(args.conll_path)