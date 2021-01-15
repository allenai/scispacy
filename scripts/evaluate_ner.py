from typing import Optional

import spacy
import importlib

from scispacy.data_util import read_full_med_mentions, read_ner_from_tsv
from scispacy.train_utils import evaluate_ner


def main(model_path: str, dataset: str, output_path: str, code: Optional[str]):
    if code is not None:
        # need to import code before loading a spacy model
        spec = importlib.util.spec_from_file_location(name, str(loc))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    nlp = spacy.load(model_path)
    if dataset.startswith("medmentions"):
        train_data, dev_data, test_data = read_full_med_mentions(data_path, None, False)
        data_split = dataset.split("-")[1]
        if data_split == "train":
            data = train_data
        elif data_split == "dev":
            data = dev_data
        elif data_split == "test":
            data = test_data
        else:
            raise Exception(f"Unrecognized split {data_split}")
    else:
        data = read_ner_from_tsv(dataset)

    evaluate_ner(nlp, data, dump_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model to evaluate")
    parser.add_argument("--dataset", type=str, help="medmentions-<train/dev/test>, or a tsv file to evalute")
    parser.add_argument("--output_path", type=str, help="Path to write results to")
    parser.add_argument("--code", type=str, default=None, help="Path to code to import before loading spacy model")

    args = parser.parse_args()
    main(args.model_path, args.dataset, args.output_path, args.code)