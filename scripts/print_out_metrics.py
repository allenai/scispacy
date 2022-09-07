import os
import json


def main():
    core_model_names = ["lg", "md", "sm", "scibert"]
    ner_model_names = ["bc5cdr", "bionlp13cg", "craft", "jnlpba"]

    base_path = "packages"
    for core_model_name in core_model_names:
        print(f"Printing results for {core_model_name}")
        with open(
            os.path.join(base_path, f"{core_model_name}_genia_results.json")
        ) as _genia_results_file:
            genia_results = json.load(_genia_results_file)

        with open(
            os.path.join(base_path, f"{core_model_name}_onto_results.json")
        ) as _onto_results_file:
            onto_results = json.load(_onto_results_file)

        with open(
            os.path.join(base_path, f"{core_model_name}_mm_results.json")
        ) as _mm_results_file:
            mm_results = json.load(_mm_results_file)

        print(f"Genia tag accuracy: {genia_results['tag_acc']}")
        print(f"Genia uas: {genia_results['dep_uas']}")
        print(f"Genia las: {genia_results['dep_las']}")
        print(f"Ontonotes uas: {onto_results['dep_uas']}")
        print(f"MedMentions F1: {mm_results['f1-measure-untyped']}")
        print()

    for ner_model_name in ner_model_names:
        print(f"Printing results for {ner_model_name}")
        with open(
            os.path.join(base_path, f"{ner_model_name}_results.json")
        ) as _ner_results_file:
            ner_results = json.load(_ner_results_file)

        print(f"NER F1: {ner_results['f1-measure-overall']}")


if __name__ == "__main__":
    main()
