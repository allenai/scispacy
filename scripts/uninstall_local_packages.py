import os

from scispacy.version import VERSION


def main():
    model_names = [
        "en_core_sci_sm",
        "en_core_sci_md",
        "en_core_sci_lg",
        "en_core_sci_scibert",
        "en_ner_bc5cdr_md",
        "en_ner_craft_md",
        "en_ner_bionlp13cg_md",
        "en_ner_jnlpba_md",
    ]

    for package_name in model_names:
        os.system(f"pip uninstall {package_name}")


if __name__ == "__main__":
    main()
