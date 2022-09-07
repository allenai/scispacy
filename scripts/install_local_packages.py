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

    full_package_paths = [
        os.path.join(
            "packages",
            f"{model_name}-{VERSION}",
            "dist",
            f"{model_name}-{VERSION}.tar.gz",
        )
        for model_name in model_names
    ]

    for package_path in full_package_paths:
        os.system(f"pip install {package_path}")


if __name__ == "__main__":
    main()
