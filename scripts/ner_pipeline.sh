#!/usr/bin/env bash

set -e

MODEL_PATH=$1
OUT_PATH=${2:-./specialised_ner}

BC5CDR_TRAIN="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/BC5CDR-IOB/train.tsv"
BC5CDR_DEV="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/BC5CDR-IOB/devel.tsv"
BC5CDR_TEST="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/BC5CDR-IOB/test.tsv"

JNLPBA_TRAIN="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/JNLPBA-IOB/train.tsv"
JNLPBA_DEV="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/JNLPBA-IOB/devel.tsv"
JNLPBA_TEST="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/JNLPBA-IOB/test.tsv"

CRAFT_TRAIN="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/CRAFT-IOB/train.tsv"
CRAFT_DEV="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/CRAFT-IOB/devel.tsv"
CRAFT_TEST="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/CRAFT-IOB/test.tsv"

BioNLP13CG_TRAIN="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/BioNLP13CG-IOB/train.tsv"
BioNLP13CG_DEV="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/BioNLP13CG-IOB/devel.tsv"
BioNLP13CG_TEST="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/ner/BioNLP13CG-IOB/test.tsv"

BC5CDR_META="./data/bc5cdr_ner.json"
JNLPBA_META="./data/jnlpba_ner.json"
CRAFT_META="./data/craft_ner.json"
BioNLP13CG_META="./data/bionlp13cg_ner.json"


if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage (run from base SciSpaCy repository): specialised_ner.sh <model_path> {path to existing model} <output directory> {default=./specialised_ner}"
  exit 1
fi
python scripts/train_specialised_ner.py --train_data_path ${BC5CDR_TRAIN} --dev_data_path ${BC5CDR_DEV} --test_data_path ${BC5CDR_TEST} --model_output_dir ${OUT_PATH}/bc5cdr --model_path ${MODEL_PATH} --iterations 7 --meta_overrides ${BC5CDR_META}
python scripts/train_specialised_ner.py --train_data_path ${JNLPBA_TRAIN} --dev_data_path ${JNLPBA_DEV} --test_data_path ${JNLPBA_TEST} --model_output_dir ${OUT_PATH}/jnlpba --model_path ${MODEL_PATH} --iterations 7 --meta_overrides ${JNLPBA_META}
python scripts/train_specialised_ner.py --train_data_path ${CRAFT_TRAIN} --dev_data_path ${CRAFT_DEV} --test_data_path ${CRAFT_TEST} --model_output_dir ${OUT_PATH}/craft --model_path ${MODEL_PATH} --iterations 7 --meta_overrides ${CRAFT_META}
python scripts/train_specialised_ner.py --train_data_path ${BioNLP13CG_TRAIN} --dev_data_path ${BioNLP13CG_DEV} --test_data_path ${BioNLP13CG_TEST} --model_output_dir ${OUT_PATH}/bionlp13cg --model_path ${MODEL_PATH} --iterations 7 --meta_overrides ${BioNLP13CG_META}


bash ./scripts/create_model_package.sh ${OUT_PATH}/craft/best
bash ./scripts/create_model_package.sh ${OUT_PATH}/bc5cdr/best
bash ./scripts/create_model_package.sh ${OUT_PATH}/jnlpba/best
bash ./scripts/create_model_package.sh ${OUT_PATH}/bionlp13cg/best


