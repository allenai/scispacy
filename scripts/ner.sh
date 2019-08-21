#!/user/bin/env bash

set -e

MODEL_PATH=$1
OUT_PATH=${2:-./ner}

MED_MENTIONS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/med_mentions.tar.gz"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage (run from base SciSpaCy repository): ner.sh <model_path> {path to existing model} <output directory> {default=./ner}"
  exit 1
fi

python scripts/train_ner.py \
  --model_output_dir ${OUT_PATH} \
  --data_path ${MED_MENTIONS} \
  --model_path ${MODEL_PATH} \
  --iterations 7 \
  --label_granularity 7
