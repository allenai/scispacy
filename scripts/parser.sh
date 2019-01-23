#!/user/bin/env bash

set -e

MODEL_PATH=$1
OUT_PATH=${2:-./parser}

GENIA_TRAIN="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/genia/train.json"
GENIA_DEV="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/genia/dev.json"
GENIA_TEST="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/genia/test.json"


if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage (run from base SciSpaCy repository): parser.sh <model_path> {path to existing model} <output directory> {default=./parser}"
  echo "(optionally export the ONTONOTES_PATH and ONTONOTES_PERCENT variables to mix ontonotes data in with training.)"
  exit 1
fi

if [[ -n "${ONTONOTES_PATH}" ]]; then
  python scripts/train_parser_and_tagger.py \
    --train_json_path ${GENIA_TRAIN} \
    --dev_json_path ${GENIA_DEV} \
    --test_json_path ${GENIA_TEST} \
    --model_path ${MODEL_PATH} \
    --model_output_dir ${OUT_PATH} \
    --ontonotes_path ${ONTONOTES_PATH} \
    --ontonotes_train_percent ${ONTONOTES_PERCENT}

else
  python scripts/train_parser_and_tagger.py \
    --train_json_path ${GENIA_TRAIN} \
    --dev_json_path ${GENIA_DEV} \
    --test_json_path ${GENIA_TEST} \
    --model_path ${MODEL_PATH} \
    --model_output_dir ${OUT_PATH}
fi

