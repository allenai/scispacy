#!/usr/bin/env bash

set -e

SIZE=$1
OUT_PATH=${2:-./build}
MODEL_NAME=${3:-scispacy_model}

if [ -d ${OUT_PATH}/${MODEL_NAME} ]; then 
  echo "Build directory ${OUT_PATH}/${MODEL_NAME} already exists. Delete if you woud like to remake the package"
  echo "Exiting"
  exit
fi

if [[ -z "${SIZE}" ]]; then
  echo "Usage (run from base SciSpaCy repository):"
  echo "pipeline.sh <size> {small|medium|large} <build_directory> {default=./build} <model name> {default='scispacy_model'}"
  exit 1
fi

bash ./scripts/base_model.sh ${SIZE} ${OUT_PATH}/base_model
bash ./scripts/parser.sh ${OUT_PATH}/base_model ${OUT_PATH}/parser
bash ./scripts/ner.sh ${OUT_PATH}/parser/best ${OUT_PATH}/ner

cp -r ${OUT_PATH}/ner/best ${OUT_PATH}/${MODEL_NAME}

bash ./scripts/create_model_package.sh ${OUT_PATH}/${MODEL_NAME}

echo "Build output present in ${OUT_PATH}, with the full model serialised to ${OUT_PATH}/${MODEL_NAME}."
echo "A spacy model package has been created in ./dist."
