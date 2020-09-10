#!/usr/bin/env bash

set -e

SIZE=$1
OUT_PATH=${2:-./base_model}

PUBMED_FREQS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/gorc_subset.freqs"
PUBMED_VECTORS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/pubmed_with_header.txt.gz"

if [ "${SIZE}" = "small" ]; then
  python scripts/init_model.py en ${OUT_PATH} ${PUBMED_FREQS} -mwf 1000 -m ./data/meta_small.json
elif [ "${SIZE}" = "medium" ]; then
  python scripts/init_model.py en ${OUT_PATH} ${PUBMED_FREQS} -v ${PUBMED_VECTORS} -x -V 50000 -mwf 150 -m ./data/meta_medium.json -vn "en_core_sci_md_vectors"
elif [ "${SIZE}" = "large" ]; then
  python scripts/init_model.py en ${OUT_PATH} ${PUBMED_FREQS} -v ${PUBMED_VECTORS} -x -V 600000 -mwf 50 -m ./data/meta_large.json -vn "en_core_sci_lg_vectors"
else
  echo "Usage (run from base SciSpaCy repository): base_model.sh <size> {small|medium|large} <build_directory> {default=./base_model}"
fi
