#!/user/bin/env bash

set -e

SIZE=$1
OUT_PATH=${2:-./base_model}

PUBMED_FREQS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/pubmed.freqs"
PUBMED_VECTORS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/pubmed_with_header.txt.gz"

if [ "${SIZE}" = "small" ]; then
  python scripts/init_model.py en ${OUT_PATH} ${PUBMED_FREQS} -m ./data/meta_small.json
elif [ "${SIZE}" = "large" ]; then
  python scripts/init_model.py en ${OUT_PATH} ${PUBMED_FREQS} -v ${PUBMED_VECTORS} -x -V 100000 -mwf 20 -m ./data/meta_large.json
else
  echo "Usage (run from base SciSpaCy repository): base_model.sh <size> {small|large} <build_directory> {default=./base_model}"
fi
