

BUILD_DIR="./build"

PUBMED_FREQS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/pubmed.freqs"
PUBMED_VECTORS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/pubmed_with_header.txt.gz"

init-small:
	python scripts/init_model.py \
	en ./build/base_small/ \
	${PUBMED_FREQS} \
	-m ./data/meta_small.json

init-large:
	python scripts/init_model.py \
	en ./build/base_large/ \
	${PUBMED_FREQS} \
	-v ${PUBMED_VECTORS} \
	-x -V 40000 -m ./data/meta_small.json

init:
	# Run python script to build vocabulary with vectors.
	# Adds custom tokeniser to base model.
	echo "Not implemented"
	python scripts/init_model.py en ./test_init_model ./pubmed.freqs -m ./meta_small.json
parser:
	# Takes in a model output by init and adds "tagger" and "parser" pipelines.
	echo "Not implemented"

ner:
	# Takes in a model output by init or parser and adds an "ner" pipeline.
	echo "Not implemented"

package:
	# Create model packages for 1) The library and 2) The Spacy model.
	bash scripts/create_model_package.sh ${BUILD_DIR}

all: init parser ner package

install:
	pip install -r requirements.in
	python -m spacy download en_core_web_sm
	python -m spacy download en_core_web_md
