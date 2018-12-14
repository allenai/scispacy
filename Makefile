

BUILD_DIR="./build"

init:
	# Run python script to build vocabulary with vectors.
	# Adds custom tokeniser to base model.
	echo "Not implemented"

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
