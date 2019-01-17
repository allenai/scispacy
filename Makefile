

BUILD_DIR="./build"

PUBMED_FREQS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/pubmed.freqs"
PUBMED_VECTORS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/pubmed_with_header.txt.gz"

GENIA_TRAIN="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/genia/train.json"
GENIA_DEV="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/genia/dev.json"
GENIA_TEST="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/genia/test.json"

SMALL_BASE="${BUILD_DIR}/small_base"
LARGE_BASE="${BUILD_DIR}/large_base"
SMALL_PARSER="${BUILD_DIR}/small_parser"
LARGE_PARSER="${BUILD_DIR}/large_parser"

init-small:
	python scripts/init_model.py \
	en ${SMALL_BASE} \
	${PUBMED_FREQS} \
	-m ./data/meta_small.json

init-large:
	python scripts/init_model.py \
	en ${LARGE_BASE} \
	${PUBMED_FREQS} \
	-v ${PUBMED_VECTORS} \
	-x -V 40000 -m ./data/meta_large.json

parser-small:
	python scripts/train_parser_and_tagger.py \
	  --train_json_path ${GENIA_TRAIN} \
	  --dev_json_path ${GENIA_DEV} \
	  --test_json_path ${GENIA_TEST} \
	  --model_path ${SMALL_BASE} \
	  --model_output_dir ${SMALL_PARSER}

parser-large:
	python scripts/train_parser_and_tagger.py \
	  --train_json_path ${GENIA_TRAIN} \
	  --dev_json_path ${GENIA_DEV} \
	  --test_json_path ${GENIA_TEST} \
	  --model_path ${LARGE_BASE} \
	  --model_output_dir ${LARGE_PARSER}

ner:
	# Takes in a model output by init or parser and adds an "ner" pipeline.
	echo "Not implemented"

package:
	# Create model packages for 1) The library and 2) The Spacy model.
	bash scripts/create_model_package.sh ${BUILD_DIR}

all-small: init-small parser-small ner package

install:
	pip install -r requirements.in
	python -m spacy download en_core_web_sm
	python -m spacy download en_core_web_md
