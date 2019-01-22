

BUILD_DIR=./build

PUBMED_FREQS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/pubmed.freqs"
PUBMED_VECTORS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/pubmed_with_header.txt.gz"

GENIA_TRAIN="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/genia/train.json"
GENIA_DEV="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/genia/dev.json"
GENIA_TEST="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/genia/test.json"

MED_MENTIONS="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/med_mentions.tar.gz"

SMALL_BASE=${BUILD_DIR}/small_base
LARGE_BASE=${BUILD_DIR}/large_base
SMALL_PARSER=${BUILD_DIR}/small_parser
LARGE_PARSER=${BUILD_DIR}/large_parser
SMALL_NER=${BUILD_DIR}/small_ner
LARGE_NER=${BUILD_DIR}/large_ner

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
	-x -V 100000 -mwf 20 -m ./data/meta_large.json

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

ner-small-from-parser:
	python scripts/train_ner.py \
	--model_output_dir ${SMALL_NER} \
	--data_path ${MED_MENTIONS} \
	--model_path ${SMALL_PARSER}/best/ \
	--iterations 7 \
	--label_granularity 7

ner-large-from-parser:
	python scripts/train_ner.py \
	--model_output_dir ${LARGE_NER} \
	--data_path ${MED_MENTIONS} \
	--model_path ${LARGE_PARSER}/best/ \
	--iterations 7 \
	--label_granularity 7

package:
	# Create model packages for 1) The library and 2) The Spacy model.
	bash scripts/create_model_package.sh ${BUILD_DIR}

all-small: init-small parser-small ner-small-from-parser
all-large: init-large parser-large ner-large-from-parser

install:
	pip install -r requirements.in
	python -m spacy download en_core_web_sm
	python -m spacy download en_core_web_md
