#!/bin/bash

if [ $1 == '' ]; then
    usage
    exit -1
fi

usage()
{
    echo $0 path_to_model
}

MODEL=$1
WORK=`mktemp -d`
CURDIR=${PWD}

mkdir -p dist

spacy package ${MODEL} ${WORK} --meta-path ${MODEL}/meta.json

INITFILE=`find ${WORK} -name __init__.py`

cp proto_model/__init__.py ${INITFILE}

cd ${WORK}/*

python setup.py sdist
cp `find . -name *.tar.gz` ${CURDIR}/dist/

cd ${CURDIR}

rm -rf ${WORK}