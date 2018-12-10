#!/usr/bin/env bash
# Run our linter over the python code.
pylint -d locally-disabled,locally-enabled -f colorized SciSpaCy
