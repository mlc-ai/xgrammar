#!/bin/bash

# build the docs to _build/html
set -euxo pipefail

make clean
make html
python3 wrap_run_llm.py
