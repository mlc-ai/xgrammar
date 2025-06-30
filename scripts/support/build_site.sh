#!/bin/bash

# build the docs and the site to site/_site

set -euxo pipefail

export PYTHONPATH=$PWD/python
cd docs && ./build_docs.sh && cd ..

cd site && jekyll b && cd ..

rm -rf site/_site/docs
cp -r docs/_build/html site/_site/docs
