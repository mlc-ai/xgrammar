#!/bin/bash

# Build the docs and the site, then serve the site locally

set -euxo pipefail

scripts/support/build_site.sh

cd site && jekyll serve --trace --skip-initial-build --host localhost --baseurl / --port 8888
