#!/bin/bash

# Usage: ./scripts/release_new_version.sh <version>

set -ex

if [ -z "$1" ]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <version>"
    exit 1
fi

# Fetch and checkout main branch
git fetch origin main
git checkout main
git tag $1 HEAD
git push origin $1
