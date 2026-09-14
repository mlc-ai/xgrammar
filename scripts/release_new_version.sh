#!/bin/bash

# Usage: ./scripts/release_new_version.sh <version>

set -ex

if [ -z "$1" ]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <version>"
    exit 1
fi

# Pull and checkout main branch
git pull origin main
git checkout main
git tag $1 HEAD
git push origin $1
