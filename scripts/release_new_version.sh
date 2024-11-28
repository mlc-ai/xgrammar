#!/bin/bash

# Usage: ./scripts/release_new_version.sh <version>
set -ex

git fetch origin main
git checkout FETCH_HEAD
git commit -m "Tag $1" --allow-empty
git tag $1 HEAD
git push origin $1
