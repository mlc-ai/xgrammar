#!/bin/bash

# Usage: ./scripts/release_new_version.sh <version>

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <version>"
    exit 1
fi

VERSION=$1
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?(\.post[0-9]+)?$ ]]; then
    echo "Error: Expected a tag such as v0.2.7, v0.2.7rc1, or v0.2.7.post1"
    exit 1
fi

if [[ "$(git branch --show-current)" != "main" || -n "$(git status --porcelain)" ]]; then
    echo "Error: Run this script from a clean main branch"
    exit 1
fi

git fetch origin main --tags
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse FETCH_HEAD)" ]]; then
    echo "Error: main must match origin/main; update and verify the release commit first"
    exit 1
fi

git tag -a "$VERSION" HEAD -m "Release $VERSION"
git push origin "refs/tags/$VERSION"
