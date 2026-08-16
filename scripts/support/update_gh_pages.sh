#!/bin/bash

# Update a gh-pages checkout with freshly built docs. The gh-pages branch is
# the persistent store for the website: the jekyll landing page lives at the
# root, docs built from main live in docs/latest/, and every released version
# keeps a frozen copy in docs/<version>/. versions.json at the root lists all
# published versions for the docs version dropdown.
#
# Usage:
#   update_gh_pages.sh latest  <docs_html_dir> <site_root_dir> <gh_pages_dir>
#   update_gh_pages.sh version <docs_html_dir> <version> <gh_pages_dir>

set -euxo pipefail

MODE=$1

if [[ "$MODE" == "latest" ]]; then
  DOCS_HTML=$2
  SITE_ROOT=$3
  GH_PAGES=$4

  # Replace the landing page at the root, keeping the versioned docs tree and
  # the generated versions.json.
  rsync -a --delete --exclude=.git --exclude=/docs --exclude=/versions.json \
    "$SITE_ROOT"/ "$GH_PAGES"/

  mkdir -p "$GH_PAGES/docs"
  rsync -a --delete "$DOCS_HTML"/ "$GH_PAGES/docs/latest/"

  # Drop leftovers from the pre-versioning layout where docs lived directly in
  # docs/. Anything that is neither "latest" nor a version directory goes away.
  find "$GH_PAGES/docs" -mindepth 1 -maxdepth 1 -regextype posix-extended \
    ! -name latest ! -name index.html ! -regex '.*/v?[0-9][^/]*' \
    -exec rm -rf {} +

  cat > "$GH_PAGES/docs/index.html" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=latest/">
<link rel="canonical" href="latest/">
<title>XGrammar documentation</title>
</head>
<body>
<a href="latest/">Redirecting to the latest documentation...</a>
</body>
</html>
EOF
elif [[ "$MODE" == "version" ]]; then
  DOCS_HTML=$2
  VERSION=$3
  GH_PAGES=$4

  # The version becomes a directory name under docs/; reject anything that
  # is not a plain version-looking single path segment.
  if [[ ! "$VERSION" =~ ^v?[0-9][A-Za-z0-9._+-]*$ ]]; then
    echo "Invalid version: $VERSION" >&2
    exit 1
  fi
  rsync -a --delete "$DOCS_HTML"/ "$GH_PAGES/docs/$VERSION/"
else
  echo "Unknown mode: $MODE" >&2
  exit 1
fi

# Regenerate versions.json from the version directories under docs/.
python3 - "$GH_PAGES" <<'EOF'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
version_pattern = re.compile(r"^v?\d")


def sort_key(name):
    core = name.lstrip("v")
    match = re.match(r"^(\d+(?:\.\d+)*)(?:([a-z]+)(\d*))?", core)
    numbers = tuple(int(part) for part in match.group(1).split("."))
    # A final release ("0.2.6") sorts above its pre-releases ("0.2.6rc1").
    # Pre-release kinds compare per PEP 440 ("a" < "b" < "rc") and their
    # numbers compare numerically ("rc10" > "rc2").
    is_final = match.group(2) is None
    suffix_kind = match.group(2) or ""
    suffix_number = int(match.group(3)) if match.group(3) else 0
    return (numbers, is_final, suffix_kind, suffix_number)


versions = sorted(
    (
        path.name
        for path in (root / "docs").iterdir()
        if path.is_dir() and version_pattern.match(path.name)
    ),
    key=sort_key,
    reverse=True,
)
entries = [{"version": "latest", "name": "latest (main)", "url": "/docs/latest/"}]
entries += [{"version": v, "name": v, "url": f"/docs/{v}/"} for v in versions]
(root / "versions.json").write_text(json.dumps(entries, indent=2) + "\n")
EOF
