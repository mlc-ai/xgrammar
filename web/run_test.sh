#!/bin/bash
set -euxo pipefail

# Ensure latest TypeScript output exists before running Jest ESM suite.
npx tsc -p tsconfig.json

# The wasm binding is produced by emcc in src/. Copy this into lib/ so
# package entry (which points at lib/) can load it during tests.
if [[ ! -f src/xgrammar_binding.js ]]; then
  echo "Missing src/xgrammar_binding.js. Run ./build.sh first." >&2
  exit 1
fi
cp -f src/xgrammar_binding.js lib/xgrammar_binding.js

# tokenizers ships CommonJS-in-ESM wrapping. Copy library file to .cjs
# so Node will load it through CommonJS loader during tests.
tokenizers_lib="node_modules/@mlc-ai/web-tokenizers/lib"
cp -f "${tokenizers_lib}/index.js" "${tokenizers_lib}/index.cjs"

node --experimental-vm-modules node_modules/jest/bin/jest
