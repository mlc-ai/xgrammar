#!/usr/bin/env bash

# Usage:
#   ./scripts/run_ctest.sh [name_of_test_to_run]

set -euxo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../build" && cmake .. -G Ninja && ninja

if [ $# -gt 0 ]; then
  # If a test name is given, run only that test
  ctest -R "$1" --verbose --timeout 30
else
  # If no argument is given, run the full test suite
  ctest --verbose --timeout 30
fi
