#!/bin/bash

# Usage: bash ./scripts/run_coverage.sh
lcov --directory . --zerocounters
ctest --test-dir build -V --timeout 60 --stop-on-failure
pytest

lcov --gcov-tool /usr/bin/gcov-13 --directory . --capture --output-file coverage.info --ignore-errors mismatch,gcov
genhtml coverage.info --output-directory coverage_report --ignore-errors version
rm coverage.info
echo "Coverage report generated at: $(pwd)/coverage_report/index.html"
