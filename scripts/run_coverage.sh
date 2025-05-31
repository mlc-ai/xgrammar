#!/bin/bash

# Usage: bash ./scripts/run_coverage.sh
pytest

lcov --gcov-tool /usr/bin/gcov-13 --directory . --capture --output-file coverage.info
genhtml coverage.info --output-directory coverage_report --ignore-errors version
rm coverage.info
echo "Coverage report generated at: $(pwd)/coverage_report/index.html"
