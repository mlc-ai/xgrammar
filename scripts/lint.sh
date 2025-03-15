#!/usr/bin/env bash

set -e
set -x

pre-commit run --all-files
ruff check . --fix
