#!/usr/bin/env bash

set -e
set -x

black .
mypy .
ruff check . --fix
