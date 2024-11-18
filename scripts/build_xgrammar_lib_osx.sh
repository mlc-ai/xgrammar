#!/bin/bash

set -e
set -u

# setup config.cmake
rm -f config.cmake
echo set\(XGRAMMAR_BUILD_PYTHON_BINDINGS ON\) >>config.cmake
echo set\(XGRAMMAR_BUILD_CXX_TESTS OFF\) >>config.cmake


# compile the xgrammar
rm -rf build
mkdir -p build
cd build

MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-${1}}

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} \
      -DHIDE_PRIVATE_SYMBOLS=ON \
      ..

make -j3
cd ..
