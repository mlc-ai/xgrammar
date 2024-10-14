#!/bin/bash
set -euxo pipefail

mkdir -p build
cd build
emcmake cmake ../.. -DBUILD_PYTHON_BINDINGS=OFF\
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -DCOMPILE_WASM_RUNTIME -DXGRAMMAR_LOG_CUSTOMIZE=1" 
emmake make xgrammar -j8
cd ..

emcc --bind -o src/xgrammar_binding.js src/xgrammar_binding.cc\
  build/libxgrammar.a\
 -O3 -s EXPORT_ES6=1 -s ERROR_ON_UNDEFINED_SYMBOLS=0 -s NO_DYNAMIC_EXECUTION=1 -s MODULARIZE=1 -s SINGLE_FILE=1 -s EXPORTED_RUNTIME_METHODS=FS -s ALLOW_MEMORY_GROWTH=1\
 -I../include -I../3rdparty/picojson -I../3rdparty/dlpack/include
