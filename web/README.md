# web-xgrammar

This folder contains the source code and emcc bindings for compiling XGrammar to Javascript/Typescript via [emscripten](https://emscripten.org/).

### Build from source
1. Install [emscripten](https://emscripten.org). It is an LLVM-based compiler that compiles C/C++ source code to WebAssembly.
    - Follow the [installation instruction](https://emscripten.org/docs/getting_started/downloads.html#installation-instructions-using-the-emsdk-recommended) to install the latest emsdk.
    - Source `emsdk_env.sh` by `source /path/to/emsdk_env.sh`, so that `emcc` is reachable from PATH and the command `emcc` works.
    - We can verify the successful installation by trying out `emcc` in the terminal.

2. Modify the content of `cmake/config.cmake` to be `web/config.cmake`.

3. Run the following

    ```bash
    source /path/to/emsdk_env.sh
    npm install
    npm run build
    ```

### Example
To try out the test webpage, run the following
```bash
cd example
npm install
npm start
```

### Testing
For testing in `node` environment, run:
```bash
npm test
```
