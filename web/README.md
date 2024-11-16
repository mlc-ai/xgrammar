# web-xgrammar

This folder contains the source code and emcc bindings for compiling XGrammar to Javascript/Typescript via [emscripten](https://emscripten.org/).

### Build from source
First modify the content of `cmake/config.cmake` to be `web/config.cmake`.

Then run the following
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
