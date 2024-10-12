# web-xgrammar

This folder contains the source code and emcc bindings for compiling XGrammar to Javascript/Typescript via [emscripten](https://emscripten.org/).

### Build from source
Run the following
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
