# JavaScript API

Beside the Python and C++ API, XGrammar also provides JavaScript/TypeScript API.

The JS SDK uses [emscripten](https://emscripten.org/) to compile the C++
code into WebAssembly. It is designed to be used for LLMs that run in the browser, such as
[WebLLM](https://github.com/mlc-ai/web-llm).

To use this SDK, run:

```bash
npm install @mlc-ai/web-xgrammar
```

For more information, see [the code](https://github.com/mlc-ai/xgrammar/tree/main/web).
