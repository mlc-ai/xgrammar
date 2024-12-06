.. _how-to-portable-api:

Portable API
============

XGrammar is implemented with a lightweight C++ core that can be integrated into many platforms.
Besides the C++ backend, we also provide ready-to-use Python and JavaScript/TypeScript libraries.

For the Python library, simply check out the :ref:`Python API Reference <apixgrammar>`. Below
we take a high-level view of the Javascript library.

.. _how-to-portable-js:

Javascript SDK for Web-based LLMs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The JS SDK is designed to be used for LLMs that run in the browser, including
`WebLLM <https://github.com/mlc-ai/web-llm>`__. WebLLM integrated with XGrammar's
JS SDK, ``web-xgrammar``. It uses `emscripten <https://emscripten.org/>`__ to compile
the C++ code into WebAssembly.

To use this SDK, simply run ``npm install @mlc-ai/web-xgrammar``. For more, see
`here <https://github.com/mlc-ai/xgrammar/tree/main/web>`__.
