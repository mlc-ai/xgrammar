To test, run `pytest .` under `xgrammar` folder. You may need to do the following:

```bash
pip install sentencepiece
pip install protobuf
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token YOUR_HF_TOKEN
```

Make sure you also have access to the gated models, which should only require you to agree
some terms on the models' website on huggingface.

Python tests should load Hugging Face tokenizers with `tokenizer_utils.load_tokenizer`.
It caches up to eight model/option combinations and returns an independent copy for each
call. This avoids repeated Hub metadata requests during parametrized tests while keeping
tokenizer mutations isolated between tests.
