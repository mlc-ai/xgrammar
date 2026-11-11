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
It caches model/option combinations for the lifetime of the test process and returns an
independent copy for each call. This avoids repeated Hub metadata requests during
parametrized tests while keeping tokenizer mutations isolated between tests.

## Offline tokenizer assets in CI

The unit-test workflow prepares tokenizer files once and distributes a portable ZIP to
every test runner. Test jobs set `XGRAMMAR_TOKENIZER_ASSETS`, `HF_HUB_OFFLINE=1`, and
`XGRAMMAR_TEST_OFFLINE=1`. The last setting rejects socket connections during pytest,
so missing assets cannot silently fall back to network downloads.

`tokenizer_assets.json` lists the model commit revisions. When adding a tokenizer to a
test, add its pinned revision to this manifest. Use `requires_token: true` for assets
used only by the existing `hf_token_required` tests. The bundle also seeds tiktoken's vocabulary cache from the pinned Phi-3-small snapshot.
The public Qwen smoke test runs
even when PR jobs have no credentials. Changing the manifest or preparation script
invalidates the bundle cache. Credentials and model weights are not included in the ZIP.

To validate locally from a fresh bundle:

```bash
python scripts/prepare_tokenizer_assets.py
python -m zipfile -e .ci/tokenizer-assets.zip .ci/extracted
XGRAMMAR_TOKENIZER_ASSETS="$PWD/.ci/extracted" TIKTOKEN_CACHE_DIR="$PWD/.ci/extracted/tiktoken" \
  HF_HUB_OFFLINE=1 XGRAMMAR_TEST_OFFLINE=1 \
  python -m pytest tests/python/test_tokenizer_utils.py
```

Without a token, preparation includes only the public smoke-test assets. With an
authorized `HF_TOKEN`, it also includes assets for the marked tests. Normal developer
test runs without `XGRAMMAR_TOKENIZER_ASSETS` retain the standard Hub loading behavior.
