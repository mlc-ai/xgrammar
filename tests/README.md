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

## Automatic tokenizer revision updates

`update_tokenizer_assets.yaml` checks tracked Hub refs weekly (Monday at 06:23 UTC),
or on manual dispatch. It queries commit SHAs sequentially without downloading files,
reuses the bounded 429 retry handling, and creates or updates a draft PR on
`ci/update-tokenizer-assets` when revisions change. Ordinary PR tests remain offline.
HF commits that change only weights may also cause revision updates.

Entries track `main` by default. Set `update_ref` to another branch/tag to track it,
or to `null` to preserve a release intentionally pinned by a test. The DeepSeek V4.1
and MiMo release tests keep their explicit pins. Without `HF_TOKEN`, the updater
checks only entries with `requires_token: false`.

Enable **Allow GitHub Actions to create and approve pull requests** in the repository's
Actions settings. The workflow uses `GITHUB_TOKEN` with contents, pull-request and
Actions write permissions. Because this token does not trigger normal PR workflows,
the updater explicitly dispatches `unit_test.yaml` on the update branch; its checks
attach to that branch's head commit. It does not dispatch docs deployment or PyPI
publishing workflows. Review and merge the update PR after tests pass.

The scheduled workflow runs only after it is merged into the default branch. Forks
can test it with manual dispatch; automatic scheduled runs are limited to upstream.
