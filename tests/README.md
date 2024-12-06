To test, run `pytest .` under `xgrammar` folder. You may need to do the following:

```bash
pip install sentencepiece
pip install protobuf
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token YOUR_HF_TOKEN
```

Make sure you also have access to the gated models, which should only require you to agree
some terms on the models' website on huggingface.
