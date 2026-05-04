# Evaluate tool-calling accuracy and efficiency on SGLang with Structural Tag

The evaluation script is modified based on the BFCL ast checker. The script uses the Structural Tag API to test tool-calling accuracy and efficiency against an SGLang OpenAI-compatible server.

## Test the accuracy

You can use `bash script.sh` directly to test the accuracy. You can also use the following commands manually:

First launch the server.
```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.6-27B --host 127.0.0.1 --port 8000
```

Than generate the raw data (w/ & w/o structural tag):
```bash
cd ./tool_call_eval
python accuracy.py --model Qwen/Qwen3.6-27B \
--tokenizer Qwen/Qwen3.6-27B \
--dataset BFCL_v3_simple --dataset-path ./data/dataset --num-gpus 1 \
--num-requests 400 --num-warmup-requests 1 --request-rate inf \
--host 127.0.0.1 --port 8000 \
--api-endpoint sglang --output ./data/accuracy_raw \
--temperature 0.001 --top-p 0.9 \
[--use-stag]
```

The raw data will be in `./data/accuracy_raw` directory. Finally process the raw data:
```bash
python check.py --dataset ALL --model ALL --dataset-path ./data/dataset \
--output-root ./data/accuracy_raw --final-root ./data/accuracy_summary
```
