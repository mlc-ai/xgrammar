"""
This example demonstrates how to use XGrammar in Huggingface's transformers, integrated with
a minimal LogitsProcessor.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import xgrammar as xgr

device = "cuda"
# device = "cpu"

# 0. Instantiate with any HF model you want
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "microsoft/Phi-3.5-mini-instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# This can be larger than tokenizer.vocab_size due to paddings
full_vocab_size = config.vocab_size

# 1. Compile grammar (NOTE: you can substitute this with other grammars like EBNF, JSON Schema)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar: xgr.CompiledGrammar = grammar_compiler.compile_builtin_json_grammar()

# 2. Prepare inputs
messages_list = []
prompts = [
    "Introduce yourself in JSON briefly as a student.",
    # Uncomment for batch generation
    # "Introduce yourself in JSON as a professor.",
]
for prompt in prompts:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    messages_list.append(messages)
texts = [
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    for messages in messages_list
]

# For batched requests, either use a model that has a padding token, or specify your own
# model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)

# 3. Instantiate logits_processor per each generate, and call generate()
xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
generated_ids = model.generate(
    **model_inputs, max_new_tokens=512, logits_processor=[xgr_logits_processor]
)

# 4. Post-process outputs and print out response
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
for response in responses:
    print(response, end="\n\n")
