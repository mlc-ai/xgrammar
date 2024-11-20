import xgrammar as xgr
import torch

from xgrammar.contrib.hf_transformers import LogitsProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Instantiate model
model_name = "/ssd1/cfruan/xgrammar/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Compile grammar
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
full_vocab_size = config.vocab_size
grammar_compiler = xgr.CachedGrammarCompiler(tokenizer_info, max_threads=1)
compiled_grammar = grammar_compiler.compile_json_grammar()

# Prepare inputs
prompt = "Introduce yourself in JSON briefly."
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate
logits_processor = LogitsProcessor(compiled_grammar, tokenizer_info, full_vocab_size)
generated_ids = model.generate(
    **model_inputs, max_new_tokens=512, logits_processor=[logits_processor]
)

# Post-processing and print out response
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
