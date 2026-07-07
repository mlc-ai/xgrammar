import sys

import pytest
import torch
import torch.nn as nn
from tokenizers import Regex, Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedConfig,
    PreTrainedModel,
    TokenizersBackend,
    pipeline,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

import xgrammar as xgr


# Set up a dummy model that generates the sequence aab
class ABTokenizer(TokenizersBackend):
    def __init__(self):
        vocab = {"<unk>": 0, "<eos>": 1, "a": 2, "b": 3}
        tokenizer_object = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_object.pre_tokenizer = Split(Regex("."), behavior="isolated")
        tokenizer_object.decoder = decoders.Fuse()
        super().__init__(
            tokenizer_object=tokenizer_object,
            unk_token="<unk>",
            pad_token="<eos>",
            eos_token="<eos>",
        )


class AABConfig(PreTrainedConfig):
    model_type = "aab"

    def __init__(self, vocab_size=4, **kwargs):
        self.vocab_size = vocab_size
        super().__init__(**kwargs)


class AAB(PreTrainedModel, GenerationMixin):
    config_class = AABConfig

    def __init__(self, config):
        super().__init__(config)
        v = 4
        self.embed = nn.Embedding(v, v)
        self.linear = nn.Linear(v, v, bias=False)
        self.post_init()

        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(v))
            self.embed.weight.requires_grad = False
            self.linear.weight.zero_()
            for i in [0, 1, 3]:
                self.linear.weight[2, i] = 1
            self.linear.weight[3, 2] = 0.1
            self.linear.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        tokens = input_ids[:, -2:]
        emb = self.embed(tokens)
        logits = self.linear(emb).sum(dim=1, keepdim=True)

        return CausalLMOutputWithPast(loss=None, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False


# Test forcing ab sequences instead
def test_constrain_aab_to_ab():
    AutoConfig.register("aab", AABConfig)
    AutoModelForCausalLM.register(AABConfig, AAB)
    tokenizer = ABTokenizer()
    config = AABConfig(vocab_size=4, pad_token_id=1, eos_token_id=1)
    model = AAB(config)

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_regex("(ab){3}")
    xgr_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, logits_processor=[xgr_processor]
    )
    results = pipe(["ba", "ab"], max_new_tokens=8, do_sample=False, batch_size=2)
    assert results[0][0]["generated_text"] == "baababab"
    assert results[1][0]["generated_text"] == "abababab"

    xgr_processor_stateless = xgr.contrib.hf.LogitsProcessor(compiled_grammar, stateless=True)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        logits_processor=[xgr_processor_stateless],
    )
    results = pipe(["ba", "ab"], max_new_tokens=8, do_sample=False, batch_size=2, num_beams=4)
    assert results[0][0]["generated_text"] == "baababab"
    assert results[1][0]["generated_text"] == "abababab"


if __name__ == "__main__":
    pytest.main(sys.argv)
