"""
This file helps integrate xgrammar in HF transformers package by extending
transformers.LogitsProcessor, which is to be fed to `model.generate()`.
"""

import transformers
import xgrammar as xgr
import torch


class LogitsProcessor(transformers.LogitsProcessor):
    """
    LogitsProcessor for processing logits in transformers' generate() method.

    Example usage
    -------------
        ```python
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
        # This can be larger than tokenizer.vocab_size due to paddings
        full_vocab_size = config.vocab_size

        grammar_compiler = xgr.CachedGrammarCompiler(tokenizer_info)
        compiled_grammar = grammar_compiler.compile_json_grammar()
        logits_processor = xgr.contrib.transformers.LogitsProcessor(
          compiled_grammar, tokenizer_info, full_vocab_size
        )
        model.generate(prompt, logit_processor=logits_processor)
        ```

        For an end-to-end example, see folder `examples/hf_transformers/`.

    Notes
    -----
        - Note that this LogitsProcessor can only be used once. For each `generate()` call,
            instantiate a new one.
        - Note that this implementation may contain extra overhead.
    """

    def __init__(
        self,
        compiled_grammar: xgr.CompiledGrammar,
        tokenizer_info: xgr.TokenizerInfo,
        full_vocab_size: int,
        batch_size: int = 1,
    ):
        """Initialize the LogitsProcessor.

        Parameters
        ----------
        compiled_grammar : xgr.CompiledGrammar
            A grammar compiled according to the given grammar and the model's tokenizer_info.
        tokenizer_info : xgr.TokenizerInfo
            The tokenizer information of the model to be used.
        full_vocab_size : int
            The full vocab size of the model (AutoConfig.vocab_size).
        batch_size : int
            The batch size to be fed into the `generate()` call that this LogitProcessor will
            be used in.
        """
        self.matchers = [
            xgr.GrammarMatcher(compiled_grammar, tokenizer_info, vocab_size=full_vocab_size)
            for _ in range(batch_size)
        ]
        self.token_bitmask = xgr.GrammarMatcher.allocate_token_bitmask(
            self.matchers[0].vocab_size, batch_size=batch_size
        )
        self.prefilled = False
        self.batch_size = batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Accept token sampled in the last iteration, fill in bitmask, and apply bitmask to logits.

        Returns:
            scores: Logits modified with bitmask.
        """
        if input_ids.shape[0] != self.batch_size:
            raise ValueError(
                "Expect input_ids.shape[0] to be LogitsProcessor.batch_size."
                + f"Got {input_ids.shape[0]} for the former, and {self.batch_size} for the latter."
                + "\nCheck your input to xgrammar.contrib.hf_transformers.LogitsProcessor()."
            )
        if scores.device.type != "cuda":
            raise ValueError("logits must be on CUDA")

        if not self.prefilled:
            # Have not sampled a token yet
            self.prefilled = True
        else:
            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    sampled_token = input_ids[i][-1]
                    assert self.matchers[i].accept_token(sampled_token)

        for i in range(self.batch_size):
            if not self.matchers[i].is_terminated():
                self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

        xgr.GrammarMatcher.apply_token_bitmask_inplace(scores, self.token_bitmask)

        # NOTE: Cannot reset here because __call__ is not invoked when stop token
        # is sampled. This is why each `generate()` call needs to instantiate an
        # LogitsProcessor

        return scores
