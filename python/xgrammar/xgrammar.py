# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The main functionality of XGrammar. The functions here are Python bindings of the C++ logic."""

import json
import math
from enum import Enum
from typing import List, Optional, Tuple, Type, Union, overload

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from . import xgrammar_bindings as _core
from .cuda.apply_token_mask_inplace import (
    apply_token_bitmask_inplace as apply_token_bitmask_inplace_cuda,
)


class XGObject:
    """The base class for all objects in XGrammar. This class provides methods to handle the
    interaction between Python and C++ objects.
    """

    @classmethod
    def from_handle(cls, handle) -> "XGObject":
        """Construct an object of the class from a C++ handle.

        Parameters
        ----------
        cls
            The class of the object.

        handle
            The C++ handle.

        Returns
        -------
        obj : XGObject
            An object of type cls.
        """
        obj = cls.__new__(cls)
        obj._handle = handle
        return obj

    def init_with_handle(self, handle):
        """Initialize an object with a handle. Used in the __init__ method of the class.

        Parameters
        ----------
        handle
            The C++ handle.
        """
        self._handle = handle

    @property
    def handle(self):
        """Get the C++ handle of the object.

        Returns
        -------
        handle
            The C++ handle.
        """
        return self._handle


class BNFGrammar(XGObject):
    """This class represents a grammar object in the Backus-Naur Form (BNF). User should provide a
    BNF/EBNF (Extended Backus-Naur Form) grammar. The provided grammar is optimized for LLM
    generation. This class is printable and serializable.

    Parameters
    ----------
    ebnf_string : str
        The grammar string in EBNF format. It should follow the format in
        https://www.w3.org/TR/xml/#sec-notation.

        Note:
        1. Use # as the comment mark
        2. Use C-style unicode escape sequence \u01AB, \U000001AB, \xAB
        3. A-B (match A and not match B) is not supported

    root_rule : str, default: "root"
        The name of the root rule in the grammar.
    """

    def __init__(self, ebnf_string: str, *, root_rule: str = "root") -> None:
        self.init_with_handle(_core.BNFGrammar(ebnf_string, root_rule))

    def to_string(self) -> str:
        """Print the BNF grammar to a string, in EBNF format.

        Returns
        -------
        grammar_string : str
            The BNF grammar string.
        """
        return self.handle.to_string()

    def __str__(self) -> str:
        """Print the BNF grammar to a string, in EBNF format.

        Returns
        -------
        grammar_string : str
            The BNF grammar string.
        """
        return self.to_string()

    def serialize(self, *, prettify: bool = False) -> str:
        """Serialize the BNF grammar to a JSON string in the raw representation.

        Parameters
        ----------
        prettify : bool
            Whether to format the JSON string. If False, all whitespaces will be removed.

        Returns
        -------
        json_string : str
            The serialized JSON string.
        """
        return self.handle.serialize(prettify)

    @staticmethod
    def deserialize(json_string: str) -> "BNFGrammar":
        """Load a BNF grammar from the raw representation in JSON format.

        Parameters
        ----------
        json_string : str
            The serialized JSON string.

        Returns
        -------
        grammar : BNFGrammar
            The loaded BNF grammar.
        """
        return BNFGrammar.from_handle(_core.BNFGrammar.deserialize(json_string))

    @staticmethod
    def _init_no_normalization(
        ebnf_string: str,
        root_rule: str = "root",
    ) -> "BNFGrammar":
        r"""Construct a BNF grammar object with a EBNF string, but not normalize it. For test
        purposes.

        Parameters
        ----------
        ebnf_string : str
            The grammar string.

        root_rule : str
            The name of the root rule. Default: "root".

        Returns
        -------
        grammar : BNFGrammar
            The parsed BNF grammar.
        """
        return BNFGrammar.from_handle(
            _core.BNFGrammar._init_no_normalization(ebnf_string, root_rule),
        )


class BuiltinGrammar:
    @staticmethod
    def json() -> BNFGrammar:
        """Get the grammar of standard JSON. This is compatible with the official JSON grammar
        in https://www.json.org/json-en.html.

        Returns
        -------
        grammar : BNFGrammar
            The JSON grammar.
        """
        return BNFGrammar.from_handle(_core.BuiltinGrammar.json())

    @staticmethod
    def json_schema(
        schema: Union[str, Type[BaseModel]],
        *,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = True,
    ) -> BNFGrammar:
        """Construct a BNF grammar from JSON schema. Pydantic model can be used to specify the
        schema.

        The format of the JSON schema can be specified with the `indent` and `separators`
        parameters. The meaning and the default values of the parameters follows the convention in
        json.dumps().

        Parameters
        ----------
        schema : Union[str, Type[BaseModel]]
            The schema string or Pydantic model.

        indent : Optional[int], default: None
            The number of spaces for indentation. If None, the output will be in one line.

        separators : Optional[Tuple[str, str]], default: None
            Two separators used in the schema: comma and colon. Examples: (",", ":"), (", ", ": ").
            If None, the default separators will be used: (",", ": ") when the indent is not None,
            and (", ", ": ") otherwise.

        strict_mode : bool, default: True
            Whether to use strict mode. In strict mode, the generated grammar will not allow
            properties and items that is not specified in the schema. This is equivalent to
            setting unevaluatedProperties and unevaluatedItems to false.

            This helps LLM to generate accurate output in the grammar-guided generation with JSON
            schema.

        Returns
        -------
        grammar : BNFGrammar
            The generated BNF grammar.
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = json.dumps(schema.model_json_schema())

        return BNFGrammar.from_handle(
            _core.BuiltinGrammar.json_schema(schema, indent, separators, strict_mode),
        )

    @staticmethod
    def _json_schema_to_ebnf(
        schema: str,
        *,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = True,
    ) -> str:
        """Convert JSON schema string to EBNF grammar string. For test purposes.

        Parameters
        ----------
        schema : str
            The schema string.

        indent : Optional[int], default: None
            The number of spaces for indentation. If None, the output will be in one line.

        separators : Optional[Tuple[str, str]], default: None
            Two separators used in the schema: comma and colon. Examples: (",", ":"), (", ", ": ").
            If None, the default separators will be used: (",", ": ") when the indent is not None,
            and (", ", ": ") otherwise.

        strict_mode : bool, default: True
            Whether to use strict mode. In strict mode, the generated grammar will not allow
            properties and items that is not specified in the schema. This is equivalent to
            setting unevaluatedProperties and unevaluatedItems to false.

            This helps LLM to generate accurate output in the grammar-guided generation with JSON
            schema.


        Returns
        -------
        ebnf_string : str
            The EBNF grammar string.
        """
        return _core.BuiltinGrammar._json_schema_to_ebnf(
            schema,
            indent,
            separators,
            strict_mode,
        )

    @staticmethod
    def _regex_to_ebnf(regex: str) -> str:
        r"""Convert a regex string to EBNF grammar string. For test purposes. The regex grammar
        follows the syntax in JavaScript (ECMA 262). Check
        https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions
        for a tutorial. Currently the following features are not supported:
        1. Backreference (\1)
        2. non-capturing group, naming capture groups and assertions ((?...))
        3. Unicode character class escape (\p{...})
        4. Word boundary (\b)
        5. Unicode property escapes (\p{...})
        6. Quantifier with range {x,y}. Now user can just repeat the element as a workaround.

        This method is primarily intended for testing and debugging purposes.

        Parameters
        ----------
        regex : str
            The regex string to be converted.

        Returns
        -------
        ebnf_string : str
            The EBNF grammar string converted from the input regex.
        """
        return _core.BuiltinGrammar._regex_to_ebnf(regex)


class VocabType(Enum):
    """The type of the vocabulary. Used in TokenizerInfo. XGrammar supports three types of
    vocabularies:

    RAW
        The vocabulary is in the raw format. The tokens in the vocabulary is the same as the input
        string. This kind of tokenizer includes the tiktoken tokenizer, e.g.
        microsoft/Phi-3-small-8k-instruct, Qwen/Qwen-7B-Chat, etc.

    BYTE_FALLBACK
        The vocabulary used in the byte fallback BPE tokenizer. The tokens are processed through
        the byte-fallback conversion. E.g. "\u001B" -> "<0x1B>", " apple" -> "▁apple". This kind of
        tokenizer includes meta-llama/Llama-2-7b-chat, microsoft/Phi-3.5-mini-instruct, etc.

    BYTE_LEVEL
        The vocabulary used in the byte level BPE tokenizer. The tokens are processed through
        the byte-to-unicode conversion, as in
        https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59

        This kind of tokenizer includes meta-llama/Meta-Llama-3-8B-Instruct,
        meta-llama/Meta-Llama-3.1-8B-Instruct, etc.
    """

    RAW = "RAW"
    BYTE_FALLBACK = "BYTE_FALLBACK"
    BYTE_LEVEL = "BYTE_LEVEL"


class TokenizerInfo(XGObject):
    """The tokenizer info, which contains the vocabulary, the type of the vocabulary, and necessary
    information for the grammar-guided generation. This class should be the first choice when
    handling tokenizers in XGrammar. It eliminates the overhead of converting the vocabulary between
    C++ and Python.

    Note that the vocabulary in TokenizerInfo is in the decoded format. Some tokenizers will encode
    the tokens in a special format. E.g. "<0x1B>" for "\u001B" in the ByteFallback tokenizer, and
    "Ġ" for " " in the Byte-Level BPE tokenizer. Huggingface tokenizer.get_vocab() will return
    the encoded vocabulary. TokenizerInfo will decode the vocabulary to the original format.

    Parameters
    ----------
    encoded_vocab : Union[List[bytes], List[str]]
        The vocabulary of the tokenizer.

    vocab_type : VocabType, default: VocabType.RAW
        The type of the vocabulary. See also VocabType.

    prepend_space_in_tokenization : bool, default: False
        Whether the tokenizer will prepend a space before the text in the tokenization process.
    """

    def __init__(
        self,
        encoded_vocab: Union[List[bytes], List[str]],
        vocab_type: VocabType = VocabType.RAW,
        prepend_space_in_tokenization: bool = False,
    ) -> None:
        self.init_with_handle(
            _core.TokenizerInfo(encoded_vocab, vocab_type.value, prepend_space_in_tokenization)
        )

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary."""
        return self.handle.vocab_size

    @property
    def vocab_type(self) -> VocabType:
        """The type of the vocabulary."""
        return VocabType(self.handle.vocab_type)

    @property
    def prepend_space_in_tokenization(self) -> bool:
        """Whether the tokenizer will prepend a space before the text in the tokenization
        process."""
        return self.handle.prepend_space_in_tokenization

    @property
    def decoded_vocab(self) -> List[bytes]:
        """The raw vocabulary of the tokenizer. This converts the tokens in the LLM's vocabulary
        back to the original format of the input text. E.g. for type ByteFallback, the token
        <0x1B> is converted back to "\u001B" in the raw vocabulary.
        """
        return self.handle.decoded_vocab

    @staticmethod
    def from_huggingface(tokenizer: PreTrainedTokenizerBase) -> "TokenizerInfo":
        """Construct the tokenizer info from the huggingface tokenizer. This constructor supports
        various tokenizer backends, including the huggingface fast tokenizer and tiktoken tokenizer.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizerBase
            The huggingface tokenizer.

        Returns
        -------
        tokenizer_info : TokenizerInfo
            The tokenizer info.
        """

        try:
            encoded_vocab = tokenizer.get_vocab()
            encoded_vocab = [
                token for token, _ in sorted(encoded_vocab.items(), key=lambda x: x[1])
            ]
        except AttributeError as e:
            msg = (
                f"Cannot get the vocabulary of the tokenizer {type(tokenizer)}. The tokenizer "
                "should have a get_vocab method."
            )
            raise ValueError(msg) from e

        if isinstance(tokenizer, PreTrainedTokenizerFast):
            # huggingface fast tokenizer
            # Note this backend_str may not contain the full vocab. Some special tokens may
            # be omitted. So we still need to pass the vocab to the constructor.
            backend_str = tokenizer.backend_tokenizer.to_str()
            return TokenizerInfo.from_handle(
                _core.TokenizerInfo.from_huggingface(encoded_vocab, backend_str)
            )
        elif (
            "vocab_file" in tokenizer.vocab_files_names
            and "tiktoken" in tokenizer.vocab_files_names["vocab_file"]
        ):
            # tiktoken tokenizer
            # e.g. Phi-3-small-8k-instruct, Qwen-7B-Chat, stablelm-2-12b-chat (previously)
            return TokenizerInfo(encoded_vocab, VocabType.RAW, False)
        else:
            # TODO(yixin): sentencepiece tokenizer
            raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")

    def dump_metadata(self) -> str:
        """Dump the metadata of the tokenizer, mainly vocab_type and
        prepend_space_in_tokenization."""
        return self.handle.dump_metadata()

    @staticmethod
    def from_vocab_and_metadata(
        encoded_vocab: List[Union[bytes, str]], metadata: str
    ) -> "TokenizerInfo":
        """Construct the tokenizer info from the vocabulary and the metadata string.

        Parameters
        ----------
        encoded_vocab : List[Union[bytes, str]]
            The vocabulary of the tokenizer.

        metadata : str
            The metadata string.
        """
        return TokenizerInfo.from_handle(
            _core.TokenizerInfo.from_vocab_and_metadata(encoded_vocab, metadata),
        )


class CompiledGrammar(XGObject):
    """The initialization context for the grammar matcher. This object is used to fast initialize
    the grammar matcher. It contains the grammar, the raw vocabulary, and the preprocessed cache
    for the generating the mask in the grammar-guided generation.

    Parameters
    ----------
    grammar : BNFGrammar
        The BNF grammar to match.

    tokenizer_info : Optional[TokenizerInfo], default: None
        The tokenizer info. If None, the grammar matcher can only handle string operations.

    max_threads : int, default: 8
        The maximum number of threads used to compile the grammar.
    """

    def __init__(
        self,
        grammar: BNFGrammar,
        tokenizer_info: Optional[TokenizerInfo] = None,
        max_threads: int = 8,
    ) -> None:
        if tokenizer_info is None:
            tokenizer_info = TokenizerInfo([])
        elif not isinstance(tokenizer_info, TokenizerInfo):
            raise ValueError(
                "Please convert the tokenizer to TokenizerInfo before passing it "
                "to CompiledGrammar."
            )

        self.init_with_handle(
            _core.CompiledGrammar(grammar.handle, tokenizer_info.handle, max_threads)
        )


class CachedGrammarCompiler(XGObject):
    """The cache for the grammar matcher initialization context. It is for eliminating the overhead
    of constructing the CompiledGrammar of the same grammar for many times. This cache
    is tokenizer-specific, i.e. different tokenizers should have different caches.

    Parameters
    ----------
    tokenizer_info : TokenizerInfo
        The tokenizer info.

    max_threads : int, default: 8
        The maximum number of threads used to compile the grammar.
    """

    def __init__(self, tokenizer_info: TokenizerInfo, max_threads: int = 8):
        if not isinstance(tokenizer_info, TokenizerInfo):
            raise ValueError(
                "Please convert the tokenizer to TokenizerInfo before passing it "
                "to CachedGrammarCompiler."
            )

        self.init_with_handle(_core.CachedGrammarCompiler(tokenizer_info.handle, max_threads))

    def compile_json_grammar(self) -> CompiledGrammar:
        """Get CompiledGrammar from the standard JSON.

        Returns
        -------
        compiled_grammar : CompiledGrammar
            The initialization context for the grammar matcher.
        """
        return CompiledGrammar.from_handle(self.handle.compile_json_grammar())

    def compile_json_schema_grammar(
        self,
        schema: Union[str, Type[BaseModel]],
        *,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = True,
    ) -> CompiledGrammar:
        """Get CompiledGrammar from the specified JSON schema and format. The indent
        and separators parameters follow the same convention as in json.dumps().

        Parameters
        ----------
        schema : Union[str, Type[BaseModel]]
            The schema string or Pydantic model.

        indent : Optional[int], default: None
            The number of spaces for indentation. If None, the output will be in one line.

        separators : Optional[Tuple[str, str]], default: None
            Two separators used in the schema: comma and colon. Examples: (",", ":"), (", ", ": ").
            If None, the default separators will be used: (",", ": ") when the indent is not None,
            and (", ", ": ") otherwise.

        strict_mode : bool, default: True
            Whether to use strict mode. In strict mode, the generated grammar will not allow
            properties and items that is not specified in the schema. This is equivalent to
            setting unevaluatedProperties and unevaluatedItems to false.

        Returns
        -------
        compiled_grammar : CompiledGrammar
            The initialization context for the grammar matcher.
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = json.dumps(schema.model_json_schema())

        return CompiledGrammar.from_handle(
            self.handle.compile_json_schema_grammar(schema, indent, separators, strict_mode)
        )

    def clear(self) -> None:
        """Clear all cached compiled grammars."""
        self.handle.clear()


class GrammarMatcher(XGObject):
    """Match the output of the LLM to the specified grammar, then generate the mask for the next
    token. This is the core class in the grammar-guided generation.

    This class maintains a stateful matcher that can accept tokens and strings, then match them
    to the specified grammar. The matcher can provide a bitmask for the next token prediction,
    so that the output of the LLM follows the specified grammar. Its state can be reset and
    rolled back by tokens. It also provides utilities for jump-forward decoding.

    After matching the whole grammar, the matcher can still accept a stop token. The token mask at
    this time will also allow stop tokens. After accepting the stop token, the matcher will
    terminate, then it cannot accept any new token or generate a new token mask.

    Under the hood, it utilizes a recursive descent parser with backtracking to match the grammar,
    with optimizations specific to LLM token mask generation.
    """

    @overload
    def __init__(
        self,
        grammar: BNFGrammar,
        tokenizer_info: Optional[TokenizerInfo] = None,
        *,
        override_stop_tokens: Union[None, int, List[int]] = None,
        terminate_without_stop_token: bool = False,
        vocab_size: Optional[int] = None,
        max_rollback_tokens: int = 0,
    ) -> None:
        """Initialize the grammar matcher with a grammar and a tokenizer or vocabulary.

        Parameters
        ----------
        grammar : BNFGrammar
            The BNF grammar to match.

        tokenizer_or_vocab : Union[None, PreTrainedTokenizerBase, TokenizerInfo, List[Union[bytes, str]]], default: None
            The tokenizer or the vocabulary. It can be None, a huggingface tokenizer, a tokenizer info,
            or a list of raw tokens.

            None means there is no vocabulary, then the grammar matcher can only handle string
            operations. If a huggingface tokenizer or a list of raw tokens are provided, a TokenizerInfo
            object will be constructed from the tokenizer or the vocabulary.

        override_stop_tokens : Union[None, int, List[int]], default: None
            The ids of the stop tokens. If None, the stop tokens are detected from the vocabulary.

        terminate_without_stop_token : bool, default: False
            Whether to accept a stop token before terminating. If True, the matcher will directly
            terminate after matching the whole grammar.

        vocab_size : Optional[int], default: None
            The size of the mask. Some LLMs may have a larger model vocabulary size (i.e. the
            dimension of the logits) than the tokenizer vocabulary size (i.e. the number of tokens
            in the vocabulary), as the model vocabulary size may be rounded up to the multiple of
            the power of 64. In this case, the model vocabulary size should be passed to align the
            mask size with the model vocabulary size.

            If None, the mask size is set to the tokenizer vocabulary size.

        max_rollback_tokens : int, default: 0
            The maximum number of tokens to rollback. When larger, more matcher states need to be
            recorded in the memory.
        """
        ...

    @overload
    def __init__(
        self,
        compiled_grammar: CompiledGrammar,
        *,
        override_stop_tokens: Union[None, int, List[int]] = None,
        terminate_without_stop_token: bool = False,
        vocab_size: Optional[int] = None,
        max_rollback_tokens: int = 0,
    ) -> None:
        """Initialize the grammar matcher with a grammar matcher initialization context. This
        initialization is very fast.

        Parameters
        ----------
        compiled_grammar : CompiledGrammar
            The initialization context for the grammar matcher.
        """
        ...

    def __init__(
        self,
        grammar_or_context: Union[BNFGrammar, CompiledGrammar],
        tokenizer_info: Optional[TokenizerInfo] = None,
        *,
        override_stop_tokens: Union[None, int, List[int]] = None,
        terminate_without_stop_token: bool = False,
        vocab_size: Optional[int] = None,
        max_rollback_tokens: int = 0,
    ) -> None:
        if isinstance(grammar_or_context, BNFGrammar):
            compiled_grammar = CompiledGrammar(grammar_or_context, tokenizer_info)
        else:
            compiled_grammar = grammar_or_context

        if isinstance(override_stop_tokens, int):
            override_stop_tokens = [override_stop_tokens]

        self.init_with_handle(
            _core.GrammarMatcher(
                compiled_grammar.handle,
                override_stop_tokens,
                terminate_without_stop_token,
                vocab_size,
                max_rollback_tokens,
            )
        )

    def accept_token(self, token_id: int, *, verbose: bool = False) -> bool:
        """Accept one token and update the state of the matcher.

        Parameters
        ----------
        token_id : int
            The id of the token to accept.

        verbose : bool, default: False
            Whether to print information about the internal state of the matcher. Helpful
            for debugging.

        Returns
        -------
        accepted : bool
            Whether the token is accepted.
        """
        return self.handle.accept_token(token_id, verbose)

    def accept_string(self, input_str: Union[str, bytes], *, verbose: bool = False) -> bool:
        """Accept a string and update the state of the matcher. The whole string is considered
        as one token in rollback. It is only used to complement the functionality of accept_token.

        Parameters
        ----------
        input_str : Union[str, bytes]
            The string to be accepted.

        verbose : bool, default: False
            Whether to print information about the internal state of the matcher. Helpful for
            debugging.

        Returns
        -------
        accepted : bool
            Whether the string is accepted.
        """
        return self.handle.accept_string(input_str, verbose)

    @staticmethod
    def allocate_token_bitmask(vocab_size: int, batch_size: Optional[int] = None) -> torch.Tensor:
        """Allocate the bitmask for the next token prediction. The bitmask is a int32 tensor on CPU
        with shape (batch_size, ceil(vocab_size / 32)). If the batch size is None, the bitmask is
        a 1D tensor with shape (ceil(vocab_size / 32),).

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary.

        batch_size : Optional[int], default: None
            The batch size of the bitmask. If None, the bitmask is a 1D tensor.

        Returns
        -------
        bitmask : torch.Tensor
            The bitmask for the next token prediction.
        """
        if batch_size is None:
            return torch.zeros(math.ceil(vocab_size / 32), dtype=torch.int32)
        else:
            return torch.zeros(batch_size, math.ceil(vocab_size / 32), dtype=torch.int32)

    def fill_next_token_bitmask(self, bitmask: torch.Tensor, batch_id: int = 0) -> None:
        """Fill the bitmask for the next token prediction.

        Parameters
        ----------
        bitmask : torch.Tensor
            The bitmask for the next token prediction. Should be a 1D or 2D tensor generated by
            allocate_token_bitmask. Should be on GPU.

        batch_id : int, default: 0
            The batch id of the bitmask. For batch inference, bitmask[batch_id] will be filled
            with the next token bitmask. Otherwise is ignored.
        """
        self.handle.fill_next_token_bitmask(bitmask, batch_id)

    @staticmethod
    def apply_token_bitmask_inplace(logits: torch.Tensor, bitmask: torch.Tensor):
        """Apply the bitmask to the logits in-place. The shape of logits and bitmask should match,
        either (vocab_size,) and (bitmask_size,) respectively, or (batch_size, vocab_size) and
        (batch_size, bitmask_size) respectively. bitmask_size = ceil(vocab_size / 32).

        Parameters
        ----------
        logits : torch.Tensor
            The tensor to apply the bitmask to. Should be on CUDA.

        bitmask : torch.Tensor
            The bitmask to apply. Should be generated by allocate_token_bitmask and
            filled by fill_next_token_bitmask.
        """
        if logits.device.type != "cuda":
            raise ValueError("logits must be on CUDA")

        if bitmask.device != logits.device:
            bitmask = bitmask.to(logits.device)

        apply_token_bitmask_inplace_cuda(logits, bitmask)

    def debug_get_masked_tokens_from_bitmask(
        self, bitmask: torch.Tensor, batch_id: int = 0
    ) -> List[int]:
        """Get the ids of the rejected tokens from the bitmask. Mainly for debug purposes.

        Parameters
        ----------
        bitmask : torch.Tensor
            The rejected token bitmask. Should be generated by allocate_token_bitmask and
            filled by fill_next_token_bitmask. Should be on CPU.

        batch_id : int, default: 0
            The batch id of the bitmask. For batch inference, bitmask[batch_id] will be used.
            Otherwise is ignored.

        Returns
        -------
        rejected_token_ids : List[int]
            A list of rejected token ids.
        """
        return self.handle.debug_get_masked_tokens_from_bitmask(bitmask, batch_id)

    def find_jump_forward_string(self) -> str:
        """Find the jump-forward string for jump-forward decoding. This is the longest string that
        certainly conforms with the current grammar from the current matcher state. This string
        can become the output of the LLM without requiring LLM decoding.

        This method does not change the matcher state.

        Returns
        -------
        jump_forward_string : str
            The jump-forward string.
        """
        return self.handle.find_jump_forward_string()

    def rollback(self, num_tokens: int = 1) -> None:
        """Rollback the matcher to a previous state by several tokens.

        Parameters
        ----------
        num_tokens : int, default: 1
            The number of tokens to rollback. It cannot exceed the current number of steps, nor can
            it exceed the specified maximum number of rollback tokens.
        """
        self.handle.rollback(num_tokens)

    def is_terminated(self) -> bool:
        """Check if the matcher has terminated. If terminate_without_stop_token is False, the
        matcher will terminate if it has accepted the stop token. Otherwise, the matcher will
        terminate after matching the whole grammar.

        Returns
        -------
        terminated : bool
            Whether the matcher has terminated.
        """
        return self.handle.is_terminated()

    def reset(self) -> None:
        """Reset the matcher to the initial state."""
        return self.handle.reset()

    @property
    def max_rollback_tokens(self) -> int:
        """Get the maximum number of rollback tokens allowed.

        Returns
        -------
        max_rollback_tokens : int
            The maximum number of rollback tokens.
        """
        return self.handle.max_rollback_tokens

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary in the generated mask.

        Returns
        -------
        vocab_size : int
            The size of the vocabulary in the generated mask.
        """
        return self.handle.vocab_size

    @property
    def stop_token_ids(self) -> List[int]:
        """The ids of the stop tokens used in the matcher. If specified, the provided stop tokens
        will be used. Otherwise, the stop tokens will be detected from the vocabulary.

        Returns
        -------
        stop_token_ids : List[int]
            The ids of the stop tokens.
        """
        return self.handle.stop_token_ids
