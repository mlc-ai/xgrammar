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
"""Classes handling the grammar guided generation."""

import json
from enum import Enum
from typing import List, Optional, Tuple, Type, Union, overload

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from . import xgrammar_bindings as _core


class XGObject:
    @classmethod
    def from_handle(cls, handle) -> "XGObject":
        """Initialize an object with a handle."""
        obj = cls.__new__(cls)
        obj._handle = handle
        return obj

    def init_with_handle(self, handle):
        self._handle = handle

    @property
    def handle(self):
        return self._handle


class BNFGrammar(XGObject):
    """This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar and
    provides utilities to parse and print the AST. User should provide a BNF/EBNF (Extended
    Backus-Naur Form) grammar, and use from_ebnf_string to parse and simplify the grammar into an
    AST of BNF grammar.
    """

    def __init__(self, ebnf_string: str, main_rule: str = "main") -> None:
        r"""Construct a BNF grammar with a EBNF-formatted string. The grammar will be normalized
        (simplified) by default.

        EBNF grammar: see https://www.w3.org/TR/xml/#sec-notation. Note:
        1. Use # as the comment mark
        2. Use C-style unicode escape sequence \u01AB, \U000001AB, \xAB
        3. A-B (match A and not match B) is not supported yet
        4. Lookahead assertion can be added at the end of a rule to speed up matching. E.g.
        ```
        main ::= "ab" a [a-z]
        a ::= "cd" (=[a-z])
        ```
        The assertion (=[a-z]) means a must be followed by [a-z].

        Parameters
        ----------
        ebnf_string : str
            The grammar string.

        main_rule : str
            The name of the main rule. Default: "main".

        Returns
        -------
        grammar : BNFGrammar
            The parsed BNF grammar.

        """
        self.init_with_handle(_core.BNFGrammar(ebnf_string, main_rule))

    def to_string(self) -> str:
        """Print the BNF grammar to a string, in standard BNF format.

        Returns
        -------
        grammar_string : str
            The BNF grammar string.

        """
        return self.handle.to_string()

    def __str__(self) -> str:
        return self.to_string()

    def serialize(self, *, prettify: bool = False) -> str:
        """Serialize the AST. Dump the raw representation of the AST to a JSON file.

        Parameters
        ----------
        prettify : bool
            Whether to format the JSON string. If False, all whitespaces will be removed.

        Returns
        -------
        json_string : str
            The JSON string.

        """
        return self.handle.serialize(prettify)

    @staticmethod
    def deserialize(json_string: str) -> "BNFGrammar":
        """Load a BNF grammar from the raw representation of the AST in JSON format.

        Parameters
        ----------
        json_string : str
            The JSON string.

        Returns
        -------
        grammar : BNFGrammar
            The loaded BNF grammar.

        """
        return BNFGrammar.from_handle(_core.BNFGrammar.deserialize(json_string))

    @staticmethod
    def _init_no_normalization(
        ebnf_string: str,
        main_rule: str = "main",
    ) -> "BNFGrammar":
        r"""Construct a BNF grammar with a EBNF-formatted string, but not normalize it.
        For test purposes.

        Parameters
        ----------
        ebnf_string : str
            The grammar string.

        main_rule : str
            The name of the main rule. Default: "main".

        Returns
        -------
        grammar : BNFGrammar
            The parsed BNF grammar.

        """
        return BNFGrammar.from_handle(
            _core.BNFGrammar._init_no_normalization(ebnf_string, main_rule),
        )


class BuiltinGrammar:
    @staticmethod
    def json() -> BNFGrammar:
        """Get the grammar of standard JSON.

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
        """Construct a BNF grammar from the json schema string. The schema string should be in the
        format of the schema of a JSON file. We will parse the schema and generate a BNF grammar.

        Parameters
        ----------
        schema : str
            The schema string.

        indent : Optional[int]
            The number of spaces for indentation. If None, the output will be in one line.
            Default: None.

        separators : Optional[Tuple[str, str]]
            Two separators used in the schema: comma and colon. Examples: (",", ":"), (", ", ": ").
            If None, the default separators will be used: (",", ": ") when the indent is not None,
            and (", ", ": ") otherwise. This follows the convention in json.dumps(). Default: None.

        strict_mode : bool
            Whether to use strict mode. In strict mode, the generated grammar will not allow
            properties and items that is not specified in the schema. This is equivalent to
            setting unevaluatedProperties and unevaluatedItems to false.

            This helps LLM to generate accurate output in the grammar-guided generation with JSON
            schema. Default: True.

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
        json_schema : str
            The JSON schema string.

        indent : Optional[int]
            The number of spaces for indentation. If None, the output will be in one line.
            Default: 2.

        separators : Optional[Tuple[str, str]]
            Two separators used in the schema: comma and colon. Examples: (",", ":"), (", ", ": ").
            If None, the default separators will be used: (",", ": ") when the indent is not None,
            and (", ", ": ") otherwise. This follows the convention in json.dumps(). Default: None.

        strict_mode : bool
            Whether to use strict mode. In strict mode, the generated grammar will not allow
            properties and items that is not specified in the schema. This is equivalent to
            setting unevaluatedProperties and unevaluatedItems to false.

            This helps LLM to generate accurate output in the grammar-guided generation with JSON
            schema. Default: True.

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


class VocabType(Enum):
    RAW = "RAW"
    BYTE_FALLBACK = "BYTE_FALLBACK"
    BYTE_LEVEL = "BYTE_LEVEL"


class TokenizerInfo(XGObject):
    def __init__(
        self,
        vocab: Union[List[bytes], List[str]],
        vocab_type: VocabType = VocabType.RAW,
        prepend_space_in_tokenization: bool = False,
    ) -> None:
        self.init_with_handle(
            _core.TokenizerInfo(vocab, vocab_type.value, prepend_space_in_tokenization)
        )

    @property
    def vocab_size(self) -> int:
        return self.handle.vocab_size

    @property
    def vocab_type(self) -> VocabType:
        return VocabType(self.handle.vocab_type)

    @property
    def prepend_space_in_tokenization(self) -> bool:
        return self.handle.prepend_space_in_tokenization

    @property
    def raw_vocab(self) -> List[bytes]:
        return self.handle.raw_vocab

    @staticmethod
    def from_huggingface(tokenizer: PreTrainedTokenizerBase) -> "TokenizerInfo":
        try:
            vocab = tokenizer.get_vocab()
            vocab = [token for token, _ in sorted(vocab.items(), key=lambda x: x[1])]
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
                _core.TokenizerInfo.from_huggingface(vocab, backend_str)
            )
        elif (
            "vocab_file" in tokenizer.vocab_files_names
            and "tiktoken" in tokenizer.vocab_files_names["vocab_file"]
        ):
            # tiktoken tokenizer
            # e.g. Phi-3-small-8k-instruct, Qwen-7B-Chat, stablelm-2-12b-chat (previously)
            return TokenizerInfo(vocab, VocabType.RAW, False)
        else:
            # TODO(yixin): sentencepiece tokenizer
            raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")

    def dump_metadata(self) -> str:
        return self.handle.dump_metadata()

    @staticmethod
    def from_vocab_and_metadata(vocab: List[Union[bytes, str]], metadata: str) -> "TokenizerInfo":
        return TokenizerInfo.from_handle(
            _core.TokenizerInfo.from_vocab_and_metadata(vocab, metadata),
        )


class GrammarMatcherInitContext(XGObject):
    def __init__(
        self,
        grammar: BNFGrammar,
        tokenizer_or_vocab: Union[
            None, PreTrainedTokenizerBase, TokenizerInfo, List[Union[bytes, str]]
        ] = None,
    ) -> None:
        # convert tokenizer_or_vocab to TokenizerInfo
        if isinstance(tokenizer_or_vocab, PreTrainedTokenizerBase):
            tokenizer_or_vocab = TokenizerInfo.from_huggingface(tokenizer_or_vocab)
        elif isinstance(tokenizer_or_vocab, list):
            tokenizer_or_vocab = TokenizerInfo(tokenizer_or_vocab)
        elif tokenizer_or_vocab is None:
            tokenizer_or_vocab = TokenizerInfo([])
        if not isinstance(tokenizer_or_vocab, TokenizerInfo):
            raise ValueError(f"Unsupported tokenizer_or_vocab type: {type(tokenizer_or_vocab)}")

        self.init_with_handle(
            _core.GrammarMatcherInitContext(grammar.handle, tokenizer_or_vocab.handle)
        )


class GrammarMatcherInitContextCache(XGObject):
    def __init__(
        self,
        tokenizer_or_vocab: Union[PreTrainedTokenizerBase, TokenizerInfo, List[Union[bytes, str]]],
    ):
        # convert tokenizer_or_vocab to TokenizerInfo
        if isinstance(tokenizer_or_vocab, PreTrainedTokenizerBase):
            tokenizer_or_vocab = TokenizerInfo.from_huggingface(tokenizer_or_vocab)
        elif isinstance(tokenizer_or_vocab, list):
            tokenizer_or_vocab = TokenizerInfo(tokenizer_or_vocab)
        if not isinstance(tokenizer_or_vocab, TokenizerInfo):
            raise ValueError(f"Unsupported tokenizer_or_vocab type: {type(tokenizer_or_vocab)}")

        self.init_with_handle(_core.GrammarMatcherInitContextCache(tokenizer_or_vocab.handle))

    def get_init_context_for_json(self) -> GrammarMatcherInitContext:
        return GrammarMatcherInitContext.from_handle(self.handle.get_init_context_for_json())

    def get_init_context_for_json_schema(
        self,
        schema: Union[str, Type[BaseModel]],
        *,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = True,
    ) -> GrammarMatcherInitContext:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = json.dumps(schema.model_json_schema())

        return GrammarMatcherInitContext.from_handle(
            self.handle.get_init_context_for_json_schema(schema, indent, separators, strict_mode)
        )


class GrammarMatcher(XGObject):
    """A stateful matcher to match tokens to the specified BNF grammar. This class is the core logic
    of the grammar-guided generation.

    This class implements the non-deterministic pushdown automaton (NPDA) matching algorithm to
    match characters to a BNF grammar. It keep track of the current state of the matching process by
    maintaining several stacks internally as possible paths in the NPDA. It also supports
    backtracking.

    It is particularly capable of finding the set of tokens that are acceptable for the next step
    and storing them in a bitmask. This aids in grammar-guided generation.

    Parameters
    ----------
    grammar : BNFGrammar
        The BNF grammar to match.

    tokenizer : Union[None, Tokenizer, List[str]]
        The tokenizer to use, or the list of tokens.

        (For debug purpose) If None, the matcher will use an empty token set, and can only accept
        and match characters. Default: None.

    max_rollback_steps : int
        The maximum number of steps to rollback when backtracking. Default: 0.

    """

    @overload
    def __init__(
        self,
        grammar: BNFGrammar,
        tokenizer_or_vocab: Union[
            None, PreTrainedTokenizerBase, TokenizerInfo, List[Union[bytes, str]]
        ] = None,
        *,
        stop_token_ids: Union[None, int, List[int]] = None,
        terminate_without_stop_token: bool = False,
        mask_vocab_size: Optional[int] = None,
        max_rollback_steps: int = 0,
    ) -> None: ...

    @overload
    def __init__(
        self,
        grammar_matcher_init_context: GrammarMatcherInitContext,
        *,
        stop_token_ids: Union[None, int, List[int]] = None,
        terminate_without_stop_token: bool = False,
        mask_vocab_size: Optional[int] = None,
        max_rollback_steps: int = 0,
    ) -> None: ...

    def __init__(
        self,
        grammar_or_context: Union[BNFGrammar, GrammarMatcherInitContext],
        tokenizer_or_vocab: Union[
            None, PreTrainedTokenizerBase, TokenizerInfo, List[Union[bytes, str]]
        ] = None,
        *,
        stop_token_ids: Union[None, int, List[int]] = None,
        terminate_without_stop_token: bool = False,
        mask_vocab_size: Optional[int] = None,
        max_rollback_steps: int = 0,
    ) -> None:
        if isinstance(grammar_or_context, BNFGrammar):
            grammar_matcher_init_context = GrammarMatcherInitContext(
                grammar_or_context, tokenizer_or_vocab
            )
        else:
            grammar_matcher_init_context = grammar_or_context

        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]

        self.init_with_handle(
            _core.GrammarMatcher(
                grammar_matcher_init_context.handle,
                stop_token_ids,
                terminate_without_stop_token,
                mask_vocab_size,
                max_rollback_steps,
            )
        )

    def accept_token(self, token_id: int, *, verbose: bool = False) -> bool:
        """Accept one token and update the state of the matcher.

        Parameters
        ----------
        token_id : int
            The id of the token to accept.

        Returns
        -------
        accepted : bool
            Whether the token is accepted.

        Note
        ----
        Termination state.

        When the end of the main rule is reached, the matcher can only accept the stop token.
        The matcher is terminated after accepting the stop token, i.e. no accept_token or
        find_next_rejected_tokens operations can be performed. The termination state can be canceled
        using Rollback().

        """
        return self.handle.accept_token(token_id, verbose)

    def accept_string(self, input_str: Union[str, bytes], *, verbose: bool = False) -> bool:
        """Accept one unicode codepoint to the current state. For test purposes.

        Parameters
        ----------
        codepoint : int
            The unicode codepoint of the character to be accepted.

        """
        return self.handle.accept_string(input_str, verbose)

    def find_next_token_bitmask(self) -> torch.Tensor:
        """Find the ids of the rejected tokens for the next step.

        Parameters
        ----------
        verbose : bool
            Whether to print information about timing and result counts to stderr.
            For debug purposes. Default: False.

        Returns
        -------
        rejected_token_bitmask : torch.Tensor
            A tensor of rejected token ids.

        """
        return self.handle.find_next_token_bitmask()

    @staticmethod
    def get_rejected_tokens_from_bitmask(bitmask: torch.Tensor, vocab_size: int) -> List[int]:
        """Get the ids of the rejected tokens from the bitmask.

        Parameters
        ----------
        bitmask : torch.Tensor
            The rejected token bitmask.

        Returns
        -------
        rejected_token_ids : List[int]
            A list of rejected token ids.

        """
        return _core.GrammarMatcher.get_rejected_tokens_from_bitmask(bitmask, vocab_size)

    # @staticmethod
    # def apply_token_bitmask(tensor: torch.Tensor, bitmask: torch.Tensor) -> torch.Tensor:
    #     """Apply the bitmask to the tensor.

    #     Parameters
    #     ----------
    #     tensor : torch.Tensor
    #         The tensor to apply the bitmask to.

    #     bitmask : torch.Tensor
    #         The bitmask to apply.

    #     Returns
    #     -------
    #     masked_tensor : torch.Tensor
    #         The masked tensor.
    #     """
    #     return None

    def find_jump_forward_string(self) -> str:
        """Find the jump-forward string for jump-forward decoding. This is the longest string that
        will be valid according to the current syntax.

        Notes
        -----
        This method does not change the grammar state.

        Returns
        -------
        jump_forward_string : str
            The jump-forward string.

        """
        return self.handle.find_jump_forward_string()

    def rollback(self, num_tokens: int = 1) -> None:
        """Rollback the matcher to a previous state.

        Parameters
        ----------
        num_tokens : int
            The number of tokens to rollback. It cannot exceed the current number of steps, nor can
            it exceed the specified maximum number of rollback steps.

        """
        self.handle.rollback(num_tokens)

    @property
    def max_rollback_steps(self) -> int:
        """Get the maximum number of rollback steps allowed.

        Returns
        -------
        max_rollback_steps : int
            The maximum number of rollback steps.

        """
        return self.handle.max_rollback_steps

    def is_terminated(self) -> bool:
        """Check if the matcher has accepted the stop token and terminated. See also
        GrammarMatcher.accept_token.

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
    def vocab_size(self) -> int:
        return self.handle.vocab_size
