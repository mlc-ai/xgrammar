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
"""Classes handling the grammar guided generation of MLC LLM serving"""

import json
import logging
from typing import List, Optional, Tuple, Type, Union

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from . import xgrammar_bindings as _core


def _init_object_with_handle(type, handle):
    """Initialize an object with a handle."""
    obj = type.__new__(type)
    obj._handle = handle
    return obj


class BNFGrammar:
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
        self._handle = _core.BNFGrammar(ebnf_string, main_rule)

    def to_string(self) -> str:
        """Print the BNF grammar to a string, in standard BNF format.

        Returns
        -------
        grammar_string : str
            The BNF grammar string.

        """
        return self._handle.to_string()

    def __str__(self) -> str:
        return self.to_string()

    def serialize(self, prettify: bool = False) -> str:
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
        return self._handle.serialize(prettify)

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
        return _init_object_with_handle(BNFGrammar, _core.BNFGrammar.deserialize(json_string))

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
        return _init_object_with_handle(
            BNFGrammar, _core.BNFGrammar._init_no_normalization(ebnf_string, main_rule),
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
        return _init_object_with_handle(BNFGrammar, _core.BuiltinGrammar.json())

    @staticmethod
    def json_schema(
        schema: Union[str, Type[BaseModel]],
        *,
        indent: Optional[int] = 2,
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

        return _init_object_with_handle(
            BNFGrammar,
            _core.BuiltinGrammar.json_schema(schema, indent, separators, strict_mode),
        )

    @staticmethod
    def _json_schema_to_ebnf(
        schema: str,
        *,
        indent: Optional[int] = 2,
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
        return _core.BuiltinGrammar._json_schema_to_ebnf(schema, indent, separators, strict_mode)

    # @staticmethod
    # def _with_suffix(base_grammar: BNFGrammar, suffix_string: str) -> BNFGrammar:
    #     """Add a suffix to the main rule of the grammar.

    #     Parameters
    #     ----------
    #     base_grammar : BNFGrammar
    #         The base grammar.

    #     suffix_string : str
    #         The suffix string.

    #     Returns
    #     -------
    #     grammar : BNFGrammar
    #         The grammar with suffix.
    #     """
    #     return _init_object_with_handle(
    #         BNFGrammar,
    #         _core.BuiltinGrammar._with_suffix(base_grammar._handle, suffix_string),
    #     )


class XGTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        try:
            backend_str = tokenizer.backend_tokenizer.to_str()
        except AttributeError:
            logging.warning(
                "Cannot detect tokenizer type from tokenizer %s. Only huggingface or "
                "sentencepiece tokenizer is supported yet. The tokenizer type is set to the "
                "default byte_fallback tokenizer.",
                type(tokenizer),
            )
            backend_str = None

        try:
            # Use the get_vocab method to get the vocabulary, since the backend_str may not
            # contain the full vocab. Some special tokens may be omitted.
            vocab = tokenizer.get_vocab()
        except AttributeError as e:
            raise ValueError(
                f"Cannot get the vocabulary of the tokenizer {type(tokenizer)}. The tokenizer "
                "should have a get_vocab method.",
            ) from e

        self._handle = _core.XGTokenizer(backend_str, vocab)

    def __str__(self) -> str:
        return self._handle.to_string()

    @property
    def decoder_type(self) -> str:
        """Get the decoder type of the tokenizer.

        Returns
        -------
        decoder_type : str
            The decoder type.

        """
        return self._handle.decoder_type

    @property
    def prepend_space_in_tokenization(self) -> bool:
        """Whether the tokenizer will prepend a space in tokenization.

        Returns
        -------
        prepend_space_in_tokenization : bool
            Whether the tokenizer will prepend a space in tokenization.

        """
        return self._handle.prepend_space_in_tokenization

    @property
    def decoded_vocab(self) -> List[bytes]:
        """Get the decoded vocabulary of the tokenizer. This property is lazy evaluated.

        Returns
        -------
        decoded_vocab : str
            The decoded vocabulary.

        """
        return self._handle.decoded_vocab


class GrammarMatcherInitContext:
    def __init__(self, grammar: BNFGrammar, decoded_vocab: Optional[List[str]]):
        self._handle = _core.GrammarMatcherInitContext(grammar._handle, decoded_vocab)


# class GrammarInitContextCache:
#     def __init__(self, vocab: List[bytes]):
#         self._handle = _core.GrammarInitContextCache(cache._handle)


class GrammarStateMatcher:
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

    def __init__(
        self,
        grammar: BNFGrammar,
        tokenizer_or_vocab: Union[None, PreTrainedTokenizerBase, List[bytes], List[str]] = None,
        *,
        stop_token_ids: Union[None, int, List[int]] = None,
        terminate_without_stop_token: bool = False,
        max_rollback_steps: int = 0,
    ):
        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]

        if tokenizer_or_vocab is None or isinstance(tokenizer_or_vocab, list):
            vocab = tokenizer_or_vocab
        elif isinstance(tokenizer_or_vocab, PreTrainedTokenizerBase):
            xg_tokenizer = XGTokenizer(tokenizer_or_vocab)
            vocab = xg_tokenizer.decoded_vocab
        else:
            raise ValueError(
                "The tokenizer_or_vocab should be None, a PreTrainedTokenizerBase, or a list of "
                "strings.",
            )

        self._handle = _core.GrammarStateMatcher(
            grammar._handle,
            vocab,
            stop_token_ids,
            terminate_without_stop_token,
            max_rollback_steps,
        )

    def accept_token(self, token_id: int, verbose: bool = False) -> bool:
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
        return self._handle.accept_token(token_id, verbose)

    def _accept_string(self, input_str: str, verbose: bool = False) -> bool:
        """Accept one unicode codepoint to the current state. For test purposes.

        Parameters
        ----------
        codepoint : int
            The unicode codepoint of the character to be accepted.

        """
        return self._handle._accept_string(input_str, verbose)

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
        return self._handle.find_next_token_bitmask()

    @staticmethod
    def get_rejected_tokens_from_bitmask(bitmask: torch.Tensor, vocab_size: bool) -> List[int]:
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
        return _core.GrammarStateMatcher.get_rejected_tokens_from_bitmask(bitmask, vocab_size)

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
        return self._handle.find_jump_forward_string()

    def rollback(self, num_tokens: int) -> None:
        """Rollback the matcher to a previous state.

        Parameters
        ----------
        num_tokens : int
            The number of tokens to rollback. It cannot exceed the current number of steps, nor can
            it exceed the specified maximum number of rollback steps.

        """
        self._handle.rollback(num_tokens)

    @property
    def max_rollback_steps(self) -> int:
        """Get the maximum number of rollback steps allowed.

        Returns
        -------
        max_rollback_steps : int
            The maximum number of rollback steps.

        """
        return self._handle.max_rollback_steps

    def is_terminated(self) -> bool:
        """Check if the matcher has accepted the stop token and terminated. See also
        GrammarStateMatcher.accept_token.

        Returns
        -------
        terminated : bool
            Whether the matcher has terminated.

        """
        return self._handle.is_terminated()

    def reset(self) -> None:
        """Reset the matcher to the initial state."""
        return self._handle.reset()

    @property
    def vocab_size(self) -> int:
        return self._handle.vocab_size
