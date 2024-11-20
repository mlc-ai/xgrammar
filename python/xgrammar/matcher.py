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

import math
from typing import List, Optional, Tuple, Union

import torch

from .base import XGRObject, _core
from .compiler import CompiledGrammar


def get_bitmask_shape(batch_size: int, vocab_size: int) -> Tuple[int, int]:
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
        The shape of the bitmask.
    """
    return (batch_size, math.ceil(vocab_size / 32))


def get_bitmask_dtype() -> torch.dtype:
    """Get the dtype of the bitmask."""
    return torch.int32


def apply_token_bitmask_inplace(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    *,
    indices: Optional[List[int]] = None,
) -> None:
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

    indices : Optional[List[int]], default: None
        The indices of the tokens to apply the bitmask to. If None, all tokens will be applied.
    """
    if logits.device.type != "cuda":
        raise ValueError("logits must be on CUDA")

    _core.matcher.apply_token_bitmask_inplace(logits, bitmask, indices)


class GrammarMatcher(XGRObject):
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

    def __init__(
        self,
        compiled_grammar: CompiledGrammar,
        *,
        override_stop_tokens: Optional[Union[int, List[int]]] = None,
        terminate_without_stop_token: bool = False,
        max_rollback_tokens: int = 0,
    ) -> None:
        """Initialize the grammar matcher with a grammar matcher initialization context. This
        initialization is very fast.

        Parameters
        ----------
        compiled_grammar : CompiledGrammar
            The initialization context for the grammar matcher.
        """
        if not isinstance(compiled_grammar, CompiledGrammar):
            raise ValueError("The grammar should be compiled before passing it to GrammarMatcher.")

        if isinstance(override_stop_tokens, int):
            override_stop_tokens = [override_stop_tokens]

        self._init_handle(
            _core.GrammarMatcher(
                compiled_grammar._handle,
                override_stop_tokens,
                terminate_without_stop_token,
                max_rollback_tokens,
            )
        )

    def accept_token(self, token_id: int, *, debug_print: bool = False) -> bool:
        """Accept one token and update the state of the matcher.

        Parameters
        ----------
        token_id : int
            The id of the token to accept.

        debug_print : bool, default: False
            Whether to print information about the internal state of the matcher. Helpful
            for debugging.

        Returns
        -------
        accepted : bool
            Whether the token is accepted.
        """
        return self._handle.accept_token(token_id, debug_print)

    def fill_next_token_bitmask(self, bitmask: torch.Tensor, index: int = 0) -> None:
        """Fill the bitmask for the next token prediction.

        Parameters
        ----------
        bitmask : torch.Tensor
            The bitmask for the next token prediction. Should be a 1D or 2D tensor generated by
            allocate_token_bitmask. Should be on GPU.

        index : int, default: 0
            The batch id of the bitmask. For batch inference, bitmask[index] will be filled
            with the next token bitmask. Otherwise is ignored.
        """
        self._handle.fill_next_token_bitmask(bitmask, index)

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
        return self._handle.find_jump_forward_string()

    def rollback(self, num_tokens: int = 1) -> None:
        """Rollback the matcher to a previous state by several tokens.

        Parameters
        ----------
        num_tokens : int, default: 1
            The number of tokens to rollback. It cannot exceed the current number of steps, nor can
            it exceed the specified maximum number of rollback tokens.
        """
        self._handle.rollback(num_tokens)

    def is_terminated(self) -> bool:
        """Check if the matcher has terminated. If terminate_without_stop_token is False, the
        matcher will terminate if it has accepted the stop token. Otherwise, the matcher will
        terminate after matching the whole grammar.

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
    def max_rollback_tokens(self) -> int:
        """Get the maximum number of rollback tokens allowed.

        Returns
        -------
        max_rollback_tokens : int
            The maximum number of rollback tokens.
        """
        return self._handle.max_rollback_tokens

    @property
    def stop_token_ids(self) -> List[int]:
        """The ids of the stop tokens used in the matcher. If specified, the provided stop tokens
        will be used. Otherwise, the stop tokens will be detected from the vocabulary.

        Returns
        -------
        stop_token_ids : List[int]
            The ids of the stop tokens.
        """
        return self._handle.stop_token_ids

    def _debug_accept_string(
        self, input_str: Union[str, bytes], *, debug_print: bool = False
    ) -> bool:
        """Accept a string and update the state of the matcher. The whole string is considered
        as one step in rollback. It is only used to complement the functionality of accept_token.

        Parameters
        ----------
        input_str : Union[str, bytes]
            The string to be accepted.

        debug_print : bool, default: False
            Whether to print information about the internal state of the matcher. Helpful for
            debugging.

        Returns
        -------
        accepted : bool
            Whether the string is accepted.
        """
        return self._handle._debug_accept_string(input_str, debug_print)
