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
from typing import Optional, Tuple, Type, Union, overload

from pydantic import BaseModel

from .base import XGRObject, _core
from .grammar import Grammar
from .tokenizer_info import TokenizerInfo


class CompiledGrammar(XGRObject):
    @property
    def grammar(self) -> Grammar:
        """The BNF grammar."""
        return Grammar._create_from_handle(self._handle.grammar)

    @property
    def tokenizer_info(self) -> TokenizerInfo:
        """The tokenizer info."""
        return TokenizerInfo._create_from_handle(self._handle.tokenizer_info)


class GrammarCompiler(XGRObject):
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

    def __init__(
        self,
        tokenizer_info: TokenizerInfo,
        *,
        max_threads: int = 8,
        cache_enabled: bool = True,
    ):
        if not isinstance(tokenizer_info, TokenizerInfo):
            raise ValueError(
                "Please convert the tokenizer to TokenizerInfo before passing it "
                "to GrammarCompiler."
            )

        self._init_handle(_core.GrammarCompiler(tokenizer_info._handle, max_threads, cache_enabled))

    def compile_json_schema(
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

        return CompiledGrammar._create_from_handle(
            self._handle.compile_json_schema(schema, indent, separators, strict_mode)
        )

    def compile_builtin_json_grammar(self) -> CompiledGrammar:
        """Get CompiledGrammar from the standard JSON.

        Returns
        -------
        compiled_grammar : CompiledGrammar
            The initialization context for the grammar matcher.
        """
        return CompiledGrammar._create_from_handle(self._handle.compile_builtin_json_grammar())

    @overload
    def compile_grammar(self, grammar: str, *, root_rule_name: str = "root") -> CompiledGrammar: ...

    @overload
    def compile_grammar(self, grammar: Grammar) -> CompiledGrammar: ...

    def compile_grammar(
        self, grammar: Union[str, Grammar], *, root_rule_name: str = "root"
    ) -> CompiledGrammar:
        """Compile a BNF grammar."""
        if isinstance(grammar, str):
            grammar = Grammar.from_ebnf(grammar, root_rule_name=root_rule_name)
        return CompiledGrammar._create_from_handle(self._handle.compile_grammar(grammar._handle))

    def clear_cache(self) -> None:
        """Clear all cached compiled grammars."""
        self._handle.clear_cache()
