"""Compiling grammar for efficient token mask generation."""

import json
from typing import Optional, Tuple, Type, Union, overload

from pydantic import BaseModel

from .base import XGRObject, _core
from .grammar import Grammar
from .tokenizer_info import TokenizerInfo


class CompiledGrammar(XGRObject):
    """This is the primary object to store compiled grammar.

    A CompiledGrammar can be used to construct GrammarMatcher
    to generate token masks efficiently.

    Note
    ----
    Do not construct this class directly, instead
    use :class:`GrammarCompiler` to construct the object.
    """

    @property
    def grammar(self) -> Grammar:
        """The original grammar."""
        return Grammar._create_from_handle(self._handle.grammar)

    @property
    def tokenizer_info(self) -> TokenizerInfo:
        """The tokenizer info associated with the compiled grammar."""
        return TokenizerInfo._create_from_handle(self._handle.tokenizer_info)


class GrammarCompiler(XGRObject):
    """The compiler for grammars. It is associated with a certain tokenizer info, and compiles
    grammars into CompiledGrammar with the tokenizer info. It allows parallel compilation with
    multiple threads, and has a cache to store the compilation result, avoiding compiling the
    same grammar multiple times.

    Parameters
    ----------
    tokenizer_info : TokenizerInfo
        The tokenizer info.

    max_threads : int, default: 8
        The maximum number of threads used to compile the grammar.

    cache_enabled : bool, default: True
        Whether to enable the cache.
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
            The compiled grammar.
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
            The compiled grammar.
        """
        return CompiledGrammar._create_from_handle(self._handle.compile_builtin_json_grammar())

    @overload
    def compile_grammar(self, ebnf_string: str, *, root_rule_name: str = "root") -> CompiledGrammar:
        """Compile a grammar from EBNF string. The EBNF string should follow the format
        in https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md.

        Parameters
        ----------
        ebnf_string : str
            The grammar string in EBNF format.

        root_rule_name : str, default: "root"
            The name of the root rule in the grammar.

        Returns
        -------
        compiled_grammar : CompiledGrammar
            The compiled grammar.
        """
        ...

    @overload
    def compile_grammar(self, grammar: Grammar) -> CompiledGrammar:
        """Compile a grammar object.

        Returns
        -------
        compiled_grammar : CompiledGrammar
            The compiled grammar.
        """
        ...

    def compile_grammar(
        self, grammar: Union[str, Grammar], *, root_rule_name: str = "root"
    ) -> CompiledGrammar:
        if isinstance(grammar, str):
            grammar = Grammar.from_ebnf(grammar, root_rule_name=root_rule_name)
        return CompiledGrammar._create_from_handle(self._handle.compile_grammar(grammar._handle))

    def clear_cache(self) -> None:
        """Clear all cached compiled grammars."""
        self._handle.clear_cache()
