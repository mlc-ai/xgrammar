"""Serializing and deserializing grammars and tokenizers."""

from typing import List, Optional, Union

from .base import _core
from .compiler import CompiledGrammar
from .grammar import Grammar
from .tokenizer_info import TokenizerInfo


class JSONSerializer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def serialize_grammar(grammar: Grammar, prettify: bool = False) -> str:
        return _core.JSONSerializer.serialize_grammar(grammar._handle, prettify)

    @staticmethod
    def deserialize_grammar(serialized_grammar: str) -> Grammar:
        return Grammar._create_from_handle(
            _core.JSONSerializer.deserialize_grammar(serialized_grammar)
        )

    @staticmethod
    def serialize_tokenizer_info(tokenizer_info: TokenizerInfo, prettify: bool = False) -> str:
        return _core.JSONSerializer.serialize_tokenizer_info(tokenizer_info._handle, prettify)

    @staticmethod
    def deserialize_tokenizer_info(
        serialized_tokenizer_info: str, encoded_vocab: Optional[List[Union[bytes, str]]] = None
    ) -> TokenizerInfo:
        return TokenizerInfo._create_from_handle(
            _core.JSONSerializer.deserialize_tokenizer_info(
                serialized_tokenizer_info, encoded_vocab or []
            )
        )

    @staticmethod
    def serialize_compiled_grammar(
        compiled_grammar: CompiledGrammar, prettify: bool = False
    ) -> str:
        return _core.JSONSerializer.serialize_compiled_grammar(compiled_grammar._handle, prettify)

    @staticmethod
    def deserialize_compiled_grammar(
        serialized_compiled_grammar: str, tokenizer: TokenizerInfo
    ) -> CompiledGrammar:
        return CompiledGrammar._create_from_handle(
            _core.JSONSerializer.deserialize_compiled_grammar(
                serialized_compiled_grammar, encoded_vocab_or_tokenizer
            )
        )
