"""This module provides classes representing grammars."""

import json
from typing import Optional, Tuple, Type, Union

from pydantic import BaseModel

from .base import XGRObject, _core


class Grammar(XGRObject):
    """This class represents a grammar object in XGrammar, and can be used later in the
    grammar-guided generation.

    The Grammar object supports context-free grammar (CFG). EBNF (extended Backus-Naur Form) is
    used as the format of the grammar. There are many specifications for EBNF in the literature,
    and we follow the specification of GBNF (GGML BNF) in
    https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md.

    When printed, the grammar will be converted to GBNF format.
    """

    def __str__(self) -> str:
        """Print the BNF grammar to a string, in EBNF format.

        Returns
        -------
        grammar_string : str
            The BNF grammar string.
        """
        return self._handle.to_string()

    @staticmethod
    def from_ebnf(ebnf_string: str, *, root_rule_name: str = "root") -> "Grammar":
        """Construct a grammar from EBNF string. The EBNF string should follow the format
        in https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md.

        Parameters
        ----------
        ebnf_string : str
            The grammar string in EBNF format.

        root_rule_name : str, default: "root"
            The name of the root rule in the grammar.
        """
        return Grammar._create_from_handle(_core.Grammar.from_ebnf(ebnf_string, root_rule_name))

    @staticmethod
    def from_json_schema(
        schema: Union[str, Type[BaseModel]],
        *,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = True,
    ) -> "Grammar":
        """Construct a grammar from JSON schema. Pydantic model or JSON schema string can be
        used to specify the schema.

        The format of the JSON schema can be specified with the `indent` and `separators`
        parameters. The meaning and the default values of the parameters follows the convention in
        json.dumps().

        It internally converts the JSON schema to a EBNF grammar.

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
        grammar : Grammar
            The constructed grammar.
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = json.dumps(schema.model_json_schema())

        return Grammar._create_from_handle(
            _core.Grammar.from_json_schema(schema, indent, separators, strict_mode),
        )

    @staticmethod
    def builtin_json_grammar() -> "Grammar":
        """Get the grammar of standard JSON. This is compatible with the official JSON grammar
        specification in https://www.json.org/json-en.html.

        Returns
        -------
        grammar : Grammar
            The JSON grammar.
        """
        return Grammar._create_from_handle(_core.Grammar.builtin_json_grammar())
