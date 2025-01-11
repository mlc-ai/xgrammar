"""This module provides classes representing grammars."""

import json
from typing import List, Optional, Tuple, Type, Union

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

        Raises
        ------
        RuntimeError
            When converting the regex pattern fails, with details about the parsing error.
        """
        return Grammar._create_from_handle(_core.Grammar.from_ebnf(ebnf_string, root_rule_name))

    @staticmethod
    def from_json_schema(
        schema: Union[str, Type[BaseModel]],
        *,
        any_whitespace: bool = True,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = True,
    ) -> "Grammar":
        """Construct a grammar from JSON schema. Pydantic model or JSON schema string can be
        used to specify the schema.

        It allows any whitespace by default. If user want to specify the format of the JSON,
        set `any_whitespace` to False and use the `indent` and `separators` parameters. The
        meaning and the default values of the parameters follows the convention in json.dumps().

        It internally converts the JSON schema to a EBNF grammar.

        Parameters
        ----------
        schema : Union[str, Type[BaseModel]]
            The schema string or Pydantic model.

        any_whitespace : bool, default: True
            Whether to use any whitespace. If True, the generated grammar will ignore the
            indent and separators parameters, and allow any whitespace.

        indent : Optional[int], default: None
            The number of spaces for indentation. If None, the output will be in one line.

            Note that specifying the indentation means forcing the LLM to generate JSON strings
            strictly formatted. However, some models may tend to generate JSON strings that
            are not strictly formatted. In this case, forcing the LLM to generate strictly
            formatted JSON strings may degrade the generation quality. See
            <https://github.com/sgl-project/sglang/issues/2216#issuecomment-2516192009> for more
            details.

        separators : Optional[Tuple[str, str]], default: None
            Two separators used in the schema: comma and colon. Examples: (",", ":"), (", ", ": ").
            If None, the default separators will be used: (",", ": ") when the indent is not None,
            and (", ", ": ") otherwise.

        strict_mode : bool, default: True
            Whether to use strict mode. In strict mode, the generated grammar will not allow
            properties and items that is not specified in the schema. This is equivalent to
            setting unevaluatedProperties and unevaluatedItems to false. It also disallows empty
            JSON objects and arrays.

            This helps LLM to generate accurate output in the grammar-guided generation with JSON
            schema.

        Returns
        -------
        grammar : Grammar
            The constructed grammar.

        Raises
        ------
        RuntimeError
            When converting the json schema fails, with details about the parsing error.
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            if hasattr(schema, "model_json_schema"):
                # pydantic 2.x
                schema = json.dumps(schema.model_json_schema())
            elif hasattr(schema, "schema_json"):
                # pydantic 1.x
                schema = json.dumps(schema.schema_json())
            else:
                raise ValueError(
                    "The schema should have a model_json_schema or json_schema method."
                )

        return Grammar._create_from_handle(
            _core.Grammar.from_json_schema(schema, any_whitespace, indent, separators, strict_mode),
        )

    @staticmethod
    def from_regex(regex_string: str) -> "Grammar":
        """Create a grammar from a regular expression string.

        Parameters
        ----------
        regex_string : str
            The regular expression pattern to create the grammar from.

        Returns
        -------
        grammar : Grammar
            The constructed grammar from the regex pattern.

        Raises
        ------
        RuntimeError
            When parsing the regex pattern fails, with details about the parsing error.
        """
        return Grammar._create_from_handle(_core.Grammar.from_regex(regex_string))

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

    @staticmethod
    def concat(*grammars: "Grammar") -> "Grammar":
        """Create a grammar that matches the concatenation of the grammars in the list. That is
        equivalent to using the `+` operator to concatenate the grammars in the list.

        Parameters
        ----------
        grammars : List[Grammar]
            The grammars to create the concatenation of.

        Returns
        -------
        grammar : Grammar
            The concatenation of the grammars.
        """
        grammar_handles = [grammar._handle for grammar in grammars]
        return Grammar._create_from_handle(_core.Grammar.concat(grammar_handles))
