"""This module provides classes representing grammars."""

import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field

from .base import XGRObject, _core


class StructuralTagItem(BaseModel):
    """A structural tag item. See :meth:`xgrammar.Grammar.from_structural_tag` for more details."""

    begin: str
    """The begin tag."""
    schema_: Union[str, Type[BaseModel], Dict[str, Any]] = Field(alias="schema")
    """The schema."""
    end: str
    """The end tag."""


def _convert_schema_to_str(schema: Union[str, Type[BaseModel], Dict[str, Any]]) -> str:
    """Convert a schema to a string representation.

    This function handles different schema input types and converts them to a JSON string:
    - Pydantic models are converted using their schema methods
    - String inputs are returned as-is (assumed to be valid JSON)
    - Dictionary inputs are converted to JSON strings

    Parameters
    ----------
    schema : Union[str, Type[BaseModel], Dict[str, Any]]
        The schema to convert, which can be a Pydantic model class,
        a JSON schema string, or a dictionary representing a JSON schema.

    Returns
    -------
    str
        The JSON schema as a string.

    Raises
    ------
    ValueError, TypeError
        If the schema type is not supported, or the dictionary is not serializable.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        if hasattr(schema, "model_json_schema"):
            return json.dumps(schema.model_json_schema())
        if hasattr(schema, "schema_json"):
            return json.dumps(schema.schema_json())
        else:
            raise ValueError("The schema should have a model_json_schema or json_schema method.")
    elif isinstance(schema, str):
        return schema
    elif isinstance(schema, dict):
        return json.dumps(schema)
    else:
        raise ValueError("The schema should be a string or a Pydantic model.")


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
        schema: Union[str, Type[BaseModel], Dict[str, Any]],
        *,
        any_whitespace: bool = True,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = True,
        print_converted_ebnf: bool = False,
    ) -> "Grammar":
        """Construct a grammar from JSON schema. Pydantic model or JSON schema string can be
        used to specify the schema.

        It allows any whitespace by default. If user want to specify the format of the JSON,
        set `any_whitespace` to False and use the `indent` and `separators` parameters. The
        meaning and the default values of the parameters follows the convention in json.dumps().

        It internally converts the JSON schema to a EBNF grammar.

        Parameters
        ----------
        schema : Union[str, Type[BaseModel], Dict[str, Any]]
            The schema string or Pydantic model or JSON schema dict.

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
            setting unevaluatedProperties and unevaluatedItems to false.

            This helps LLM to generate accurate output in the grammar-guided generation with JSON
            schema.

        print_converted_ebnf : bool, default: False
            If True, the converted EBNF string will be printed. For debugging purposes.

        Returns
        -------
        grammar : Grammar
            The constructed grammar.

        Raises
        ------
        RuntimeError
            When converting the json schema fails, with details about the parsing error.
        """
        schema_str = _convert_schema_to_str(schema)
        return Grammar._create_from_handle(
            _core.Grammar.from_json_schema(
                schema_str, any_whitespace, indent, separators, strict_mode, print_converted_ebnf
            )
        )

    @staticmethod
    def from_regex(regex_string: str, *, print_converted_ebnf: bool = False) -> "Grammar":
        """Create a grammar from a regular expression string.

        Parameters
        ----------
        regex_string : str
            The regular expression pattern to create the grammar from.

        print_converted_ebnf : bool, default: False
            This method will convert the regex pattern to EBNF first. If this is true, the converted
            EBNF string will be printed. For debugging purposes. Default: False.

        Returns
        -------
        grammar : Grammar
            The constructed grammar from the regex pattern.

        Raises
        ------
        RuntimeError
            When parsing the regex pattern fails, with details about the parsing error.
        """
        return Grammar._create_from_handle(
            _core.Grammar.from_regex(regex_string, print_converted_ebnf)
        )

    @staticmethod
    def from_structural_tag(tags: List[StructuralTagItem], triggers: List[str]) -> "Grammar":
        """Create a grammar from structural tags. The structural tag handles the dispatching
        of different grammars based on the tags and triggers: it initially allows any output,
        until a trigger is encountered, then dispatch to the corresponding tag; when the end tag
        is encountered, the grammar will allow any following output, until the next trigger is
        encountered.

        The tags parameter is used to specify the output pattern. It is especially useful for LLM
        function calling, where the pattern is:
        <function=func_name>{"arg1": ..., "arg2": ...}</function>.
        This pattern consists of three parts: a begin tag (<function=func_name>), a parameter list
        according to some schema ({"arg1": ..., "arg2": ...}), and an end tag (</function>). This
        pattern can be described in a StructuralTagItem with a begin tag, a schema, and an end tag.
        The structural tag is able to handle multiple such patterns by passing them into multiple
        tags.

        The triggers parameter is used to trigger the dispatching of different grammars. The trigger
        should be a prefix of a provided begin tag. When the trigger is encountered, the
        corresponding tag should be used to constrain the following output. There can be multiple
        tags matching the same trigger. Then if the trigger is encountered, the following output
        should match one of the tags. For example, in function calling, the triggers can be
        ["<function="]. Then if "<function=" is encountered, the following output must match one
        of the tags (e.g. <function=get_weather>{"city": "Beijing"}</function>).

        The corrrespondence of tags and triggers is automatically determined: all tags with the
        same trigger will be grouped together. User should make sure any trigger is not a prefix
        of another trigger: then the corrrespondence of tags and triggers will be ambiguous.

        To use this grammar in grammar-guided generation, the GrammarMatcher constructed from
        structural tag will generate a mask for each token. When the trigger is not encountered,
        the mask will likely be all-1 and not have to be used (fill_next_token_bitmask returns
        False, meaning no token is masked). When a trigger is encountered, the mask should be
        enforced (fill_next_token_bitmask will return True, meaning some token is masked) to the
        output logits.

        The benefit of this method is the token boundary between tags and triggers is automatically
        handled. The user does not need to worry about the token boundary.

        Parameters
        ----------
        tags : List[StructuralTagItem]
            The structural tags.

        triggers : List[str]
            The triggers.

        Returns
        -------
        grammar : Grammar
            The constructed grammar.

        Examples
        --------
        >>> class Schema1(BaseModel):
        ...     arg1: str
        ...     arg2: int
        >>> class Schema2(BaseModel):
        ...     arg3: float
        ...     arg4: List[str]
        >>> tags = [
        ...     StructuralTagItem(begin="<function=f>", schema=Schema1, end="</function>"),
        ...     StructuralTagItem(begin="<function=g>", schema=Schema2, end="</function>"),
        ... ]
        >>> triggers = ["<function="]
        >>> grammar = Grammar.from_structural_tag(tags, triggers)
        """
        tags_tuple = [(tag.begin, _convert_schema_to_str(tag.schema_), tag.end) for tag in tags]
        return Grammar._create_from_handle(_core.Grammar.from_structural_tag(tags_tuple, triggers))

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

    @staticmethod
    def union(*grammars: "Grammar") -> "Grammar":
        """Create a grammar that matches any of the grammars in the list. That is equivalent to
        using the `|` operator to concatenate the grammars in the list.

        Parameters
        ----------
        grammars : List[Grammar]
            The grammars to create the union of.

        Returns
        -------
        grammar : Grammar
            The union of the grammars.
        """
        grammar_handles = [grammar._handle for grammar in grammars]
        return Grammar._create_from_handle(_core.Grammar.union(grammar_handles))

    def serialize_json(self) -> str:
        """Serialize the grammar to a JSON string.

        Returns
        -------
        json_string : str
            The JSON string.
        """
        return self._handle.serialize_json()

    @staticmethod
    def deserialize_json(json_string: str) -> "Grammar":
        """Deserialize a grammar from a JSON string.

        Parameters
        ----------
        json_string : str
            The JSON string.

        Returns
        -------
        grammar : Grammar
            The deserialized grammar.

        Raises
        ------
        InvalidJSONError
            When the JSON string is invalid.
        DeserializeFormatError
            When the JSON string does not follow the serialization format of the grammar.
        DeserializeVersionError
            When the __VERSION__ field in the JSON string is not the same as the current version.
        """
        return Grammar._create_from_handle(_core.Grammar.deserialize_json(json_string))
