"""This module provides classes representing grammars."""

import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union, overload

from pydantic import BaseModel
from typing_extensions import deprecated

from .base import XGRObject, _core
from .structural_tag import StructuralTag, StructuralTagItem


def _convert_instance_to_str(instance: Union[str, Dict[str, Any], StructuralTag]) -> str:
    """Convert a instance to a string representation. It returns the schema in string format because
    it's faster to send to C++.

    This function handles different instance input types and converts them to a JSON string:
    - StructuralTag.
    - String inputs are returned as-is (assumed to be valid JSON)
    - Dictionary inputs are converted to JSON strings

    Parameters
    ----------
    instance : Union[str, StructuralTag, Dict[str, Any]]
        The instance to convert, which can be a StructuralTag,
        a JSON schema string, or a dictionary representing a JSON schema.

    Returns
    -------
    str
        The JSON schema as a string.

    Raises
    ------
    ValueError
        When the instance type is not supported.
    TypeError
        When he dictionary is not serializable.
    """
    if isinstance(instance, dict):
        return json.dumps(instance)
    elif isinstance(instance, str):
        return instance
    elif isinstance(instance, StructuralTag):
        return instance.model_dump_json()
    else:
        raise ValueError("Invalid instance type")


def _convert_schema_to_str(schema: Union[str, Type[BaseModel], Dict[str, Any]]) -> str:
    """Convert a schema to a string representation. It returns the schema in string format because
    it's faster to send to C++.

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
    ValueError
        When the schema type is not supported.
    TypeError
        When the dictionary is not serializable.
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


def _get_structural_tag_str_from_args(args: List[Any], kwargs: Dict[str, Any]) -> str:
    """Get the structural tag string from the arguments. It returns the structural tag in string
    format because it's faster to send to C++.

    Parameters
    ----------
    args : List[Any]
        The positional arguments.
    kwargs : Dict[str, Any]
        The keyword arguments.

    Returns
    -------
    str
        The structural tag string.

    Raises
    ------
    TypeError
        When the arguments are invalid.
    """
    if len(args) == 1:
        if isinstance(args[0], (str, dict, StructuralTag)):
            return _convert_instance_to_str(args[0])
        else:
            raise TypeError("Invalid argument type for from_structural_tag")
    elif len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list):
        return StructuralTag.from_legacy_structural_tag(args[0], args[1]).model_dump_json(
            indent=None
        )
    elif "structural_tag" in kwargs:
        return _convert_instance_to_str(kwargs["structural_tag"])
    elif "tags" in kwargs and "triggers" in kwargs:
        return StructuralTag.from_legacy_structural_tag(
            kwargs["tags"], kwargs["triggers"]
        ).model_dump_json(indent=None)
    else:
        raise TypeError("Invalid arguments for from_structural_tag")


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
        max_whitespace_cnt: Optional[int] = None,
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

        max_whitespace_cnt : Optional[int], default: None
            The maximum number of whitespace characters allowed between elements, such like keys, values, separators and so on.
            If None, there is no limit on the number of whitespace characters.
            If specified, it will limit the number of whitespace characters to at most max_whitespace_cnt.
            It should be a positive integer.

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
                schema_str,
                any_whitespace,
                indent,
                separators,
                strict_mode,
                max_whitespace_cnt,
                print_converted_ebnf,
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

    @overload
    @staticmethod
    def from_structural_tag(
        structural_tag: Union[StructuralTag, str, Dict[str, Any]]
    ) -> "Grammar": ...

    @overload
    @staticmethod
    @deprecated(
        "from_structural_tag(tags, triggers) is deprecated. Construct structural tag with the "
        "StructuralTag class instead."
    )
    def from_structural_tag(tags: List[StructuralTagItem], triggers: List[str]) -> "Grammar": ...

    @staticmethod
    def from_structural_tag(*args, **kwargs) -> "Grammar":
        """Create a grammar from a structural tag. See the Structural Tag Usage in XGrammar
        documentation for its usage.

        This method supports two calling patterns:

        1. Single structural tag parameter:
           from_structural_tag(structural_tag)

        2. Legacy pattern (deprecated):
           from_structural_tag(tags, triggers)

        Parameters
        ----------
        structural_tag : Union[StructuralTag, str, Dict[str, Any]]
            The structural tag either as a StructuralTag object, or a JSON string or a dictionary.

        tags : List[StructuralTagItem]
            (Deprecated) The structural tags. Use StructuralTag class instead.

        triggers : List[str]
            (Deprecated) The triggers. Use StructuralTag class instead.

        Returns
        -------
        grammar : Grammar
            The constructed grammar from the structural tag.

        Raises
        ------
        InvalidJSONError
            When the structural tag is not a valid JSON string.
        InvalidStructuralTagError
            When the structural tag is not valid.
        TypeError
            When the arguments are invalid.

        Notes
        -----
        The legacy pattern from_structural_tag(tags, triggers) is deprecated. Use the StructuralTag
        class to construct structural tags instead.

        For the deprecated pattern: The structural tag handles the dispatching of different grammars
        based on the tags and triggers: it initially allows any output, until a trigger is
        encountered, then dispatch to the corresponding tag; when the end tag is encountered, the
        grammar will allow any following output, until the next trigger is encountered. See the
        Advanced Topics of the Structural Tag in XGrammar documentation for its semantic.
        Structural Tag in XGrammar documentation for its semantic.
        """
        structural_tag_str = _get_structural_tag_str_from_args(args, kwargs)
        return Grammar._create_from_handle(_core.Grammar.from_structural_tag(structural_tag_str))

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
