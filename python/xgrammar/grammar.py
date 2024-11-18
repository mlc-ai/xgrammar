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
from typing import Optional, Tuple, Type, Union

from pydantic import BaseModel

from .base import XGObject, _core


class BNFGrammar(XGObject):
    """This class represents a grammar object in the Backus-Naur Form (BNF). User should provide a
    BNF/EBNF (Extended Backus-Naur Form) grammar. The provided grammar is optimized for LLM
    generation. This class is printable and serializable.

    Parameters
    ----------
    ebnf_string : str
        The grammar string in EBNF format. It should follow the format in
        https://www.w3.org/TR/xml/#sec-notation.

        Note:
        1. Use # as the comment mark
        2. Use C-style unicode escape sequence \u01AB, \U000001AB, \xAB
        3. A-B (match A and not match B) is not supported

    root_rule : str, default: "root"
        The name of the root rule in the grammar.
    """

    @staticmethod
    def from_ebnf(ebnf_string: str, *, root_rule: str = "root") -> "BNFGrammar":
        return BNFGrammar.from_handle(_core.BNFGrammar(ebnf_string, root_rule))

    @staticmethod
    def builtin_json_grammar() -> "BNFGrammar":
        """Get the grammar of standard JSON. This is compatible with the official JSON grammar
        in https://www.json.org/json-en.html.

        Returns
        -------
        grammar : BNFGrammar
            The JSON grammar.
        """
        return BNFGrammar.from_handle(_core.BuiltinGrammar.json())

    @staticmethod
    def from_json_schema(
        schema: Union[str, Type[BaseModel]],
        *,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = True,
    ) -> "BNFGrammar":
        """Construct a BNF grammar from JSON schema. Pydantic model can be used to specify the
        schema.

        The format of the JSON schema can be specified with the `indent` and `separators`
        parameters. The meaning and the default values of the parameters follows the convention in
        json.dumps().

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
        grammar : BNFGrammar
            The generated BNF grammar.
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = json.dumps(schema.model_json_schema())

        return BNFGrammar.from_handle(
            _core.BuiltinGrammar.json_schema(schema, indent, separators, strict_mode),
        )

    def __str__(self) -> str:
        """Print the BNF grammar to a string, in EBNF format.

        Returns
        -------
        grammar_string : str
            The BNF grammar string.
        """
        return self.handle.to_string()

    @staticmethod
    def _init_no_normalization(
        ebnf_string: str,
        root_rule: str = "root",
    ) -> "BNFGrammar":
        r"""Construct a BNF grammar object with a EBNF string, but not normalize it. For test
        purposes.

        Parameters
        ----------
        ebnf_string : str
            The grammar string.

        root_rule : str
            The name of the root rule. Default: "root".

        Returns
        -------
        grammar : BNFGrammar
            The parsed BNF grammar.
        """
        return BNFGrammar.from_handle(
            _core.BNFGrammar._init_no_normalization(ebnf_string, root_rule),
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
        schema : str
            The schema string.

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
        ebnf_string : str
            The EBNF grammar string.
        """
        return _core.BuiltinGrammar._json_schema_to_ebnf(
            schema,
            indent,
            separators,
            strict_mode,
        )

    @staticmethod
    def _regex_to_ebnf(regex: str) -> str:
        r"""Convert a regex string to EBNF grammar string. For test purposes. The regex grammar
        follows the syntax in JavaScript (ECMA 262). Check
        https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions
        for a tutorial. Currently the following features are not supported:
        1. Backreference (\1)
        2. non-capturing group, naming capture groups and assertions ((?...))
        3. Unicode character class escape (\p{...})
        4. Word boundary (\b)
        5. Unicode property escapes (\p{...})
        6. Quantifier with range {x,y}. Now user can just repeat the element as a workaround.

        This method is primarily intended for testing and debugging purposes.

        Parameters
        ----------
        regex : str
            The regex string to be converted.

        Returns
        -------
        ebnf_string : str
            The EBNF grammar string converted from the input regex.
        """
        return _core.BuiltinGrammar._regex_to_ebnf(regex)
