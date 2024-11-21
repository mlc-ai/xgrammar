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

from .base import XGRObject, _core


class Grammar(XGRObject):
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

    root_rule_name : str, default: "root"
        The name of the root rule in the grammar.
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
        return Grammar._create_from_handle(_core.Grammar.from_ebnf(ebnf_string, root_rule_name))

    @staticmethod
    def from_json_schema(
        schema: Union[str, Type[BaseModel]],
        *,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        strict_mode: bool = True,
    ) -> "Grammar":
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
        grammar : Grammar
            The generated BNF grammar.
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = json.dumps(schema.model_json_schema())

        return Grammar._create_from_handle(
            _core.Grammar.from_json_schema(schema, indent, separators, strict_mode),
        )

    @staticmethod
    def builtin_json_grammar() -> "Grammar":
        """Get the grammar of standard JSON. This is compatible with the official JSON grammar
        in https://www.json.org/json-en.html.

        Returns
        -------
        grammar : Grammar
            The JSON grammar.
        """
        return Grammar._create_from_handle(_core.Grammar.builtin_json_grammar())
