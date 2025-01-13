"""Testing utilities."""

import time
from typing import List, Optional, Tuple, Union

import torch

from .base import _core
from .compiler import CompiledGrammar, GrammarCompiler
from .grammar import Grammar
from .matcher import GrammarMatcher, bitmask_dtype, get_bitmask_shape
from .tokenizer_info import TokenizerInfo


def _json_schema_to_ebnf(
    schema: str,
    *,
    any_whitespace: bool = True,
    indent: Optional[int] = None,
    separators: Optional[Tuple[str, str]] = None,
    strict_mode: bool = True,
) -> str:
    """Convert JSON schema string to BNF grammar string. For test purposes.

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
    bnf_string : str
        The BNF grammar string.
    """
    return _core.testing._json_schema_to_ebnf(
        schema,
        any_whitespace,
        indent,
        separators,
        strict_mode,
    )


def _regex_to_ebnf(regex: str, with_rule_name: bool = True) -> str:
    r"""Convert a regex string to BNF grammar string. For test purposes. The regex grammar
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
    bnf_string : str
        The BNF grammar string converted from the input regex.
    """
    return _core.testing._regex_to_ebnf(regex, with_rule_name)


def _is_grammar_accept_string(
    grammar: Union[Grammar, str],
    input_str: str,
    *,
    debug_print: bool = False,
    print_time: bool = False,
) -> bool:
    """Check if a grammar accepts a string. For test purposes.

    Parameters
    ----------
    grammar : Union[Grammar, str]
        The grammar to check. Can be either a Grammar object or a BNF grammar string.
    input_str : str
        The input string to check.
    debug_print : bool, default: False
        Whether to print debug information during matching.
    print_time : bool, default: False
        Whether to print timing information.

    Returns
    -------
    bool
        True if the grammar accepts the string, False otherwise.
    """

    if isinstance(grammar, str):
        grammar = Grammar.from_ebnf(grammar)
    grammar_compiler = GrammarCompiler(TokenizerInfo([]), cache_enabled=False)
    compiled_grammar = grammar_compiler.compile_grammar(grammar)
    matcher = GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)

    if print_time:
        start = time.monotonic_ns()
    accepted = matcher._debug_accept_string(input_str, debug_print=debug_print)

    if print_time:
        end = time.monotonic_ns()
        print(f"Accepting {input_str}, result: {accepted}, time: {(end - start) / 1e3} us")

    if not accepted:
        return False
    return matcher.is_terminated()


def _get_masked_tokens_from_bitmask(
    bitmask: torch.Tensor, vocab_size: int, index: int = 0
) -> List[int]:
    """Get the ids of the rejected tokens from the bitmask. Mainly for debug purposes.

    Parameters
    ----------
    bitmask : torch.Tensor
        The rejected token bitmask. Should be generated by allocate_token_bitmask and
        filled by fill_next_token_bitmask. Should be on CPU.

    index : int, default: 0
        The batch index of the bitmask. For batch inference, bitmask[index] will be used.
        Otherwise is ignored.

    Returns
    -------
    rejected_token_ids : List[int]
        A list of rejected token ids.
    """
    if bitmask.device.type != "cpu":
        raise ValueError("bitmask should be on CPU.")
    if bitmask.dtype != bitmask_dtype:
        raise ValueError(f"bitmask should be of type {bitmask_dtype}.")
    return _core.testing._get_masked_tokens_from_bitmask(
        bitmask.data_ptr(), list(bitmask.shape), vocab_size, index
    )


def _get_matcher_from_grammar_and_tokenizer_info(
    grammar: Union[Grammar, str], tokenizer_info: Optional[TokenizerInfo] = None, **kwargs
) -> GrammarMatcher:
    """Create a GrammarMatcher from a grammar and tokenizer info.

    Parameters
    ----------
    grammar : Union[Grammar, str]
        The grammar to create the matcher from. Can be either a Grammar object or a string
        containing EBNF grammar.
    tokenizer_info : Optional[TokenizerInfo], default: None
        Information about the tokenizer to use with this grammar. If None, an empty
        TokenizerInfo will be created.
    **kwargs
        Additional keyword arguments to pass to the GrammarMatcher constructor.

    Returns
    -------
    matcher : GrammarMatcher
        The created grammar matcher.
    """
    if tokenizer_info is None:
        tokenizer_info = TokenizerInfo([])
    grammar_compiler = GrammarCompiler(tokenizer_info, cache_enabled=False)
    compiled_grammar = grammar_compiler.compile_grammar(grammar)
    return GrammarMatcher(compiled_grammar, **kwargs)


def _get_grammar_union(*grammars: "Grammar") -> "Grammar":
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


def _get_allow_empty_rule_ids(compiled_grammar: CompiledGrammar) -> List[int]:
    return _core.testing._get_allow_empty_rule_ids(compiled_grammar._handle)
