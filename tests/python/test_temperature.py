import math
from typing import Dict, Optional, Union

import pytest
import torch

import xgrammar as xgr

VOCAB = ["a", "b", "x", "z", " "]
TOKENIZER_INFO = xgr.TokenizerInfo(VOCAB, vocab_size=len(VOCAB), stop_token_ids=[])


def _compile_lark(
    grammar: str,
    *,
    named_grammars: Optional[Dict[str, Union[xgr.Grammar, str]]] = None,
    cache_enabled: bool = False,
) -> xgr.CompiledGrammar:
    grammar_object = xgr.Grammar.from_lark(
        grammar, tokenizer_info=TOKENIZER_INFO, named_grammars=named_grammars
    )
    return xgr.GrammarCompiler(TOKENIZER_INFO, cache_enabled=cache_enabled).compile_grammar(
        grammar_object
    )


def test_temperature_defaults_to_none() -> None:
    compiled_grammar = _compile_lark('start: "a"')
    matcher = xgr.GrammarMatcher(compiled_grammar)

    assert matcher.temperature is None
    assert matcher.accept_string("a")
    assert matcher.temperature is None


def test_default_temperature() -> None:
    compiled_grammar = _compile_lark('start: "a"')
    matcher = xgr.GrammarMatcher(compiled_grammar, default_temperature=0.25)

    assert matcher.temperature == pytest.approx(0.25)
    assert matcher.accept_string("a")
    assert matcher.temperature == pytest.approx(0.25)


@pytest.mark.parametrize("temperature", [-0.1, math.inf, -math.inf, math.nan])
def test_invalid_default_temperature(temperature: float) -> None:
    compiled_grammar = _compile_lark('start: "a"')

    with pytest.raises(RuntimeError, match="finite non-negative"):
        xgr.GrammarMatcher(compiled_grammar, default_temperature=temperature)


@pytest.mark.parametrize(
    "temperature, expected", [(0.0, 0.0), (0.125, 0.125), (1e-2, 0.01), (2.0, 2.0)]
)
def test_lark_temperature_number_forms(temperature: float, expected: float) -> None:
    compiled_grammar = _compile_lark(f'start[temperature={temperature}]: "a"')
    matcher = xgr.GrammarMatcher(compiled_grammar, default_temperature=0.5)

    assert matcher.temperature == pytest.approx(expected)


def test_temperature_enters_and_leaves_named_subgrammar() -> None:
    item_grammar = xgr.Grammar.from_lark('start: "x"+')
    compiled_grammar = _compile_lark(
        'start: "a" value "z"\nvalue[temperature=0.6]: @item', named_grammars={"item": item_grammar}
    )
    matcher = xgr.GrammarMatcher(compiled_grammar, default_temperature=0.1)

    assert matcher.temperature == pytest.approx(0.1)
    assert matcher.accept_string("a")
    assert matcher.temperature == pytest.approx(0.6)
    assert matcher.accept_string("xx")
    assert matcher.temperature == pytest.approx(0.6)
    assert matcher.accept_string("z")
    assert matcher.temperature == pytest.approx(0.1)

    matcher.rollback()
    assert matcher.temperature == pytest.approx(0.6)
    matcher.reset()
    assert matcher.temperature == pytest.approx(0.1)


@pytest.mark.parametrize(
    "body, value", [('%json {"type": "integer"}', "1"), ('%lark { start: "x" }', "x")]
)
def test_temperature_on_inline_subgrammar(body: str, value: str) -> None:
    compiled_grammar = _compile_lark(f'start: "a" value "z"\nvalue[temperature=0.6]: {body}')
    matcher = xgr.GrammarMatcher(compiled_grammar, default_temperature=0.1)

    assert matcher.accept_string("a")
    assert matcher.temperature == pytest.approx(0.6)
    assert matcher.accept_string(value + "z")
    assert matcher.temperature == pytest.approx(0.1)


def test_inner_temperature_overrides_outer_temperature() -> None:
    item_grammar = xgr.Grammar.from_lark('start[temperature=0.3]: "x"')
    compiled_grammar = _compile_lark(
        'start: "a" value "z"\nvalue[temperature=0.6]: @item', named_grammars={"item": item_grammar}
    )
    matcher = xgr.GrammarMatcher(compiled_grammar, default_temperature=0.1)

    assert matcher.accept_string("a")
    assert matcher.temperature == pytest.approx(0.3)
    assert matcher.accept_string("x")
    assert matcher.temperature == pytest.approx(0.1)


def test_ambiguous_paths_use_maximum_temperature() -> None:
    compiled_grammar = _compile_lark(
        'start: left | right\nleft[temperature=0.2]: "a"+\n' 'right[temperature=0.9]: "a" "b"'
    )
    matcher = xgr.GrammarMatcher(compiled_grammar, default_temperature=0.1)

    assert matcher.temperature == pytest.approx(0.9)
    assert matcher.accept_string("a")
    assert matcher.temperature == pytest.approx(0.9)
    assert matcher.accept_string("a")
    assert matcher.temperature == pytest.approx(0.2)


def test_temperature_works_with_ignored_tokens() -> None:
    compiled_grammar = _compile_lark(
        '%import common.WS\n%ignore WS\nstart: hot "z"\nhot[temperature=0.6]: "x"'
    )
    matcher = xgr.GrammarMatcher(compiled_grammar, default_temperature=0.1)

    assert matcher.temperature == pytest.approx(0.6)
    assert matcher.accept_string("x")
    assert matcher.temperature == pytest.approx(0.1)
    assert matcher.accept_string(" ")
    assert matcher.temperature == pytest.approx(0.1)


@pytest.mark.parametrize(
    "grammar, message",
    [
        ('start[temperature=-0.1]: "a"', "temperature must be a finite non-negative number"),
        ('start[temperature=1e999]: "a"', "temperature must be a finite non-negative number"),
        (
            'start[temperature=0.1, temperature=0.2]: "a"',
            "temperature attribute is specified more than once",
        ),
        ('start[temperature=abc]: "a"', "expected number after temperature="),
        (
            'start: complex\ncomplex[temperature=0.5]: "a" child\nchild: "b"',
            "temperature is only supported on terminals and subgrammars",
        ),
    ],
)
def test_invalid_lark_temperature(grammar: str, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        xgr.Grammar.from_lark(grammar, tokenizer_info=TOKENIZER_INFO)


def test_temperature_survives_serialization_and_cached_compilation() -> None:
    grammar = xgr.Grammar.from_lark('start[temperature=0.75]: "a"')
    serialized = grammar.serialize_json()
    restored = xgr.Grammar.deserialize_json(serialized)
    compiler = xgr.GrammarCompiler(TOKENIZER_INFO, cache_enabled=True)

    assert "[temperature=0.75]" in str(restored)
    compiled_grammar = compiler.compile_grammar(restored)
    restored_compiled_grammar = xgr.CompiledGrammar.deserialize_json(
        compiled_grammar.serialize_json(), TOKENIZER_INFO
    )
    matcher = xgr.GrammarMatcher(restored_compiled_grammar)
    assert matcher.temperature == pytest.approx(0.75)


@pytest.mark.parametrize("max_threads", [1, 2])
def test_batch_matcher_returns_temperatures(max_threads: int) -> None:
    grammar_temperature = xgr.GrammarMatcher(
        _compile_lark('start[temperature=0.2]: "a"'), default_temperature=0.8
    )
    default_temperature = xgr.GrammarMatcher(_compile_lark('start: "a"'), default_temperature=0.4)
    no_temperature = xgr.GrammarMatcher(_compile_lark('start: "a"'))
    matchers = [grammar_temperature, default_temperature, no_temperature]
    bitmask = xgr.allocate_token_bitmask(3, len(VOCAB))

    temperatures = xgr.BatchGrammarMatcher(max_threads).batch_fill_next_token_bitmask(
        matchers, bitmask, indices=[2, 0, 1]
    )

    assert temperatures[0] == pytest.approx(0.2)
    assert temperatures[1] == pytest.approx(0.4)
    assert temperatures[2] is None


def test_traverse_draft_tree_fills_temperatures() -> None:
    item_grammar = xgr.Grammar.from_lark('start: "x"+')
    compiled_grammar = _compile_lark(
        'start: "a" value "z"\nvalue[temperature=0.6]: @item', named_grammars={"item": item_grammar}
    )
    matcher = xgr.GrammarMatcher(compiled_grammar, default_temperature=0.1)
    retrieve_next_token = torch.tensor([1, 2, 3, -1], dtype=torch.int64)
    retrieve_next_sibling = torch.full((4,), -1, dtype=torch.int64)
    draft_tokens = torch.tensor([0, 0, 2, 3], dtype=torch.int64)
    bitmask = xgr.allocate_token_bitmask(4, len(VOCAB))
    temperatures = torch.empty(4, dtype=torch.float32)

    completed = matcher.traverse_draft_tree(
        retrieve_next_token, retrieve_next_sibling, draft_tokens, bitmask, temperatures=temperatures
    )

    assert completed
    torch.testing.assert_close(temperatures, torch.tensor([0.1, 0.6, 0.6, 0.1]))
    assert matcher.temperature == pytest.approx(0.1)


def test_traverse_draft_tree_uses_nan_for_missing_and_rejected_nodes() -> None:
    compiled_grammar = _compile_lark('start: "a"')
    matcher = xgr.GrammarMatcher(compiled_grammar)
    retrieve_next_token = torch.tensor([1, -1], dtype=torch.int64)
    retrieve_next_sibling = torch.full((2,), -1, dtype=torch.int64)
    draft_tokens = torch.tensor([1, 0], dtype=torch.int64)
    bitmask = xgr.allocate_token_bitmask(2, len(VOCAB))
    temperatures = torch.zeros(2, dtype=torch.float32)

    assert matcher.traverse_draft_tree(
        retrieve_next_token, retrieve_next_sibling, draft_tokens, bitmask, temperatures=temperatures
    )
    assert torch.isnan(temperatures).all()


@pytest.mark.parametrize(
    "temperatures", [torch.empty(2, dtype=torch.float64), torch.empty((2, 1), dtype=torch.float32)]
)
def test_traverse_draft_tree_validates_temperature_tensor(temperatures: torch.Tensor) -> None:
    compiled_grammar = _compile_lark('start: "a"')
    matcher = xgr.GrammarMatcher(compiled_grammar)
    retrieve_next_token = torch.tensor([1, -1], dtype=torch.int64)
    retrieve_next_sibling = torch.full((2,), -1, dtype=torch.int64)
    draft_tokens = torch.tensor([0, 0], dtype=torch.int64)
    bitmask = xgr.allocate_token_bitmask(2, len(VOCAB))

    with pytest.raises(RuntimeError, match="temperatures tensor"):
        matcher.traverse_draft_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            bitmask,
            temperatures=temperatures,
        )
