"""Masks built from character-class token summaries must equal the eager masks.

A rule whose body is one character class, or one-or-more of the same class, gets its mask from
a vocabulary summary of that class when the grammar is compiled dynamically. Eager compilation
never uses that builder, so it is the oracle here.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import xgrammar as xgr

# Token id == byte value for the first 256 tokens, so texts can be fed byte by byte.
VOCAB = [bytes([i]) for i in range(256)] + [
    # ASCII: shared prefixes, a duplicate, class members followed by the template's delimiters.
    b"ab",
    b"ab",
    b"abc",
    b"abd",
    b"ab!",
    b"ab)",
    b"a)x",
    b"a>",
    b"zz",
    b"12",
    b"12)",
    b"1a",
    b"a1",
    b"A_9",
    b"  ",
    b" a",
    b'a"',
    b'"a',
    b"a\\",
    b"a\x1f",
    b"a\x7f",
    b"a\x00",
    b"~~",
    b"-]",
    b"()",
    b"))",
    # Complete UTF-8 characters at the encoding-length boundaries.
    b"\xc2\x80",  # U+0080
    b"\xdf\xbf",  # U+07FF
    b"\xe0\xa0\x80",  # U+0800
    b"\xef\xbf\xbf",  # U+FFFF
    b"\xf0\x90\x80\x80",  # U+10000
    b"\xf4\x8f\xbf\xbf",  # U+10FFFF
    "中".encode(),
    "文".encode(),
    "日".encode(),
    "中文".encode(),
    "中日".encode(),
    "中)".encode(),
    "a中".encode(),
    "中a".encode(),
    "😀".encode(),
    "😀😂".encode(),
    "😀a".encode(),
    # Tokens that end inside a character.
    b"\xe4\xb8",  # U+4E00..U+4E3F, contains 中
    b"\xe6\x96",  # U+6580..U+65BF, contains 文
    b"\xe0\xa0",  # U+0800..U+083F
    b"\xf0\x9f",  # U+1F000..U+1FFFF
    b"\xf0\x9f\x98",  # U+1F600..U+1F63F
    b"\xf4\x8f\xbf",  # includes U+10FFFF
    b"\xed\xa0",  # surrogate prefix: preserve the parser's codepoint semantics
    b"a\xe4\xb8",
    "中".encode() + b"\xe4",
    b"\xf4\x90",  # only above U+10FFFF
    b"\xe0\x80",  # only overlong encodings
    b"\xf0\x80",
    b"\xf0\x80\x80",
    # Invalid UTF-8.
    b"\xe4a",  # continuation byte missing
    b"\xc2\x20",
    b"\xe4\xb8A",  # invalid last byte of a three-byte character
    b"\xf0\x9fA",
    b"\xf0\x9f\x98A",  # invalid last byte of a four-byte character
    b"a\xff",
    b"a\xe4A",  # a complete class member before malformed UTF-8
    b"a\xe0\x80",
    b"\xc0\x80",  # overlong
    b"\xe0\x80\x80",  # overlong
    b"\xed\xa0\x80",  # surrogate U+D800 (accepted by the existing byte parser)
    b"\xf0\x8f\xbf\xbf",  # overlong four-byte form
    b"\xf4\x90\x80\x80",  # above U+10FFFF
    b"\xf8\x88\x80\x80\x80",  # five-byte form
    b"<stop>",
]
STOP = len(VOCAB) - 1

# Every rule under test is referenced twice and never first in a sequence, so that it is neither
# inlined nor given a lookahead by the grammar normalizer.
TEMPLATE = 'root ::= "(" r ")" | "<" r ">"\n'
SHAPES = {"single": "r ::= {c}", "plus": "r ::= {c} | {c} r"}


def source_for(character_class, shape):
    return TEMPLATE + SHAPES[shape].format(c=character_class)


def compile_both(info, source, cache_enabled=False):
    eager = xgr.GrammarCompiler(info, cache_enabled=False).compile_grammar(source)
    dynamic = xgr.GrammarCompiler(
        info, cache_enabled=cache_enabled, enable_dynamic_compilation=True
    ).compile_grammar(source)
    return eager, dynamic


def mask_bits(matcher, vocab_size):
    """Compare actual vocabulary IDs, excluding unused bitmask padding bits."""
    bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    words = [int(word) & 0xFFFFFFFF for word in bitmask[0]]
    return {
        token_id for token_id in range(vocab_size) if (words[token_id // 32] >> (token_id % 32)) & 1
    }


def assert_same_masks(eager, dynamic, vocab_size, prefix=b"(", replay=False, follow_tokens=False):
    """Check the mask against eager compilation and, optionally, every parser transition."""
    expected = xgr.GrammarMatcher(eager)
    actual = xgr.GrammarMatcher(dynamic)
    assert expected.accept_string(prefix)
    assert actual.accept_string(prefix)
    allowed = mask_bits(expected, vocab_size)
    assert mask_bits(actual, vocab_size) == allowed, prefix
    if replay or follow_tokens:
        for token_id in range(vocab_size):
            oracle = expected.fork()
            accepted = oracle.accept_token(token_id)
            assert accepted == (token_id in allowed), (prefix, token_id)
            if follow_tokens and accepted and not oracle.is_terminated():
                candidate = actual.fork()
                assert candidate.accept_token(token_id), (prefix, token_id)
                assert mask_bits(candidate, vocab_size) == mask_bits(oracle, vocab_size), (
                    prefix,
                    token_id,
                )
    return allowed


CHARACTER_CLASSES = [
    # Positive, ASCII only: summarized byte by byte with prefix reuse between sorted tokens.
    r"[a-z]",
    r"[a-cx-z]",  # disjoint ranges and the gap between them
    r"[\x00-\x1f]",
    r"[\x00-\x7f]",
    r"[\-\]\\]",  # escaped members
    r"[()!<>]",  # the characters that follow the rule
    r"[]",  # matches nothing
    # Positive with characters beyond ASCII.
    r"[\x7f-\x80]",
    r"[\x80-߿]",
    r"[ࠀ-￿]",
    r"[\U00010000-\U0010ffff]",
    r"[\x00-\U0010ffff]",
    r"[中-文]",
    r"[a-z中-文]",
    r"[😀-😂]",
    r"[\ud800-\udfff]",  # retain the existing parser's treatment of surrogate codepoints
    # Negated.
    r"[^]",  # matches everything
    r"[^a]",
    r"[^()!<>]",
    r'[^"\\\x00-\x1f]',  # a JSON string character: accepts every printable ASCII token
    r"[^\x00-\x7f]",
    r"[^\x80-\U0010ffff]",
    r"[^\x00-\U0010ffff]",  # matches nothing
    r"[^中-文]",
    r"[^一-丿]",  # excludes everything the partial token e4 b8 can become
    r"[^一-丟丠-丿]",  # the same through two adjacent ranges
    r"[^一-丬]",  # excludes only part of it
    r"[^\U0001f600-\U0001f63f]",
    r"[^\ud800-\udfff]",
]


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("character_class", CHARACTER_CLASSES)
def test_character_class_masks_match_eager_and_replay(character_class, shape):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    eager, dynamic = compile_both(info, source_for(character_class, shape))
    # Explore every accepted first token, including each UTF-8 lead byte and rule-crossing token.
    # This is exhaustive over this vocabulary, rather than sampling token IDs.
    assert_same_masks(eager, dynamic, len(VOCAB), replay=True, follow_tokens=True)


# Rules that look like the supported shapes but must take the generic builder.
FALLBACK_SOURCES = [
    # Lookahead and counted repetition already have oracle tests in test_dynamic_compilation.py.
    # One choice that is not exactly one class.
    TEMPLATE + 'r ::= [a-z] "x"',
    TEMPLATE + "r ::= [a]",  # normalized to a string
    TEMPLATE + "r ::= [a-z]*",
    # Two choices that are not class | class self.
    TEMPLATE + "r ::= [a-z] | [0-9] r",  # different classes
    TEMPLATE + "r ::= [a-z] | [a-z] [0-9]",  # second element is not a rule
    TEMPLATE + 'r ::= [a-z] | [a-z] other\nother ::= "x" | "y" other',  # not itself
    TEMPLATE + "r ::= [a-z] | [0-9]",  # no recursive choice
    TEMPLATE + "r ::= [a-z] r | [0-9] r",  # no base choice
    TEMPLATE + "r ::= [a-z] | [a-z0-9] r",  # classes of different sizes
    TEMPLATE + 'r ::= "a" | [a-z] r',  # no base class
    TEMPLATE + 'r ::= [a-z] | "ab" r',
    # More than two choices, and a body that is not a list of choices.
    TEMPLATE + 'r ::= [a-z] | [a-z] r | "ab"',
    TEMPLATE + 'r ::= TagDispatch(("<a>", other))\nother ::= "x"',
    # Character budgets must disable the direct path even for an otherwise eligible class.
    TEMPLATE + "r[max_chars=2] ::= [a-z] | [a-z] r",
]


@pytest.mark.parametrize("source", FALLBACK_SOURCES)
def test_unsupported_rule_shapes_match_eager(source):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    eager, dynamic = compile_both(info, source)
    assert_same_masks(eager, dynamic, len(VOCAB), follow_tokens=True)


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize(
    "accepted_count,rejected_count,store_type",
    [
        (999, 1001, 0),  # accepted indices, just below the bitset cutoff
        (1000, 1001, 2),  # exactly the bitset cutoff
        (1001, 1001, 2),  # a large tie still needs a bitset
        (1001, 1000, 2),  # rejection count exactly at the cutoff
        (1001, 999, 1),  # rejected indices, just below the cutoff
        (999, 999, 1),  # small ties favor rejected indices
    ],
)
def test_mask_storage_boundaries_and_special_ids(shape, accepted_count, rejected_count, store_type):
    # Duplicates have distinct token IDs. Reordering makes token IDs differ from sorted indices.
    # Rejections on both sides of the accepted tokens exercise both rejected-index loops.
    vocab = ["中"] * accepted_count + ["(", ")", "\uffff"] + ["!"] * (rejected_count - 3)
    vocab += ["中)", "中x"]  # uncertain: only the first can complete the parent
    vocab = vocab[::2] + vocab[1::2]
    vocab += ["", "<stop>", "<stop2>"]
    stops = [len(vocab) - 2, len(vocab) - 1]
    info = xgr.TokenizerInfo(vocab, vocab_size=len(vocab) + 5, stop_token_ids=stops)
    eager, dynamic = compile_both(info, source_for("[一-鿿]", shape))
    allowed = assert_same_masks(eager, dynamic, info.vocab_size)
    assert allowed == {i for i, token in enumerate(vocab) if token in ("中", "中)")}
    assert assert_same_masks(eager, dynamic, info.vocab_size, prefix="(中)") == set(stops)

    # Check the representation as well as its public behavior: parity alone could pass if
    # these cases silently stopped exercising the new bitset constructor or rejected list.
    serialized = dynamic.serialize_json()
    payload = json.loads(serialized)
    rule_id = next(i for i, rule in enumerate(payload["grammar"]["rules"]) if rule[0] == "r")
    start_state = payload["grammar"]["per_rule_fsms"][rule_id][0][1]
    masks = [
        mask
        for state, mask in payload["adaptive_token_mask_cache"]
        if state[0] == rule_id and state[2] == start_state
    ]
    assert len(masks) == 1
    assert masks[0]["store_type"] == store_type

    # One round trip for each representation is sufficient; do not multiply every boundary
    # case by serialization and compiler-cache settings.
    if shape == "plus" and (accepted_count, rejected_count) in (
        (999, 1001),
        (1000, 1001),
        (1001, 999),
    ):
        restored_info = xgr.TokenizerInfo.deserialize_json(info.serialize_json())
        restored = xgr.CompiledGrammar.deserialize_json(serialized, restored_info)
        assert assert_same_masks(eager, restored, info.vocab_size) == allowed


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize(
    "character", ["\u0080", "\u07ff", "\u0800", "中", "\uffff", "\U00010000", "😀", "\U0010ffff"]
)
def test_every_utf8_split_and_rule_completion(character, shape):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    eager, dynamic = compile_both(info, source_for(r"[\x00-\U0010ffff]", shape))
    encoded = character.encode()
    for split in range(1, len(encoded) + 1):
        assert_same_masks(eager, dynamic, len(VOCAB), prefix=b"(" + encoded[:split], replay=True)
    allowed = assert_same_masks(
        eager, dynamic, len(VOCAB), prefix=b"(" + encoded + b")", replay=True
    )
    assert STOP in allowed


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_summaries_are_shared_between_grammars_and_outlive_the_compiler(cache_enabled):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    compiler = xgr.GrammarCompiler(
        info, cache_enabled=cache_enabled, enable_dynamic_compilation=True
    )
    # The same class in different surroundings, in both shapes, and twice within one grammar.
    sources = [
        source_for("[a-z]", "plus"),
        source_for("[a-z]", "single"),
        'root ::= "(" r ")" | "<" r "]" | "{" s "}" | "[" s ">"\n'
        "r ::= [a-z] | [a-z] r\ns ::= [a-z] | [a-z] s",
        source_for("[^a-z]", "plus"),
        source_for("[0-9]", "plus"),
        source_for("[a-z中-文]", "plus"),
    ]
    dynamic = [compiler.compile_grammar(source) for source in sources]
    eager = [
        xgr.GrammarCompiler(info, cache_enabled=False).compile_grammar(source) for source in sources
    ]
    compiler.clear_cache()
    del compiler

    barrier = Barrier(6)

    def compare(index):
        barrier.wait()
        which = index % len(sources)
        for prefix in (b"(", b"{") if which == 2 else (b"(",):
            assert_same_masks(
                eager[which], dynamic[which], len(VOCAB), prefix=prefix, follow_tokens=True
            )

    # Masks are built on first use, so the first lookups of each grammar race.
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(compare, range(12)))
