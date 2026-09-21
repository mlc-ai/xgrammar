/**
 * \file tests/cpp/test_repeat_fast_path.cc
 * \brief Correctness tests of the repeat fast-path token-acceptance certificate as an enabled
 * optimization inside FillBitmaskForStates.
 *
 * The production helper CanFastAcceptRepeatToken is acceptance-only: a
 * positive verdict proves the token is accepted and the byte-level trial is
 * skipped; a negative verdict leaves the ordinary trial completely unchanged.
 * These tests assert the observable contract of the enabled fast path:
 *   - the produced bitmask is exactly what the ordinary parser produces
 *     (accept/reject decisions are never changed by the fast path), and
 *   - the LCP / rollback / capacity / aggregation machinery stays consistent
 *     across fills that interleave fast-path and fallback tokens.
 * Which tokens are fast-path-eligible is fixed by construction (a clean
 * repeat-body seed plus a pure-ASCII token inside the repeat bound), so each
 * test knows which fills exercise the fast path even though the assertions
 * only observe the resulting masks and parser behavior.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "grammar_builder.h"
#include "xgrammar/compiler.h"
#include "xgrammar/grammar.h"
#include "xgrammar/matcher.h"
#include "xgrammar/tokenizer_info.h"

using namespace xgrammar;

namespace {

/*! Fill the next-token bitmask. */
void FillMask(GrammarMatcher* matcher, int vocab_size, std::vector<int32_t>* mask) {
  const int32_t words = (vocab_size + 31) / 32;
  mask->assign(words, 0);
  int64_t shape = words;
  DLTensor tensor{};
  tensor.data = mask->data();
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 1;
  tensor.shape = &shape;
  tensor.dtype = {kDLInt, 32, 1};
  matcher->FillNextTokenBitmask(&tensor, 0, false);
}

bool MaskHas(const std::vector<int32_t>& mask, int token_id) {
  return ((mask[token_id >> 5] >> (token_id & 31)) & 1) != 0;
}

std::vector<int32_t> MaskCopy(const std::vector<int32_t>& mask) { return mask; }

/*! Build a matcher over the grammar with a RAW vocab (optionally marking stop tokens). */
std::unique_ptr<GrammarMatcher> MakeMatcher(
    const Grammar& grammar,
    const std::vector<std::string>& vocab,
    const std::vector<int>& stop_token_ids = {}
) {
  std::optional<std::vector<int>> stops;
  if (!stop_token_ids.empty()) {
    stops = stop_token_ids;
  }
  TokenizerInfo tokenizer(
      vocab, VocabType::RAW, static_cast<int>(vocab.size()), std::move(stops), false
  );
  GrammarCompiler compiler(tokenizer, /*max_threads=*/1, /*cache_enabled=*/false);
  return std::make_unique<GrammarMatcher>(compiler.CompileGrammar(grammar));
}

int AddToken(std::vector<std::string>* vocab, const std::string& s) {
  vocab->push_back(s);
  return static_cast<int>(vocab->size()) - 1;
}

/*! root ::= "A" body{L,U} "Z", with body ::= the given character class.
 *  With U > 128 the compiler unzips the repeat: the inner registration carries
 *  upper = U - 128 and the first 128 body chars run through the unrolled
 *  prefix leg. */
Grammar MakeRepeatGrammar(
    const std::vector<GrammarBuilder::CharacterClassElement>& elems,
    bool is_negative,
    int32_t lo,
    int32_t hi
) {
  GrammarBuilder b;
  const int32_t body =
      b.AddRule("body", b.AddChoices({b.AddSequence({b.AddCharacterClass(elems, is_negative)})}));
  const int32_t rep = b.AddRepeat(body, lo, hi);
  b.AddRule(
      "root", b.AddChoices({b.AddSequence({b.AddByteString("A"), rep, b.AddByteString("Z")})})
  );
  return b.Get("root");
}

/*! The bounded JSON string shape with the real delimiters:
 *  root ::= "\"" body{L,U} "\"", body ::= [^"\\\r\n]. */
Grammar MakeBoundedJsonStringGrammar(int32_t lo, int32_t hi) {
  GrammarBuilder b;
  const int32_t body = b.AddRule(
      "body",
      b.AddChoices({b.AddSequence(
          {b.AddCharacterClass({{34, 34}, {92, 92}, {10, 10}, {13, 13}}, /*is_negative=*/true)}
      )})
  );
  const int32_t rep = b.AddRepeat(body, lo, hi);
  b.AddRule(
      "root", b.AddChoices({b.AddSequence({b.AddByteString("\""), rep, b.AddByteString("\"")})})
  );
  return b.Get("root");
}

// Pre-LCP safety: T1 and T3 are ordinary trials that share T3's prefix with
// the middle token, which the fast path accepts without any bookkeeping. If
// the fast path left prev_token / prev_matched_size / trial rows broken, T3's
// LCP reuse would pop the wrong rows and falsely reject it.
TEST(RepeatFastPathTest, CertifiedBetweenFallbacksSharedPrefix) {
  Grammar grammar = MakeRepeatGrammar({{'a', 'z'}}, /*is_negative=*/false, 1, 1000);
  std::vector<std::string> vocab;
  const int a = AddToken(&vocab, "A");
  const int t1 = AddToken(&vocab, std::string(200, 'a') + "Z");
  const int t2 = AddToken(&vocab, std::string(300, 'a'));
  const int t3 = AddToken(&vocab, std::string(300, 'a') + "Z");
  const int t4 = AddToken(&vocab, std::string(900, 'a') + "Z");
  const int t5 = AddToken(&vocab, std::string(1001, 'a') + "Z");
  const int z = AddToken(&vocab, "Z");
  auto matcher = MakeMatcher(grammar, vocab);
  std::vector<int32_t> mask;
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  ASSERT_TRUE(matcher->AcceptToken(a));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);

  // Sorted-vocab order is T1 < T2 < T3 < T4 < T5, so one uncertain-token loop
  // covers the middle token between fallback tokens with shared prefixes.
  // T2 is fast-path-eligible by construction (clean body seed, 300 'a's).
  EXPECT_TRUE(MaskHas(mask, t1)) << "fallback token with boundary suffix must stay accepted";
  EXPECT_TRUE(MaskHas(mask, t2)) << "the 300-char token must be accepted";
  EXPECT_TRUE(MaskHas(mask, t3)
  ) << "token sharing the 300-char token's prefix must not be falsely rejected";
  EXPECT_TRUE(MaskHas(mask, t4)
  ) << "900 body chars are within maxLength 1000 (beyond the inner registration's 872 "
       "capacity), so the ordinary trial accepts";
  EXPECT_FALSE(MaskHas(mask, t5)) << "1001 body chars exceed the repeat bound: rejected";
  (void)z;  // "Z" cannot close the rule before the first body char (lo = 1)
  // The parser history must remain intact: consuming the fallback boundary
  // token after the middle token completes the root rule.
  ASSERT_TRUE(matcher->AcceptToken(t3));
  EXPECT_TRUE(matcher->IsCompleted());
}

// A rejected token leaves partial trial rows and a rejected subtree range;
// the following middle token is fast-path-accepted without any bookkeeping;
// the last token sharing its prefix must still be correctly rejected at its
// trailing byte via the LCP state left by the last ordinary trial (T1).
TEST(RepeatFastPathTest, FailedPrefixThenCertifiedThenFallback) {
  Grammar grammar = MakeRepeatGrammar({{'n', 'z'}}, /*is_negative=*/false, 1, 1000);
  std::vector<std::string> vocab;
  const int a = AddToken(&vocab, "A");
  const int t1 = AddToken(&vocab, std::string(50, 's') + "a" + std::string(50, 's'));
  const int t2 = AddToken(&vocab, std::string(200, 's'));
  const int t3 = AddToken(&vocab, std::string(200, 's') + "a");
  const int z = AddToken(&vocab, "Z");
  auto matcher = MakeMatcher(grammar, vocab);
  std::vector<int32_t> mask;
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  ASSERT_TRUE(matcher->AcceptToken(a));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);

  EXPECT_FALSE(MaskHas(mask, t1)) << "token rejected at byte 50 ('a' outside [n-z])";
  EXPECT_TRUE(MaskHas(mask, t2)) << "the 200-char token must be accepted";
  EXPECT_FALSE(MaskHas(mask, t3)
  ) << "token sharing the 200-char token's prefix but ending in a rejected byte must "
       "be rejected via the LCP state left by the last ORDINARY token (T1)";
  (void)z;  // "Z" cannot close the rule before the first body char (lo = 1)
  // Mask computation is repeatable: a second fill must produce the same mask.
  const auto mask_before = MaskCopy(mask);
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  EXPECT_EQ(mask, mask_before) << "a second fill must not perturb the mask or the parser";
}

// Cleanup safety: when no body token goes through the byte-level trial, the
// end-of-seed PopLastStates must pop exactly the singleton seed row, never
// real parser history. Consuming an accepted token and closing the rule
// verifies the history stayed intact.
TEST(RepeatFastPathTest, AllUncertainTokensCertifiedCleanup) {
  Grammar grammar = MakeRepeatGrammar({{'a', 'z'}}, /*is_negative=*/false, 1, 1000);
  std::vector<std::string> vocab;
  const int a = AddToken(&vocab, "A");
  const int a200 = AddToken(&vocab, std::string(200, 'a'));
  const int a256 = AddToken(&vocab, std::string(256, 'a'));
  const int a300 = AddToken(&vocab, std::string(300, 'a'));
  const int z = AddToken(&vocab, "Z");
  auto matcher = MakeMatcher(grammar, vocab);
  std::vector<int32_t> mask;
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  ASSERT_TRUE(matcher->AcceptToken(a));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);

  EXPECT_TRUE(MaskHas(mask, a200));
  EXPECT_TRUE(MaskHas(mask, a256));
  EXPECT_TRUE(MaskHas(mask, a300));
  EXPECT_FALSE(MaskHas(mask, z)) << "lo = 1: 'Z' cannot close before a body char";
  // Consume an accepted token and fill again: the same three tokens must
  // still be accepted from the live registration (repeat count reread, not
  // cached).
  ASSERT_TRUE(matcher->AcceptToken(a256));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  EXPECT_TRUE(MaskHas(mask, a200));
  EXPECT_TRUE(MaskHas(mask, a256));
  EXPECT_TRUE(MaskHas(mask, a300));
  EXPECT_TRUE(MaskHas(mask, z)) << "'Z' closes the rule after a body char";
  // The parser history must be intact after the all-accepted fill.
  ASSERT_TRUE(matcher->AcceptToken(z));
  EXPECT_TRUE(matcher->IsCompleted());
}

// Rollback: after a rollback the fill runs against reused parser rows, so the
// live registration's repeat count must be reread (not cached from the
// rolled-back fill), and the mask at the same parser position must be
// identical to the pre-rollback fill.
TEST(RepeatFastPathTest, RollbackReusedRows) {
  Grammar grammar = MakeRepeatGrammar({{'a', 'z'}}, /*is_negative=*/false, 1, 1000);
  std::vector<std::string> vocab;
  const int a = AddToken(&vocab, "A");
  const int a256 = AddToken(&vocab, std::string(256, 'a'));
  const int a700 = AddToken(&vocab, std::string(700, 'a'));
  const int z = AddToken(&vocab, "Z");
  auto matcher = MakeMatcher(grammar, vocab);
  std::vector<int32_t> mask;
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  ASSERT_TRUE(matcher->AcceptToken(a));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  EXPECT_TRUE(MaskHas(mask, a256));
  EXPECT_TRUE(MaskHas(mask, a700))
      << "q = 0: 700 is within the inner registration's capacity (872)";
  const auto mask_before = MaskCopy(mask);

  ASSERT_TRUE(matcher->AcceptToken(a256));  // q becomes 256
  matcher->Rollback(1);                     // back to q = 0, rows reused
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  EXPECT_EQ(
      mask, mask_before
  ) << "after rollback the fill must reproduce the pre-rollback mask exactly";

  // Re-consume and fill again: q = 256 now. The parser still ACCEPTS a700
  // (256 + 700 = 956 <= 1000) even though it is beyond the inner
  // registration's remaining capacity (872 - 256).
  ASSERT_TRUE(matcher->AcceptToken(a256));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  EXPECT_TRUE(MaskHas(mask, a256));
  EXPECT_TRUE(MaskHas(mask, a700)) << "956 <= 1000: parser accepts via the ordinary trial";
  (void)z;
}

// Strict capacity: inner upper = 872 (hi = 1000 unzipped). The fast path may
// only accept via the strict bound m < upper - q of the inner registration,
// so this boundary checks that tokens near the bound keep their parser
// verdicts: at q = 0 every token up to maxLength is accepted, and at q = 871
// only tokens that keep the total within maxLength are accepted.
TEST(RepeatFastPathTest, CapacityBoundary) {
  Grammar grammar = MakeRepeatGrammar({{'a', 'z'}}, /*is_negative=*/false, 1, 1000);
  std::vector<std::string> vocab;
  const int a = AddToken(&vocab, "A");
  const int r871 = AddToken(&vocab, std::string(871, 'a'));
  const int r872 = AddToken(&vocab, std::string(872, 'a'));
  const int r873 = AddToken(&vocab, std::string(873, 'a'));
  const int one = AddToken(&vocab, "a");
  const int z = AddToken(&vocab, "Z");
  auto matcher = MakeMatcher(grammar, vocab);
  std::vector<int32_t> mask;
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  ASSERT_TRUE(matcher->AcceptToken(a));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);

  EXPECT_TRUE(MaskHas(mask, r871));
  EXPECT_TRUE(MaskHas(mask, r872)) << "872 body chars are within maxLength: parser accepts";
  EXPECT_TRUE(MaskHas(mask, r873)) << "873 body chars are within maxLength: parser accepts";

  // Shift the boundary: consume exactly upper - 1 = 871 body chars.
  ASSERT_TRUE(matcher->AcceptToken(r871));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  EXPECT_FALSE(MaskHas(mask, r871)) << "q = 871: another 871 body chars would total 1742 > 1000";
  EXPECT_FALSE(MaskHas(mask, r872)) << "another 872 body chars would total 1743 > 1000";
  EXPECT_TRUE(MaskHas(mask, one)) << "872 body chars total is within maxLength: parser accepts";
  EXPECT_TRUE(MaskHas(mask, z));
}

// UTF-8 fallback: codepoint tokens and delimiter bytes keep the parser's mask
// semantics unchanged. A Unicode body char consumed earlier must not change
// the verdict for a later clean ASCII body-restart seed.
TEST(RepeatFastPathTest, Utf8PreservesMaskSemantics) {
  Grammar grammar = MakeBoundedJsonStringGrammar(1, 100000);
  std::vector<std::string> vocab;
  const int quote = AddToken(&vocab, "\"");
  const int eacute = AddToken(&vocab, "\xC3\xA9");
  const int a_eacute = AddToken(&vocab, "a\xC3\xA9");
  const int star256 = AddToken(&vocab, std::string(256, '*'));
  auto matcher = MakeMatcher(grammar, vocab);
  std::vector<int32_t> mask;
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  ASSERT_TRUE(matcher->AcceptToken(quote));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);

  EXPECT_TRUE(MaskHas(mask, eacute)) << "the parser accepts the UTF-8 codepoint";
  EXPECT_TRUE(MaskHas(mask, a_eacute));
  EXPECT_TRUE(MaskHas(mask, star256));
  ASSERT_TRUE(matcher->AcceptToken(eacute));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  EXPECT_TRUE(MaskHas(mask, star256))
      << "after a UTF-8 body char, a clean body-restart seed must still accept the ASCII run";
}

TEST(RepeatFastPathTest, Delimiters) {
  // Delimiter bytes at the beginning, end, or middle of an otherwise
  // certifiable run keep the parser's reject verdicts.
  Grammar grammar = MakeBoundedJsonStringGrammar(1, 100000);
  std::vector<std::string> vocab;
  const int quote = AddToken(&vocab, "\"");
  const int star256 = AddToken(&vocab, std::string(256, '*'));
  const std::string run = std::string(200, '*');
  const int nl_run = AddToken(&vocab, "\n" + run);
  const int run_nl = AddToken(&vocab, run + "\n");
  const int run_nl_run = AddToken(&vocab, run.substr(0, 100) + "\n" + run.substr(0, 100));
  const int quote_run = AddToken(&vocab, "\"" + run);
  auto matcher = MakeMatcher(grammar, vocab);
  std::vector<int32_t> mask;
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  ASSERT_TRUE(matcher->AcceptToken(quote));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);

  EXPECT_TRUE(MaskHas(mask, star256)) << "the pure ASCII run is accepted";
  EXPECT_FALSE(MaskHas(mask, nl_run)) << "delimiter at the beginning: parser rejects";
  EXPECT_FALSE(MaskHas(mask, run_nl)) << "delimiter at the end: parser rejects";
  EXPECT_FALSE(MaskHas(mask, run_nl_run)) << "delimiter in the middle: parser rejects";
  EXPECT_FALSE(MaskHas(mask, quote_run)) << "quote delimiter: parser rejects";
}

TEST(RepeatFastPathTest, AmbiguousEdges) {
  // body ::= [a-b] [a-b] | [a-b]: the body FSM start has TWO char edges
  // matching 'a' and 'b' (different targets), so no single byte path can be
  // certified. The parser accepts via its class union over the alternatives;
  // the verdicts are preserved.
  GrammarBuilder b;
  const int32_t cls =
      b.AddCharacterClass({{static_cast<int32_t>('a'), static_cast<int32_t>('b')}}, false);
  const int32_t body =
      b.AddRule("body", b.AddChoices({b.AddSequence({cls, cls}), b.AddSequence({cls})}));
  const int32_t rep = b.AddRepeat(body, 0, 5);
  b.AddRule(
      "root", b.AddChoices({b.AddSequence({b.AddByteString("A"), rep, b.AddByteString("Z")})})
  );
  std::vector<std::string> vocab = {"A", "Z", "a", "b", "ab", "!"};
  auto matcher = MakeMatcher(b.Get("root"), vocab, {5});
  std::vector<int32_t> mask;
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  ASSERT_TRUE(matcher->AcceptToken(0));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);

  EXPECT_TRUE(MaskHas(mask, 2)) << "'a' matches the [a-b] alternative: parser accepts";
  EXPECT_TRUE(MaskHas(mask, 3)) << "'b' matches the [a-b] alternative: parser accepts";
  EXPECT_TRUE(MaskHas(mask, 4)) << "'ab' matches [a-b][a-b]: parser accepts";
  EXPECT_TRUE(MaskHas(mask, 1)) << "lo = 0: 'Z' can close with zero body chars";
}

// Semantic machinery: budget / capture / lazy / temperature rules activate
// machinery the fast path cannot reason about, so these grammars check that
// the mask semantics stay identical to the ordinary parser.
TEST(RepeatFastPathTest, SemanticMachineryPreservesMaskSemantics) {
  // --- max_tokens (matcher-wide budget machinery) ---
  {
    GrammarBuilder b;
    const int32_t body = b.AddRule(
        "body", b.AddChoices({b.AddSequence({b.AddCharacterClass({{'a', 'z'}}, false)})})
    );
    const int32_t rep = b.AddRepeat(body, 1, 1000);
    b.AddRule(
        "root", b.AddChoices({b.AddSequence({b.AddByteString("A"), rep, b.AddByteString("Z")})})
    );
    b.UpdateMaxTokens(body, 5);
    std::vector<std::string> vocab;
    const int a = AddToken(&vocab, "A");
    const int a200 = AddToken(&vocab, std::string(200, 'a'));
    const int z = AddToken(&vocab, "Z");
    auto matcher = MakeMatcher(b.Get("root"), vocab);
    std::vector<int32_t> mask;
    FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
    ASSERT_TRUE(matcher->AcceptToken(a));
    FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
    EXPECT_TRUE(MaskHas(mask, a200));
    (void)z;  // lo = 1: "Z" cannot close before a body char
  }
  // --- capture (matcher-wide capture tracking) ---
  // The capture is attached to a separate referenced rule so matcher-wide
  // capture tracking is exercised without changing the repeat-body shape.
  {
    GrammarBuilder b;
    const int32_t body = b.AddRule(
        "body", b.AddChoices({b.AddSequence({b.AddCharacterClass({{'a', 'z'}}, false)})})
    );
    const int32_t cap =
        b.AddRule("cap", b.AddChoices({b.AddSequence({b.AddCharacterClass({{'a', 'z'}}, false)})}));
    const int32_t rep = b.AddRepeat(body, 1, 1000);
    b.AddRule(
        "root",
        b.AddChoices(
            {b.AddSequence({b.AddByteString("A"), rep, b.AddByteString("Z")}),
             b.AddSequence({b.AddByteString("Q"), b.AddRuleRef(cap), b.AddByteString("Q")})}
        )
    );
    b.UpdateCaptureName(cap, "cap");
    std::vector<std::string> vocab;
    const int a = AddToken(&vocab, "A");
    const int a200 = AddToken(&vocab, std::string(200, 'a'));
    const int z = AddToken(&vocab, "Z");
    auto matcher = MakeMatcher(b.Get("root"), vocab);
    std::vector<int32_t> mask;
    FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
    ASSERT_TRUE(matcher->AcceptToken(a));
    FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
    EXPECT_TRUE(MaskHas(mask, a200));
    (void)z;  // lo = 1: "Z" cannot close before a body char
  }
  // --- lazy body rule (child rule metadata) ---
  {
    GrammarBuilder b;
    const int32_t body = b.AddRule(
        "body", b.AddChoices({b.AddSequence({b.AddCharacterClass({{'a', 'z'}}, false)})})
    );
    const int32_t rep = b.AddRepeat(body, 1, 1000);
    b.AddRule(
        "root", b.AddChoices({b.AddSequence({b.AddByteString("A"), rep, b.AddByteString("Z")})})
    );
    b.UpdateLazy(body, true);
    std::vector<std::string> vocab;
    const int a = AddToken(&vocab, "A");
    const int a200 = AddToken(&vocab, std::string(200, 'a'));
    const int z = AddToken(&vocab, "Z");
    auto matcher = MakeMatcher(b.Get("root"), vocab);
    std::vector<int32_t> mask;
    FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
    ASSERT_TRUE(matcher->AcceptToken(a));
    FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
    EXPECT_TRUE(MaskHas(mask, a200)) << "lazy rules keep their ordinary mask semantics";
    (void)z;  // lo = 1: "Z" cannot close before a body char
  }
  // --- temperature on the body rule (child rule metadata) ---
  {
    GrammarBuilder b;
    const int32_t body = b.AddRule(
        "body", b.AddChoices({b.AddSequence({b.AddCharacterClass({{'a', 'z'}}, false)})})
    );
    const int32_t rep = b.AddRepeat(body, 1, 1000);
    b.AddRule(
        "root", b.AddChoices({b.AddSequence({b.AddByteString("A"), rep, b.AddByteString("Z")})})
    );
    b.UpdateRuleTemperature(body, 1.0f);
    std::vector<std::string> vocab;
    const int a = AddToken(&vocab, "A");
    const int a200 = AddToken(&vocab, std::string(200, 'a'));
    const int z = AddToken(&vocab, "Z");
    auto matcher = MakeMatcher(b.Get("root"), vocab);
    std::vector<int32_t> mask;
    FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
    ASSERT_TRUE(matcher->AcceptToken(a));
    FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
    EXPECT_TRUE(MaskHas(mask, a200));
    (void)z;  // lo = 1: "Z" cannot close before a body char
  }
}

// Mask union: two live root seeds after "A" (the repeat branch and the
// literal "B" branch) plus the predicted body-restart seed. The final mask is
// the UNION of the accepted token sets, so an acceptance by one seed must
// survive another seed rejecting the same token.
TEST(RepeatFastPathTest, MaskStorageAggregation) {
  GrammarBuilder b;
  const int32_t body =
      b.AddRule("body", b.AddChoices({b.AddSequence({b.AddCharacterClass({{'a', 'c'}}, false)})}));
  const int32_t rep = b.AddRepeat(body, 1, 1000);
  b.AddRule(
      "root",
      b.AddChoices(
          {b.AddSequence({b.AddByteString("A"), rep, b.AddByteString("Z")}),
           b.AddSequence({b.AddByteString("A"), b.AddByteString("B")})}
      )
  );
  std::vector<std::string> vocab;
  const int a = AddToken(&vocab, "A");
  const int bb = AddToken(&vocab, "B");
  const int z = AddToken(&vocab, "Z");
  const int a200 = AddToken(&vocab, std::string(200, 'a'));
  const int a300 = AddToken(&vocab, std::string(300, 'a'));
  const int c200 = AddToken(&vocab, std::string(200, 'c'));
  auto matcher = MakeMatcher(b.Get("root"), vocab);
  std::vector<int32_t> mask;
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);
  ASSERT_TRUE(matcher->AcceptToken(a));
  FillMask(matcher.get(), static_cast<int>(vocab.size()), &mask);

  EXPECT_TRUE(MaskHas(mask, a200))
      << "union: an acceptance by one seed must survive another seed rejecting the same token";
  EXPECT_TRUE(MaskHas(mask, a300))
      << "union: an acceptance by one seed must survive another seed rejecting the same token";
  EXPECT_TRUE(MaskHas(mask, c200));
  EXPECT_TRUE(MaskHas(mask, bb)) << "the 'B' branch seed keeps its own accepted token";
  (void)z;  // lo = 1: "Z" cannot close before a body char
  // The "B" branch seed rejects the body tokens; the union must keep the body
  // seed's acceptances.
}

}  // namespace
