/*!
 * Copyright (c) 2026 by Contributors
 * \file tests/cpp/test_grammar_compiler_structural_cache.cc
 * \brief Preserve bounded-pattern and cyclic-rule behavior with structural caching enabled.
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fsm_builder.h"
#include "grammar_builder.h"

namespace xgrammar {
namespace {

constexpr int64_t kCacheBytes = 128 * 1024 * 1024;

const std::vector<std::string> kVocab = {
    "\"", "-", " ", "a", "b", "c", "\\", "n", "(", ")", "x", "y", "aa", "bb", "- ", "ab", "<eos>"
};
constexpr int32_t kStopToken = 16;

TokenizerInfo TestTokenizerInfo() {
  return TokenizerInfo(kVocab, VocabType::RAW, kVocab.size(), std::vector<int32_t>{kStopToken});
}

std::pair<bool, std::vector<int32_t>> NextTokenMask(GrammarMatcher* matcher) {
  int64_t shape = GetBitmaskSize(kVocab.size());
  std::vector<int32_t> mask(shape, -1);
  DLTensor tensor{};
  tensor.data = mask.data();
  tensor.device = DLDevice{kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = GetBitmaskDLType();
  tensor.shape = &shape;
  bool needs_apply = matcher->FillNextTokenBitmask(&tensor);
  return {needs_apply, std::move(mask)};
}

bool MatchesEntireString(const CompiledGrammar& grammar, const std::string& input) {
  GrammarMatcher matcher(grammar, std::nullopt, /*terminate_without_stop_token=*/true);
  return matcher.AcceptString(input) && matcher.IsTerminated();
}

void ExpectSameTokenTrace(
    const CompiledGrammar& cached,
    const CompiledGrammar& uncached,
    const std::string& input,
    bool expected_acceptance
) {
  SCOPED_TRACE(input);
  GrammarMatcher cached_matcher(cached);
  GrammarMatcher uncached_matcher(uncached);
  for (size_t index = 0; index < input.size(); ++index) {
    SCOPED_TRACE(index);
    auto cached_mask = NextTokenMask(&cached_matcher);
    auto uncached_mask = NextTokenMask(&uncached_matcher);
    EXPECT_EQ(cached_mask.first, uncached_mask.first);
    EXPECT_EQ(cached_mask.second, uncached_mask.second);

    auto token = std::find(kVocab.begin(), kVocab.end(), std::string(1, input[index]));
    ASSERT_NE(token, kVocab.end());
    int32_t token_id = static_cast<int32_t>(token - kVocab.begin());
    bool cached_accepted = cached_matcher.AcceptToken(token_id);
    bool uncached_accepted = uncached_matcher.AcceptToken(token_id);
    ASSERT_EQ(cached_accepted, uncached_accepted);
    EXPECT_EQ(cached_matcher.IsCompleted(), uncached_matcher.IsCompleted());
    if (!cached_accepted) {
      EXPECT_FALSE(expected_acceptance);
      return;
    }
  }

  EXPECT_EQ(cached_matcher.IsCompleted(), expected_acceptance);
  EXPECT_EQ(uncached_matcher.IsCompleted(), expected_acceptance);
  auto cached_mask = NextTokenMask(&cached_matcher);
  auto uncached_mask = NextTokenMask(&uncached_matcher);
  EXPECT_EQ(cached_mask.first, uncached_mask.first);
  EXPECT_EQ(cached_mask.second, uncached_mask.second);
  if (cached_matcher.IsCompleted() && uncached_matcher.IsCompleted()) {
    EXPECT_TRUE(cached_matcher.AcceptToken(kStopToken));
    EXPECT_TRUE(uncached_matcher.AcceptToken(kStopToken));
    EXPECT_TRUE(cached_matcher.IsTerminated());
    EXPECT_TRUE(uncached_matcher.IsTerminated());
  }
}

TEST(GrammarCompilerStructuralCache, BoundedPatternSchemaCompilesAndMatchesBoundaries) {
  // A large bounded regex introduces referenced rules during schema lowering. Enabling
  // structural caching must not turn an otherwise valid schema into a compiler error.
  constexpr const char* kSchema = R"({
    "type": "string",
    "pattern": "^- [^\\r\\n]{1,900}$",
    "maxLength": 902
  })";
  GrammarCompiler uncached_compiler(TestTokenizerInfo(), 1, false);
  auto uncached = uncached_compiler.CompileJSONSchema(kSchema);
  GrammarCompiler cached_compiler(TestTokenizerInfo(), 1, true, kCacheBytes);
  auto cached = cached_compiler.CompileJSONSchema(kSchema);

  const std::vector<std::pair<std::string, bool>> cases = {
      {R"("- a")", true},
      {"\"- " + std::string(900, 'a') + "\"", true},
      {R"("- ")", false},
      {"\"- " + std::string(901, 'a') + "\"", false},
      {R"("a")", false},
      {R"("- a\nb")", false},
      {R"("- a"b")", false},
      {R"("- a\b")", false}
  };
  for (const auto& [input, expected] : cases) {
    EXPECT_EQ(MatchesEntireString(uncached, input), expected) << input;
    EXPECT_EQ(MatchesEntireString(cached, input), expected) << input;
  }
}

TEST(GrammarCompilerStructuralCache, SequentialBoundedSchemasMatchUncachedTokenMasks) {
  // Warm one compiler with different minimum and maximum bounds, then exercise both
  // schemas again. A reused mask must reflect the active schema, not the previous one.
  constexpr const char* kFirstSchema = R"({
    "type": "string", "pattern": "^- [ab]{1,129}$", "maxLength": 131
  })";
  constexpr const char* kSecondSchema = R"({
    "type": "string", "pattern": "^- [ab]{2,130}$", "maxLength": 132
  })";
  GrammarCompiler cached_compiler(TestTokenizerInfo(), 1, true, kCacheBytes);
  GrammarCompiler uncached_compiler(TestTokenizerInfo(), 1, false);

  auto first_uncached = uncached_compiler.CompileJSONSchema(kFirstSchema);
  auto first_cached = cached_compiler.CompileJSONSchema(kFirstSchema);
  auto second_uncached = uncached_compiler.CompileJSONSchema(kSecondSchema);
  auto second_cached = cached_compiler.CompileJSONSchema(kSecondSchema);
  auto first_cached_again = cached_compiler.CompileJSONSchema(kFirstSchema);

  const std::vector<std::pair<std::string, bool>> first_cases = {
      {R"("- a")", true},
      {"\"- " + std::string(129, 'a') + "\"", true},
      {R"("- ")", false},
      {"\"- " + std::string(130, 'a') + "\"", false},
      {R"("- ac")", false}
  };
  for (const auto& [input, expected] : first_cases) {
    ExpectSameTokenTrace(first_cached, first_uncached, input, expected);
    ExpectSameTokenTrace(first_cached_again, first_uncached, input, expected);
  }

  const std::vector<std::pair<std::string, bool>> second_cases = {
      {R"("- ab")", true},
      {"\"- " + std::string(130, 'b') + "\"", true},
      {R"("- a")", false},
      {"\"- " + std::string(131, 'b') + "\"", false},
      {R"("- ac")", false}
  };
  for (const auto& [input, expected] : second_cases) {
    ExpectSameTokenTrace(second_cached, second_uncached, input, expected);
  }
}

TEST(GrammarCompilerStructuralCache, CyclicRuleMatchesUncachedTokenMasks) {
  // Non-regular recursion has no complete reusable FSM hash. Its unknown reference
  // must retain native matching behavior without a crash or stale cross-grammar masks.
  constexpr const char* kFirstGrammar = R"EBNF(
root ::= "(" node ")"
node ::= "a" node "b" | "x"
)EBNF";
  constexpr const char* kSecondGrammar = R"EBNF(
root ::= "(" node ")"
node ::= "a" node "b" | "y"
)EBNF";
  GrammarCompiler cached_compiler(TestTokenizerInfo(), 1, true, kCacheBytes);
  GrammarCompiler uncached_compiler(TestTokenizerInfo(), 1, false);
  auto first_uncached = uncached_compiler.CompileGrammar(kFirstGrammar);
  auto first_cached = cached_compiler.CompileGrammar(kFirstGrammar);
  auto second_uncached = uncached_compiler.CompileGrammar(kSecondGrammar);
  auto second_cached = cached_compiler.CompileGrammar(kSecondGrammar);

  for (const auto* input : {"(x)", "(axb)", "(aaxbb)"}) {
    ExpectSameTokenTrace(first_cached, first_uncached, input, true);
  }
  for (const auto* input : {"(y)", "(ayb)", "(aaybb)"}) {
    ExpectSameTokenTrace(second_cached, second_uncached, input, true);
  }
  for (const auto* input : {"(x)", "(aybb)", "(aayb)", "(ab)", "("}) {
    ExpectSameTokenTrace(second_cached, second_uncached, input, false);
  }
}

TEST(GrammarCompilerStructuralCache, UnfilteredBoundedRegexMatchesUncachedTokenMasks) {
  GrammarCompiler cached_compiler(TestTokenizerInfo(), 1, true, kCacheBytes);
  GrammarCompiler uncached_compiler(TestTokenizerInfo(), 1, false);
  auto first_uncached = uncached_compiler.CompileRegex("^[ab]{1,129}$");
  auto first_cached = cached_compiler.CompileRegex("^[ab]{1,129}$");
  auto second_uncached = uncached_compiler.CompileRegex("^[ab]{2,130}$");
  auto second_cached = cached_compiler.CompileRegex("^[ab]{2,130}$");

  for (const auto& [input, expected] : std::vector<std::pair<std::string, bool>>{
           {"a", true},
           {std::string(129, 'a'), true},
           {"", false},
           {std::string(130, 'a'), false},
           {"ac", false}
       }) {
    ExpectSameTokenTrace(first_cached, first_uncached, input, expected);
  }
  for (const auto& [input, expected] : std::vector<std::pair<std::string, bool>>{
           {"ab", true},
           {std::string(130, 'b'), true},
           {"a", false},
           {std::string(131, 'b'), false},
           {"ac", false}
       }) {
    ExpectSameTokenTrace(second_cached, second_uncached, input, expected);
  }
}

TEST(GrammarCompilerStructuralCache, ArbitraryForbiddenBytesConstrainCountedSubrules) {
  std::bitset<256> forbidden;
  forbidden.set('b');
  auto fsm = RegexFSMBuilder::BuildWithForbiddenChars("^[abc]{1,130}$", forbidden).Unwrap();
  EXPECT_TRUE(fsm.AcceptString("a"));
  EXPECT_TRUE(fsm.AcceptString(std::string(130, 'c')));
  EXPECT_FALSE(fsm.AcceptString(""));
  EXPECT_FALSE(fsm.AcceptString(std::string(131, 'a')));
  EXPECT_FALSE(fsm.AcceptString("ab"));

  GrammarBuilder builder;
  auto rule_count = builder.NumRules();
  auto built =
      RegexFSMBuilder::BuildWithForbiddenChars("^[abc]{1,130}$", forbidden, &builder, "filtered");
  ASSERT_TRUE(built.IsOk());
  auto with_builder = std::move(built).Unwrap();
  EXPECT_TRUE(with_builder.AcceptString("ac"));
  EXPECT_FALSE(with_builder.AcceptString("ab"));
  EXPECT_EQ(builder.NumRules(), rule_count);
  EXPECT_TRUE(
      RegexFSMBuilder::BuildWithForbiddenChars("^[abc]{1,100001}$", forbidden, &builder, "filtered")
          .IsErr()
  );
}

}  // namespace
}  // namespace xgrammar
