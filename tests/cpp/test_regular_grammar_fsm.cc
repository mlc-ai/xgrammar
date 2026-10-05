/*!
 * \file tests/cpp/test_regular_grammar_fsm.cc
 * \brief Preserve EBNF terminal and call semantics when flattening regular grammars.
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include "grammar_functor.h"

using namespace xgrammar;

TEST(RegularGrammarFSM, RepeatedCallsHaveSeparateReturns) {
  auto grammar = Grammar::FromEBNF(R"(
root ::= part "x" part "y"
part ::= "a" part | "b"
)");
  auto result = GrammarFSMBuilder::FromRegularGrammar(grammar);
  ASSERT_TRUE(result.IsOk());
  auto fsm = std::move(result).Unwrap();
  for (const auto* text : {"bxby", "abxby", "bxaaby", "aaabxaaaby"}) {
    EXPECT_TRUE(fsm.AcceptString(text)) << text;
  }
  for (const auto* text : {"", "by", "bxy", "bxay", "abxay", "bybx", "bxbyy"}) {
    EXPECT_FALSE(fsm.AcceptString(text)) << text;
  }
}

TEST(RegularGrammarFSM, RecursiveCalleeCannotUseCallersOptionalExit) {
  auto grammar = Grammar::FromEBNF(R"(
root ::= part? "z" part*
part ::= "a" part | "b"
)");
  auto result = GrammarFSMBuilder::FromRegularGrammar(grammar);
  ASSERT_TRUE(result.IsOk());
  auto fsm = std::move(result).Unwrap();
  for (const auto* text : {"z", "bz", "abz", "zb", "zaab", "bzab", "abzbaab"}) {
    EXPECT_TRUE(fsm.AcceptString(text)) << text;
  }
  for (const auto* text : {"az", "za", "bza", "aaz", "zba"}) {
    EXPECT_FALSE(fsm.AcceptString(text)) << text;
  }
}

TEST(RegularGrammarFSM, RegexFallbackMatchesOriginalGrammar) {
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);
  const std::vector<std::string> patterns = {
      R"((ab|c){1,3}d?)",
      R"((a*b)+c{2,})",
      R"((ab|){0,3}z)",
      R"([é你]+|\u0061\s\D)",
      R"([^你])",
      R"(^[a-zA-Z0-9\.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$)"
  };
  const std::vector<std::string> samples = {
      "",
      "ab",
      "abc",
      "abccd",
      "d",
      "cccc",
      "bcc",
      "aabbccc",
      "abcc",
      "acccc",
      "z",
      "abz",
      "ababz",
      "abababz",
      "ababababz",
      "é你é",
      "你",
      "a b",
      "a 1",
      "a\tb",
      "name@example.com",
      "name@example",
      "name@.com",
      "<|close|>"
  };
  for (const auto& pattern : patterns) {
    auto grammar = Grammar::FromRegex(pattern);
    auto result = GrammarFSMBuilder::FromRegularGrammar(grammar);
    ASSERT_TRUE(result.IsOk()) << pattern;
    auto fsm = std::move(result).Unwrap();
    GrammarMatcher matcher(compiler.CompileGrammar(grammar), std::nullopt, true);
    for (const auto& sample : samples) {
      matcher.Reset();
      bool expected = matcher.AcceptString(sample) && matcher.IsTerminated();
      EXPECT_EQ(fsm.AcceptString(sample), expected) << pattern << " input=" << sample;
    }
  }
}

TEST(RegularGrammarFSM, RejectsUnsupportedRecursionAndBounds) {
  for (const auto* source :
       {"root ::= \"a\" root \"b\" | \"\"",
        "root ::= \"a\" child | \"\"\nchild ::= \"b\" root | \"c\"",
        "root ::= Token(1)"}) {
    EXPECT_TRUE(GrammarFSMBuilder::FromRegularGrammar(Grammar::FromEBNF(source)).IsErr()) << source;
  }
  auto grammar = Grammar::FromRegex("(ab){100}");
  EXPECT_TRUE(GrammarFSMBuilder::FromRegularGrammar(grammar, 10).IsErr());
}
