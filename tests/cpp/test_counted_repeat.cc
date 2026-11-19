/*!
 *  Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 * \file tests/cpp/test_counted_repeat.cc
 * \brief Counted repetitions retain their completion frame across ordinary tail calls.
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <optional>
#include <string>
#include <vector>

#include "grammar_builder.h"
#include "grammar_impl.h"

namespace xgrammar {
namespace {

TokenizerInfo SmallTokenizer() {
  std::vector<std::string> vocab;
  for (int ch = 32; ch < 127; ++ch) {
    vocab.emplace_back(1, static_cast<char>(ch));
  }
  vocab.emplace_back("<eos>");
  return TokenizerInfo(
      vocab,
      VocabType::RAW,
      std::nullopt,
      std::vector<int32_t>{static_cast<int32_t>(vocab.size() - 1)}
  );
}

bool AcceptsComplete(const CompiledGrammar& grammar, const std::string& input) {
  GrammarMatcher matcher(grammar, std::nullopt, true);
  return matcher.AcceptString(input) && matcher.IsTerminated();
}

std::string Repeat(const std::string& value, int count) {
  std::string result;
  for (int i = 0; i < count; ++i) {
    result += value;
  }
  return result;
}

bool HasCountedRepeat(const CompiledGrammar& compiled) {
  const auto& fsm = compiled.GetGrammar()->complete_fsm;
  for (int state = 0; state < fsm.NumStates(); ++state) {
    for (const auto& edge : fsm.GetEdges(state)) {
      if (edge.IsRepeatRef()) {
        return true;
      }
    }
  }
  return false;
}

TEST(XGrammarCountedRepeatTest, LiteralAndNonterminalTailBounds) {
  for (bool cache_enabled : {false, true}) {
    GrammarCompiler compiler(SmallTokenizer(), 1, cache_enabled);
    for (int count : {2, 128, 129, 255, 512}) {
      SCOPED_TRACE(count);
      SCOPED_TRACE(cache_enabled);
      for (const auto& body :
           {std::string("\",\" \"0\""), std::string("\",\" tail"), std::string("tail \",\"")}) {
        SCOPED_TRACE(body);
        auto compiled = compiler.CompileGrammar(
            "root ::= (" + body + "){" + std::to_string(count) + "}\ntail ::= \"0\"\n"
        );
        if (count > 128) {
          ASSERT_TRUE(HasCountedRepeat(compiled));
        }
        const std::string atom = body.front() == 't' ? "0," : ",0";
        EXPECT_TRUE(AcceptsComplete(compiled, Repeat(atom, count)));
        EXPECT_FALSE(AcceptsComplete(compiled, Repeat(atom, count - 1)));
        EXPECT_FALSE(AcceptsComplete(compiled, Repeat(atom, count + 1)));
      }
    }
  }
}

TEST(XGrammarCountedRepeatTest, RangeAndNullableTail) {
  for (bool cache_enabled : {false, true}) {
    GrammarCompiler compiler(SmallTokenizer(), 1, cache_enabled);
    auto range = compiler.CompileGrammar("root ::= (\",\" tail){129,131}\ntail ::= \"0\"\n");
    ASSERT_TRUE(HasCountedRepeat(range));
    for (int count : {0, 127, 128, 129, 130, 131, 132, 512}) {
      SCOPED_TRACE(count);
      EXPECT_EQ(AcceptsComplete(range, Repeat(",0", count)), count >= 129 && count <= 131);
    }
    auto nullable = compiler.CompileGrammar("root ::= (\",\" tail){129}\ntail ::= \"\" | \"0\"\n");
    ASSERT_TRUE(HasCountedRepeat(nullable));
    for (int count : {128, 129, 130}) {
      SCOPED_TRACE(count);
      EXPECT_EQ(AcceptsComplete(nullable, Repeat(",0", count)), count == 129);
      EXPECT_EQ(AcceptsComplete(nullable, Repeat(",", count)), count == 129);
    }
    auto maximum = compiler.CompileGrammar("root ::= (\",\" tail){0,512}\ntail ::= \"0\"\n");
    ASSERT_TRUE(HasCountedRepeat(maximum));
    for (int count : {0, 1, 127, 128, 129, 511, 512, 513, 1739}) {
      SCOPED_TRACE(count);
      EXPECT_EQ(AcceptsComplete(maximum, Repeat(",0", count)), count <= 512);
    }
  }
}

TEST(XGrammarCountedRepeatTest, NestedTailAndOrdinaryCaller) {
  GrammarCompiler compiler(SmallTokenizer(), 1, false);
  auto nested = compiler.CompileGrammar(
      "root ::= \"x\" middle \"|\" (\"a\" middle){129}\n"
      "middle ::= \"(\" tail\n"
      "tail ::= \"b\" | \"bc\"\n"
  );
  ASSERT_TRUE(HasCountedRepeat(nested));
  EXPECT_TRUE(AcceptsComplete(nested, "x(b|" + Repeat("a(b", 129)));
  EXPECT_TRUE(AcceptsComplete(nested, "x(bc|" + Repeat("a(bc", 129)));
  EXPECT_FALSE(AcceptsComplete(nested, "x(b|" + Repeat("a(b", 128)));
  EXPECT_FALSE(AcceptsComplete(nested, "x(b|" + Repeat("a(b", 130)));
}

TEST(XGrammarCountedRepeatTest, ContextFreeTailBounds) {
  for (bool cache_enabled : {false, true}) {
    GrammarCompiler compiler(SmallTokenizer(), 1, cache_enabled);
    auto compiled = compiler.CompileGrammar(
        "root ::= (\",\" tail){129}\n"
        "tail ::= \"0\" | \"(\" tail \")\"\n"
    );
    ASSERT_TRUE(HasCountedRepeat(compiled));
    for (const auto& atom : {std::string(",0"), std::string(",((0))")}) {
      EXPECT_TRUE(AcceptsComplete(compiled, Repeat(atom, 129)));
      EXPECT_FALSE(AcceptsComplete(compiled, Repeat(atom, 128)));
      EXPECT_FALSE(AcceptsComplete(compiled, Repeat(atom, 130)));
    }
  }
}

TEST(XGrammarCountedRepeatTest, SimultaneousOrdinaryAndCountedParents) {
  GrammarBuilder builder;
  const auto root = builder.AddEmptyRule("root");
  const auto unit = builder.AddEmptyRule("unit");
  const auto repeated = builder.AddEmptyRule("repeated");
  const auto tail = builder.AddEmptyRule("tail");
  builder.UpdateRuleBody(
      tail, builder.AddChoices({builder.AddSequence({builder.AddByteString("0")})})
  );
  builder.UpdateRuleBody(
      unit,
      builder.AddChoices({builder.AddSequence({builder.AddByteString(","), builder.AddRuleRef(tail)}
      )})
  );
  builder.UpdateRuleBody(
      repeated, builder.AddChoices({builder.AddSequence({builder.AddRepeat(unit, 129, 129)})})
  );
  builder.UpdateRuleBody(
      root,
      builder.AddChoices({
          builder.AddSequence(
              {builder.AddByteString("x"), builder.AddRuleRef(repeated), builder.AddByteString("!")}
          ),
          builder.AddSequence(
              {builder.AddByteString("x"), builder.AddRuleRef(unit), builder.AddByteString("?")}
          ),
      })
  );
  GrammarCompiler compiler(SmallTokenizer(), 1, false);
  auto compiled = compiler.CompileGrammar(builder.Get(root));
  ASSERT_TRUE(HasCountedRepeat(compiled));
  EXPECT_TRUE(AcceptsComplete(compiled, "x" + Repeat(",0", 129) + "!"));
  EXPECT_TRUE(AcceptsComplete(compiled, "x,0?"));
  EXPECT_FALSE(AcceptsComplete(compiled, "x" + Repeat(",0", 128) + "!"));
  EXPECT_FALSE(AcceptsComplete(compiled, "x" + Repeat(",0", 130) + "!"));
  EXPECT_FALSE(AcceptsComplete(compiled, "x,0,0?"));
}

TEST(XGrammarCountedRepeatTest, FullyNullableBody) {
  GrammarCompiler compiler(SmallTokenizer(), 1, false);
  auto compiled =
      compiler.CompileGrammar("root ::= \"[\" (\"a\"? tail){129} \"]\"\ntail ::= \"\" | \"0\"\n");
  ASSERT_TRUE(HasCountedRepeat(compiled));
  EXPECT_TRUE(AcceptsComplete(compiled, "[]"));
  for (const auto& atom : {std::string("a"), std::string("0"), std::string("a0")}) {
    EXPECT_TRUE(AcceptsComplete(compiled, "[" + Repeat(atom, 129) + "]"));
    EXPECT_FALSE(AcceptsComplete(compiled, "[" + Repeat(atom, 130) + "]"));
  }
}

TEST(XGrammarCountedRepeatTest, NestedCountedBodies) {
  GrammarCompiler compiler(SmallTokenizer(), 1, false);
  auto compiled = compiler.CompileGrammar(
      "root ::= \"[\" (\"a\" inner){129} \"]\"\n"
      "inner ::= \"(\" (\"b\" tail){129} \")\"\n"
      "tail ::= \"0\" | \"01\"\n"
  );
  ASSERT_TRUE(HasCountedRepeat(compiled));
  const std::string atom = "a(" + Repeat("b0", 129) + ")";
  EXPECT_TRUE(AcceptsComplete(compiled, "[" + Repeat(atom, 129) + "]"));
  EXPECT_FALSE(AcceptsComplete(compiled, "[" + Repeat(atom, 128) + "]"));
  EXPECT_FALSE(AcceptsComplete(compiled, "[" + Repeat(atom, 130) + "]"));
  EXPECT_FALSE(AcceptsComplete(compiled, "[a(" + Repeat("b0", 128) + ")" + Repeat(atom, 128) + "]")
  );
}

TEST(XGrammarCountedRepeatTest, RollbackAndRejectedInputAreTransactional) {
  GrammarCompiler compiler(SmallTokenizer(), 1, false);
  auto compiled =
      compiler.CompileGrammar("root ::= \"[\" (\",\" tail){129} \"]\"\ntail ::= \"0\"\n");
  ASSERT_TRUE(HasCountedRepeat(compiled));
  GrammarMatcher matcher(compiled, std::nullopt, true);
  ASSERT_TRUE(matcher.AcceptString("[" + Repeat(",0", 128)));
  ASSERT_TRUE(matcher.AcceptString(",0]"));
  ASSERT_TRUE(matcher.IsTerminated());
  matcher.Rollback(1);
  ASSERT_FALSE(matcher.IsTerminated());
  EXPECT_FALSE(matcher.AcceptString(",0,0]"));
  ASSERT_TRUE(matcher.AcceptString(",0]"));
  ASSERT_TRUE(matcher.IsTerminated());
  matcher.Rollback(1);
  ASSERT_TRUE(matcher.AcceptString(",0"));
  ASSERT_FALSE(matcher.IsTerminated());
  ASSERT_TRUE(matcher.AcceptString("]"));
  ASSERT_TRUE(matcher.IsTerminated());
  matcher.Reset();
  EXPECT_TRUE(matcher.AcceptString("[" + Repeat(",0", 129) + "]"));
  EXPECT_TRUE(matcher.IsTerminated());
}

TEST(XGrammarCountedRepeatTest, LargeJSONSchemaArrayBounds) {
  for (bool cache_enabled : {false, true}) {
    GrammarCompiler compiler(SmallTokenizer(), 1, cache_enabled);
    for (bool exact : {false, true}) {
      const std::string minimum = exact ? "\"minItems\":512," : "";
      auto compiled = compiler.CompileJSONSchema(
          "{\"type\":\"object\",\"properties\":{\"values\":{\"type\":\"array\"," + minimum +
          "\"maxItems\":512,\"items\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":999999999}}},"
          "\"required\":[\"values\"],\"additionalProperties\":false}"
      );
      ASSERT_TRUE(HasCountedRepeat(compiled));
      for (int count : {0, 1, 127, 128, 129, 511, 512, 513, 1739}) {
        SCOPED_TRACE(count);
        const auto values = count == 0 ? "" : "0" + Repeat(",0", count - 1);
        const auto input = "{\"values\":[" + values + "]}";
        EXPECT_EQ(AcceptsComplete(compiled, input), exact ? count == 512 : count <= 512);
      }
    }
  }
}

}  // namespace
}  // namespace xgrammar
