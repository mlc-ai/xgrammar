/*!
 * Copyright (c) 2026 by Contributors
 * \file tests/cpp/test_json_schema_string_length.cc
 * \brief JSON validity and decoded-character bounds for length-constrained strings.
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace xgrammar {
namespace {

bool MatchesString(const CompiledGrammar& grammar, const std::string& input) {
  GrammarMatcher matcher(grammar, std::nullopt, /*terminate_without_stop_token=*/true);
  return matcher.AcceptString(input) && matcher.IsTerminated();
}

class BoundedJSONStringTest : public testing::TestWithParam<bool> {
 protected:
  GrammarCompiler compiler_{TokenizerInfo(std::vector<std::string>{}), 1, false};

  CompiledGrammar Compile(const std::string& schema) {
    if (!GetParam()) {
      return compiler_.CompileJSONSchema(schema);
    }
    return compiler_.CompileStructuralTag(
        R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
        R"({"type":"regex","pattern":"[ \\t\\n\\r]*"},)"
        R"({"type":"json_schema","json_schema":)" +
        schema + R"(,"style":"json"}]}})"
    );
  }
};

TEST_P(BoundedJSONStringTest, RejectsEveryRawControlAndAcceptsItsEscape) {
  for (const std::string bounds :
       {R"("minLength":1)", R"("maxLength":1)", R"("minLength":1,"maxLength":1)"}) {
    auto grammar = Compile(R"({"type":"string",)" + bounds + "}");
    for (int control = 0; control < 32; ++control) {
      SCOPED_TRACE(bounds + " control=" + std::to_string(control));
      EXPECT_FALSE(MatchesString(grammar, "\"" + std::string(1, char(control)) + "\""));
      char escaped[9];
      std::snprintf(escaped, sizeof(escaped), "\"\\u%04x\"", control);
      EXPECT_TRUE(MatchesString(grammar, escaped));
    }
  }
}

TEST_P(BoundedJSONStringTest, CountsDecodedCodePointsAcrossLengthBoundaries) {
  // Bodies representing exactly one Unicode code point, in every JSON escape form.
  const std::vector<std::string> characters = {
      "a",
      "é",
      "中",
      "😀",
      R"(\")",
      R"(\\)",
      R"(\/)",
      R"(\b)",
      R"(\f)",
      R"(\n)",
      R"(\r)",
      R"(\t)",
      R"(\u0061)",
      R"(\u00E9)",
      R"(\uD7FF)",
      R"(\uE000)",
      R"(\uFFFF)",
      R"(\ud800\udc00)",
      R"(\uDBFF\uDFFF)",
      R"(\ud83D\uDe00)"
  };
  for (auto [minimum, maximum] :
       std::vector<std::pair<int, int>>{{0, 0}, {0, 1}, {1, 1}, {2, 2}, {1, -1}, {2, -1}, {1, 3}}) {
    std::string schema = R"({"type":"string","minLength":)" + std::to_string(minimum);
    if (maximum >= 0) {
      schema += R"(,"maxLength":)" + std::to_string(maximum);
    }
    auto grammar = Compile(schema + "}");
    for (const auto& character : characters) {
      std::string body;
      for (int length = 0; length <= 4; ++length) {
        SCOPED_TRACE(schema + " character=" + character + " length=" + std::to_string(length));
        EXPECT_EQ(
            MatchesString(grammar, "\"" + body + "\""),
            length >= minimum && (maximum < 0 || length <= maximum)
        );
        body += character;
      }
    }
  }
}

TEST_P(BoundedJSONStringTest, RejectsMalformedEscapesAndUnpairedSurrogates) {
  auto grammar = Compile(R"({"type":"string","minLength":1,"maxLength":8})");
  for (const std::string body :
       {R"(\x41)",
        R"(\v)",
        R"(\0)",
        R"(\u12)",
        R"(\uGGGG)",
        R"(\uD800)",
        R"(\udfff)",
        R"(\uDC00\uD800)",
        R"(\uD800\u0041)",
        R"(\uD800\uD800)"}) {
    SCOPED_TRACE(body);
    EXPECT_FALSE(MatchesString(grammar, "\"" + body + "\""));
  }
}

TEST_P(BoundedJSONStringTest, NestedStringsKeepIndependentBounds) {
  auto grammar = Compile(R"({"type":"object","properties":{
    "values":{"type":"array","items":{"type":"string","minLength":1,"maxLength":1}}
  },"required":["values"],"additionalProperties":false})");
  EXPECT_TRUE(MatchesString(grammar, R"({"values":["\t","\uD83D\uDE00","é"]})"));
  EXPECT_FALSE(MatchesString(grammar, R"({"values":["\t",""]})"));
  EXPECT_FALSE(MatchesString(grammar, R"({"values":["\t","\uD83D\uDE00a"]})"));
}

INSTANTIATE_TEST_SUITE_P(JSONSchemaAndStructuralTag, BoundedJSONStringTest, testing::Bool());

TEST(BoundedJSONStringMaskTest, MasksWholeAndSplitEscapesAtCharacterBoundary) {
  std::vector<std::string> vocab;
  for (int byte = 0; byte < 128; ++byte) {
    vocab.emplace_back(1, char(byte));
  }
  vocab.insert(vocab.end(), {R"(\uD83D)", R"(\uDE00)", R"(\t)", "😀", "<eos>"});
  const int eos = int(vocab.size()) - 1;
  GrammarCompiler compiler(
      TokenizerInfo(vocab, VocabType::RAW, int(vocab.size()), std::vector<int32_t>{eos}), 1, false
  );
  auto grammar = compiler.CompileJSONSchema(R"({"type":"string","minLength":1,"maxLength":1})");
  GrammarMatcher matcher(grammar);
  std::vector<int32_t> mask((vocab.size() + 31) / 32);
  int64_t shape[2] = {1, int64_t(mask.size())};
  DLTensor tensor{mask.data(), {kDLCPU, 0}, 2, {kDLInt, 32, 1}, shape, nullptr, 0};
  auto allowed = [&](int token) { return (uint32_t(mask[token / 32]) >> (token % 32)) & 1U; };

  ASSERT_TRUE(matcher.AcceptToken('"'));
  matcher.FillNextTokenBitmask(&tensor);
  for (int control = 0; control < 32; ++control) {
    EXPECT_FALSE(allowed(control)) << control;
  }
  EXPECT_TRUE(allowed('\\'));
  EXPECT_TRUE(allowed(128));   // High surrogate prefix: must wait for its low surrogate.
  EXPECT_FALSE(allowed(129));  // Lone low surrogate.
  EXPECT_TRUE(allowed(130));   // Complete escaped tab.
  EXPECT_TRUE(allowed(131));   // Complete UTF-8 code point.
  EXPECT_FALSE(allowed('"'));

  ASSERT_TRUE(matcher.AcceptToken(128));
  matcher.FillNextTokenBitmask(&tensor);
  EXPECT_TRUE(allowed(129));
  EXPECT_FALSE(allowed('a'));
  EXPECT_FALSE(allowed('"'));
  ASSERT_TRUE(matcher.AcceptToken(129));
  matcher.FillNextTokenBitmask(&tensor);
  EXPECT_TRUE(allowed('"'));
  EXPECT_FALSE(allowed('a'));
  EXPECT_FALSE(allowed(130));
  EXPECT_FALSE(allowed(131));
  ASSERT_TRUE(matcher.AcceptToken('"'));
  ASSERT_TRUE(matcher.AcceptToken(eos));
  EXPECT_TRUE(matcher.IsTerminated());
}

}  // namespace
}  // namespace xgrammar
