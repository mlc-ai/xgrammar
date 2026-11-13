/**
 * \file tests/cpp/test_structural_tag.cc
 * \brief Test structural tag conversion and parser state growth.
 */

#include <gtest/gtest.h>
#include <xgrammar/grammar.h>

#include <string>
#include <variant>

#include "earley_parser.h"
#include "grammar_functor.h"

namespace xgrammar {
namespace {

class StructuralTagPrefixTest : public ::testing::TestWithParam<int> {};

TEST_P(StructuralTagPrefixTest, SharedPrefixBoundsPlainTextStates) {
  const int num_tags = GetParam();
  std::string tags;
  for (int i = 0; i < num_tags; ++i) {
    if (i != 0) tags += ",";
    tags += R"({"begin":"<tool_call>tool_)" + std::to_string(i) +
            R"(","content":{"type":"json_schema","json_schema":{"type":"object",
                 "properties":{"value":{"type":"string"}},"required":["value"],
                 "additionalProperties":false}},"end":"</tool_call>"})";
  }
  auto result = Grammar::FromStructuralTag(
      R"({"type":"structural_tag","format":{"type":"sequence","elements":[
           {"type":"any_text"},{"type":"tags_with_separator","separator":"","tags":[)" +
      tags + "]}]}}"
  );
  ASSERT_TRUE(std::holds_alternative<Grammar>(result));
  auto grammar = GrammarOptimizer::Apply(std::get<Grammar>(result));
  EarleyParser parser(grammar);
  // One free-text state and one shared-prefix state, independent of the tool count.
  EXPECT_EQ(parser.GetLatestScanableStates().size(), 2);
  for (unsigned char ch : std::string("hello world")) {
    ASSERT_TRUE(parser.Advance(ch));
    EXPECT_EQ(parser.GetLatestScanableStates().size(), 2);
  }
  for (unsigned char ch : std::string("<tool_call>tool_0{\"value\":\"x\"}</tool_call>")) {
    ASSERT_TRUE(parser.Advance(ch));
  }
  EXPECT_TRUE(parser.IsCompleted());
}

INSTANTIATE_TEST_SUITE_P(ToolCounts, StructuralTagPrefixTest, ::testing::Values(8, 32, 128, 318));

}  // namespace
}  // namespace xgrammar
