#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "earley_parser.h"
#include "grammar_functor.h"

TEST(RepeatDetectorTest, PreservesStatesAcrossSetTransitionAndResetsCopies) {
  xgrammar::RepeatDetector detector(2);
  const xgrammar::ParserState first_state(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  const xgrammar::ParserState second_state(11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
  const xgrammar::ParserState third_state(21, 22, 23, 24, 25, 26, 27, 28, 29, 30);

  const xgrammar::ParserState* stored_first = detector.InsertIfAbsent(first_state);
  const xgrammar::ParserState* stored_second = detector.InsertIfAbsent(second_state);
  ASSERT_NE(stored_first, nullptr);
  ASSERT_NE(stored_second, nullptr);
  EXPECT_EQ(detector.InsertIfAbsent(first_state), nullptr);

  const xgrammar::ParserState* stored_third = detector.InsertIfAbsent(third_state);
  ASSERT_NE(stored_third, nullptr);
  EXPECT_TRUE(xgrammar::StateEqualForParsing()(*stored_first, first_state));
  EXPECT_TRUE(xgrammar::StateEqualForParsing()(*stored_second, second_state));
  EXPECT_TRUE(xgrammar::StateEqualForParsing()(*stored_third, third_state));

  xgrammar::RepeatDetector copied_detector(detector);
  EXPECT_NE(copied_detector.InsertIfAbsent(first_state), nullptr);

  detector.Clear();
  EXPECT_NE(detector.InsertIfAbsent(first_state), nullptr);
}

TEST(EarleyParserTest, HandlesMoreThanFiftyStatesAcrossCopyFailureAndReset) {
  constexpr int32_t branch_count = 64;
  std::string grammar_text = "root ::= ";
  for (int32_t branch_index = 0; branch_index < branch_count; ++branch_index) {
    if (branch_index != 0) {
      grammar_text += " | ";
    }
    grammar_text += "branch_" + std::to_string(branch_index);
  }
  for (int32_t branch_index = 0; branch_index < branch_count; ++branch_index) {
    std::string suffix;
    suffix += static_cast<char>('A' + branch_index / 26);
    suffix += static_cast<char>('a' + branch_index % 26);
    grammar_text += "\nbranch_" + std::to_string(branch_index) + " ::= prefix_" +
                    std::to_string(branch_index) + " \"" + suffix + "\"";
    grammar_text += "\nprefix_" + std::to_string(branch_index) + " ::= \"a\"";
  }

  xgrammar::Grammar grammar =
      xgrammar::GrammarOptimizer::Apply(xgrammar::Grammar::FromEBNF(grammar_text));
  xgrammar::EarleyParser parser(grammar);
  ASSERT_GT(parser.GetLatestScanableStates().size(), 50);

  xgrammar::EarleyParser copied_parser(parser);
  EXPECT_FALSE(parser.Advance('!'));
  EXPECT_FALSE(copied_parser.Advance('!'));

  for (char character : std::string("aCl")) {
    ASSERT_TRUE(parser.Advance(static_cast<uint8_t>(character)));
    ASSERT_TRUE(copied_parser.Advance(static_cast<uint8_t>(character)));
    if (character == 'a') {
      EXPECT_GT(parser.GetLatestScanableStates().size(), 50);
      EXPECT_GT(copied_parser.GetLatestScanableStates().size(), 50);
    }
  }
  EXPECT_TRUE(parser.IsCompleted());
  EXPECT_TRUE(copied_parser.IsCompleted());

  parser.Reset();
  for (char character : std::string("aAa")) {
    ASSERT_TRUE(parser.Advance(static_cast<uint8_t>(character)));
  }
  EXPECT_TRUE(parser.IsCompleted());
}
