/**
 * \file tests/cpp/test_fsm_builder.cc
 * \brief Test FSM builders: regex, trie, etc.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "fsm.h"
#include "fsm_builder.h"
#include "grammar_functor.h"
#include "xgrammar/grammar.h"

using namespace xgrammar;

TEST(XGrammarFSMBuilderTest, TestTrieFSMBuilder) {
  TrieFSMBuilder trie_builder;
  std::vector<std::string> patterns = {"hello", "hi", "哈哈", "哈", "hili", "good"};
  auto fsm_result = trie_builder.Build(patterns, {});
  EXPECT_TRUE(fsm_result.has_value());
  auto fsm = std::move(fsm_result).value();

  // Test1: The printed result of FSM

  // Test2: The printed result of CompactFSM
  CompactFSMWithStartEnd compact_fsm(fsm.GetFsm().ToCompact(), fsm.GetStart(), fsm.GetEnds());

  // Test3: Walk through the FSM
  int state = fsm.GetStart();
  EXPECT_EQ(state, 0);

  // Test "hello"
  state = fsm.GetStart();
  EXPECT_EQ(fsm.GetFsm().GetNextState(state, 'h'), 1);
  EXPECT_EQ(fsm.GetFsm().GetNextState(1, 'e'), 2);
  EXPECT_EQ(fsm.GetFsm().GetNextState(2, 'l'), 3);
  EXPECT_EQ(fsm.GetFsm().GetNextState(3, 'l'), 4);
  EXPECT_EQ(fsm.GetFsm().GetNextState(4, 'o'), 5);
  EXPECT_TRUE(fsm.IsEndState(5));

  // Test "hil"
  state = fsm.GetStart();
  EXPECT_EQ(fsm.GetFsm().GetNextState(state, 'h'), 1);
  EXPECT_EQ(fsm.GetFsm().GetNextState(1, 'i'), 6);
  EXPECT_EQ(fsm.GetFsm().GetNextState(6, 'l'), 13);
  EXPECT_FALSE(fsm.IsEndState(13));

  // Test walk failure
  state = fsm.GetStart();
  EXPECT_EQ(fsm.GetFsm().GetNextState(state, 'g'), 15);
  EXPECT_EQ(fsm.GetFsm().GetNextState(15, 'o'), 16);
  EXPECT_EQ(fsm.GetFsm().GetNextState(16, 'e'), -1);
}

TEST(XGrammarFSMBuilderTest, TestTagDispatchFSMBuilder1) {
  // Case 1. loop_after_dispatch = true
  Grammar::Impl::TagDispatch tag_dispatch = {
      /* tag_rule_pairs = */ {{"hel", 1}, {"hi", 2}, {"哈", 3}},
      /* loop_after_dispatch = */ true,
      /* excludes = */ {}
  };
  auto fsm_result = GrammarFSMBuilder::TagDispatch(tag_dispatch);
  EXPECT_TRUE(fsm_result.has_value());
  auto fsm = std::move(fsm_result).value();
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed = R"(FSM(num_states=8, start=0, end=[0, 1, 2, 5, 6], edges=[
0: [[\0-g]->0, 'h'->1, [i-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
1: [[\0-d]->0, 'e'->2, [f-g]->0, 'h'->1, 'i'->4, [j-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
2: [[\0-g]->0, 'h'->1, [i-k]->0, 'l'->3, [m-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
3: [Rule(1)->0]
4: [Rule(2)->0]
5: [[\0-g]->0, 'h'->1, [i-\x92]->0, '\x93'->6, [\x94-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
6: [[\0-g]->0, 'h'->1, [i-\x87]->0, '\x88'->7, [\x89-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
7: [Rule(3)->0]
]))";

  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestTagDispatchFSMBuilder2) {
  // Case 2. loop_after_dispatch = false
  Grammar::Impl::TagDispatch tag_dispatch = {
      /* tag_rule_pairs = */ {{"hel", 1}, {"hi", 2}, {"哈", 3}},
      /* loop_after_dispatch = */ false,
      /* excludes = */ {}
  };
  auto fsm_result = GrammarFSMBuilder::TagDispatch(tag_dispatch);
  EXPECT_TRUE(fsm_result.has_value());
  auto fsm = std::move(fsm_result).value();
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed =
      R"(FSM(num_states=11, start=0, end=[0, 1, 2, 5, 6, 8, 9, 10], edges=[
0: [[\0-g]->0, 'h'->1, [i-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
1: [[\0-d]->0, 'e'->2, [f-g]->0, 'h'->1, 'i'->4, [j-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
2: [[\0-g]->0, 'h'->1, [i-k]->0, 'l'->3, [m-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
3: [Rule(1)->8]
4: [Rule(2)->9]
5: [[\0-g]->0, 'h'->1, [i-\x92]->0, '\x93'->6, [\x94-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
6: [[\0-g]->0, 'h'->1, [i-\x87]->0, '\x88'->7, [\x89-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
7: [Rule(3)->10]
8: []
9: []
10: []
]))";

  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestTagDispatchFSMBuilder3) {
  // Case 3. string excludes are compiled into the trie
  Grammar::Impl::TagDispatch tag_dispatch = {
      /* tag_rule_pairs = */ {{"hel", 1}, {"hi", 2}, {"哈", 3}},
      /* loop_after_dispatch = */ true,
      /* excludes = */ {"hos", "eos"}
  };
  auto fsm_result = GrammarFSMBuilder::TagDispatch(tag_dispatch);
  EXPECT_TRUE(fsm_result.has_value());
  auto fsm = std::move(fsm_result).value();
  auto fsm_printed = fsm.ToString();
  EXPECT_NE(fsm_printed.find("Rule(1)->0"), std::string::npos);
  EXPECT_NE(fsm_printed.find("Rule(2)->0"), std::string::npos);
  EXPECT_NE(fsm_printed.find("Rule(3)->0"), std::string::npos);
}

TEST(XGrammarFSMBuilderTest, TestTokenTagDispatchFSMBuilder) {
  Grammar::Impl::TokenTagDispatch ttd = {
      /* trigger_rule_pairs = */ {{3, 1}, {5, 2}},
      /* loop_after_dispatch = */ false,
      /* excludes = */ {7}
  };
  auto fsm_result = GrammarFSMBuilder::TokenTagDispatch(ttd);
  EXPECT_TRUE(fsm_result.has_value());
  auto fsm = std::move(fsm_result).value();
  auto fsm_printed = fsm.ToString();
  EXPECT_NE(fsm_printed.find("Token"), std::string::npos);
  EXPECT_NE(fsm_printed.find("ExcludeToken"), std::string::npos);
}
using GrammarExpr = Grammar::Impl::GrammarExpr;
using GrammarExprType = Grammar::Impl::GrammarExprType;

TEST(XGrammarFSMBuilderTest, TestByteStringFSMBuilder1) {
  int32_t byte_string[] = {'h', 'e', 'l', 'l', 'o'};
  GrammarExpr grammar_expr = {GrammarExprType::kByteString, byte_string, 5};
  auto fsm = GrammarFSMBuilder::ByteString(grammar_expr);
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed =
      R"(FSM(num_states=6, start=0, end=[5], edges=[
0: ['h'->1]
1: ['e'->2]
2: ['l'->3]
3: ['l'->4]
4: ['o'->5]
5: []
]))";
  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestByteStringFSMBuilder2) {
  std::string byte_string = "你好";
  std::vector<int32_t> byte_string_vec(byte_string.begin(), byte_string.end());
  GrammarExpr grammar_expr = {
      GrammarExprType::kByteString,
      byte_string_vec.data(),
      static_cast<int32_t>(byte_string_vec.size())
  };
  auto fsm = GrammarFSMBuilder::ByteString(grammar_expr);
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed =
      R"(FSM(num_states=7, start=0, end=[6], edges=[
0: ['\xe4'->1]
1: ['\xbd'->2]
2: ['\xa0'->3]
3: ['\xe5'->4]
4: ['\xa5'->5]
5: ['\xbd'->6]
6: []
]))";
  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestRuleRefFSMBuilder) {
  int32_t rule_ref = 1;
  GrammarExpr grammar_expr = {GrammarExprType::kRuleRef, &rule_ref, 1};
  auto fsm = GrammarFSMBuilder::RuleRef(grammar_expr);
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed =
      R"(FSM(num_states=2, start=0, end=[1], edges=[
0: [Rule(1)->1]
1: []
]))";
  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestCharacterClassFSMBuilder1) {
  std::vector<int32_t> datas = {0, 'a', 'z', 'A', 'Z'};
  GrammarExpr grammar_expr = {
      GrammarExprType::kCharacterClass, datas.data(), static_cast<int32_t>(datas.size())
  };
  auto fsm = GrammarFSMBuilder::CharacterClass(grammar_expr);
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed =
      R"(FSM(num_states=2, start=0, end=[1], edges=[
0: [[a-z]->1, [A-Z]->1]
1: []
]))";
  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestCharacterClassFSMBuilder2) {
  std::vector<int32_t> datas = {0, 'a', 'z', 'A', 'Z'};
  GrammarExpr grammar_expr = {
      GrammarExprType::kCharacterClassStar, datas.data(), static_cast<int32_t>(datas.size())
  };
  auto fsm = GrammarFSMBuilder::CharacterClass(grammar_expr);
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed =
      R"(FSM(num_states=1, start=0, end=[0], edges=[
0: [[a-z]->0, [A-Z]->0]
]))";
  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestCharacterClassFSMBuilder3) {
  std::vector<int32_t> datas = {1, 'a', 'z', 'A', 'Z'};
  GrammarExpr grammar_expr = {
      GrammarExprType::kCharacterClass, datas.data(), static_cast<int32_t>(datas.size())
  };
  auto fsm = GrammarFSMBuilder::CharacterClass(grammar_expr);
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed =
      R"(FSM(num_states=8, start=0, end=[1], edges=[
0: [[\0-@]->1, [[-`]->1, [{-\x7f]->1, [\xc0-\xdf]->2, [\xe0-\xef]->3, [\xf0-\xf7]->5]
1: []
2: [[\x80-\xbf]->1]
3: [[\x80-\xbf]->4]
4: [[\x80-\xbf]->1]
5: [[\x80-\xbf]->6]
6: [[\x80-\xbf]->7]
7: [[\x80-\xbf]->1]
]))";
  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestCharacterClassFSMBuilder4) {
  std::vector<int32_t> datas = {1, 'a', 'z', 'A', 'Z'};
  GrammarExpr grammar_expr = {
      GrammarExprType::kCharacterClassStar, datas.data(), static_cast<int32_t>(datas.size())
  };
  auto fsm = GrammarFSMBuilder::CharacterClass(grammar_expr);
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed =
      R"(FSM(num_states=7, start=0, end=[0], edges=[
0: [[\0-@]->0, [[-`]->0, [{-\x7f]->0, [\xc0-\xdf]->1, [\xe0-\xef]->2, [\xf0-\xf7]->4]
1: [[\x80-\xbf]->0]
2: [[\x80-\xbf]->3]
3: [[\x80-\xbf]->0]
4: [[\x80-\xbf]->5]
5: [[\x80-\xbf]->6]
6: [[\x80-\xbf]->0]
]))";
  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestSequenceFSMBuilder) {
  std::string test_grammar = R"(
    root ::= rule1 rule2 rule3
    rule1 ::= "a" [a-z]* rule3
    rule2 ::= "c" [A-Z] rule3
    rule3 ::= "a" rule3
  )";
  auto grammar = Grammar::FromEBNF(test_grammar);
  std::string expected_fsm_root = R"(FSM(num_states=4, start=2, end=[3], edges=[
0: [Rule(2)->1]
1: [Rule(3)->3]
2: [Rule(1)->0]
3: []
]))";
  auto fsm_root_result = GrammarFSMBuilder::Choices(
      grammar->GetGrammarExpr(grammar->GetRootRule().body_expr_id), grammar
  );
  EXPECT_TRUE(fsm_root_result.has_value());
  EXPECT_EQ(fsm_root_result->ToString(), expected_fsm_root);

  auto fsm_rule1_result = GrammarFSMBuilder::Choices(
      grammar->GetGrammarExpr(grammar->GetRule(1).body_expr_id), grammar
  );
  std::string expected_fsm_rule1 = R"(FSM(num_states=3, start=1, end=[2], edges=[
0: [Rule(3)->2, [a-z]->0]
1: ['a'->0]
2: []
]))";

  EXPECT_TRUE(fsm_rule1_result.has_value());
  EXPECT_EQ(fsm_rule1_result->ToString(), expected_fsm_rule1);

  auto fsm_rule2_result = GrammarFSMBuilder::Choices(
      grammar->GetGrammarExpr(grammar->GetRule(2).body_expr_id), grammar
  );
  std::string expected_fsm_rule2 = R"(FSM(num_states=4, start=2, end=[3], edges=[
0: [[A-Z]->1]
1: [Rule(3)->3]
2: ['c'->0]
3: []
]))";

  EXPECT_TRUE(fsm_rule2_result.has_value());
  EXPECT_EQ(fsm_rule2_result->ToString(), expected_fsm_rule2);

  auto fsm_rule3_result = GrammarFSMBuilder::Choices(
      grammar->GetGrammarExpr(grammar->GetRule(3).body_expr_id), grammar
  );
  std::string expected_fsm_rule3 = R"(FSM(num_states=3, start=1, end=[2], edges=[
0: [Rule(3)->2]
1: ['a'->0]
2: []
]))";

  EXPECT_TRUE(fsm_rule3_result.has_value());
  EXPECT_EQ(fsm_rule3_result->ToString(), expected_fsm_rule3);
}

TEST(XGrammarFSMBuilderTest, TestChoicesFSMBuilder) {
  std::string test_grammar = R"(
      root ::= rule1 | rule2
      rule1 ::= "" | "hello" rule2
      rule2 ::= [a-z]* "A" | "B" rule2
  )";
  auto grammar = Grammar::FromEBNF(test_grammar);
  auto fsm_root_result = GrammarFSMBuilder::Choices(
      grammar->GetGrammarExpr(grammar->GetRootRule().body_expr_id), grammar
  );
  std::string expected_fsm_root = R"(FSM(num_states=3, start=0, end=[1, 2], edges=[
0: [Rule(1)->1, Rule(2)->2]
1: []
2: []
]))";

  EXPECT_TRUE(fsm_root_result.has_value());
  EXPECT_EQ(fsm_root_result->ToString(), expected_fsm_root);

  auto fsm_rule1_result = GrammarFSMBuilder::Choices(
      grammar->GetGrammarExpr(grammar->GetRule(1).body_expr_id), grammar
  );
  std::string expected_fsm_rule1 = R"(FSM(num_states=7, start=0, end=[0, 6], edges=[
0: ['h'->2]
1: [Rule(2)->6]
2: ['e'->3]
3: ['l'->4]
4: ['l'->5]
5: ['o'->1]
6: []
]))";

  EXPECT_TRUE(fsm_rule1_result.has_value());
  EXPECT_EQ(fsm_rule1_result->ToString(), expected_fsm_rule1);

  auto fsm_rule2_result = GrammarFSMBuilder::Choices(
      grammar->GetGrammarExpr(grammar->GetRule(2).body_expr_id), grammar
  );
  std::string expected_fsm_rule2 = R"(FSM(num_states=4, start=1, end=[0], edges=[
0: []
1: [Eps->2, 'B'->3]
2: ['A'->0, [a-z]->2]
3: [Rule(2)->0]
]))";

  EXPECT_TRUE(fsm_rule2_result.has_value());
  EXPECT_EQ(fsm_rule2_result->ToString(), expected_fsm_rule2);
}
