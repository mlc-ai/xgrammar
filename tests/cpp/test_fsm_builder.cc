/**
 * \file tests/cpp/test_fsm_builder.cc
 * \brief Test FSM builders: regex, trie, etc.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

#include "fsm.h"
#include "fsm_builder.h"

using namespace xgrammar;

TEST(XGrammarFSMBuilderTest, TestTrieFSMBuilder) {
  TrieFSMBuilder trie_builder;
  std::vector<std::string> patterns = {"hello", "hi", "哈哈", "哈", "hili", "good"};
  auto fsm_result = trie_builder.Build(patterns);
  EXPECT_TRUE(fsm_result.has_value());
  auto fsm = std::move(fsm_result).value();

  // Test1: The printed result of FSM

  // Test2: The printed result of CompactFSM
  CompactFSMWithStartEnd compact_fsm(fsm->ToCompact(), fsm.GetStart(), fsm.GetEnds());

  // Test3: Walk through the FSM
  int state = fsm.GetStart();
  EXPECT_EQ(state, 0);

  // Test "hello"
  state = fsm.GetStart();
  EXPECT_EQ(fsm->GetNextState(state, 'h'), 1);
  EXPECT_EQ(fsm->GetNextState(1, 'e'), 2);
  EXPECT_EQ(fsm->GetNextState(2, 'l'), 3);
  EXPECT_EQ(fsm->GetNextState(3, 'l'), 4);
  EXPECT_EQ(fsm->GetNextState(4, 'o'), 5);
  EXPECT_TRUE(fsm.IsEndState(5));

  // Test "hil"
  state = fsm.GetStart();
  EXPECT_EQ(fsm->GetNextState(state, 'h'), 1);
  EXPECT_EQ(fsm->GetNextState(1, 'i'), 6);
  EXPECT_EQ(fsm->GetNextState(6, 'l'), 13);
  EXPECT_FALSE(fsm.IsEndState(13));

  // Test walk failure
  state = fsm.GetStart();
  EXPECT_EQ(fsm->GetNextState(state, 'g'), 15);
  EXPECT_EQ(fsm->GetNextState(15, 'o'), 16);
  EXPECT_EQ(fsm->GetNextState(16, 'e'), -1);
}

TEST(XGrammarFSMBuilderTest, TestTagDispatchFSMBuilder1) {
  // Case 1. stop_eos = true, loop_after_dispatch = true
  Grammar::Impl::TagDispatch tag_dispatch = {
      /* tag_rule_pairs = */ {{"hel", 1}, {"hi", 2}, {"哈", 3}},
      /* stop_eos = */ true,
      /* stop_str = */ {},
      /* loop_after_dispatch = */ true,
  };
  auto fsm_result = TagDispatchFSMBuilder::Build(tag_dispatch);
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
  // Case 2. stop_eos = true, loop_after_dispatch = false
  Grammar::Impl::TagDispatch tag_dispatch = {
      /* tag_rule_pairs = */ {{"hel", 1}, {"hi", 2}, {"哈", 3}},
      /* stop_eos = */ true,
      /* stop_str = */ {},
      /* loop_after_dispatch = */ false,
  };
  auto fsm_result = TagDispatchFSMBuilder::Build(tag_dispatch);
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
  // Case 3. stop_eos = false, loop_after_dispatch = true
  Grammar::Impl::TagDispatch tag_dispatch = {
      /* tag_rule_pairs = */ {{"hel", 1}, {"hi", 2}, {"哈", 3}},
      /* stop_eos = */ false,
      /* stop_str = */ {"hos", "eos"},
      /* loop_after_dispatch = */ true,
  };
  auto fsm_result = TagDispatchFSMBuilder::Build(tag_dispatch);
  EXPECT_TRUE(fsm_result.has_value());
  auto fsm = std::move(fsm_result).value();
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed = R"(FSM(num_states=13, start=0, end=[9, 12], edges=[
0: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
1: [[\0-d]->0, 'e'->2, [f-g]->0, 'h'->1, 'i'->4, [j-n]->0, 'o'->8, [p-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
2: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-k]->0, 'l'->3, [m-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
3: [Rule(1)->0]
4: [Rule(2)->0]
5: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-\x92]->0, '\x93'->6, [\x94-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
6: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-\x87]->0, '\x88'->7, [\x89-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
7: [Rule(3)->0]
8: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-r]->0, 's'->9, [t-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
9: []
10: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-n]->0, 'o'->11, [p-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
11: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-r]->0, 's'->12, [t-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
12: []
]))";

  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}

TEST(XGrammarFSMBuilderTest, TestTagDispatchFSMBuilder4) {
  // Case 4. stop_eos = false, loop_after_dispatch = false
  Grammar::Impl::TagDispatch tag_dispatch = {
      /* tag_rule_pairs = */ {{"hel", 1}, {"hi", 2}, {"哈", 3}},
      /* stop_eos = */ false,
      /* stop_str = */ {"hos", "eos"},
      /* loop_after_dispatch = */ false,
  };
  auto fsm_result = TagDispatchFSMBuilder::Build(tag_dispatch);
  EXPECT_TRUE(fsm_result.has_value());
  auto fsm = std::move(fsm_result).value();
  auto fsm_printed = fsm.ToString();
  std::string expected_fsm_printed =
      R"(FSM(num_states=20, start=0, end=[9, 12, 16, 19], edges=[
0: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
1: [[\0-d]->0, 'e'->2, [f-g]->0, 'h'->1, 'i'->4, [j-n]->0, 'o'->8, [p-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
2: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-k]->0, 'l'->3, [m-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
3: [Rule(1)->13]
4: [Rule(2)->13]
5: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-\x92]->0, '\x93'->6, [\x94-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
6: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-\x87]->0, '\x88'->7, [\x89-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
7: [Rule(3)->13]
8: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-r]->0, 's'->9, [t-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
9: []
10: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-n]->0, 'o'->11, [p-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
11: [[\0-d]->0, 'e'->10, [f-g]->0, 'h'->1, [i-r]->0, 's'->12, [t-\xe4]->0, '\xe5'->5, [\xe6-\xff]->0]
12: []
13: ['h'->14, 'e'->17]
14: ['o'->15]
15: ['s'->16]
16: []
17: ['o'->18]
18: ['s'->19]
19: []
]))";

  EXPECT_EQ(fsm_printed, expected_fsm_printed);
}
