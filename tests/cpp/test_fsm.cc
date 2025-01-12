#include <gtest/gtest.h>

#include "fsm.h"

using namespace xgrammar;

TEST(XGrammarFSMTest, BuildTrieTest) {
  std::vector<std::string> patterns = {"hello", "hi", "哈哈", "哈", "hili", "good"};
  auto fsm = BuildTrie(patterns);

  // Test1: The printed result of FSM
  std::stringstream ss;
  ss << fsm;
  std::string fsm_str = ss.str();
  std::string fsm_str_expected = R"(FSM(num_nodes=19, start=0, end=[5, 6, 12, 9, 14, 18], edges=[
0: [(104)->1, (229)->7, (103)->15]
1: [(101)->2, (105)->6]
2: [(108)->3]
3: [(108)->4]
4: [(111)->5]
5: []
6: [(108)->13]
7: [(147)->8]
8: [(136)->9]
9: [(229)->10]
10: [(147)->11]
11: [(136)->12]
12: []
13: [(105)->14]
14: []
15: [(111)->16]
16: [(111)->17]
17: [(100)->18]
18: []
]))";

  EXPECT_EQ(fsm_str, fsm_str_expected);

  // Test2: The printed result of CompactFSM
  auto compact_fsm = fsm.ToCompact();
  ss.str("");
  ss << compact_fsm;
  std::string compact_fsm_str = ss.str();
  std::string compact_fsm_str_expected =
      R"(CompactFSM(num_nodes=19, start=0, end=[5, 6, 12, 9, 14, 18], edges=[
0: [(103)->15, (104)->1, (229)->7]
1: [(101)->2, (105)->6]
2: [(108)->3]
3: [(108)->4]
4: [(111)->5]
5: []
6: [(108)->13]
7: [(147)->8]
8: [(136)->9]
9: [(229)->10]
10: [(147)->11]
11: [(136)->12]
12: []
13: [(105)->14]
14: []
15: [(111)->16]
16: [(111)->17]
17: [(100)->18]
18: []
]))";

  EXPECT_EQ(compact_fsm_str, compact_fsm_str_expected);

  // Test3: Walk through the FSM
  int state = fsm.StartNode();
  EXPECT_EQ(state, 0);

  // Test "hello"
  state = fsm.StartNode();
  EXPECT_EQ(fsm.Transition(state, 'h'), 1);
  EXPECT_EQ(fsm.Transition(1, 'e'), 2);
  EXPECT_EQ(fsm.Transition(2, 'l'), 3);
  EXPECT_EQ(fsm.Transition(3, 'l'), 4);
  EXPECT_EQ(fsm.Transition(4, 'o'), 5);
  EXPECT_TRUE(fsm.IsEndNode(5));

  // Test "hil"
  state = fsm.StartNode();
  EXPECT_EQ(fsm.Transition(state, 'h'), 1);
  EXPECT_EQ(fsm.Transition(1, 'i'), 6);
  EXPECT_EQ(fsm.Transition(6, 'l'), 13);
  EXPECT_FALSE(fsm.IsEndNode(13));

  // Test walk failure
  state = fsm.StartNode();
  EXPECT_EQ(fsm.Transition(state, 'g'), 15);
  EXPECT_EQ(fsm.Transition(15, 'o'), 16);
  EXPECT_EQ(fsm.Transition(16, 'e'), -1);
}
