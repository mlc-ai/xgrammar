/**
 * \file tests/cpp/test_fsm_builder.cc
 * \brief Test FSM builders: regex, trie, etc.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <variant>
#include <vector>

#include "fsm.h"
#include "fsm_builder.h"
#include "grammar_builder.h"
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

TEST(XGrammarFSMBuilderTest, TestEOSExpressionBuildsEOSEdge) {
  auto grammar = Grammar::FromEBNF(R"(root ::= "a" EOS() "b")");
  auto fsm_result = GrammarFSMBuilder::Choices(
      grammar->GetGrammarExpr(grammar->GetRootRule().body_expr_id), grammar
  );
  ASSERT_TRUE(fsm_result.has_value());

  int32_t eos_edge_count = 0;
  for (const auto& edges : fsm_result->GetFsm().GetEdges()) {
    for (const auto& edge : edges) {
      eos_edge_count += edge.IsEOS();
    }
  }
  EXPECT_EQ(eos_edge_count, 1);
  EXPECT_NE(fsm_result->ToString().find("EOS->"), std::string::npos);
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

TEST(XGrammarFSMBuilderTest, TestByteStringChoicesBuildTrieDirectly) {
  auto grammar = Grammar::FromEBNF(R"(
    root ::= "" | "a" | "ab" | "ac" | "bcd" | "ab"
  )");
  auto fsm_result = GrammarFSMBuilder::Choices(
      grammar->GetGrammarExpr(grammar->GetRootRule().body_expr_id), grammar
  );
  ASSERT_TRUE(fsm_result.has_value());

  for (const char* accepted : {"", "a", "ab", "ac", "bcd"}) {
    EXPECT_TRUE(fsm_result->AcceptString(accepted)) << accepted;
  }
  for (const char* rejected : {"b", "abc", "ad", "bc", "bcde"}) {
    EXPECT_FALSE(fsm_result->AcceptString(rejected)) << rejected;
  }
}

TEST(XGrammarFSMBuilderTest, TestRegexBuildWithForbiddenChars) {
  // \S matches any non-whitespace byte. With the JSON forbidden characters removed, the
  // quote and the backslash must be rejected while other printable characters stay.
  const auto& forbidden = GrammarFSMBuilder::JSONStringForbiddenChars();
  auto fsm_wse = RegexFSMBuilder::BuildWithForbiddenChars("\\S+", forbidden).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc"));
  EXPECT_TRUE(fsm_wse.AcceptString("a!~z"));
  EXPECT_FALSE(fsm_wse.AcceptString("a\"b"));
  EXPECT_FALSE(fsm_wse.AcceptString("a\\b"));
  EXPECT_FALSE(fsm_wse.AcceptString("a b"));

  // A positive class spanning the quote and the backslash is split around them.
  fsm_wse = RegexFSMBuilder::BuildWithForbiddenChars("[ -~]+", forbidden).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("A z!"));
  EXPECT_FALSE(fsm_wse.AcceptString("\""));
  EXPECT_FALSE(fsm_wse.AcceptString("\\"));

  // . matches any byte; the control characters must be rejected as well.
  fsm_wse = RegexFSMBuilder::BuildWithForbiddenChars(".", forbidden).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("\t"));
  EXPECT_FALSE(fsm_wse.AcceptString("\n"));
  EXPECT_FALSE(fsm_wse.AcceptString("\""));

  // Epsilon transitions are preserved: the pattern still accepts the empty string.
  fsm_wse = RegexFSMBuilder::BuildWithForbiddenChars("(ab)*", forbidden).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(""));
  EXPECT_TRUE(fsm_wse.AcceptString("abab"));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));

  // A pattern that requires a forbidden character becomes the empty language.
  fsm_wse = RegexFSMBuilder::BuildWithForbiddenChars("a\"b", forbidden).Unwrap();
  EXPECT_FALSE(fsm_wse.AcceptString("a\"b"));
  EXPECT_FALSE(fsm_wse.AcceptString("ab"));
  EXPECT_FALSE(fsm_wse.AcceptString(""));

  // Multi-byte UTF-8 characters (bytes >= 0x80) are not affected by the JSON exclusion.
  fsm_wse = RegexFSMBuilder::BuildWithForbiddenChars(".+", forbidden).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("你好"));
}

TEST(XGrammarFSMBuilderTest, TestGrammarFSMBuilderRegex) {
  // The compiled regex automaton must preserve the language after simplification.
  auto fsm_wse = GrammarFSMBuilder::Regex("(ab)+").Unwrap();
  EXPECT_FALSE(fsm_wse.AcceptString(""));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  EXPECT_TRUE(fsm_wse.AcceptString("ab"));
  EXPECT_TRUE(fsm_wse.AcceptString("abab"));

  fsm_wse = GrammarFSMBuilder::Regex("[0-9]{5}$").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("12345"));
  EXPECT_FALSE(fsm_wse.AcceptString("1234"));
  EXPECT_FALSE(fsm_wse.AcceptString("123456"));

  // json_string=true matches decoded characters through their valid JSON source spellings.
  fsm_wse = GrammarFSMBuilder::Regex("\\S+", /*json_string=*/true).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc"));
  EXPECT_TRUE(fsm_wse.AcceptString("a\\\"b"));
  EXPECT_TRUE(fsm_wse.AcceptString("a\\\\b"));
  EXPECT_TRUE(fsm_wse.AcceptString("a\\bb"));
  EXPECT_FALSE(fsm_wse.AcceptString("a\"b"));
  EXPECT_FALSE(fsm_wse.AcceptString(std::string("a") + '\b' + "b"));
  EXPECT_FALSE(fsm_wse.AcceptString("a\\qb"));

  // Without the flag, the quote is accepted.
  fsm_wse = GrammarFSMBuilder::Regex("\\S+").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a\"b"));

  // Invalid patterns report an error.
  EXPECT_TRUE(GrammarFSMBuilder::Regex("+a").IsErr());
}

TEST(XGrammarFSMBuilderTest, TestRegexJSONStringSpellings) {
  auto fsm_wse = RegexFSMBuilder::BuildForJSONString("a[\\s\\S]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("ab"));
  EXPECT_TRUE(fsm_wse.AcceptString("a\\\""));
  EXPECT_TRUE(fsm_wse.AcceptString("a\\n"));
  EXPECT_TRUE(fsm_wse.AcceptString("a\\u000A"));
  EXPECT_TRUE(fsm_wse.AcceptString("a\\u000a"));
  EXPECT_TRUE(fsm_wse.AcceptString("a你"));
  EXPECT_TRUE(fsm_wse.AcceptString("a\\u4F60"));
  EXPECT_TRUE(fsm_wse.AcceptString("a😀"));
  EXPECT_TRUE(fsm_wse.AcceptString("a\\uD83D\\uDE00"));
  EXPECT_FALSE(fsm_wse.AcceptString("a\""));
  EXPECT_FALSE(fsm_wse.AcceptString("a\\q"));
  EXPECT_FALSE(fsm_wse.AcceptString("a\\uD83D"));
  EXPECT_FALSE(fsm_wse.AcceptString("a\\uDE00"));

  fsm_wse = RegexFSMBuilder::BuildForJSONString("[\\u01FE-\\u0201]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("\\u01fE"));
  EXPECT_TRUE(fsm_wse.AcceptString("\\u0200"));
  EXPECT_TRUE(fsm_wse.AcceptString("ȁ"));
  EXPECT_FALSE(fsm_wse.AcceptString("\\u01FD"));
  EXPECT_FALSE(fsm_wse.AcceptString("\\u0202"));

  fsm_wse = RegexFSMBuilder::BuildForJSONString("[\\u{103FF}-\\u{10401}]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("\\uD800\\uDFFF"));
  EXPECT_TRUE(fsm_wse.AcceptString("\\uD801\\uDC00"));
  EXPECT_TRUE(fsm_wse.AcceptString("\\uD801\\uDC01"));
  EXPECT_FALSE(fsm_wse.AcceptString("\\uD800\\uDFFE"));
  EXPECT_FALSE(fsm_wse.AcceptString("\\uD801\\uDC02"));

  // Repetition counts decoded code points, not source bytes or escape characters.
  fsm_wse = RegexFSMBuilder::BuildForJSONString(".{2}").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a你"));
  EXPECT_TRUE(fsm_wse.AcceptString("\\u0061\\u4F60"));
  EXPECT_TRUE(fsm_wse.AcceptString("\\n\\uD83D\\uDE00"));
  EXPECT_FALSE(fsm_wse.AcceptString("\\u0061"));
  EXPECT_FALSE(fsm_wse.AcceptString("abc"));

  // ASCII case folding applies to raw and Unicode-escaped spellings alike.
  fsm_wse = RegexFSMBuilder::BuildForJSONString("(?i)a").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_TRUE(fsm_wse.AcceptString("A"));
  EXPECT_TRUE(fsm_wse.AcceptString("\\u0061"));
  EXPECT_TRUE(fsm_wse.AcceptString("\\u0041"));
  EXPECT_FALSE(fsm_wse.AcceptString("b"));

  // A search-style expression must not lose its required middle literal during optimization.
  constexpr const char* kSearchRegex = "(?:[\\s\\S]*)(?:x_)(?:[\\s\\S]*)";
  fsm_wse = RegexFSMBuilder::BuildForJSONString(kSearchRegex).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("prefix_x_suffix"));
  EXPECT_FALSE(fsm_wse.AcceptString("before_y_after"));
  fsm_wse = GrammarFSMBuilder::Regex(kSearchRegex, /*json_string=*/true).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("prefix_x_suffix"));
  EXPECT_FALSE(fsm_wse.AcceptString("before_y_after"));

  std::string invalid_utf8_pattern(1, static_cast<char>(0xFF));
  EXPECT_TRUE(RegexFSMBuilder::BuildForJSONString(invalid_utf8_pattern).IsErr());
}

TEST(XGrammarFSMBuilderTest, TestRegexRepeatZero) {
  // {0} / {0,0} matches exactly the empty string (ECMA-262 semantics, and consistent
  // with the EBNF parser and the CFG fallback path).
  auto fsm_wse = RegexFSMBuilder::Build("a{0}").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(""));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));

  fsm_wse = RegexFSMBuilder::Build("a{0,0}").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(""));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));

  fsm_wse = RegexFSMBuilder::Build("(ab){0}").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(""));
  EXPECT_FALSE(fsm_wse.AcceptString("ab"));

  // In a concatenation the {0} element contributes nothing.
  fsm_wse = RegexFSMBuilder::Build("ba{0}c").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("bc"));
  EXPECT_FALSE(fsm_wse.AcceptString("bac"));

  // The empty-string FSM survives the simplification passes.
  fsm_wse = GrammarFSMBuilder::Regex("ba{0}c").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("bc"));
  EXPECT_FALSE(fsm_wse.AcceptString("bac"));

  // A lower bound larger than the upper bound is an error in every regex dialect.
  EXPECT_TRUE(RegexFSMBuilder::Build("a{3,1}").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("a{1,0}").IsErr());
}

TEST(XGrammarFSMBuilderTest, TestRegexByteMode) {
  auto fsm_wse = RegexFSMBuilder::Build("[^\\x00-\\x7F]+", /*byte_mode=*/true).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("\x80"));
  EXPECT_TRUE(fsm_wse.AcceptString("\xff\x80"));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString(""));

  // Dot and complements use a one-byte universe, while the default remains codepoint-oriented.
  fsm_wse = GrammarFSMBuilder::Regex(".", /*json_string=*/false, /*byte_mode=*/true).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("\x80"));
  EXPECT_TRUE(fsm_wse.AcceptString("\n"));
  EXPECT_FALSE(fsm_wse.AcceptString("é"));
  fsm_wse = GrammarFSMBuilder::Regex(".").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("é"));
  EXPECT_FALSE(fsm_wse.AcceptString("\x80"));

  // Byte escapes are raw, UTF-8 literals retain their encoded byte sequence, and ASCII case
  // folding remains available.
  fsm_wse = RegexFSMBuilder::Build("(?i)\\x80\\xFFéA", /*byte_mode=*/true).Unwrap();
  EXPECT_TRUE(
      fsm_wse.AcceptString("\x80\xff\xc3\xa9"
                           "a")
  );
  EXPECT_TRUE(
      fsm_wse.AcceptString("\x80\xff\xc3\xa9"
                           "A")
  );
  EXPECT_FALSE(
      fsm_wse.AcceptString("\xc2\x80\xff\xc3\xa9"
                           "a")
  );

  fsm_wse = RegexFSMBuilder::Build("[^\\x00-\\xFF]", /*byte_mode=*/true).Unwrap();
  EXPECT_FALSE(fsm_wse.AcceptString(""));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("\xff"));

  EXPECT_TRUE(RegexFSMBuilder::Build("\\p{L}", true).IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("\\x{80}", true).IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("[é]", true).IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("a^b", true).IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("a$b", true).IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("\\bword\\b", true).IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("(?=a)", true).IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("\\1", true).IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("\\q", true).IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("[a-\\d]", true).IsErr());
  EXPECT_TRUE(GrammarFSMBuilder::Regex("a", /*json_string=*/true, /*byte_mode=*/true).IsErr());
}

TEST(XGrammarFSMBuilderTest, TestRegexEmptyGroupQuantifiersDoNotRetarget) {
  // A discarded empty group would make the following '*' accidentally bind to the preceding 'a'.
  auto fsm_wse = RegexFSMBuilder::Build("a()*", /*byte_mode=*/true).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString(""));
  EXPECT_FALSE(fsm_wse.AcceptString("aa"));

  fsm_wse = RegexFSMBuilder::Build("a()+", /*byte_mode=*/true).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("aa"));
  fsm_wse = RegexFSMBuilder::Build("a(){0,500}", /*byte_mode=*/true).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("aa"));
}

TEST(XGrammarFSMBuilderTest, TestRawByteRegexSerializationRoundTrip) {
  std::string raw_pattern;
  raw_pattern.push_back(static_cast<char>(0x80));
  raw_pattern.push_back('\0');
  raw_pattern.push_back(static_cast<char>(0xFF));

  GrammarBuilder builder;
  int32_t regex = builder.AddRegex(raw_pattern, /*json_string=*/false, /*byte_mode=*/true);
  int32_t root = builder.AddRule("root", regex);
  Grammar grammar = builder.Get(root);
  std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("Regex(\"\\\\x80\\0\\\\xFF\", byte_mode=true)"), std::string::npos);

  Grammar restored_ebnf = Grammar::FromEBNF(printed);
  EXPECT_EQ(restored_ebnf.ToString(), printed);
  auto restored_json = Grammar::DeserializeJSON(grammar.SerializeJSON());
  ASSERT_TRUE(std::holds_alternative<Grammar>(restored_json));
  EXPECT_EQ(std::get<Grammar>(restored_json).ToString(), printed);
}

TEST(XGrammarFSMBuilderTest, TestRegexCaseInsensitiveFlag) {
  // The (?i) prefix folds ASCII letters in literals, alternations and repeats.
  auto fsm_wse = RegexFSMBuilder::Build("(?i)abc").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc"));
  EXPECT_TRUE(fsm_wse.AcceptString("ABC"));
  EXPECT_TRUE(fsm_wse.AcceptString("aBc"));
  EXPECT_FALSE(fsm_wse.AcceptString("abd"));

  fsm_wse = RegexFSMBuilder::Build("(?i)(ab|cd)+").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("aBcD"));
  EXPECT_FALSE(fsm_wse.AcceptString("ac"));

  // Positive classes fold both the explicit letters and letter ranges.
  fsm_wse = RegexFSMBuilder::Build("(?i)[a-dx]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("b"));
  EXPECT_TRUE(fsm_wse.AcceptString("B"));
  EXPECT_TRUE(fsm_wse.AcceptString("X"));
  EXPECT_FALSE(fsm_wse.AcceptString("e"));
  EXPECT_FALSE(fsm_wse.AcceptString("E"));

  // Negated classes exclude both cases of the folded letters.
  fsm_wse = RegexFSMBuilder::Build("(?i)[^k]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("k"));
  EXPECT_FALSE(fsm_wse.AcceptString("K"));

  // Simple Unicode case-fold equivalents are accepted.
  fsm_wse = RegexFSMBuilder::Build("(?i)Σ").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("Σ"));
  EXPECT_TRUE(fsm_wse.AcceptString("σ"));
  EXPECT_TRUE(fsm_wse.AcceptString("ς"));

  // Without the prefix the match stays case-sensitive.
  fsm_wse = RegexFSMBuilder::Build("abc").Unwrap();
  EXPECT_FALSE(fsm_wse.AcceptString("ABC"));
}

TEST(XGrammarFSMBuilderTest, TestRegexScopedFlags) {
  auto fsm_wse = RegexFSMBuilder::Build("(?i:a(?-i:b)c)d").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("AbCd"));
  EXPECT_TRUE(fsm_wse.AcceptString("abcd"));
  EXPECT_FALSE(fsm_wse.AcceptString("ABCd"));
  EXPECT_FALSE(fsm_wse.AcceptString("AbCD"));

  fsm_wse = RegexFSMBuilder::Build("a(?i:b)c").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("aBc"));
  EXPECT_FALSE(fsm_wse.AcceptString("Abc"));

  std::string rewritten = RewriteRegexDots("a(?s:.)(?-s:.)b", false);
  EXPECT_EQ(rewritten, "a(?s:.)(?-s:[^\\n])b");
  fsm_wse = RegexFSMBuilder::Build(rewritten).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a\nxb"));
  EXPECT_FALSE(fsm_wse.AcceptString("ax\nb"));

  rewritten = RewriteRegexDots("(?-s:.)(?s:.)", true);
  EXPECT_EQ(rewritten, "(?-s:[^\\n])(?s:.)");

  rewritten = RewriteRegexDots("(?R:.)(?-R:.)", false);
  EXPECT_EQ(rewritten, "(?R:[^\\r\\n])(?-R:[^\\n])");
  fsm_wse = RegexFSMBuilder::Build(rewritten).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a\r"));
  EXPECT_FALSE(fsm_wse.AcceptString("\ra"));

  rewritten = RewriteRegexDots("[](?=.].", false);
  EXPECT_EQ(rewritten, "[](?=.][^\\n]");

  fsm_wse = RegexFSMBuilder::Build("(?x:a b # comment\n c)").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc"));
  EXPECT_FALSE(fsm_wse.AcceptString("a b c"));

  fsm_wse = RegexFSMBuilder::Build("(?x:ab(?-x:c d)e f)").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc def"));
  EXPECT_FALSE(fsm_wse.AcceptString("abcdef"));

  EXPECT_EQ(RewriteRegexExtended("a(?-x: )b", true), "(?x)a(?-x: )b");

  EXPECT_TRUE(ContainsRegexMultilineLineAnchor("^a$", true));
  EXPECT_FALSE(ContainsRegexMultilineLineAnchor("(?-m:^a$)", true));
  EXPECT_TRUE(ContainsRegexMultilineLineAnchor("(?-m:^a$)(?m:^b$)", true));
  EXPECT_FALSE(ContainsRegexMultilineLineAnchor("(?m:(?-m:(?<λ.[>^a$)))"));
  EXPECT_FALSE(ContainsRegexMultilineLineAnchor("[]^]", true));

  // regex-syntax verbose mode ignores unescaped whitespace and comments inside classes too.
  fsm_wse = RegexFSMBuilder::Build("(?x:[ a # comment\n b])").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_TRUE(fsm_wse.AcceptString("b"));
  EXPECT_FALSE(fsm_wse.AcceptString(" "));
  EXPECT_FALSE(fsm_wse.AcceptString("#"));

  fsm_wse = RegexFSMBuilder::Build("(?x:a\u00a0b)").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("ab"));
  EXPECT_FALSE(fsm_wse.AcceptString("a\u00a0b"));
}

TEST(XGrammarFSMBuilderTest, TestRegexEscapes) {
  // \xHH, \uHHHH and \u{...} escapes, both standalone and inside classes.
  auto fsm_wse = RegexFSMBuilder::Build("\\x41\\u0042\\u{43}").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("ABC"));

  fsm_wse = RegexFSMBuilder::Build("\\u{1F600}").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("😀"));
  EXPECT_FALSE(fsm_wse.AcceptString("😁"));

  fsm_wse = RegexFSMBuilder::Build("[\\u0041-\\u0043]+").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("ABC"));
  EXPECT_FALSE(fsm_wse.AcceptString("D"));

  // A quantifier after a multi-byte character applies to the whole codepoint.
  fsm_wse = RegexFSMBuilder::Build("好*").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(""));
  EXPECT_TRUE(fsm_wse.AcceptString("好好"));
  EXPECT_FALSE(fsm_wse.AcceptString("\xbd"));

  // Unicode-mode shorthands follow regex-syntax's Unicode 16.0.0 tables.
  fsm_wse = RegexFSMBuilder::Build("\\d").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("১"));
  EXPECT_FALSE(fsm_wse.AcceptString("A"));

  fsm_wse = RegexFSMBuilder::Build("\\w+").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("ł\u0301"));
  EXPECT_FALSE(fsm_wse.AcceptString("😀"));
  EXPECT_FALSE(fsm_wse.AcceptString("-"));

  fsm_wse = RegexFSMBuilder::Build("\\s+").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(" \t\n\r\f\v\u0085\u00a0\u2003\u3000"));
  EXPECT_FALSE(fsm_wse.AcceptString(std::string(1, '\0')));
  EXPECT_FALSE(fsm_wse.AcceptString("\x01"));

  // \S is the codepoint-domain complement, so non-ASCII characters are accepted.
  fsm_wse = RegexFSMBuilder::Build("\\S").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("好"));
  EXPECT_TRUE(fsm_wse.AcceptString(std::string(1, '\0')));
  EXPECT_FALSE(fsm_wse.AcceptString(" "));

  // Byte mode deliberately retains ASCII shorthand semantics.
  fsm_wse = RegexFSMBuilder::Build("\\w", /*byte_mode=*/true).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("A"));
  EXPECT_FALSE(fsm_wse.AcceptString("ł"));
}

TEST(XGrammarFSMBuilderTest, TestRegexUnicodePropertiesUnsupported) {
  constexpr const char* kExpected =
      "Regex parsing error at position 1: Unicode property escapes \\p and \\P are not supported";
  auto error_message = [](const std::string& regex, bool byte_mode = false) {
    return std::string(RegexFSMBuilder::Build(regex, byte_mode).UnwrapErr().what());
  };

  EXPECT_EQ(error_message("\\pL"), kExpected);
  EXPECT_EQ(error_message("\\p{Letter}"), kExpected);
  EXPECT_EQ(error_message("\\P{Greek}"), kExpected);
  EXPECT_EQ(error_message("\\p{L", /*byte_mode=*/true), kExpected);
}

TEST(XGrammarFSMBuilderTest, TestRegexCharacterClassSets) {
  auto fsm_wse = RegexFSMBuilder::Build("[[:alpha:]]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("A"));
  EXPECT_FALSE(fsm_wse.AcceptString("1"));

  fsm_wse = RegexFSMBuilder::Build("[[:^digit:]]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("A"));
  EXPECT_FALSE(fsm_wse.AcceptString("1"));

  fsm_wse = RegexFSMBuilder::Build("[a-y&&xyz]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("x"));
  EXPECT_TRUE(fsm_wse.AcceptString("y"));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("z"));

  fsm_wse = RegexFSMBuilder::Build("[0-9--4]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("3"));
  EXPECT_TRUE(fsm_wse.AcceptString("5"));
  EXPECT_FALSE(fsm_wse.AcceptString("4"));

  fsm_wse = RegexFSMBuilder::Build("[a-g~~b-h]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_TRUE(fsm_wse.AcceptString("h"));
  EXPECT_FALSE(fsm_wse.AcceptString("b"));
  EXPECT_FALSE(fsm_wse.AcceptString("g"));

  fsm_wse = RegexFSMBuilder::Build("[x[^xyz]]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("x"));
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("y"));
  EXPECT_FALSE(fsm_wse.AcceptString("z"));

  // Binary set operators have equal precedence and associate from left to right.
  fsm_wse = RegexFSMBuilder::Build("[a-z--b-y&&x-z]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("z"));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("x"));

  fsm_wse = RegexFSMBuilder::Build("(?i)[a-z--A]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("b"));
  EXPECT_TRUE(fsm_wse.AcceptString("B"));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("A"));

  // A leading ']' and leading '-' characters are literals. Unknown POSIX-like spellings fall
  // back to ordinary nested-class syntax, matching regex-syntax.
  fsm_wse = RegexFSMBuilder::Build("[]a]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("]"));
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  fsm_wse = RegexFSMBuilder::Build("[](?=.]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("]"));
  EXPECT_TRUE(fsm_wse.AcceptString("."));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  fsm_wse = RegexFSMBuilder::Build("[--]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("-"));
  fsm_wse = RegexFSMBuilder::Build("[[:loower:]]").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString(":"));
  EXPECT_TRUE(fsm_wse.AcceptString("w"));
  EXPECT_FALSE(fsm_wse.AcceptString("a"));

  fsm_wse = RegexFSMBuilder::Build("[a&&b]").Unwrap();
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  EXPECT_FALSE(fsm_wse.AcceptString("b"));
  EXPECT_TRUE(RegexFSMBuilder::Build("[a&&]").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("[&&a]").IsErr());
}

TEST(XGrammarFSMBuilderTest, TestRegexUnicodeModeFlagsAndAnchors) {
  auto fsm_wse = RegexFSMBuilder::Build("(?-u:\\w)").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("A"));
  EXPECT_FALSE(fsm_wse.AcceptString("é"));
  EXPECT_TRUE(RegexFSMBuilder::Build("(?-u:.)").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("(?-u:[^A])").IsErr());

  fsm_wse = RegexFSMBuilder::Build("(?u:\\w)", /*byte_mode=*/true).Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("A"));
  EXPECT_TRUE(fsm_wse.AcceptString("é"));

  fsm_wse = RegexFSMBuilder::Build("\\Aabc\\z").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc"));
  EXPECT_FALSE(fsm_wse.AcceptString("xabc"));

  fsm_wse = RegexFSMBuilder::Build("\\x{41}\\U0001F600\\U{1F600}").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("A😀😀"));
  EXPECT_FALSE(fsm_wse.AcceptString("A😀"));
}

TEST(XGrammarFSMBuilderTest, TestRegexUnsupportedFeatures) {
  // Word boundaries and backreferences raise errors.
  EXPECT_TRUE(RegexFSMBuilder::Build("a\\b").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("a\\B").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("(a)\\1").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("(?<name>a)\\k<name>").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("(?<=a)b").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("[]").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("[^]").IsErr());

  // Lookahead assertions are ignored with a warning (treated as the empty string).
  auto fsm_wse = RegexFSMBuilder::Build("a(?=b)c").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("ac"));
  fsm_wse = RegexFSMBuilder::Build("a(?!b)c").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("ac"));
  fsm_wse = RegexFSMBuilder::Build("a(?=(?<λ.[>b))c").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("ac"));

  // Named groups compile like plain groups; the name is ignored.
  fsm_wse = RegexFSMBuilder::Build("(?<name>ab)+").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abab"));
  fsm_wse = RegexFSMBuilder::Build("(?P<name>ab)c").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc"));
  fsm_wse = RegexFSMBuilder::Build("(?<λ.名[1]>ab)c").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc"));
  fsm_wse = RegexFSMBuilder::Build("(?<λ²>a)").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  fsm_wse = RegexFSMBuilder::Build("(?<Ⅰ>a)").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  fsm_wse = RegexFSMBuilder::Build("(?x:(?<λ.[> a b ) c)").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc"));
  EXPECT_TRUE(RegexFSMBuilder::Build("(?<́name>a)").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("(?<²name>a)").IsErr());
  EXPECT_TRUE(RegexFSMBuilder::Build("(?<same>a)(?<same>b)").IsErr());

  // Mid-pattern anchors are ignored with a warning.
  fsm_wse = RegexFSMBuilder::Build("a^b$c").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("abc"));

  // Non-greedy quantifiers behave like their greedy counterparts.
  fsm_wse = RegexFSMBuilder::Build(
                "a*?b+?c?"
                "?(de){1,2}?"
  )
                .Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("aabbcde"));
  EXPECT_TRUE(fsm_wse.AcceptString("bdede"));

  // Empty alternatives match the empty string.
  fsm_wse = RegexFSMBuilder::Build("(a|)b").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("ab"));
  EXPECT_TRUE(fsm_wse.AcceptString("b"));
  fsm_wse = RegexFSMBuilder::Build("a|").Unwrap();
  EXPECT_TRUE(fsm_wse.AcceptString("a"));
  EXPECT_TRUE(fsm_wse.AcceptString(""));
}

TEST(XGrammarFSMBuilderTest, TestRegexErrorMessages) {
  // Parse errors mirror the RegexConverter format and carry the 1-based position.
  auto error_message = [](const std::string& regex) {
    return std::string(RegexFSMBuilder::Build(regex).UnwrapErr().what());
  };
  EXPECT_EQ(
      error_message("+a"),
      "Regex parsing error at position 1: There is nothing to repeat before '+'"
  );
  EXPECT_EQ(error_message("ab)"), "Regex parsing error at position 3: Unmatched ')'");
  EXPECT_EQ(error_message("a]"), "Regex parsing error at position 2: Unmatched ']'");
  EXPECT_EQ(
      error_message("a(bc"), "Regex parsing error at position 2: The parenthesis is not closed"
  );
  EXPECT_EQ(error_message("a[bc"), "Regex parsing error at position 2: Unclosed '['");
  EXPECT_EQ(error_message("a[]"), "Regex parsing error at position 2: Unclosed '['");
  EXPECT_EQ(
      error_message("a{,2}"),
      "Regex parsing error at position 2: Invalid repetition count: expected a number after '{'"
  );
  EXPECT_EQ(
      error_message("a{3,1}"),
      "Regex parsing error at position 2: Invalid repetition count: the lower bound 3 is larger "
      "than the upper bound 1"
  );
  EXPECT_EQ(
      error_message("a{1,9999999999}"),
      "Regex parsing error at position 2: Invalid repetition count: the count 9999999999 is too "
      "large"
  );
  EXPECT_EQ(
      error_message("a\\b"),
      "Regex parsing error at position 2: Word boundary assertion \\b is not supported in regex"
  );
  EXPECT_EQ(
      error_message("(?<name)a"), "Regex parsing error at position 1: Invalid named capturing group"
  );
  // The position includes the "(?i)" prefix when present.
  EXPECT_EQ(
      error_message("(?i)+a"),
      "Regex parsing error at position 5: There is nothing to repeat before '+'"
  );
}

TEST(XGrammarFSMBuilderTest, TestRegexMatchesEmpty) {
  EXPECT_TRUE(RegexFSMBuilder::MatchesEmpty("a*").Unwrap());
  EXPECT_TRUE(RegexFSMBuilder::MatchesEmpty("(a|)").Unwrap());
  EXPECT_TRUE(RegexFSMBuilder::MatchesEmpty("a{0,10}").Unwrap());
  EXPECT_TRUE(RegexFSMBuilder::MatchesEmpty("(ab)?").Unwrap());
  EXPECT_FALSE(RegexFSMBuilder::MatchesEmpty("a+").Unwrap());
  EXPECT_FALSE(RegexFSMBuilder::MatchesEmpty("a{1,10}").Unwrap());
  EXPECT_FALSE(RegexFSMBuilder::MatchesEmpty("ab|cd").Unwrap());
  // No FSM is built, so huge bounded repetitions are cheap to check.
  EXPECT_FALSE(RegexFSMBuilder::MatchesEmpty("(abc){2,1000000}").Unwrap());
  EXPECT_TRUE(RegexFSMBuilder::MatchesEmpty("(abc){0,1000000}").Unwrap());
  EXPECT_TRUE(RegexFSMBuilder::MatchesEmpty("(?i)A*").Unwrap());
  EXPECT_TRUE(RegexFSMBuilder::MatchesEmpty("\\b").IsErr());
}

TEST(XGrammarFSMBuilderTest, TestRegexLargeRepeatWithoutBuilder) {
  // Without a GrammarBuilder, moderate repetitions are unrolled physically.
  auto fsm_wse = RegexFSMBuilder::Build("a{2,300}").Unwrap();
  EXPECT_FALSE(fsm_wse.AcceptString("a"));
  EXPECT_TRUE(fsm_wse.AcceptString("aa"));
  EXPECT_TRUE(fsm_wse.AcceptString(std::string(300, 'a')));
  EXPECT_FALSE(fsm_wse.AcceptString(std::string(301, 'a')));

  // Repetitions whose estimated state count is too large report an error instead of hanging.
  EXPECT_TRUE(RegexFSMBuilder::Build("(abcdefghij){1,100000}").IsErr());
}
