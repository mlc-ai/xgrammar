/**
 * \file tests/cpp/test_grammar_fsm_hasher.cc
 * \brief Regression tests for canonical grammar FSM hashing.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "grammar_functor.h"
#include "grammar_impl.h"
#include "xgrammar/grammar.h"

using namespace xgrammar;

namespace {

int32_t FindRule(const Grammar& grammar, const std::string& name) {
  for (int32_t rule_id = 0; rule_id < grammar->NumRules(); ++rule_id) {
    if (grammar->GetRule(rule_id).name == name) {
      return rule_id;
    }
  }
  return -1;
}

int CountRuleEdges(const CompactFSMWithStartEnd& fsm) {
  int result = 0;
  for (int32_t state = 0; state < fsm.NumStates(); ++state) {
    for (const auto& edge : fsm.GetFsm().GetEdges(state)) {
      result += edge.IsRuleRef();
    }
  }
  return result;
}

}  // namespace

TEST(XGrammarFSMHasherTest, RuleEdgeScratchIsScopedToEachState) {
  auto grammar = Grammar::FromEBNF(R"(
root ::= left | right
left ::= "a" first "x" second | "b" second "y" third
right ::= "b" second "y" third | "a" first "x" second
first ::= "1"
second ::= "2"
third ::= "3"
)");
  GrammarFSMBuilder::Apply(&grammar);
  GrammarFSMHasher::Apply(&grammar);

  const int32_t left = FindRule(grammar, "left");
  const int32_t right = FindRule(grammar, "right");
  ASSERT_GE(left, 0);
  ASSERT_GE(right, 0);
  ASSERT_TRUE(grammar->per_rule_fsms[left].has_value());
  ASSERT_TRUE(grammar->per_rule_fsms[right].has_value());
  EXPECT_GE(CountRuleEdges(grammar->per_rule_fsms[left]->GetFsm()), 4);
  EXPECT_GE(CountRuleEdges(grammar->per_rule_fsms[right]->GetFsm()), 4);

  ASSERT_TRUE(grammar->per_rule_fsm_hashes[left].has_value());
  ASSERT_TRUE(grammar->per_rule_fsm_hashes[right].has_value());
  EXPECT_EQ(grammar->per_rule_fsm_hashes[left], grammar->per_rule_fsm_hashes[right]);

  const auto first_hashes = grammar->per_rule_fsm_hashes;
  const auto first_state_ids = grammar->per_rule_fsm_new_state_ids;
  GrammarFSMHasher::Apply(&grammar);
  EXPECT_EQ(grammar->per_rule_fsm_hashes, first_hashes);
  EXPECT_EQ(grammar->per_rule_fsm_new_state_ids, first_state_ids);
}
