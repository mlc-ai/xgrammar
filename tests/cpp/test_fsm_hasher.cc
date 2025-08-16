/**
 * \file tests/cpp/test_fsm_hasher.cc
 * \brief Test the FSM hasher.
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <optional>

#include "grammar_functor.h"
#include "xgrammar/grammar.h"

using namespace xgrammar;

TEST(XGrammarFSMHasherTest, TestTerminalfsm) {
  Grammar grammar[2] = {
      Grammar::FromEBNF("root ::= [a-z] | [0-9]"), Grammar::FromEBNF("root ::= [0-9] | [a-z]")
  };
  GrammarFSMBuilder::Apply(&grammar[0]);
  GrammarFSMBuilder::Apply(&grammar[1]);
  GrammarFSMHasher::Apply(&grammar[0]);
  GrammarFSMHasher::Apply(&grammar[1]);
  EXPECT_EQ(grammar[0]->per_rule_fsm_hashes[0], grammar[1]->per_rule_fsm_hashes[0]);
  EXPECT_EQ(grammar[0]->per_rule_fsm_new_state_ids[0]->size(), 2);
  EXPECT_EQ(grammar[0]->per_rule_fsm_new_state_ids[0], grammar[1]->per_rule_fsm_new_state_ids[0]);
}

TEST(XGrammarFSMHasherTest, TestNonTerminalAndRecursionFsm) {
  Grammar grammar[2] = {
      Grammar::FromEBNF("root ::= root1 | [a-z]\n root1 ::= [0-9] root1"),
      Grammar::FromEBNF("root1 ::= [0-9] root1\n root ::= root1 | [a-z]")
  };
  GrammarFSMBuilder::Apply(&grammar[0]);
  GrammarFSMBuilder::Apply(&grammar[1]);
  GrammarFSMHasher::Apply(&grammar[0]);
  GrammarFSMHasher::Apply(&grammar[1]);
  EXPECT_EQ(grammar[0]->per_rule_fsm_hashes[0], grammar[1]->per_rule_fsm_hashes[1]);
  EXPECT_EQ(grammar[0]->per_rule_fsm_hashes[1], grammar[1]->per_rule_fsm_hashes[0]);
}

TEST(XGrammarFSMHasherTest, TestLoopReferenceFsm) {
  std::string ebnf_grammar = R"(
    root ::= rule1 | rule3
    rule1 ::= [0-9] rule2
    rule2 ::= [A-Z] | rule1
    rule3 ::= [0-9] rule3
    )";
  std::string ebnf_grammar2 = R"(
    rule2 ::= [0-9] rule3
    rule3 ::= [A-Z] | rule2
    root ::= rule2 | rule1
    rule1 ::= [0-9] rule1
)";
  auto grammar = Grammar::FromEBNF(ebnf_grammar);
  auto grammar2 = Grammar::FromEBNF(ebnf_grammar2);
  GrammarFSMBuilder::Apply(&grammar);
  GrammarFSMHasher::Apply(&grammar);
  GrammarFSMBuilder::Apply(&grammar2);
  GrammarFSMHasher::Apply(&grammar2);
  EXPECT_NE(grammar->per_rule_fsm_hashes[0], std::nullopt);
  EXPECT_NE(grammar->per_rule_fsm_hashes[1], std::nullopt);
  EXPECT_NE(grammar->per_rule_fsm_hashes[2], std::nullopt);
  EXPECT_NE(grammar->per_rule_fsm_hashes[3], std::nullopt);
  EXPECT_EQ(grammar->per_rule_fsm_hashes[0], grammar2->per_rule_fsm_hashes[2]);
  EXPECT_EQ(grammar->per_rule_fsm_hashes[1], grammar2->per_rule_fsm_hashes[0]);
  EXPECT_EQ(grammar->per_rule_fsm_hashes[2], grammar2->per_rule_fsm_hashes[1]);
  EXPECT_EQ(grammar->per_rule_fsm_hashes[3], grammar2->per_rule_fsm_hashes[3]);
}

TEST(XGrammarFSMHasherTest, TestComplexFsm) {
  std::string ebnf_grammar[2] = {
      R"(
    root ::= "" | root1 | root | [a-z]
    root1 ::= [0-9] | [A-Z] | root2 | ""
    root2 ::= "" | "testing"
)",
      R"(
    root2 ::= "testing" | ""
    root ::= [a-z] | root1 | root | ""
    root1 ::=  root2 | "" | [0-9] | [A-Z]
)"
  };
  Grammar grammar[2] = {Grammar::FromEBNF(ebnf_grammar[0]), Grammar::FromEBNF(ebnf_grammar[1])};
  GrammarFSMBuilder::Apply(&grammar[0]);
  GrammarFSMHasher::Apply(&grammar[0]);
  GrammarFSMBuilder::Apply(&grammar[1]);
  GrammarFSMHasher::Apply(&grammar[1]);
  EXPECT_EQ(grammar[0]->per_rule_fsm_hashes[0], grammar[1]->per_rule_fsm_hashes[1]);
  EXPECT_EQ(grammar[0]->per_rule_fsm_hashes[1], grammar[1]->per_rule_fsm_hashes[2]);
  EXPECT_EQ(grammar[0]->per_rule_fsm_hashes[2], grammar[1]->per_rule_fsm_hashes[0]);
}
