/*!
 *  Copyright (c) 2026 by Contributors
 * \file tests/cpp/test_grammar_builder.cc
 * \brief Tests for direct, nested GrammarBuilder expression specifications.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "grammar_builder.h"
#include "grammar_impl.h"

namespace xgrammar {
namespace {

using namespace grammar_spec;

TEST(XGrammarGrammarBuilderTest, MaterializesNestedSpecs) {
  GrammarBuilder builder;
  builder.AddRule(
      "root",
      Choices(
          Sequence(ByteString("abc"), CharacterClass({{'a', 'z'}})),
          Regex("[0-9]+", /*json_string=*/true),
          Sequence(CharacterClassStar({{'0', '9'}}), EmptyStr())
      )
  );

  EXPECT_EQ(
      builder.Get("root").ToString(),
      "root ::= ((\"abc\" [a-z]) | Regex(\"[0-9]+\", json_string=true) | ([0-9]* \"\"))\n"
  );
}

TEST(XGrammarGrammarBuilderTest, BindsSelfReference) {
  GrammarBuilder builder;
  int32_t item_rule_id = builder.AddRule("item", ByteString("a"));
  int32_t list_rule_id =
      builder.AddRule("list", Choices(EmptyStr(), Sequence(RuleRef(item_rule_id), SelfRef())));
  Grammar grammar = builder.Get(list_rule_id);

  EXPECT_EQ(grammar.ToString(), "item ::= \"a\"\nlist ::= (\"\" | (item list))\n");
}

TEST(XGrammarGrammarBuilderTest, LowersRepeatSpecs) {
  GrammarBuilder builder;
  int32_t item_rule_id = builder.AddRule("item", ByteString("a"));
  int32_t root_rule_id = builder.AddRule(
      "root",
      Sequence(
          Repeat(RuleRef(item_rule_id), 2, 4),
          Repeat(ByteString("b"), 0, -1),
          Repeat(ByteString("c"), 0, 1)
      )
  );
  Grammar grammar = builder.Get(root_rule_id);
  const Grammar::Impl& impl = *grammar.operator->();

  auto root_sequence = impl.GetGrammarExpr(impl.GetRule(root_rule_id).body_expr_id);
  ASSERT_EQ(root_sequence.type, Grammar::Impl::GrammarExprType::kSequence);
  ASSERT_EQ(root_sequence.size(), 3);

  auto bounded_repeat = impl.GetGrammarExpr(root_sequence[0]);
  ASSERT_EQ(bounded_repeat.type, Grammar::Impl::GrammarExprType::kRepeat);
  EXPECT_EQ(bounded_repeat[0], item_rule_id);
  EXPECT_EQ(bounded_repeat[1], 2);
  EXPECT_EQ(bounded_repeat[2], 4);

  EXPECT_EQ(impl.GetGrammarExpr(root_sequence[1]).type, Grammar::Impl::GrammarExprType::kRuleRef);
  EXPECT_EQ(impl.GetGrammarExpr(root_sequence[2]).type, Grammar::Impl::GrammarExprType::kRuleRef);
}

TEST(XGrammarGrammarBuilderTest, MaterializesTokenSpecs) {
  GrammarBuilder builder;
  int32_t root_rule_id = builder.AddRule("root", Choices(Token({1, 2, 3}), ExcludeToken({4, 5})));
  Grammar grammar = builder.Get(root_rule_id);
  const Grammar::Impl& impl = *grammar.operator->();

  auto choices = impl.GetGrammarExpr(impl.GetRule(root_rule_id).body_expr_id);
  ASSERT_EQ(choices.type, Grammar::Impl::GrammarExprType::kChoices);
  ASSERT_EQ(choices.size(), 2);
  auto token = impl.GetGrammarExpr(choices[0]);
  auto exclude_token = impl.GetGrammarExpr(choices[1]);
  EXPECT_EQ(token.type, Grammar::Impl::GrammarExprType::kToken);
  EXPECT_EQ(std::vector<int32_t>(token.begin(), token.end()), (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(exclude_token.type, Grammar::Impl::GrammarExprType::kExcludeToken);
  EXPECT_EQ(
      std::vector<int32_t>(exclude_token.begin(), exclude_token.end()), (std::vector<int32_t>{4, 5})
  );
}

}  // namespace
}  // namespace xgrammar
