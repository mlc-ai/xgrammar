/*!
 *  Copyright (c) 2026 by Contributors
 * \file tests/cpp/test_grammar_builder.cc
 * \brief Tests for GrammarBuilder: nested GrammarExprSpec construction, self references,
 * sub grammar addition, typed getters, and normalization on Get().
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <string>
#include <vector>

#include "grammar_builder.h"
#include "grammar_impl.h"

using namespace xgrammar;

using namespace xgrammar::grammar_spec;

TEST(XGrammarGrammarBuilderTest, NestedSpec) {
  GrammarBuilder builder;
  builder.AddRule(
      "root",
      Choices(
          Sequence(ByteString("abc"), CharacterClass({{'a', 'z'}})),
          ByteString("def"),
          Sequence(CharacterClassStar({{'0', '9'}}), EmptyStr())
      )
  );
  auto grammar = builder.Get("root");
  EXPECT_EQ(grammar.ToString(), "root ::= ((\"abc\" [a-z]) | \"def\" | ([0-9]* \"\"))\n");
}

TEST(XGrammarGrammarBuilderTest, SelfRef) {
  GrammarBuilder builder;
  auto item_rule_id = builder.AddRule("item", ByteString("a"));
  builder.AddRule("list", Choices(EmptyStr(), Sequence(RuleRef(item_rule_id), SelfRef())));
  auto grammar = builder.Get("list");
  EXPECT_EQ(grammar.ToString(), "item ::= \"a\"\nlist ::= (\"\" | (item list))\n");
}

TEST(XGrammarGrammarBuilderTest, MixExprIdsInSpec) {
  GrammarBuilder builder;
  auto expr_id1 = builder.AddByteString("x");
  auto expr_id2 = builder.AddCharacterClass({{'a', 'z'}}, /*is_negative=*/true);
  builder.AddRule("root", Choices(expr_id1, expr_id2, ByteString("y")));
  auto grammar = builder.Get("root");
  EXPECT_EQ(grammar.ToString(), "root ::= (\"x\" | [^a-z] | \"y\")\n");
}

TEST(XGrammarGrammarBuilderTest, RepeatSpec) {
  GrammarBuilder builder;
  auto item_rule_id = builder.AddRule("item", ByteString("a"));
  builder.AddRule(
      "root", Sequence(Repeat(RuleRef(item_rule_id), 2, 4), Repeat(ByteString("b"), 0, -1))
  );
  auto grammar = builder.Get("root");
  // The non-rule-ref repeat element is wrapped into a new rule root_1.
  EXPECT_EQ(
      grammar.ToString(), "item ::= \"a\"\nroot ::= (item{2, 4} root_1{0, -1})\nroot_1 ::= \"b\"\n"
  );
}

TEST(XGrammarGrammarBuilderTest, TagDispatchSpec) {
  GrammarBuilder builder;
  auto tag_a_rule_id = builder.AddRule("tag_a", ByteString("A"));
  auto tag_b_rule_id = builder.AddRule("tag_b", ByteString("B"));
  builder.AddRule(
      "root", TagDispatch{{{"<a>", tag_a_rule_id}, {"<b>", tag_b_rule_id}}, true, {"</end>"}}
  );
  auto grammar = builder.Get("root");
  EXPECT_EQ(
      grammar.ToString(),
      "tag_a ::= \"A\"\n"
      "tag_b ::= \"B\"\n"
      "root ::= TagDispatch(\n"
      "  (\"<a>\", tag_a),\n"
      "  (\"<b>\", tag_b),\n"
      "  loop_after_dispatch=true,\n"
      "  excludes=(\"</end>\")\n"
      ")\n"
  );
}

TEST(XGrammarGrammarBuilderTest, GetWithNormalize) {
  GrammarBuilder builder;
  builder.AddRule("root", Choices(ByteString("a"), Choices(ByteString("b"), ByteString("c"))));
  auto grammar = builder.Get("root", /*normalize=*/true);
  EXPECT_EQ(grammar.ToString(), "root ::= ((\"a\") | (\"b\") | (\"c\"))\n");
}

TEST(XGrammarGrammarBuilderTest, AddSubGrammar) {
  auto sub_grammar1 = Grammar::FromEBNF("root ::= \"a\" sub\nsub ::= \"b\"\n");
  auto sub_grammar2 = Grammar::FromEBNF("root ::= \"c\" sub\nsub ::= \"d\"\n");

  GrammarBuilder builder;
  auto root_rule_id = builder.AddEmptyRule("root");
  auto sub_root_id1 = builder.AddSubGrammar(sub_grammar1);
  auto sub_root_id2 = builder.AddSubGrammar(sub_grammar2);
  builder.UpdateRuleBody(
      root_rule_id,
      builder.AddExpr(Choices(Sequence(RuleRef(sub_root_id1)), Sequence(RuleRef(sub_root_id2))))
  );
  auto grammar = builder.Get(root_rule_id);

  // Conflicting rule names are renamed, and rule refs are remapped to the new rule ids.
  EXPECT_EQ(
      grammar.ToString(),
      "root ::= ((root_1) | (root_2))\n"
      "root_1 ::= ((\"a\" sub))\n"
      "sub ::= ((\"b\"))\n"
      "root_2 ::= ((\"c\" sub_1))\n"
      "sub_1 ::= ((\"d\"))\n"
  );
}

TEST(XGrammarGrammarBuilderTest, AddSubGrammarWithLookaheadAndTagDispatch) {
  auto sub_grammar = Grammar::FromEBNF(
      "root ::= TagDispatch((\"<a>\", tag_a), loop_after_dispatch=false)\n"
      "tag_a ::= \"A\" (=\"end\")\n"
  );
  GrammarBuilder builder;
  builder.AddRule("root", ByteString("prefix"));
  auto sub_root_id = builder.AddSubGrammar(sub_grammar);
  auto grammar = builder.Get(sub_root_id);

  // The tag dispatch's rule refs are remapped, and the lookahead assertion is preserved.
  EXPECT_EQ(
      grammar.ToString(),
      "root ::= \"prefix\"\n"
      "root_1 ::= TagDispatch(\n"
      "  (\"<a>\", tag_a),\n"
      "  loop_after_dispatch=false,\n"
      "  excludes=()\n"
      ")\n"
      "tag_a ::= ((\"A\")) (=(\"end\"))\n"
  );
}

TEST(XGrammarGrammarBuilderTest, TypedGetters) {
  GrammarBuilder builder;
  auto byte_string_id = builder.AddByteString("ab");
  auto character_class_id =
      builder.AddCharacterClass({{'a', 'z'}, {'0', '9'}}, /*is_negative=*/true);
  auto sequence_id = builder.AddSequence({byte_string_id, character_class_id});
  auto choices_id = builder.AddChoices({sequence_id});
  auto rule_id = builder.AddRule("root", choices_id);
  auto rule_ref_id = builder.AddRuleRef(rule_id);
  auto repeat_id = builder.AddRepeat(rule_id, 1, 5);
  auto token_id = builder.AddToken({1, 2, 3});
  auto exclude_token_id = builder.AddExcludeToken({4, 5});
  auto grammar = builder.Get("root");
  const Grammar::Impl& impl = *grammar.operator->();

  EXPECT_EQ(impl.GetByteString(byte_string_id).ToString(), "ab");

  auto character_class = impl.GetCharacterClass(character_class_id);
  EXPECT_TRUE(character_class.is_negative);
  ASSERT_EQ(character_class.size(), 2);
  EXPECT_EQ(character_class[0].lower, 'a');
  EXPECT_EQ(character_class[0].upper, 'z');
  EXPECT_EQ(character_class[1].lower, '0');
  EXPECT_EQ(character_class[1].upper, '9');

  auto sequence = impl.GetSequence(sequence_id);
  ASSERT_EQ(sequence.size(), 2);
  EXPECT_EQ(sequence[0], byte_string_id);
  EXPECT_EQ(sequence[1], character_class_id);

  auto choices = impl.GetChoices(choices_id);
  ASSERT_EQ(choices.size(), 1);
  EXPECT_EQ(choices[0], sequence_id);

  EXPECT_EQ(impl.GetRuleRef(rule_ref_id), rule_id);

  auto repeat = impl.GetRepeat(repeat_id);
  EXPECT_EQ(repeat.rule_id, rule_id);
  EXPECT_EQ(repeat.min_repeat_count, 1);
  EXPECT_EQ(repeat.max_repeat_count, 5);

  EXPECT_EQ(impl.GetToken(token_id).ToVector(), (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(impl.GetExcludeToken(exclude_token_id).ToVector(), (std::vector<int32_t>{4, 5}));
}
