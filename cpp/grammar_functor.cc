/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_functor.cc
 */

#include "grammar_functor.h"

#include <xgrammar/xgrammar.h>

#include "grammar_data_structure.h"
#include "support/encoding.h"

namespace xgrammar {

/*************************** Impl of grammar functors ***************************/

/*!
 * \brief Eliminates single-element sequence or choice or character class in the grammar.
 * \example `A ::= choices("a")` --> `A ::= "a"` (the body is a string)
 * \example `A ::= sequence("a")` --> `A ::= "a"` (the body is a string)
 * \example `A ::= [a-a]` --> `A ::= "a"` (the body is a string)
 */
class SingleElementExprEliminator : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

 private:
  // Keep the sequence expr in lookahead assertion
  int32_t VisitLookaheadAssertion(int32_t lookahead_assertion_id) final {
    if (lookahead_assertion_id == -1) {
      return -1;
    }
    auto rule_expr = old_grammar_->GetRuleExpr(lookahead_assertion_id);
    XGRAMMAR_CHECK(rule_expr.type == RuleExprType::kSequence);

    std::vector<int32_t> sequence_ids;
    for (int32_t i : rule_expr) {
      sequence_ids.push_back(VisitExpr(i));
    }
    return builder_.AddSequence(sequence_ids);
  }

  int32_t VisitSequence(const RuleExpr& rule_expr) final {
    std::vector<int32_t> sequence_ids;
    for (int32_t i : rule_expr) {
      sequence_ids.push_back(VisitExpr(i));
    }
    if (sequence_ids.size() == 1) {
      return sequence_ids[0];
    }
    return builder_.AddSequence(sequence_ids);
  }

  int32_t VisitChoices(const RuleExpr& rule_expr) final {
    std::vector<int32_t> choice_ids;
    for (int32_t i : rule_expr) {
      choice_ids.push_back(VisitExpr(i));
    }
    if (choice_ids.size() == 1) {
      return choice_ids[0];
    }
    return builder_.AddChoices(choice_ids);
  }

  int32_t VisitCharacterClass(const RuleExpr& rule_expr) final {
    if (rule_expr.data_len == 3 && rule_expr[0] == 0 && rule_expr[1] == rule_expr[2]) {
      std::string str = PrintAsUTF8(rule_expr[1]);
      std::vector<int32_t> bytes;
      bytes.reserve(str.size());
      for (char c : str) {
        bytes.push_back(static_cast<int32_t>(c));
      }
      return builder_.AddByteString(bytes);
    }
    return builder_.AddRuleExpr(rule_expr);
  }
};

/*!
 * \brief Unwrap the rules containing nested expressions. After unwrapping, each rule will be in
 * the form: `rule_name ::= ("" | (element1_1 element1_2 ...) | (element2_1 element2_2 ...) | ...)`.
 *
 * I.e. a list of choices, each choice is a sequence of elements. Elements can be a character class
 * or a rule reference. And if the rule can be empty, the first choice will be an empty string.
 *
 * \example The rule `A ::= ((a) (((b)) (c)) "")` will be replaced by `A ::= ((a b c))`. One choice
 * containing a sequence of three elements. The empty string is removed.
 * \example The rule `A ::= (a | (b | (c | "")))` will be replaced by
 * `A ::= ("" | (a) | (b) | (c))`. The first choice is an empty string, and each of the other three
 * choices is a sequence containing a single element.
 * \example The rule `A ::= (a | (b (c | d)))` will be replaced by
 * `A ::= ((a) | (b B)), B ::= ((c) | (d))`. A new rule B is created to represent the nested
 * choices.
 */
class NestedRuleUnwrapper : public GrammarMutator {
 public:
  using GrammarMutator::GrammarMutator;

  Grammar Apply(const Grammar& grammar) final {
    Init(grammar);
    for (int i = 0; i < static_cast<int>(old_grammar_->NumRules()); ++i) {
      builder_.AddEmptyRule(old_grammar_->GetRule(i).name);
    }
    for (int i = 0; i < static_cast<int>(old_grammar_->NumRules()); ++i) {
      auto rule = old_grammar_->GetRule(i);
      auto rule_expr = old_grammar_->GetRuleExpr(rule.body_expr_id);
      cur_rule_name_ = rule.name;
      auto new_body_expr_id = VisitRuleBody(rule_expr);
      builder_.UpdateRuleBody(i, new_body_expr_id);
      builder_.AddLookaheadAssertion(i, VisitLookaheadAssertion(rule.lookahead_assertion_id));
    }
    return builder_.Get(old_grammar_->GetRootRule().name);
  }

 private:
  int32_t VisitLookaheadAssertion(int32_t lookahead_assertion_id) final {
    if (lookahead_assertion_id == -1) {
      return -1;
    }
    auto assertion_expr = old_grammar_->GetRuleExpr(lookahead_assertion_id);
    return builder_.AddSequence(VisitSequence_(assertion_expr));
  }

  /*! \brief Visit a RuleExpr as a rule body. */
  int32_t VisitRuleBody(const RuleExpr& rule_expr) {
    switch (rule_expr.type) {
      case RuleExprType::kSequence:
        return builder_.AddChoices({builder_.AddSequence(VisitSequence_(rule_expr))});
      case RuleExprType::kChoices:
        return builder_.AddChoices(VisitChoices_(rule_expr));
      case RuleExprType::kEmptyStr:
        return builder_.AddChoices({builder_.AddEmptyStr()});
      case RuleExprType::kByteString:
      case RuleExprType::kCharacterClass:
      case RuleExprType::kCharacterClassStar:
      case RuleExprType::kRuleRef:
        return builder_.AddChoices({builder_.AddSequence({builder_.AddRuleExpr(rule_expr)})});
      case RuleExprType::kTagDispatch:
        return VisitTagDispatch(rule_expr);
      default:
        XGRAMMAR_LOG(FATAL) << "Unexpected sequence type: " << static_cast<int>(rule_expr.type);
    }
  }

  /*!
   * \brief Visit a RuleExpr containing choices.
   * \returns A list of new choice RuleExpr ids.
   */
  std::vector<int32_t> VisitChoices_(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_choice_ids;
    bool found_empty = false;
    for (auto i : rule_expr) {
      auto choice_expr = old_grammar_->GetRuleExpr(i);
      switch (choice_expr.type) {
        case RuleExprType::kSequence:
          VisitSequenceInChoices(choice_expr, &new_choice_ids, &found_empty);
          break;
        case RuleExprType::kChoices:
          VisitChoicesInChoices(choice_expr, &new_choice_ids, &found_empty);
          break;
        case RuleExprType::kEmptyStr:
          found_empty = true;
          break;
        case RuleExprType::kByteString:
        case RuleExprType::kCharacterClass:
        case RuleExprType::kCharacterClassStar:
        case RuleExprType::kRuleRef:
          VisitElementInChoices(choice_expr, &new_choice_ids);
          break;
        case RuleExprType::kTagDispatch:
          XGRAMMAR_LOG(FATAL) << "TagDispatch should not be in choices";
        default:
          XGRAMMAR_LOG(FATAL) << "Unexpected choice type: " << static_cast<int>(choice_expr.type);
      }
    }
    if (found_empty) {
      new_choice_ids.insert(new_choice_ids.begin(), builder_.AddEmptyStr());
    }
    XGRAMMAR_ICHECK(new_choice_ids.size() >= 1);
    return new_choice_ids;
  }

  /*! \brief Visit a sequence RuleExpr that is one of a list of choices. */
  void VisitSequenceInChoices(
      const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids, bool* found_empty
  ) {
    auto sub_sequence_ids = VisitSequence_(rule_expr);
    if (sub_sequence_ids.size() == 0) {
      *found_empty = true;
    } else {
      new_choice_ids->push_back(builder_.AddSequence(sub_sequence_ids));
    }
  }

  /*! \brief Visit a choice RuleExpr that is one of a list of choices. */
  void VisitChoicesInChoices(
      const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids, bool* found_empty
  ) {
    auto sub_choice_ids = VisitChoices_(rule_expr);
    bool contains_empty = builder_.GetRuleExpr(sub_choice_ids[0]).type == RuleExprType::kEmptyStr;
    if (contains_empty) {
      *found_empty = true;
      new_choice_ids->insert(
          new_choice_ids->end(), sub_choice_ids.begin() + 1, sub_choice_ids.end()
      );
    } else {
      new_choice_ids->insert(new_choice_ids->end(), sub_choice_ids.begin(), sub_choice_ids.end());
    }
  }

  /*! \brief Visit an atom element RuleExpr that is one of a list of choices. */
  void VisitElementInChoices(const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids) {
    auto sub_expr_id = builder_.AddRuleExpr(rule_expr);
    new_choice_ids->push_back(builder_.AddSequence({sub_expr_id}));
  }

  /*!
   * \brief Visit a RuleExpr containing a sequence.
   * \returns A list of new sequence RuleExpr ids.
   */
  std::vector<int32_t> VisitSequence_(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_sequence_ids;
    for (auto i : rule_expr) {
      auto element_expr = old_grammar_->GetRuleExpr(i);
      switch (element_expr.type) {
        case RuleExprType::kSequence:
          VisitSequenceInSequence(element_expr, &new_sequence_ids);
          break;
        case RuleExprType::kChoices:
          VisitChoiceInSequence(element_expr, &new_sequence_ids);
          break;
        case RuleExprType::kEmptyStr:
          break;
        case RuleExprType::kByteString:
        case RuleExprType::kCharacterClass:
        case RuleExprType::kCharacterClassStar:
        case RuleExprType::kRuleRef:
          VisitElementInSequence(element_expr, &new_sequence_ids);
          break;
        case RuleExprType::kTagDispatch:
          XGRAMMAR_LOG(FATAL) << "TagDispatch should not be in sequence";
        default:
          XGRAMMAR_LOG(FATAL) << "Unexpected sequence type: "
                              << static_cast<int>(element_expr.type);
      }
    }
    return new_sequence_ids;
  }

  /*! \brief Visit a sequence RuleExpr that is one element in another sequence. */
  void VisitSequenceInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    auto sub_sequence_ids = VisitSequence_(rule_expr);
    new_sequence_ids->insert(
        new_sequence_ids->end(), sub_sequence_ids.begin(), sub_sequence_ids.end()
    );
  }

  /*! \brief Visit a choice RuleExpr that is one element in a sequence. */
  void VisitChoiceInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    auto sub_choice_ids = VisitChoices_(rule_expr);
    if (sub_choice_ids.size() == 1) {
      auto choice_element_expr = builder_.GetRuleExpr(sub_choice_ids[0]);
      if (choice_element_expr.type != RuleExprType::kEmptyStr) {
        new_sequence_ids->insert(
            new_sequence_ids->end(), choice_element_expr.begin(), choice_element_expr.end()
        );
      }
    } else {
      auto new_choice_id = builder_.AddChoices(sub_choice_ids);
      auto new_choice_rule_id = builder_.AddRuleWithHint(cur_rule_name_ + "_choice", new_choice_id);
      new_sequence_ids->push_back(builder_.AddRuleRef(new_choice_rule_id));
    }
  }

  /*! \brief Visit an atom element RuleExpr that is in a sequence. */
  void VisitElementInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    new_sequence_ids->push_back(builder_.AddRuleExpr(rule_expr));
  }
};

class ByteStringFuser : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

 private:
  /*!
   * \brief Visit a RuleExpr containing a sequence.
   * \returns A list of new sequence RuleExpr ids.
   */
  int32_t VisitSequence(const RuleExpr& rule_expr) final {
    std::vector<int32_t> new_sequence_ids;
    std::vector<int32_t> cur_byte_string;
    for (auto i : rule_expr) {
      auto element_expr = old_grammar_->GetRuleExpr(i);
      if (element_expr.type == RuleExprType::kByteString) {
        cur_byte_string.insert(cur_byte_string.end(), element_expr.begin(), element_expr.end());
        continue;
      } else {
        if (!cur_byte_string.empty()) {
          new_sequence_ids.push_back(builder_.AddByteString(cur_byte_string));
          cur_byte_string.clear();
        }
        new_sequence_ids.push_back(builder_.AddRuleExpr(element_expr));
      }
    }
    if (!cur_byte_string.empty()) {
      new_sequence_ids.push_back(builder_.AddByteString(cur_byte_string));
    }
    return builder_.AddSequence(new_sequence_ids);
  }
};

class GrammarNormalizerImpl : public GrammarMutator {
 public:
  GrammarNormalizerImpl() = default;

  Grammar Apply(const Grammar& grammar) final {
    std::vector<std::unique_ptr<GrammarMutator>> normalizer_mutators = GetNormalizerList();
    old_grammar_ = grammar;
    for (auto& mutator : normalizer_mutators) {
      old_grammar_ = mutator->Apply(old_grammar_);
    }
    return old_grammar_;
  }

 private:
  // Return the list of all normalizers in the class. The normalizers are applied one by one.
  std::vector<std::unique_ptr<GrammarMutator>> GetNormalizerList() {
    std::vector<std::unique_ptr<GrammarMutator>> normalizer_mutators;
    normalizer_mutators.emplace_back(std::make_unique<SingleElementExprEliminator>());
    normalizer_mutators.emplace_back(std::make_unique<NestedRuleUnwrapper>());
    normalizer_mutators.emplace_back(std::make_unique<ByteStringFuser>());
    return normalizer_mutators;
  }
};

class SubGrammarAdder : public GrammarMutator {
 public:
  SubGrammarAdder() = default;

 protected:
  /*!
   * \brief Visit a subgrammar and add the rules to the builder.
   * \param grammar The subgrammar to visit.
   * \return The new id of the root rule of this subgrammar.
   */
  int32_t VisitSubGrammar(const Grammar& grammar) {
    old_grammar_ = grammar;
    new_rule_ids_names.reserve(grammar->NumRules());
    new_rule_ids_names.clear();
    for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
      auto new_name = builder_.GetNewRuleName(grammar->GetRule(i).name);
      auto new_id = builder_.AddEmptyRule(new_name);
      new_rule_ids_names.emplace_back(new_id, new_name);
    }
    for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
      auto rule = grammar->GetRule(i);
      cur_rule_name_ = new_rule_ids_names[i].second;
      auto new_body_expr_id = VisitExpr(rule.body_expr_id);
      builder_.UpdateRuleBody(new_rule_ids_names[i].first, new_body_expr_id);
      auto new_lookahead_assertion_id = VisitLookaheadAssertion(rule.lookahead_assertion_id);
      builder_.AddLookaheadAssertion(new_rule_ids_names[i].first, new_lookahead_assertion_id);
    }
    return new_rule_ids_names[grammar->GetRootRuleId()].first;
  }

  std::vector<std::pair<int32_t, std::string>> new_rule_ids_names;
};

class GrammarUnionFunctorImpl : public SubGrammarAdder {
 public:
  GrammarUnionFunctorImpl() = default;

  Grammar Apply(const std::vector<Grammar>& grammars) {
    builder_ = GrammarBuilder();
    auto root_rule_id = builder_.AddEmptyRule("root");

    std::vector<int32_t> new_root_choices;
    new_root_choices.reserve(grammars.size());

    for (const auto& grammar : grammars) {
      auto new_root_id_for_grammar = VisitSubGrammar(grammar);
      auto new_rule_ref = builder_.AddRuleRef(new_root_id_for_grammar);
      auto new_rule_ref_seq = builder_.AddSequence({new_rule_ref});
      new_root_choices.push_back(new_rule_ref_seq);
    }

    builder_.UpdateRuleBody(root_rule_id, builder_.AddChoices(new_root_choices));
    return builder_.Get(root_rule_id);
  }
};

class GrammarConcatFunctorImpl : public SubGrammarAdder {
 public:
  GrammarConcatFunctorImpl() = default;

  Grammar Apply(const std::vector<Grammar>& grammars) {
    builder_ = GrammarBuilder();
    auto root_rule_id = builder_.AddEmptyRule("root");

    std::vector<int32_t> new_root_sequence;
    new_root_sequence.reserve(grammars.size());

    for (const auto& grammar : grammars) {
      auto new_root_id_for_grammar = VisitSubGrammar(grammar);
      auto new_rule_ref = builder_.AddRuleRef(new_root_id_for_grammar);
      new_root_sequence.push_back(new_rule_ref);
    }

    auto new_root_seq = builder_.AddSequence(new_root_sequence);
    builder_.UpdateRuleBody(root_rule_id, builder_.AddChoices({new_root_seq}));

    return builder_.Get(root_rule_id);
  }
};

/*************************** Forward grammar functors to their impl ***************************/

Grammar GrammarNormalizer::Apply(const Grammar& grammar) {
  return GrammarNormalizerImpl().Apply(grammar);
}

Grammar GrammarUnionFunctor::Apply(const std::vector<Grammar>& grammars) {
  return GrammarUnionFunctorImpl().Apply(grammars);
}

Grammar GrammarConcatFunctor::Apply(const std::vector<Grammar>& grammars) {
  return GrammarConcatFunctorImpl().Apply(grammars);
}

}  // namespace xgrammar
