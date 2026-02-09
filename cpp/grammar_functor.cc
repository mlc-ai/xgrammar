/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_functor.cc
 */

#include "grammar_functor.h"

#include <xgrammar/xgrammar.h>

#include <bitset>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <stack>
#include <tuple>
#include <vector>

#include "compiled_grammar_impl.h"
#include "fsm_builder.h"
#include "grammar_builder.h"
#include "grammar_impl.h"
#include "support/container.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

using GrammarExpr = Grammar::Impl::GrammarExpr;
using ExprType = Grammar::Impl::GrammarExprType;

/*************************** Impl of grammar constructors ***************************/

/*!
 * \brief Base class for grammar mutators that add subgrammars.
 *
 * Provides functionality to visit a subgrammar and add its rules to the builder
 * while maintaining proper rule references and names.
 */
class SubGrammarAdderImpl : public GrammarMutator {
 public:
  SubGrammarAdderImpl() = default;

  /*!
   * \brief Visit a subgrammar and add the rules to the builder.
   * \param grammar The subgrammar to visit.
   * \return The new id of the root rule of this subgrammar.
   */
  int32_t ApplyWithBuilder(GrammarBuilder* builder, const Grammar& sub_grammar) {
    InitGrammar(sub_grammar);
    InitBuilder(builder);
    new_rule_ids_names.reserve(base_grammar_->NumRules());
    new_rule_ids_names.clear();
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto new_name = builder_->GetNewRuleName(base_grammar_->GetRule(i).name);
      auto new_id = builder_->AddEmptyRule(new_name);
      new_rule_ids_names.emplace_back(new_id, new_name);
    }
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      cur_rule_name_ = new_rule_ids_names[i].second;
      auto new_body_expr_id = VisitExpr(rule.body_expr_id);
      builder_->UpdateRuleBody(new_rule_ids_names[i].first, new_body_expr_id);
      auto new_lookahead_assertion_id = VisitLookaheadAssertion(rule.lookahead_assertion_id);
      builder_->UpdateLookaheadAssertion(new_rule_ids_names[i].first, new_lookahead_assertion_id);
    }
    return new_rule_ids_names[base_grammar_->GetRootRuleId()].first;
  }

  int32_t VisitRuleRef(const GrammarExpr& grammar_expr) final {
    return builder_->AddRuleRef(new_rule_ids_names[grammar_expr[0]].first);
  }

  int32_t VisitRepeat(const GrammarExpr& grammar_expr) final {
    return builder_->AddRepeat(
        new_rule_ids_names[grammar_expr[0]].first, grammar_expr[1], grammar_expr[2]
    );
  }

  int32_t VisitTagDispatch(const GrammarExpr& grammar_expr) final {
    Grammar::Impl::TagDispatch old_tag_dispatch = base_grammar_->GetTagDispatch(grammar_expr);
    Grammar::Impl::TagDispatch new_tag_dispatch;
    new_tag_dispatch.stop_eos = old_tag_dispatch.stop_eos;
    for (const auto& [tag, rule_id] : old_tag_dispatch.tag_rule_pairs) {
      new_tag_dispatch.tag_rule_pairs.emplace_back(tag, new_rule_ids_names[rule_id].first);
    }
    new_tag_dispatch.stop_str = old_tag_dispatch.stop_str;
    new_tag_dispatch.loop_after_dispatch = old_tag_dispatch.loop_after_dispatch;
    new_tag_dispatch.excluded_str = old_tag_dispatch.excluded_str;
    return builder_->AddTagDispatch(new_tag_dispatch);
  }

  std::vector<std::pair<int32_t, std::string>> new_rule_ids_names;
};

/*!
 * \brief Implementation of grammar union operation.
 *
 * Creates a new grammar that accepts strings from any of the input grammars.
 * The resulting grammar has a new root rule that chooses between the root rules
 * of all input grammars.
 */
class GrammarUnionFunctorImpl : public GrammarMutator {
 public:
  GrammarUnionFunctorImpl() = default;

  Grammar Apply(const std::vector<Grammar>& grammars) {
    InitGrammar();
    InitBuilder();
    auto root_rule_id = builder_->AddEmptyRule("root");

    std::vector<int32_t> new_root_choices;
    new_root_choices.reserve(grammars.size());

    for (const auto& grammar : grammars) {
      auto new_root_id_for_grammar = SubGrammarAdderImpl().ApplyWithBuilder(builder_, grammar);
      auto new_rule_ref = builder_->AddRuleRef(new_root_id_for_grammar);
      auto new_rule_ref_seq = builder_->AddSequence({new_rule_ref});
      new_root_choices.push_back(new_rule_ref_seq);
    }

    builder_->UpdateRuleBody(root_rule_id, builder_->AddChoices(new_root_choices));
    return builder_->Get(root_rule_id);
  }

  // Avoid hiding the original Apply(const Grammar&)
  Grammar Apply(const Grammar& grammar) final {
    XGRAMMAR_LOG(FATAL) << "Should not be called";
    XGRAMMAR_UNREACHABLE();
  }
};

/*!
 * \brief Implementation of grammar concatenation operation.
 *
 * Creates a new grammar that accepts strings that are concatenations of strings
 * from the input grammars in order. The resulting grammar has a new root rule
 * that concatenates the root rules of all input grammars.
 */
class GrammarConcatFunctorImpl : public GrammarMutator {
 public:
  GrammarConcatFunctorImpl() = default;

  Grammar Apply(const std::vector<Grammar>& grammars) {
    InitGrammar();
    InitBuilder();
    auto root_rule_id = builder_->AddEmptyRule("root");

    std::vector<int32_t> new_root_sequence;
    new_root_sequence.reserve(grammars.size());

    for (const auto& grammar : grammars) {
      auto new_root_id_for_grammar = SubGrammarAdderImpl().ApplyWithBuilder(builder_, grammar);
      auto new_rule_ref = builder_->AddRuleRef(new_root_id_for_grammar);
      new_root_sequence.push_back(new_rule_ref);
    }

    auto new_root_seq = builder_->AddSequence(new_root_sequence);
    builder_->UpdateRuleBody(root_rule_id, builder_->AddChoices({new_root_seq}));

    return builder_->Get(root_rule_id);
  }

  // Avoid hiding the original Apply(const Grammar&)
  Grammar Apply(const Grammar& grammar) final {
    XGRAMMAR_LOG(FATAL) << "Should not be called";
    XGRAMMAR_UNREACHABLE();
  }
};

/*************************** Impl of grammar normalizers ***************************/

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
  int32_t VisitSequence(const GrammarExpr& grammar_expr) final {
    std::vector<int32_t> sequence_ids;
    for (int32_t i : grammar_expr) {
      sequence_ids.push_back(VisitExpr(i));
    }
    if (sequence_ids.size() == 1) {
      return sequence_ids[0];
    }
    return builder_->AddSequence(sequence_ids);
  }

  int32_t VisitChoices(const GrammarExpr& grammar_expr) final {
    std::vector<int32_t> choice_ids;
    for (int32_t i : grammar_expr) {
      choice_ids.push_back(VisitExpr(i));
    }
    if (choice_ids.size() == 1) {
      return choice_ids[0];
    }
    return builder_->AddChoices(choice_ids);
  }

  int32_t VisitCharacterClass(const GrammarExpr& grammar_expr) final {
    if (grammar_expr.data_len == 3 && grammar_expr[0] == 0 && grammar_expr[1] == grammar_expr[2]) {
      std::string str = CharToUTF8(grammar_expr[1]);
      std::vector<int32_t> bytes;
      bytes.reserve(str.size());
      for (char c : str) {
        bytes.push_back(static_cast<int32_t>(c));
      }
      return builder_->AddByteString(bytes);
    }
    return builder_->AddGrammarExpr(grammar_expr);
  }
};

/*!
 * \brief Take a grammar from SingleElementExprEliminator and normalize the structure of the
 * grammar.
 *
 * \note The normalized form:
 * Each rule should be either:
 * - A sequence of choices, each choice is a sequence of elements. Elements can be a character
 *   class, a byte string, or a rule reference. Only the first choice can be an empty string,
 *   indicating the rule can be empty. E.g.
 *   `rule_name ::= ("" | (element1_1 element1_2 ...) | (element2_1 element2_2 ...) | ...)`
 * - A macro. Now only TagDispatch is supported.
 *
 * The lookahead assertion should be a sequence.
 *
 * New rules may be created to make every rule fit the normalized form.
 *
 * \example `A ::= ((a) (((b)) (c)) "")` -> `A ::= ((a b c))`
 * \example `A ::= (a | (b | (c | "")))` -> `A ::= ("" | (a) | (b) | (c))`
 * \example `A ::= (a | (b (c | d)))` -> `A ::= ((a) | (b A_1)), A_1 ::= ((c) | (d))`
 * \example `A ::= (a | TagDispatch((tag1, rule1)))` -> `A ::= ((a) | (A_1)), A_1 ::=
 * TagDispatch((tag1, rule1))`
 */
class StructureNormalizerImpl : public GrammarMutator {
 public:
  using GrammarMutator::GrammarMutator;

  Grammar Apply(const Grammar& grammar) final {
    auto grammar_new = SingleElementExprEliminator().Apply(grammar);
    InitGrammar(grammar_new);
    InitBuilder();
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      builder_->AddEmptyRule(base_grammar_->GetRule(i).name);
    }
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
      cur_rule_name_ = rule.name;
      auto new_body_expr_id = VisitRuleBody(grammar_expr);
      builder_->UpdateRuleBody(i, new_body_expr_id);
      builder_->UpdateLookaheadAssertion(i, VisitLookaheadAssertion(rule.lookahead_assertion_id));
    }
    return builder_->Get(base_grammar_->GetRootRule().name);
  }

 private:
  int32_t VisitLookaheadAssertion(int32_t lookahead_assertion_id) final {
    if (lookahead_assertion_id == -1) {
      return -1;
    }
    auto assertion_expr = base_grammar_->GetGrammarExpr(lookahead_assertion_id);
    switch (assertion_expr.type) {
      case GrammarExprType::kSequence:
        return builder_->AddSequence(VisitSequence_(assertion_expr));
      case GrammarExprType::kChoices:
        XGRAMMAR_LOG(FATAL) << "Choices in lookahead assertion are not supported yet";
        XGRAMMAR_UNREACHABLE();
      case GrammarExprType::kEmptyStr:
        XGRAMMAR_LOG(FATAL) << "Empty string should not be in lookahead assertion";
        XGRAMMAR_UNREACHABLE();
      case GrammarExprType::kTagDispatch:
        XGRAMMAR_LOG(FATAL) << "TagDispatch should not be in lookahead assertion";
        XGRAMMAR_UNREACHABLE();
      case GrammarExprType::kByteString:
      case GrammarExprType::kCharacterClass:
      case GrammarExprType::kCharacterClassStar:
      case GrammarExprType::kRuleRef:
      case GrammarExprType::kRepeat:
        return builder_->AddSequence({builder_->AddGrammarExpr(assertion_expr)});
      default:
        XGRAMMAR_LOG(FATAL) << "Unexpected lookahead assertion type: "
                            << static_cast<int>(assertion_expr.type);
        XGRAMMAR_UNREACHABLE();
    }
  }

  /*! \brief Visit a GrammarExpr as a rule body. */
  int32_t VisitRuleBody(const GrammarExpr& grammar_expr) {
    switch (grammar_expr.type) {
      case GrammarExprType::kSequence:
        return builder_->AddChoices({builder_->AddSequence(VisitSequence_(grammar_expr))});
      case GrammarExprType::kChoices:
        return builder_->AddChoices(VisitChoices_(grammar_expr));
      case GrammarExprType::kEmptyStr:
        return builder_->AddChoices({builder_->AddEmptyStr()});
      case GrammarExprType::kByteString:
      case GrammarExprType::kCharacterClass:
      case GrammarExprType::kCharacterClassStar:
      case GrammarExprType::kRuleRef:
      case GrammarExprType::kRepeat:
        return builder_->AddChoices({builder_->AddSequence({builder_->AddGrammarExpr(grammar_expr)})
        });
      case GrammarExprType::kTagDispatch:
        return VisitTagDispatch(grammar_expr);
      default:
        XGRAMMAR_LOG(FATAL) << "Unexpected sequence type: " << static_cast<int>(grammar_expr.type);
        XGRAMMAR_UNREACHABLE();
    }
  }

  /*!
   * \brief Visit a GrammarExpr containing choices.
   * \returns A list of new choice GrammarExpr ids.
   */
  std::vector<int32_t> VisitChoices_(const GrammarExpr& grammar_expr) {
    std::vector<int32_t> new_choice_ids;
    bool found_empty = false;
    for (auto i : grammar_expr) {
      auto choice_expr = base_grammar_->GetGrammarExpr(i);
      switch (choice_expr.type) {
        case GrammarExprType::kSequence:
          VisitSequenceInChoices(choice_expr, &new_choice_ids, &found_empty);
          break;
        case GrammarExprType::kChoices:
          VisitChoicesInChoices(choice_expr, &new_choice_ids, &found_empty);
          break;
        case GrammarExprType::kEmptyStr:
          found_empty = true;
          break;
        case GrammarExprType::kByteString:
        case GrammarExprType::kCharacterClass:
        case GrammarExprType::kCharacterClassStar:
        case GrammarExprType::kRuleRef:
        case GrammarExprType::kRepeat:
          VisitElementInChoices(choice_expr, &new_choice_ids);
          break;
        case GrammarExprType::kTagDispatch: {
          auto tag_dispatch_expr_id = VisitTagDispatch(choice_expr);
          auto new_rule_id = builder_->AddRuleWithHint(cur_rule_name_, tag_dispatch_expr_id);
          auto new_sequence_id = builder_->AddSequence({builder_->AddRuleRef(new_rule_id)});
          new_choice_ids.push_back(new_sequence_id);
          break;
        }
        default:
          XGRAMMAR_LOG(FATAL) << "Unexpected choice type: " << static_cast<int>(choice_expr.type);
      }
    }
    if (found_empty) {
      new_choice_ids.insert(new_choice_ids.begin(), builder_->AddEmptyStr());
    }
    XGRAMMAR_ICHECK(new_choice_ids.size() >= 1);
    return new_choice_ids;
  }

  /*! \brief Visit a sequence GrammarExpr that is one of a list of choices. */
  void VisitSequenceInChoices(
      const GrammarExpr& grammar_expr, std::vector<int32_t>* new_choice_ids, bool* found_empty
  ) {
    auto sub_sequence_ids = VisitSequence_(grammar_expr);
    if (sub_sequence_ids.size() == 0) {
      *found_empty = true;
    } else {
      new_choice_ids->push_back(builder_->AddSequence(sub_sequence_ids));
    }
  }

  /*! \brief Visit a choice GrammarExpr that is one of a list of choices. */
  void VisitChoicesInChoices(
      const GrammarExpr& grammar_expr, std::vector<int32_t>* new_choice_ids, bool* found_empty
  ) {
    auto sub_choice_ids = VisitChoices_(grammar_expr);
    bool contains_empty =
        builder_->GetGrammarExpr(sub_choice_ids[0]).type == GrammarExprType::kEmptyStr;
    if (contains_empty) {
      *found_empty = true;
      new_choice_ids->insert(
          new_choice_ids->end(), sub_choice_ids.begin() + 1, sub_choice_ids.end()
      );
    } else {
      new_choice_ids->insert(new_choice_ids->end(), sub_choice_ids.begin(), sub_choice_ids.end());
    }
  }

  /*! \brief Visit an atom element GrammarExpr that is one of a list of choices. */
  void VisitElementInChoices(
      const GrammarExpr& grammar_expr, std::vector<int32_t>* new_choice_ids
  ) {
    auto sub_expr_id = builder_->AddGrammarExpr(grammar_expr);
    new_choice_ids->push_back(builder_->AddSequence({sub_expr_id}));
  }

  /*!
   * \brief Visit a GrammarExpr containing a sequence.
   * \returns A list of new sequence GrammarExpr ids.
   */
  std::vector<int32_t> VisitSequence_(const GrammarExpr& grammar_expr) {
    std::vector<int32_t> new_sequence_ids;
    for (auto i : grammar_expr) {
      auto element_expr = base_grammar_->GetGrammarExpr(i);
      switch (element_expr.type) {
        case GrammarExprType::kSequence:
          VisitSequenceInSequence(element_expr, &new_sequence_ids);
          break;
        case GrammarExprType::kChoices:
          VisitChoiceInSequence(element_expr, &new_sequence_ids);
          break;
        case GrammarExprType::kEmptyStr:
          break;
        case GrammarExprType::kByteString:
        case GrammarExprType::kCharacterClass:
        case GrammarExprType::kCharacterClassStar:
        case GrammarExprType::kRuleRef:
        case GrammarExprType::kRepeat:
          VisitElementInSequence(element_expr, &new_sequence_ids);
          break;
        case GrammarExprType::kTagDispatch: {
          auto tag_dispatch_expr_id = VisitTagDispatch(element_expr);
          auto new_rule_id = builder_->AddRuleWithHint(cur_rule_name_, tag_dispatch_expr_id);
          new_sequence_ids.push_back(builder_->AddRuleRef(new_rule_id));
          break;
        }
        default:
          XGRAMMAR_LOG(FATAL) << "Unexpected sequence type: "
                              << static_cast<int>(element_expr.type);
      }
    }
    return new_sequence_ids;
  }

  /*! \brief Visit a sequence GrammarExpr that is one element in another sequence. */
  void VisitSequenceInSequence(
      const GrammarExpr& grammar_expr, std::vector<int32_t>* new_sequence_ids
  ) {
    auto sub_sequence_ids = VisitSequence_(grammar_expr);
    new_sequence_ids->insert(
        new_sequence_ids->end(), sub_sequence_ids.begin(), sub_sequence_ids.end()
    );
  }

  /*! \brief Visit a choice GrammarExpr that is one element in a sequence. */
  void VisitChoiceInSequence(
      const GrammarExpr& grammar_expr, std::vector<int32_t>* new_sequence_ids
  ) {
    auto sub_choice_ids = VisitChoices_(grammar_expr);
    if (sub_choice_ids.size() == 1) {
      auto choice_element_expr = builder_->GetGrammarExpr(sub_choice_ids[0]);
      if (choice_element_expr.type != GrammarExprType::kEmptyStr) {
        new_sequence_ids->insert(
            new_sequence_ids->end(), choice_element_expr.begin(), choice_element_expr.end()
        );
      }
    } else {
      auto new_choice_id = builder_->AddChoices(sub_choice_ids);
      auto new_choice_rule_id = builder_->AddRuleWithHint(cur_rule_name_, new_choice_id);
      new_sequence_ids->push_back(builder_->AddRuleRef(new_choice_rule_id));
    }
  }

  /*! \brief Visit an atom element GrammarExpr that is in a sequence. */
  void VisitElementInSequence(
      const GrammarExpr& grammar_expr, std::vector<int32_t>* new_sequence_ids
  ) {
    new_sequence_ids->push_back(builder_->AddGrammarExpr(grammar_expr));
  }
};

/*!
 * \brief A class that normalizes a grammar by applying a series of transformations.
 *
 * The normalizer applies the following transformations in order:
 * 1. SingleElementExprEliminator - Eliminates single element expressions
 * 2. NestedRuleUnwrapper - Unwraps nested rules
 */
class GrammarNormalizerImpl {
 public:
  GrammarNormalizerImpl() = default;

  Grammar Apply(const Grammar& grammar) {
    auto renamed_grammar = RootRuleRenamer::Apply(grammar);
    return StructureNormalizerImpl().Apply(renamed_grammar);
  }
};

/*************************** Impl of grammar optimizers ***************************/

/*!
 * \brief Inline rules that can be inlined.
 *
 * Now we only inline rule references that:
 * 1. at the beginning of a sequence
 * 2. The rule should be a sequence of choices, cannot be empty, cannot refer to other rules
 */
class RuleInlinerImpl : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

 private:
  int32_t VisitChoices(const GrammarExpr& grammar_expr) final {
    std::vector<int32_t> new_choice_ids;
    for (int i : grammar_expr) {
      auto choice_expr = base_grammar_->GetGrammarExpr(i);
      if (choice_expr.type == GrammarExprType::kEmptyStr) {
        new_choice_ids.push_back(VisitExpr(i));
        continue;
      }
      XGRAMMAR_ICHECK(choice_expr.type == GrammarExprType::kSequence);
      auto first_element = base_grammar_->GetGrammarExpr(choice_expr[0]);
      if (first_element.type != GrammarExprType::kRuleRef) {
        new_choice_ids.push_back(VisitExpr(choice_expr));
        continue;
      }
      auto rule_ref_id = first_element[0];
      if (can_rule_be_inlined_.count(rule_ref_id) == 0) {
        can_rule_be_inlined_[rule_ref_id] = CheckIfRuleCanBeInlined(rule_ref_id);
      }
      if (!can_rule_be_inlined_[rule_ref_id]) {
        new_choice_ids.push_back(VisitExpr(choice_expr));
        continue;
      }

      // Do inlining
      std::vector<int32_t> other_elements;
      for (int i = 1; i < choice_expr.size(); ++i) {
        other_elements.push_back(VisitExpr(choice_expr[i]));
      }

      auto ref_rule = base_grammar_->GetRule(rule_ref_id);
      auto ref_grammar_expr = base_grammar_->GetGrammarExpr(ref_rule.body_expr_id);

      for (auto ref_choice_id : ref_grammar_expr) {
        auto ref_choice_expr = base_grammar_->GetGrammarExpr(ref_choice_id);
        XGRAMMAR_ICHECK(ref_choice_expr.type == GrammarExprType::kSequence);
        std::vector<int32_t> choice_to_add;
        for (auto ref_element_id : ref_choice_expr) {
          choice_to_add.push_back(VisitExpr(ref_element_id));
        }
        choice_to_add.insert(choice_to_add.end(), other_elements.begin(), other_elements.end());
        new_choice_ids.push_back(builder_->AddSequence(choice_to_add));
      }
    }
    return builder_->AddChoices(new_choice_ids);
  }

  /**
   * The rule should be: a sequence of choices, cannot be empty, cannot refer to other rules
   */
  bool CheckIfRuleCanBeInlined(int32_t rule_id) {
    auto rule = base_grammar_->GetRule(rule_id);
    auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
    if (grammar_expr.type != GrammarExprType::kChoices) {
      return false;
    }
    if (grammar_expr.size() == 0) {
      return false;
    }
    for (auto choice_id : grammar_expr) {
      auto choice_expr = base_grammar_->GetGrammarExpr(choice_id);
      if (choice_expr.type == GrammarExprType::kEmptyStr) {
        return false;
      }
      XGRAMMAR_ICHECK(choice_expr.type == GrammarExprType::kSequence);
      for (auto element_id : choice_expr) {
        auto element_expr = base_grammar_->GetGrammarExpr(element_id);
        if (element_expr.type == GrammarExprType::kRuleRef) {
          return false;
        }
      }
    }
    return true;
  }

  std::unordered_map<int32_t, bool> can_rule_be_inlined_;
};

/*!
 * \brief Analyze all referenced rules or the main rule. Return a list of all referenced rule ids.
 * This is useful for dead code elimination.
 */
class UsedRulesAnalyzer : public GrammarVisitor<std::vector<int32_t>> {
 public:
  UsedRulesAnalyzer() = default;

  std::vector<int32_t> Apply(const Grammar& grammar) final {
    InitGrammar(grammar);

    std::set<int32_t> visited;

    std::queue<int32_t>().swap(visit_queue_);

    visit_queue_.push(base_grammar_->GetRootRuleId());
    while (!visit_queue_.empty()) {
      auto rule_id = visit_queue_.front();
      visit_queue_.pop();
      if (visited.count(rule_id)) {
        continue;
      }
      visited.insert(rule_id);
      auto rule = base_grammar_->GetRule(rule_id);
      VisitExpr(rule.body_expr_id);
      if (rule.lookahead_assertion_id != -1) {
        VisitExpr(rule.lookahead_assertion_id);
      }
    }

    return std::vector<int32_t>(visited.begin(), visited.end());
  }

  void VisitTagDispatch(const GrammarExpr& grammar_expr) {
    for (int i = 0;
         i < grammar_expr.size() - Grammar::Impl::TagDispatch::kTagDispatchExtraParameter;
         i += 2) {
      visit_queue_.push(grammar_expr[i + 1]);
    }
  }

  void VisitRuleRef(const GrammarExpr& grammar_expr) { visit_queue_.push(grammar_expr[0]); }

  void VisitRepeat(const GrammarExpr& grammar_expr) { visit_queue_.push(grammar_expr[0]); }

 private:
  std::queue<int32_t> visit_queue_;
};

class DeadCodeEliminatorImpl : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

  Grammar Apply(const Grammar& grammar) final {
    InitGrammar(grammar);
    InitBuilder();
    auto used_rules = UsedRulesAnalyzer().Apply(grammar);
    rule_id_map_.clear();
    for (auto rule_id : used_rules) {
      rule_id_map_[rule_id] = builder_->AddEmptyRule(grammar->GetRule(rule_id).name);
    }
    for (auto rule_id : used_rules) {
      auto rule = grammar->GetRule(rule_id);
      auto new_body_expr_id = VisitExpr(rule.body_expr_id);
      builder_->UpdateRuleBody(rule_id_map_[rule_id], new_body_expr_id);
      builder_->UpdateLookaheadAssertion(
          rule_id_map_[rule_id], VisitLookaheadAssertion(rule.lookahead_assertion_id)
      );
    }
    XGRAMMAR_CHECK(rule_id_map_.count(grammar->GetRootRuleId()) > 0);
    return builder_->Get(rule_id_map_[grammar->GetRootRuleId()]);
  }

  int32_t VisitTagDispatch(const GrammarExpr& grammar_expr) final {
    Grammar::Impl::TagDispatch tag_dispatch = base_grammar_->GetTagDispatch(grammar_expr);
    for (auto& [tag, rule_id] : tag_dispatch.tag_rule_pairs) {
      XGRAMMAR_DCHECK(rule_id_map_.count(rule_id) > 0);
      rule_id = rule_id_map_[rule_id];
    }

    return builder_->AddTagDispatch(tag_dispatch);
  }

  int32_t VisitRuleRef(const GrammarExpr& grammar_expr) final {
    XGRAMMAR_DCHECK(rule_id_map_.count(grammar_expr[0]) > 0);
    auto new_rule_id = rule_id_map_[grammar_expr[0]];
    return builder_->AddRuleRef(new_rule_id);
  }

  int32_t VisitRepeat(const GrammarExpr& grammar_expr) final {
    XGRAMMAR_DCHECK(rule_id_map_.count(grammar_expr[0]) > 0);
    auto new_rule_id = rule_id_map_[grammar_expr[0]];
    return builder_->AddRepeat(new_rule_id, grammar_expr[1], grammar_expr[2]);
  }

 private:
  std::unordered_map<int32_t, int32_t> rule_id_map_;
};

class LookaheadAssertionAnalyzerImpl : public GrammarMutator {
 public:
  using GrammarMutator::GrammarMutator;

  Grammar Apply(const Grammar& grammar) final {
    InitGrammar(grammar);
    InitBuilder(grammar);
    auto root_rule = grammar->GetRootRule();
    auto root_grammar_expr = base_grammar_->GetGrammarExpr(root_rule.body_expr_id);
    if (root_grammar_expr.type == GrammarExprType::kTagDispatch) {
      return grammar;
    }
    for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
      auto rule = grammar->GetRule(i);
      if (i == grammar->GetRootRuleId()) {
        continue;
      }
      if (rule.lookahead_assertion_id != -1) {
        builder_->UpdateLookaheadExact(i, IsExactLookaheadAssertion(i));
        continue;
      }
      auto look_head_assertion_id = DetectLookaheadAssertion(i);
      if (look_head_assertion_id != -1) {
        builder_->UpdateLookaheadAssertion(i, look_head_assertion_id);
        builder_->UpdateLookaheadExact(i);
      }
    }
    return builder_->Get(grammar->GetRootRuleId());
  }

  bool IsExactLookaheadAssertion(int32_t rule_id) {
    XGRAMMAR_DCHECK(base_grammar_->GetRule(rule_id).lookahead_assertion_id != -1);
    bool found = false;
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
      if (grammar_expr.type == GrammarExprType::kTagDispatch) {
        for (int j = 1;
             j < grammar_expr.size() - Grammar::Impl::TagDispatch::kTagDispatchExtraParameter;
             j += 2) {
          if (grammar_expr[j] == rule_id) {
            return false;
          }
        }
        continue;
      }
      XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kChoices);
      for (auto sequence_id : grammar_expr) {
        auto sequence_expr = base_grammar_->GetGrammarExpr(sequence_id);
        if (sequence_expr.type != GrammarExprType::kSequence) {
          continue;
        }
        auto last_element = base_grammar_->GetGrammarExpr(sequence_expr.end()[-1]);
        if (last_element.type == GrammarExprType::kRuleRef && last_element[0] == rule_id &&
            i != rule_id) {
          return false;
        }

        for (int j = 0; j < sequence_expr.size() - 1; ++j) {
          auto element_expr = base_grammar_->GetGrammarExpr(sequence_expr[j]);
          if (element_expr.type != GrammarExprType::kRuleRef || element_expr[0] != rule_id) {
            continue;
          }
          if (found) {
            return false;
          }
          found = true;
        }
      }
    }
    return found;
  }

  int32_t DetectLookaheadAssertion(int32_t rule_id) {
    std::vector<int32_t> found_sequence;  // Element ids
    bool found = false;
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
      if (grammar_expr.type == GrammarExprType::kTagDispatch) {
        for (int j = 1;
             j < grammar_expr.size() - Grammar::Impl::TagDispatch::kTagDispatchExtraParameter;
             j += 2) {
          if (grammar_expr[j] == rule_id) {
            return -1;
          }
        }
        continue;
      }
      XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kChoices);
      for (auto sequence_id : grammar_expr) {
        auto sequence_expr = base_grammar_->GetGrammarExpr(sequence_id);
        if (sequence_expr.type != GrammarExprType::kSequence) {
          continue;
        }
        auto last_element = base_grammar_->GetGrammarExpr(sequence_expr.end()[-1]);
        if (last_element.type == GrammarExprType::kRuleRef && last_element[0] == rule_id &&
            i != rule_id) {
          return -1;
        }

        for (int j = 0; j < sequence_expr.size() - 1; ++j) {
          auto element_expr = base_grammar_->GetGrammarExpr(sequence_expr[j]);
          if (element_expr.type != GrammarExprType::kRuleRef || element_expr[0] != rule_id) {
            continue;
          }
          if (found) {
            return -1;
          }
          found = true;
          for (int k = j + 1; k < sequence_expr.size(); ++k) {
            found_sequence.push_back(sequence_expr[k]);
          }
        }
      }
    }

    if (!found) {
      return -1;
    }
    return builder_->AddSequence(found_sequence);
  }
};

/*!
 * \brief Finds the rule reference graph of a grammar.
 *
 * The rule reference graph shows which rules reference which other rules.
 * The returned graph is inverted: it points from referee to referer.
 */
class RuleRefGraphFinder : public GrammarVisitor<std::vector<std::vector<int32_t>>> {
 public:
  RuleRefGraphFinder() = default;

  std::vector<std::vector<int32_t>> Apply(const Grammar& grammar) {
    InitGrammar(grammar);
    rule_visit_graph_ = std::vector<std::vector<int32_t>>(base_grammar_->NumRules());
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
      cur_rule_id_ = i;
      VisitExpr(grammar_expr);
    }
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      std::sort(rule_visit_graph_[i].begin(), rule_visit_graph_[i].end());
      auto end_it = std::unique(rule_visit_graph_[i].begin(), rule_visit_graph_[i].end());
      rule_visit_graph_[i].erase(end_it, rule_visit_graph_[i].end());
    }
    return std::move(rule_visit_graph_);
  }

 private:
  void VisitRuleRef(const GrammarExpr& grammar_expr) {
    rule_visit_graph_[grammar_expr[0]].push_back(cur_rule_id_);
  }

  void VisitRepeat(const GrammarExpr& grammar_expr) {
    rule_visit_graph_[grammar_expr[0]].push_back(cur_rule_id_);
  }

  void VisitTagDispatch(const GrammarExpr& grammar_expr) {
    for (int i = 1;
         i < grammar_expr.size() - Grammar::Impl::TagDispatch::kTagDispatchExtraParameter;
         i += 2) {
      rule_visit_graph_[grammar_expr[i]].push_back(cur_rule_id_);
    }
  }

  // Inversed reference graph: pointing from referee to referer
  std::vector<std::vector<int32_t>> rule_visit_graph_;
  int32_t cur_rule_id_;
};

/*!
 * \brief Analyzes which rules in a grammar can match the empty string.
 */
class AllowEmptyRuleAnalyzerImpl : public GrammarVisitor<std::vector<int32_t>> {
 public:
  AllowEmptyRuleAnalyzerImpl() = default;

  std::vector<int32_t> Apply(const Grammar& grammar) final {
    InitGrammar(grammar);

    // Step 1: Find rules that explicitly allow empty string
    std::unordered_set<int32_t> empty_rule_id_set;
    FindExplicitEmptyRules(&empty_rule_id_set);

    // Step 2: Find rules that indirectly allow empty string. Using the Bellman-Ford algorithm
    // on the rule reference graph.
    std::vector<std::vector<int32_t>> rule_ref_graph = RuleRefGraphFinder().Apply(grammar);
    FindIndirectEmptyRules(&empty_rule_id_set, rule_ref_graph);

    auto result = std::vector<int32_t>(empty_rule_id_set.begin(), empty_rule_id_set.end());
    std::sort(result.begin(), result.end());
    return result;
  }

  void FindExplicitEmptyRules(std::unordered_set<int32_t>* empty_rule_id_set) {
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);
      if (grammar_expr.type == GrammarExprType::kTagDispatch) {
        continue;
      }

      XGRAMMAR_DCHECK(grammar_expr.type == GrammarExprType::kChoices);
      if (base_grammar_->GetGrammarExpr(grammar_expr[0]).type == GrammarExprType::kEmptyStr) {
        empty_rule_id_set->insert(i);
        continue;
      }

      for (auto seq_id : grammar_expr) {
        auto seq_expr = base_grammar_->GetGrammarExpr(seq_id);
        if (std::all_of(seq_expr.begin(), seq_expr.end(), [&](int32_t i) {
              return base_grammar_->GetGrammarExpr(i).type == GrammarExprType::kCharacterClassStar;
            })) {
          empty_rule_id_set->insert(i);
          break;
        }
      }
    }
  }

  bool SeqExprIsEpsilon(
      const GrammarExpr& seq_expr, const std::unordered_set<int32_t>& empty_rule_id_set
  ) {
    if (seq_expr.type == GrammarExprType::kEmptyStr) {
      return true;
    }
    XGRAMMAR_DCHECK(seq_expr.type == GrammarExprType::kSequence);

    return std::all_of(seq_expr.begin(), seq_expr.end(), [&](int32_t i) {
      auto element_expr = base_grammar_->GetGrammarExpr(i);
      return (element_expr.type == GrammarExprType::kRuleRef &&
              empty_rule_id_set.count(element_expr[0])) ||
             element_expr.type == GrammarExprType::kCharacterClassStar ||
             (element_expr.type == GrammarExprType::kRepeat &&
              (empty_rule_id_set.count(element_expr[0]) || element_expr[1] == 0));
    });
  }

  void FindIndirectEmptyRules(
      std::unordered_set<int32_t>* empty_rule_id_set,
      const std::vector<std::vector<int32_t>>& rule_ref_graph
  ) {
    std::queue<int32_t> queue;
    for (auto i : *empty_rule_id_set) {
      queue.push(i);
    }

    while (!queue.empty()) {
      auto rule_id = queue.front();
      queue.pop();
      XGRAMMAR_DCHECK(rule_id >= 0 && rule_id < static_cast<int>(rule_ref_graph.size()));
      for (auto referer_rule_id : rule_ref_graph[rule_id]) {
        if (empty_rule_id_set->count(referer_rule_id)) {
          continue;
        }
        auto rule = base_grammar_->GetRule(referer_rule_id);
        auto grammar_expr = base_grammar_->GetGrammarExpr(rule.body_expr_id);

        XGRAMMAR_DCHECK(grammar_expr.type != GrammarExprType::kTagDispatch)
            << "TagDispatch rules should already exist in empty_rule_id_set";

        bool is_epsilon = std::any_of(grammar_expr.begin(), grammar_expr.end(), [&](int32_t i) {
          auto seq_expr = base_grammar_->GetGrammarExpr(i);
          return SeqExprIsEpsilon(seq_expr, *empty_rule_id_set);
        });

        if (is_epsilon) {
          empty_rule_id_set->insert(referer_rule_id);
          queue.push(referer_rule_id);
        }
      }
    }
  }
};

// Convert a Unicode codepoint to the packed UTF-8 format used by AddCharacterRange.
// The packed format stores UTF-8 bytes as: (byte0 << 24) | (byte1 << 16) | (byte2 << 8) | byte3
// where byte0 is the first UTF-8 byte (leading byte) and subsequent bytes are continuation bytes.
inline uint32_t CodepointToPackedUTF8(uint32_t codepoint) {
  if (codepoint <= 0x7F) {
    // 1-byte sequence (ASCII)
    return codepoint;
  } else if (codepoint <= 0x7FF) {
    // 2-byte sequence: byte0 = 110xxxxx, byte1 = 10xxxxxx
    uint8_t byte0 = 0xC0 | ((codepoint >> 6) & 0x1F);
    uint8_t byte1 = 0x80 | (codepoint & 0x3F);
    return (static_cast<uint32_t>(byte0) << 8) | byte1;
  } else if (codepoint <= 0xFFFF) {
    // 3-byte sequence: byte0 = 1110xxxx, byte1 = 10xxxxxx, byte2 = 10xxxxxx
    uint8_t byte0 = 0xE0 | ((codepoint >> 12) & 0x0F);
    uint8_t byte1 = 0x80 | ((codepoint >> 6) & 0x3F);
    uint8_t byte2 = 0x80 | (codepoint & 0x3F);
    return (static_cast<uint32_t>(byte0) << 16) | (static_cast<uint32_t>(byte1) << 8) | byte2;
  } else {
    // 4-byte sequence: byte0 = 11110xxx, byte1-3 = 10xxxxxx
    uint8_t byte0 = 0xF0 | ((codepoint >> 18) & 0x07);
    uint8_t byte1 = 0x80 | ((codepoint >> 12) & 0x3F);
    uint8_t byte2 = 0x80 | ((codepoint >> 6) & 0x3F);
    uint8_t byte3 = 0x80 | (codepoint & 0x3F);
    return (static_cast<uint32_t>(byte0) << 24) | (static_cast<uint32_t>(byte1) << 16) |
           (static_cast<uint32_t>(byte2) << 8) | byte3;
  }
}

class GrammarFSMBuilderImpl {
 public:
  const static uint32_t kMax1ByteUnicode = 0x7F;
  const static uint32_t kMin2BytesUnicode = 0xC080;
  const static uint32_t kMax2BytesUnicode = 0xDFBF;
  const static uint32_t kMin3BytesUnicode = 0xE08080;
  const static uint32_t kMax3BytesUnicode = 0xEFBFBF;
  const static uint32_t kMin4BytesUnicode = 0xF0808080;
  const static uint32_t kMax4BytesUnicode = 0xF7BFBFBF;

  void Apply(Grammar* grammar) {
    FSM complete_fsm;
    std::vector<std::optional<FSMWithStartEnd>> per_rule_fsms((*grammar)->NumRules());
    std::vector<int> state_mapping;

    for (int i = 0; i < (*grammar)->NumRules(); ++i) {
      auto rule = (*grammar)->GetRule(i);
      auto grammar_expr = (*grammar)->GetGrammarExpr(rule.body_expr_id);
      if (grammar_expr.type == Grammar::Impl::GrammarExprType::kTagDispatch) {
        auto rule_fsm = TagDispatch((*grammar)->GetTagDispatch(grammar_expr));
        XGRAMMAR_CHECK(rule_fsm.has_value()) << "Failed to build tag dispatch fsm for rule " << i;
        per_rule_fsms[i] = rule_fsm->AddToCompleteFSM(&complete_fsm, &state_mapping);
      } else {
        XGRAMMAR_DCHECK(grammar_expr.type == Grammar::Impl::GrammarExprType::kChoices);
        auto rule_fsm = Choices(grammar_expr, *grammar);
        if (rule_fsm.has_value()) {
          per_rule_fsms[i] = rule_fsm->AddToCompleteFSM(&complete_fsm, &state_mapping);
        }
      }
    }

    // Compress to compact fsm
    CompactFSM compact_complete_fsm = complete_fsm.ToCompact();
    std::vector<std::optional<CompactFSMWithStartEnd>> compact_per_rule_fsms((*grammar)->NumRules()
    );
    for (int i = 0; i < (*grammar)->NumRules(); ++i) {
      if (per_rule_fsms[i]) {
        compact_per_rule_fsms[i] = CompactFSMWithStartEnd(
            compact_complete_fsm, per_rule_fsms[i]->GetStart(), per_rule_fsms[i]->GetEnds()
        );
      }
    }

    (*grammar)->complete_fsm = std::move(compact_complete_fsm);
    (*grammar)->per_rule_fsms = std::move(compact_per_rule_fsms);
  }

  /* Basic Building functions.*/
  static FSMWithStartEnd RuleRef(const GrammarExpr& expr);
  static FSMWithStartEnd CharacterClass(const GrammarExpr& expr);
  static FSMWithStartEnd ByteString(const GrammarExpr& expr);
  static std::optional<FSMWithStartEnd> Sequence(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> Choices(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> TagDispatch(const Grammar::Impl::TagDispatch& tag_dispatch);
  static void AddCharacterRange(FSMWithStartEnd& fsm, int from, int to, uint32_t min, uint32_t max);
  /* Building tool funtions.*/
  static std::optional<FSMWithStartEnd> BuildTagDispatchWithEOSStop(
      const std::vector<std::pair<std::string, int>>& tag_dispatch_rules,
      bool loop_after_dispatch,
      const std::vector<std::string>& excluded_strings
  );
  static std::optional<FSMWithStartEnd> BuildTagDispatchWithStopString(
      const std::vector<std::pair<std::string, int>>& tag_dispatch_rules,
      const std::vector<std::string>& stop_strings,
      bool loop_after_dispatch,
      const std::vector<std::string>& excluded_strings
  );
  static FSMWithStartEnd BuildNegativeCharacterClass(const GrammarExpr& expr);
};

// This function will add a range [min, max] of characters to the FSM, and the length
// of the characters are the same.
void AddSameLengthCharacterRange(
    FSMWithStartEnd& fsm, int from, int to, uint32_t min, uint32_t max
) {
  uint8_t byte_min[4] = {
      static_cast<uint8_t>(min & 0xFF),
      static_cast<uint8_t>(min >> 8),
      static_cast<uint8_t>(min >> 16),
      static_cast<uint8_t>(min >> 24)
  };
  uint8_t byte_max[4] = {
      static_cast<uint8_t>(max & 0xFF),
      static_cast<uint8_t>(max >> 8),
      static_cast<uint8_t>(max >> 16),
      static_cast<uint8_t>(max >> 24)
  };

  // ASCII.
  if (byte_max[1] == 0) {
    fsm.GetFsm().AddEdge(from, to, byte_min[0], byte_max[0]);
    return;
  }

  if (byte_max[3] != 0) {
    // 4-byte unicode.
    if (byte_max[3] == byte_min[3]) {
      int tmp_state = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state, byte_min[3], byte_max[3]);
      min = (min & 0x00FFFFFF);
      max = (max & 0x00FFFFFF);
      AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
      return;
    }
    if ((min & 0x00FFFFFF) != 0x808080) {
      int tmp_state_min = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state_min, byte_min[3], byte_min[3]);
      AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FFFFFF), 0x00BFBFBF);
    } else {
      byte_min[3]--;
    }
    if ((max & 0x00FFFFFF) != 0xBFBFBF) {
      int tmp_state_max = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state_max, byte_max[3], byte_max[3]);
      AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x00808080, (max & 0x00FFFFFF));
    } else {
      byte_max[3]++;
    }
    if (byte_max[3] - byte_min[3] > 1) {
      int tmp_state_mid = fsm.AddState();
      // First byte.
      fsm.GetFsm().AddEdge(from, tmp_state_mid, byte_min[3] + 1, byte_max[3] - 1);
      int tmp_state_mid2 = fsm.AddState();
      // Second byte.
      fsm.GetFsm().AddEdge(tmp_state_mid, tmp_state_mid2, 0x80, 0xBF);
      int tmp_state_mid3 = fsm.AddState();
      // Third byte.
      fsm.GetFsm().AddEdge(tmp_state_mid2, tmp_state_mid3, 0x80, 0xBF);
      // Last byte.
      fsm.GetFsm().AddEdge(tmp_state_mid3, to, 0x80, 0xBF);
    }
    return;
  }
  if (byte_max[2] != 0) {
    // 3 byte unicode.
    if (byte_max[2] == byte_min[2]) {
      int tmp_state = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state, byte_min[2], byte_max[2]);
      min = (min & 0x00FFFF);
      max = (max & 0x00FFFF);
      AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
      return;
    }
    if ((min & 0x00FFFF) != 0x8080) {
      int tmp_state_min = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state_min, byte_min[2], byte_min[2]);
      AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FFFF), 0x00BFBF);
    } else {
      byte_min[2]--;
    }
    if ((max & 0x00FFFF) != 0xBFBF) {
      int tmp_state_max = fsm.AddState();
      fsm.GetFsm().AddEdge(from, tmp_state_max, byte_max[2], byte_max[2]);
      AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x0080, (max & 0x00FFFF));
    } else {
      byte_max[2]++;
    }
    if (byte_max[2] - byte_min[2] > 1) {
      int tmp_state_mid = fsm.AddState();
      // First byte.
      fsm.GetFsm().AddEdge(from, tmp_state_mid, byte_min[2] + 1, byte_max[2] - 1);
      int tmp_state_mid2 = fsm.AddState();
      // Second byte.
      fsm.GetFsm().AddEdge(tmp_state_mid, tmp_state_mid2, 0x80, 0xBF);
      // Last byte.
      fsm.GetFsm().AddEdge(tmp_state_mid2, to, 0x80, 0xBF);
    }
    return;
  }

  // 2 byte unicode.
  if (byte_max[1] == byte_min[1]) {
    int tmp_state = fsm.AddState();
    fsm.GetFsm().AddEdge(from, tmp_state, byte_min[1], byte_max[1]);
    min = (min & 0x00FF);
    max = (max & 0x00FF);
    AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
    return;
  }
  if ((min & 0x00FF) != 0x80) {
    int tmp_state_min = fsm.AddState();
    fsm.GetFsm().AddEdge(from, tmp_state_min, byte_min[1], byte_min[1]);
    AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FF), 0x00BF);
  } else {
    byte_min[1]--;
  }
  if ((max & 0x00FF) != 0xBF) {
    int tmp_state_max = fsm.AddState();
    fsm.GetFsm().AddEdge(from, tmp_state_max, byte_max[1], byte_max[1]);
    AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x0080, (max & 0x00FF));
  } else {
    byte_max[1]++;
  }
  if (byte_max[1] - byte_min[1] > 1) {
    int tmp_state_mid = fsm.AddState();
    // First byte.
    fsm.GetFsm().AddEdge(from, tmp_state_mid, byte_min[1] + 1, byte_max[1] - 1);
    fsm.GetFsm().AddEdge(tmp_state_mid, to, 0x80, 0xBF);
  }
  return;
}

// This function will add a range [min, max] of unicode characters to the FSM.
void GrammarFSMBuilderImpl::AddCharacterRange(
    FSMWithStartEnd& fsm, int from, int to, uint32_t min, uint32_t max
) {
  XGRAMMAR_CHECK(min <= max) << "Invalid character range: min (" << min << ") > max (" << max
                             << ")";
  // Ensure max and min are valid unicode value.
  if (max > kMax4BytesUnicode) {
    max = kMax4BytesUnicode;
  } else if (max > kMax3BytesUnicode) {
    if (max < kMin4BytesUnicode) {
      max = kMax3BytesUnicode;
    }
  } else if (max > kMax2BytesUnicode) {
    if (max < kMin3BytesUnicode) {
      max = kMax2BytesUnicode;
    }
  } else if (max < kMin2BytesUnicode && (max > kMax1ByteUnicode)) {
    max = kMax1ByteUnicode;
  }

  if (min > kMax4BytesUnicode) {
    min = kMax4BytesUnicode;
  } else if (min > kMax3BytesUnicode) {
    if (min < kMin4BytesUnicode) {
      min = kMin4BytesUnicode;
    }
  } else if (min > kMax2BytesUnicode) {
    if (min < kMin3BytesUnicode) {
      min = kMin3BytesUnicode;
    }
  } else if (min < kMin2BytesUnicode && (min > kMax1ByteUnicode)) {
    min = kMin2BytesUnicode;
  }

  // Step2. Divide the range into several ranges, which contain characters with different lengths.
  if (max <= kMax1ByteUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, max);
    return;
  }
  if (max <= kMax2BytesUnicode) {
    if (min >= kMin2BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, max);
    } else {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, max);
    }
    return;
  }
  if (max <= kMax3BytesUnicode) {
    if (min >= kMin3BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, max);
    } else if (min >= kMin2BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax2BytesUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, max);
    } else {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, kMax2BytesUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, max);
    }
    return;
  }
  XGRAMMAR_CHECK(max <= kMax4BytesUnicode);
  if (min >= kMin4BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, max);
  } else if (min >= kMin3BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  } else if (min >= kMin2BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax2BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  } else {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, kMax2BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  }
  return;
}

FSMWithStartEnd GrammarFSMBuilderImpl::BuildNegativeCharacterClass(const GrammarExpr& expr) {
  XGRAMMAR_DCHECK(
      expr.type == ExprType::kCharacterClass || expr.type == ExprType::kCharacterClassStar
  );
  XGRAMMAR_DCHECK(expr[0]);  // Negative character class should be true.
  std::bitset<128> char_set;
  for (int i = 1; i < static_cast<int>(expr.size()); i += 2) {
    uint8_t byte_min = static_cast<uint8_t>(expr[i]);
    uint8_t byte_max = static_cast<uint8_t>(expr[i + 1]);
    if (byte_max > 128) {
      XGRAMMAR_LOG(WARNING) << "Negative Character class contains byte greater than 127, "
                            << "clamping to 127.";
      byte_max = 127;
    }
    for (uint8_t j = byte_min; j <= byte_max; ++j) {
      char_set.set(j);
    }
  }

  // Construct the basic FSM.
  FSMWithStartEnd result_fsm;
  int start_state = result_fsm.AddState();
  bool is_star = expr.type == ExprType::kCharacterClassStar;
  result_fsm.SetStartState(start_state);
  int end_state = -1;
  if (is_star) {
    end_state = start_state;
  } else {
    end_state = result_fsm.AddState();
  }
  result_fsm.AddEndState(end_state);
  int left_bound = -1;
  for (int i = 0; i < 128; ++i) {
    if (!char_set[i]) {
      left_bound = i;
      int right_bound = i + 1;
      while (right_bound < 128 && !char_set[right_bound]) {
        right_bound++;
      }
      result_fsm.GetFsm().AddEdge(
          start_state,
          end_state,
          static_cast<uint8_t>(left_bound),
          static_cast<uint8_t>(right_bound - 1)
      );
      i = right_bound;
    }
  }
  AddCharacterRange(result_fsm, start_state, end_state, kMin2BytesUnicode, kMax4BytesUnicode);
  return result_fsm;
}

FSMWithStartEnd GrammarFSMBuilderImpl::CharacterClass(const GrammarExpr& expr) {
  bool is_negative = expr[0];
  FSMWithStartEnd result_fsm;
  if (is_negative) {
    result_fsm = BuildNegativeCharacterClass(expr);
    return result_fsm;
  }
  int start_state = result_fsm.AddState();
  result_fsm.SetStartState(start_state);
  bool is_star = expr.type == ExprType::kCharacterClassStar;
  int end_state = -1;
  if (is_star) {
    end_state = start_state;
  } else {
    end_state = result_fsm.AddState();
  }
  result_fsm.AddEndState(end_state);
  for (int i = 1; i < static_cast<int>(expr.size()); i += 2) {
    uint32_t codepoint_min = static_cast<uint32_t>(expr[i]);
    uint32_t codepoint_max = static_cast<uint32_t>(expr[i + 1]);
    // Convert Unicode codepoints to packed UTF-8 format for AddCharacterRange
    uint32_t packed_min = CodepointToPackedUTF8(codepoint_min);
    uint32_t packed_max = CodepointToPackedUTF8(codepoint_max);
    AddCharacterRange(result_fsm, start_state, end_state, packed_min, packed_max);
  }
  return result_fsm;
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::Sequence(
    const GrammarExpr& expr, const Grammar& grammar
) {
  std::vector<FSMWithStartEnd> fsm_lists;

  // Build the fsm of sub-expressions.
  for (const auto& sequence_id : expr) {
    const auto& sequence_expr = grammar->GetGrammarExpr(sequence_id);
    switch (sequence_expr.type) {
      case (ExprType::kByteString): {
        fsm_lists.push_back(ByteString(sequence_expr));
        break;
      }
      case (ExprType::kRuleRef): {
        fsm_lists.push_back(RuleRef(sequence_expr));
        break;
      }
      case (ExprType::kCharacterClass):
      case (ExprType::kCharacterClassStar): {
        fsm_lists.push_back(CharacterClass(sequence_expr));
        break;
      }
      default: {
        return std::nullopt;
      }
    }
  }

  // Check if the sequence is empty.
  if (fsm_lists.empty()) {
    FSMWithStartEnd empty_fsm;
    empty_fsm.AddState();
    empty_fsm.SetStartState(0);
    empty_fsm.AddEndState(0);
    return empty_fsm;
  }

  return FSMWithStartEnd::Concat(fsm_lists);
}

FSMWithStartEnd GrammarFSMBuilderImpl::RuleRef(const GrammarExpr& expr) {
  FSMWithStartEnd result_fsm;
  result_fsm.AddState();
  result_fsm.AddState();
  result_fsm.SetStartState(0);
  result_fsm.AddEndState(1);
  result_fsm.GetFsm().AddRuleEdge(0, 1, expr[0]);
  return result_fsm;
}

FSMWithStartEnd GrammarFSMBuilderImpl::ByteString(const GrammarExpr& expr) {
  XGRAMMAR_DCHECK(expr.type == ExprType::kByteString);
  FSMWithStartEnd result_fsm;
  int current_state = result_fsm.AddState();
  result_fsm.SetStartState(current_state);
  for (const auto& byte : expr) {
    int next_state = result_fsm.AddState();
    result_fsm.GetFsm().AddEdge(
        current_state, next_state, static_cast<uint8_t>(byte), static_cast<uint8_t>(byte)
    );
    current_state = next_state;
  }
  result_fsm.AddEndState(current_state);
  return result_fsm;
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::Choices(
    const GrammarExpr& expr, const Grammar& grammar
) {
  XGRAMMAR_DCHECK(expr.type == ExprType::kChoices);
  std::vector<FSMWithStartEnd> fsm_list;
  bool nullable = false;
  for (const auto& choice_id : expr) {
    const auto& choice_expr = grammar->GetGrammarExpr(choice_id);
    // The choice expression should be either a sequence or an empty string.
    if (choice_expr.type == ExprType::kEmptyStr) {
      nullable = true;
      continue;
    }
    XGRAMMAR_DCHECK(choice_expr.type == ExprType::kSequence);
    auto fsm_result = Sequence(choice_expr, grammar);
    if (!fsm_result.has_value()) {
      return std::nullopt;
    }
    fsm_list.push_back(std::move(fsm_result.value()));
  }

  if (fsm_list.empty()) {
    // It's an empty rule.
    FSMWithStartEnd empty_fsm;
    empty_fsm.AddState();
    empty_fsm.SetStartState(0);
    empty_fsm.AddEndState(0);
    return empty_fsm;
  }
  if (nullable) {
    FSMWithStartEnd null_fsm;
    null_fsm.AddState();
    null_fsm.SetStartState(0);
    null_fsm.AddEndState(0);
    fsm_list.push_back(std::move(null_fsm));
  }

  auto result = FSMWithStartEnd::Union(fsm_list);
  result = result.SimplifyEpsilon();
  result = result.MergeEquivalentSuccessors();
  auto result_raw = result.MinimizeDFA();
  if (result_raw.IsOk()) {
    result = std::move(result_raw).Unwrap();
  }
  return result;
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::BuildTagDispatchWithStopString(
    const std::vector<std::pair<std::string, int>>& tag_dispatch_rules,
    const std::vector<std::string>& stop_strings,
    bool loop_after_dispatch,
    const std::vector<std::string>& excluded_strings
) {
  XGRAMMAR_DCHECK(stop_strings.size() > 0);
  std::vector<std::string> tag_names;
  tag_names.reserve(tag_dispatch_rules.size());
  for (const auto& [tag_name, tag_id] : tag_dispatch_rules) {
    tag_names.push_back(tag_name);
  }
  for (const auto& stop_string : stop_strings) {
    tag_names.push_back(stop_string);
  }
  std::vector<int> trie_end_states;
  auto trie_result =
      TrieFSMBuilder::Build(tag_names, excluded_strings, &trie_end_states, false, true);
  if (!trie_result.has_value()) {
    return std::nullopt;
  }
  auto trie_fsm = trie_result->GetFsm();
  auto start = trie_result->GetStart();
  std::unordered_set<int> old_ends;
  for (int end = 0; end < trie_result->NumStates(); end++) {
    if (trie_result->IsEndState(end)) {
      old_ends.insert(end);
    }
  }
  std::vector<bool> ends(trie_fsm.NumStates(), false);

  // The final end states are the end of each stop string.
  for (int i = static_cast<int>(tag_dispatch_rules.size());
       i < static_cast<int>(trie_end_states.size());
       i++) {
    ends[trie_end_states[i]] = true;
  }

  if (loop_after_dispatch) {
    for (int i = 0; i < static_cast<int>(tag_dispatch_rules.size()); i++) {
      trie_fsm.AddRuleEdge(trie_end_states[i], start, tag_dispatch_rules[i].second);
    }
  } else {
    // We should first build a new FSM that only contains the stop strings.
    tag_names.clear();
    for (const auto& stop_string : stop_strings) {
      tag_names.push_back(stop_string);
    }
    std::vector<int> stop_end_states;
    auto stop_trie_result =
        TrieFSMBuilder::Build(tag_names, excluded_strings, nullptr, false, false);
    XGRAMMAR_DCHECK(stop_trie_result.has_value());
    auto stop_trie_fsm = stop_trie_result->GetFsm();
    auto stop_trie_start = stop_trie_result->GetStart();
    std::unordered_set<int> stop_trie_ends;
    for (int end = 0; end < stop_trie_result->NumStates(); end++) {
      if (stop_trie_result->IsEndState(end)) {
        stop_trie_ends.insert(end);
      }
    }

    std::vector<int> stop_trie_to_trie_map;
    trie_fsm.AddFSM(stop_trie_fsm, &stop_trie_to_trie_map);
    ends.resize(trie_fsm.NumStates(), false);
    int start_of_stop_trie = stop_trie_to_trie_map[stop_trie_start];
    for (auto state : stop_trie_ends) {
      ends[stop_trie_to_trie_map[state]] = true;
    }

    for (int i = 0; i < static_cast<int>(tag_dispatch_rules.size()); i++) {
      trie_fsm.AddRuleEdge(trie_end_states[i], start_of_stop_trie, tag_dispatch_rules[i].second);
    }
  }

  return FSMWithStartEnd(trie_fsm, start, ends);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::BuildTagDispatchWithEOSStop(
    const std::vector<std::pair<std::string, int>>& tag_dispatch_rules,
    bool loop_after_dispatch,
    const std::vector<std::string>& excluded_strings
) {
  std::vector<std::string> tag_names;
  tag_names.reserve(tag_dispatch_rules.size());
  for (const auto& [tag_name, tag_id] : tag_dispatch_rules) {
    tag_names.push_back(tag_name);
  }
  std::vector<int> end_states;
  auto trie_result = TrieFSMBuilder::Build(tag_names, excluded_strings, &end_states, false, true);
  if (!trie_result.has_value()) {
    return std::nullopt;
  }
  auto trie_fsm = trie_result->GetFsm();
  auto start = trie_result->GetStart();
  std::unordered_set<int> old_ends;
  std::vector<bool> ends(trie_fsm.NumStates(), false);
  for (int end = 0; end < trie_result->NumStates(); end++) {
    if (trie_result->IsEndState(end)) {
      old_ends.insert(end);
    }
  }

  // The final end states are all but old_ends.
  for (int i = 0; i < trie_fsm.NumStates(); i++) {
    if (old_ends.count(i) == 0) {
      ends[i] = true;
    }
  }

  // Add rule ref edges
  for (int i = 0; i < static_cast<int>(tag_dispatch_rules.size()); i++) {
    int next_state;
    if (loop_after_dispatch) {
      next_state = start;
    } else {
      next_state = trie_fsm.AddState();
      ends.push_back(true);
    }
    trie_fsm.AddRuleEdge(end_states[i], next_state, tag_dispatch_rules[i].second);
  }

  return FSMWithStartEnd(trie_fsm, start, ends);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilderImpl::TagDispatch(
    const Grammar::Impl::TagDispatch& tag_dispatch
) {
  if (tag_dispatch.stop_eos) {
    return BuildTagDispatchWithEOSStop(
        tag_dispatch.tag_rule_pairs, tag_dispatch.loop_after_dispatch, tag_dispatch.excluded_str
    );
  } else {
    return BuildTagDispatchWithStopString(
        tag_dispatch.tag_rule_pairs,
        tag_dispatch.stop_str,
        tag_dispatch.loop_after_dispatch,
        tag_dispatch.excluded_str
    );
  }
}

class RepetitionNormalizerImpl {
 public:
  void Apply(Grammar* grammar) {
    auto& grammar_ref = *grammar;
    for (int i = 0; i < grammar_ref->NumGrammarExprs(); ++i) {
      auto expr = grammar_ref->GetGrammarExpr(i);
      if (expr.type != Grammar::Impl::GrammarExprType::kRepeat) {
        continue;
      }
      int repeat_rule_id = expr[0];
      grammar_ref->GetRule(repeat_rule_id).is_exact_lookahead = true;
      if (std::binary_search(
              grammar_ref->allow_empty_rule_ids.begin(),
              grammar_ref->allow_empty_rule_ids.end(),
              repeat_rule_id
          )) {
        // The repeated rule can be empty, so we need to normalize it.
        expr.SetData(1, 0);  // Set min repeat to 0
      }
    }
  }
};

class GrammarOptimizerImpl {
 public:
  static Grammar Apply(const Grammar& grammar) {
    Grammar result = ByteStringFuser::Apply(grammar);
    result = RuleInliner::Apply(result);
    result = DeadCodeEliminator::Apply(result);
    result = LookaheadAssertionAnalyzer::Apply(result);
    result->allow_empty_rule_ids = AllowEmptyRuleAnalyzer::Apply(result);
    RepetitionNormalizer::Apply(&result);
    GrammarFSMBuilder::Apply(&result);
    result->optimized = true;
    return result;
  }
};

class ByteStringFuserImpl : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

 private:
  /*!
   * \brief Visit a GrammarExpr containing a sequence.
   * \returns A list of new sequence GrammarExpr ids.
   */
  int32_t VisitSequence(const GrammarExpr& grammar_expr) final {
    std::vector<int32_t> new_sequence_ids;
    std::vector<int32_t> cur_byte_string;
    for (auto i : grammar_expr) {
      auto element_expr = base_grammar_->GetGrammarExpr(i);
      if (element_expr.type == GrammarExprType::kByteString) {
        cur_byte_string.insert(cur_byte_string.end(), element_expr.begin(), element_expr.end());
        continue;
      } else {
        if (!cur_byte_string.empty()) {
          new_sequence_ids.push_back(builder_->AddByteString(cur_byte_string));
          cur_byte_string.clear();
        }
        new_sequence_ids.push_back(builder_->AddGrammarExpr(element_expr));
      }
    }
    if (!cur_byte_string.empty()) {
      new_sequence_ids.push_back(builder_->AddByteString(cur_byte_string));
    }
    return builder_->AddSequence(new_sequence_ids);
  }
};

class RootRuleRenamerImpl {
 public:
  static Grammar Apply(const Grammar& grammar) {
    // If the root name is "root", return directly.
    if (grammar->GetRootRule().name == "root") {
      return grammar;
    }

    // Collect all the rule names.
    std::unordered_set<std::string> rule_names;
    int root_name_rule_id = -1;
    for (int i = 0; i < grammar->NumRules(); i++) {
      const auto& rule_name = grammar->GetRule(i).name;
      if (rule_name == "root") {
        root_name_rule_id = i;
      }
      rule_names.insert(rule_name);
    }

    // Rename the rules.
    Grammar grammar_copy = grammar;
    grammar_copy->GetRule(grammar_copy->GetRootRuleId()).name = "root";
    if (root_name_rule_id != -1) {
      std::string rule_prefix = "root_";
      for (int i = 0; i <= grammar_copy->NumRules(); i++) {
        std::string new_rule_name = rule_prefix + std::to_string(i);
        if (rule_names.find(new_rule_name) == rule_names.end()) {
          grammar_copy->GetRule(root_name_rule_id).name = new_rule_name;
          break;
        }
        XGRAMMAR_DCHECK(false
        ) << "The rule must be renamed successfully after (n + 1) times of iterations.";
      }
    }
    return grammar_copy;
  }
};

class GrammarFSMHasherImpl {
 public:
  void Apply(Grammar* grammar);
  static std::optional<uint64_t> HashSequence(const Grammar& grammar, int32_t sequence_id);

  static const int16_t kNotEndStateFlag = -0x100;
  static const int16_t kEndStateFlag = -0x200;
  static const int16_t kSelfRecursionFlag = -0x300;
  static const int16_t kSimpleCycleFlag = -0x400;
  static const int16_t kUnKnownFlag = -0x500;

 private:
  Grammar* grammar_;
  std::vector<bool> visited_;
  std::vector<std::vector<int32_t>> ref_graph_from_referrer_to_referee_;
  std::vector<std::vector<int32_t>> ref_graph_from_referee_to_referrer_;
  std::vector<std::vector<FSMEdge>> sorted_edges_;
  std::vector<bool> has_inward_edges_;

  /*!
   * \brief Get the hash value of a fsm, with a given grammar.
   */
  uint64_t HashFsm(int fsm_index);

  /*!
   * \brief Find a simple cycle in the reference graph, And hash the
   * fsms in the simple cycle.
   */
  bool FindSimpleCycle();

  /*!
   * \brief Hash the fsms in the simple cycle.
   */
  void HashSimpleCycle(const std::vector<int32_t>& simple_cycle);

  /*!
   * \brief Find a simple fsm that can be hashed. If it can't, it will
   * call FindSimpleCycle() and try to simplify the graph, and then try to
   * find a simple fsm again.
   */
  int32_t FindSimpleFsmCanBeHashed();

  std::pair<bool, uint64_t> IsPartialHashable(int fsm_index);
};

bool GrammarFSMHasherImpl::FindSimpleCycle() {
  // Try to find a simple cycle.
  std::vector<bool> not_simple_cycle = visited_;
  for (size_t i = 0; i < ref_graph_from_referee_to_referrer_.size(); i++) {
    if (not_simple_cycle[i]) {
      continue;
    }
    // Not a simple cycle if it has more than one referee.
    std::stack<int32_t> dfs_stack;
    std::vector<int32_t> simple_cycle;
    auto in_stack = std::vector<bool>(ref_graph_from_referee_to_referrer_.size(), false);
    dfs_stack.push(static_cast<int32_t>(i));
    int32_t current_fsm_index = i;
    in_stack[current_fsm_index] = true;
    while ((ref_graph_from_referrer_to_referee_[current_fsm_index].size() == 1) &&
           !not_simple_cycle[current_fsm_index]) {
      XGRAMMAR_CHECK(current_fsm_index != ref_graph_from_referrer_to_referee_[current_fsm_index][0])
          << "Self-recursion cycle found in the reference graph, which is not allowed.";
      not_simple_cycle[current_fsm_index] = true;
      current_fsm_index = ref_graph_from_referrer_to_referee_[current_fsm_index][0];
      if (in_stack[current_fsm_index]) {
        simple_cycle.push_back(current_fsm_index);
        while (dfs_stack.top() != current_fsm_index) {
          simple_cycle.push_back(dfs_stack.top());
          dfs_stack.pop();
        }
        // Found a simple cycle.
        break;
      } else {
        dfs_stack.push(current_fsm_index);
        in_stack[current_fsm_index] = true;
      }
    }
    if (!simple_cycle.empty()) {
      HashSimpleCycle(simple_cycle);
      return true;
    }
  }
  return false;
}

void GrammarFSMHasherImpl::HashSimpleCycle(const std::vector<int32_t>& simple_cycle) {
  // Initialize the cycle hash.
  for (const auto& cycle_id : simple_cycle) {
    visited_[cycle_id] = true;
    grammar_->ImplPtr()->per_rule_fsm_hashes[cycle_id] = kSimpleCycleFlag;
  }

  std::vector<uint64_t> local_cycle_hash;
  local_cycle_hash.reserve(simple_cycle.size());
  for (const auto& cycle_id : simple_cycle) {
    local_cycle_hash.push_back(HashFsm(cycle_id));
  }
  std::vector<uint64_t> local_cycle_hash_copy = local_cycle_hash;
  for (int i = 0; i < static_cast<int>(local_cycle_hash.size()); i++) {
    uint64_t current_hash = 0;
    for (int j = 0; j < static_cast<int>(local_cycle_hash.size()); j++) {
      current_hash =
          HashCombine(current_hash, local_cycle_hash_copy[(i + j) % local_cycle_hash.size()]);
    }
    local_cycle_hash[i] = current_hash;
  }

  for (int i = 0; i < static_cast<int>(simple_cycle.size()); i++) {
    grammar_->ImplPtr()->per_rule_fsm_hashes[simple_cycle[i]] = local_cycle_hash[i];
    for (const auto& referer : ref_graph_from_referee_to_referrer_[simple_cycle[i]]) {
      ref_graph_from_referrer_to_referee_[referer].erase(std::find_if(
          ref_graph_from_referrer_to_referee_[referer].begin(),
          ref_graph_from_referrer_to_referee_[referer].end(),
          [&](int32_t rule_id) { return rule_id == simple_cycle[i]; }
      ));
    }
  }
}

int32_t GrammarFSMHasherImpl::FindSimpleFsmCanBeHashed() {
  bool possible_to_find = true;
  while (possible_to_find) {
    possible_to_find = false;
    for (size_t i = 0; i < ref_graph_from_referrer_to_referee_.size(); i++) {
      if (visited_[i]) {
        continue;
      }
      if (ref_graph_from_referrer_to_referee_[i].empty()) {
        return i;
      }
      if (ref_graph_from_referrer_to_referee_[i].size() == 1 &&
          ref_graph_from_referrer_to_referee_[i][0] == static_cast<int32_t>(i)) {
        // Self-recursion fsm.
        return static_cast<int32_t>(i);
      }
    }
    // Try to find a simple cycle. We must ensure there are not self-recursion cycles.
    possible_to_find = FindSimpleCycle();
  }
  return -1;
}

void GrammarFSMHasherImpl::Apply(Grammar* grammar) {
  grammar_ = grammar;
  grammar->ImplPtr()->per_rule_fsm_hashes =
      std::vector<std::optional<uint64_t>>((*grammar)->NumRules());
  grammar->ImplPtr()->per_rule_fsm_new_state_ids.resize((*grammar)->NumRules());
  ref_graph_from_referee_to_referrer_.clear();
  ref_graph_from_referrer_to_referee_.clear();
  sorted_edges_.clear();
  visited_ = std::vector<bool>((*grammar)->NumRules(), false);
  has_inward_edges_ = std::vector<bool>((*grammar)->complete_fsm.NumStates(), false);
  for (int i = 0; i < grammar_->ImplPtr()->complete_fsm.NumStates(); i++) {
    for (const auto& edge : grammar->ImplPtr()->complete_fsm.GetEdges(i)) {
      has_inward_edges_[edge.target] = true;
    }
  }

  // Get the reference graph.
  ref_graph_from_referee_to_referrer_ = RuleRefGraphFinder().Apply(*grammar);
  ref_graph_from_referrer_to_referee_ = std::vector<std::vector<int32_t>>((*grammar)->NumRules());
  for (int referee = 0; referee < static_cast<int>(ref_graph_from_referee_to_referrer_.size());
       ++referee) {
    for (int referer : ref_graph_from_referee_to_referrer_[referee]) {
      ref_graph_from_referrer_to_referee_[referer].push_back(referee);
    }
  }

  // Sort the edges.
  const auto& complete_fsm = grammar->ImplPtr()->complete_fsm;
  sorted_edges_.reserve(complete_fsm.NumStates());
  for (int i = 0; i < complete_fsm.NumStates(); i++) {
    const auto& edges = complete_fsm.GetEdges(i);
    sorted_edges_.emplace_back();
    sorted_edges_.back().reserve(edges.size());
    for (const auto& edge : edges) {
      sorted_edges_.back().emplace_back(edge);
    }
    std::sort(sorted_edges_.back().begin(), sorted_edges_.back().end());
  }

  // Disable non-fsms.
  for (size_t i = 0; i < grammar->ImplPtr()->per_rule_fsms.size(); i++) {
    if (!grammar->ImplPtr()->per_rule_fsms[i].has_value()) {
      visited_[i] = true;
    }
  }

  // Find the fsm which can be hashed: a terminal fsm, or a self-recursion fsm.
  auto current_operating_index = FindSimpleFsmCanBeHashed();
  while (current_operating_index != -1) {
    visited_[current_operating_index] = true;

    grammar->ImplPtr()->per_rule_fsm_hashes[current_operating_index] =
        HashFsm(current_operating_index);
    // Remove the fsm from the reference graph.
    for (const auto& referer : ref_graph_from_referee_to_referrer_[current_operating_index]) {
      ref_graph_from_referrer_to_referee_[referer].erase(std::find_if(
          ref_graph_from_referrer_to_referee_[referer].begin(),
          ref_graph_from_referrer_to_referee_[referer].end(),
          [&](int32_t rule_id) { return rule_id == current_operating_index; }
      ));
    }

    // Find if there are more fsms can be hashed.
    current_operating_index = FindSimpleFsmCanBeHashed();
  }

  // Try to hash the remaining fsms: they must contain something can't be hashed, like repetition.
  // We can do this: if the fsm's start state has no inward edges, and all the ref edges are hashed
  // except the edges at the start state, we can hash it.
  std::vector<std::pair<int32_t, uint64_t>> partial_hashed_list;
  for (int i = 0; i < (*grammar)->NumRules(); i++) {
    if (grammar->ImplPtr()->per_rule_fsm_hashes[i].has_value()) {
      continue;
    }
    if (!grammar->ImplPtr()->per_rule_fsms[i].has_value()) {
      continue;
    }
    if (has_inward_edges_[grammar->ImplPtr()->per_rule_fsms[i]->GetStart()]) {
      continue;
    }
    const auto& [can_be_hashed, hash_value] = IsPartialHashable(i);
    if (can_be_hashed) {
      partial_hashed_list.emplace_back(i, hash_value);
    }
  }
  for (const auto& [rule_id, hash_value] : partial_hashed_list) {
    grammar->ImplPtr()->per_rule_fsm_hashes[rule_id] = hash_value;
  }
}

std::pair<bool, uint64_t> GrammarFSMHasherImpl::IsPartialHashable(int fsm_index) {
  uint64_t hash_result = 0;
  XGRAMMAR_DCHECK(fsm_index >= 0 && fsm_index < (*grammar_)->NumRules())
      << "Invalid fsm index: " << fsm_index << " num_rules: " << (*grammar_)->NumRules();
  const auto& fsm = grammar_->ImplPtr()->per_rule_fsms[fsm_index];
  XGRAMMAR_DCHECK(fsm.has_value());
  std::map<int32_t, int32_t> original_state_id_to_new_id;
  original_state_id_to_new_id[fsm->GetStart()] = 0;
  std::queue<int32_t> bfs_queue;
  std::set<std::pair<int32_t, int32_t>> hash_and_target;
  bfs_queue.push(fsm->GetStart());
  // Perform a bfs to hash all the edges.
  while (!bfs_queue.empty()) {
    int current_old_state_id = bfs_queue.front();
    bool is_start = current_old_state_id == fsm->GetStart();
    int current_new_state_id = original_state_id_to_new_id[current_old_state_id];
    bfs_queue.pop();

    // Check if the current state is an end state.
    if (fsm->IsEndState(current_old_state_id)) {
      hash_result = HashCombine(
          hash_result, current_new_state_id, kEndStateFlag, kEndStateFlag, current_new_state_id
      );
    } else {
      hash_result = HashCombine(
          hash_result,
          current_new_state_id,
          kNotEndStateFlag,
          kNotEndStateFlag,
          current_new_state_id
      );
    }

    // Hash the edges.

    // First, check the edges which are rule references. To keep consistent, we need to sort them
    // with hashes.
    int32_t unhashed_rules_count = 0;
    for (const auto& edge : sorted_edges_[current_old_state_id]) {
      if (!edge.IsRuleRef()) {
        continue;
      }
      if (edge.GetRefRuleId() == fsm_index) {
        hash_and_target.insert({kSelfRecursionFlag, edge.target});
        continue;
      }
      if (!grammar_->ImplPtr()->per_rule_fsm_hashes[edge.GetRefRuleId()].has_value()) {
        // Can't be hashed.
        if (!is_start) {
          return {false, 0};
        } else {
          unhashed_rules_count++;
          if (unhashed_rules_count > 1) {
            return {false, 0};
          }
          hash_and_target.insert({kUnKnownFlag, edge.target});
        }
        continue;
      }
      hash_and_target.insert(
          {grammar_->ImplPtr()->per_rule_fsm_hashes[edge.GetRefRuleId()].value(), edge.target}
      );
    }

    // Hash them.
    for (const auto& [hash, target] : hash_and_target) {
      if (original_state_id_to_new_id.find(target) == original_state_id_to_new_id.end()) {
        original_state_id_to_new_id[target] =
            static_cast<int32_t>(original_state_id_to_new_id.size());
        bfs_queue.push(target);
      }
      int32_t target_new_id = original_state_id_to_new_id[target];
      hash_result = HashCombine(hash_result, current_new_state_id, hash, target_new_id);
    }

    // Then, check the edges which are not rule references.
    for (const auto& edge : sorted_edges_[current_old_state_id]) {
      // Visit a new node.
      if (original_state_id_to_new_id.find(edge.target) == original_state_id_to_new_id.end()) {
        original_state_id_to_new_id[edge.target] =
            static_cast<int32_t>(original_state_id_to_new_id.size());
        bfs_queue.push(edge.target);
      }
      int32_t target_new_id = original_state_id_to_new_id[edge.target];
      if (edge.IsRuleRef()) {
        continue;
      }
      hash_result = HashCombine(
          hash_result,
          current_new_state_id,
          static_cast<int32_t>(edge.min),
          static_cast<int32_t>(edge.max),
          target_new_id
      );
    }
  }
  std::vector<std::pair<int32_t, int32_t>> new_id_mapping;
  new_id_mapping.reserve(original_state_id_to_new_id.size());
  for (const auto& [original_state_id, new_state_id] : original_state_id_to_new_id) {
    new_id_mapping.emplace_back(original_state_id, new_state_id);
  }
  grammar_->ImplPtr()->per_rule_fsm_new_state_ids[fsm_index] = new_id_mapping;
  return {true, hash_result};
}

uint64_t GrammarFSMHasherImpl::HashFsm(int fsm_index) {
  uint64_t hash_result = 0;
  XGRAMMAR_DCHECK(fsm_index >= 0 && fsm_index < (*grammar_)->NumRules())
      << "Invalid fsm index: " << fsm_index << " num_rules: " << (*grammar_)->NumRules();
  const auto& fsm = grammar_->ImplPtr()->per_rule_fsms[fsm_index];
  XGRAMMAR_DCHECK(fsm.has_value());
  std::map<int32_t, int32_t> original_state_id_to_new_id;
  original_state_id_to_new_id[fsm->GetStart()] = 0;
  std::queue<int32_t> bfs_queue;
  std::set<std::pair<int32_t, int32_t>> hash_and_target;
  bfs_queue.push(fsm->GetStart());

  // Perform a bfs to hash all the edges.
  while (!bfs_queue.empty()) {
    int current_old_state_id = bfs_queue.front();
    int current_new_state_id = original_state_id_to_new_id[current_old_state_id];
    bfs_queue.pop();

    // Check if the current state is an end state.
    if (fsm->IsEndState(current_old_state_id)) {
      hash_result = HashCombine(
          hash_result, current_new_state_id, kEndStateFlag, kEndStateFlag, current_new_state_id
      );
    } else {
      hash_result = HashCombine(
          hash_result,
          current_new_state_id,
          kNotEndStateFlag,
          kNotEndStateFlag,
          current_new_state_id
      );
    }

    // Hash the edges.

    // First, check the edges which are rule references. To keep consistent, we need to sort them
    // with hashes.
    for (const auto& edge : sorted_edges_[current_old_state_id]) {
      if (!edge.IsRuleRef()) {
        continue;
      }
      if (edge.GetRefRuleId() == fsm_index) {
        hash_and_target.insert({kSelfRecursionFlag, edge.target});
        continue;
      }
      XGRAMMAR_CHECK(grammar_->ImplPtr()->per_rule_fsm_hashes[edge.GetRefRuleId()].has_value());
      hash_and_target.insert(
          {grammar_->ImplPtr()->per_rule_fsm_hashes[edge.GetRefRuleId()].value(), edge.target}
      );
    }

    // Hash them.
    for (const auto& [hash, target] : hash_and_target) {
      if (original_state_id_to_new_id.find(target) == original_state_id_to_new_id.end()) {
        original_state_id_to_new_id[target] =
            static_cast<int32_t>(original_state_id_to_new_id.size());
        bfs_queue.push(target);
      }
      int32_t target_new_id = original_state_id_to_new_id[target];
      hash_result = HashCombine(hash_result, current_new_state_id, hash, target_new_id);
    }

    // Then, check the edges which are not rule references.
    for (const auto& edge : sorted_edges_[current_old_state_id]) {
      // Visit a new node.
      if (original_state_id_to_new_id.find(edge.target) == original_state_id_to_new_id.end()) {
        original_state_id_to_new_id[edge.target] =
            static_cast<int32_t>(original_state_id_to_new_id.size());
        bfs_queue.push(edge.target);
      }
      int32_t target_new_id = original_state_id_to_new_id[edge.target];
      if (edge.IsRuleRef()) {
        continue;
      }
      hash_result = HashCombine(
          hash_result,
          current_new_state_id,
          static_cast<int32_t>(edge.min),
          static_cast<int32_t>(edge.max),
          target_new_id
      );
    }
  }
  std::vector<std::pair<int32_t, int32_t>> new_id_mapping;
  new_id_mapping.reserve(original_state_id_to_new_id.size());
  for (const auto& [original_state_id, new_state_id] : original_state_id_to_new_id) {
    new_id_mapping.emplace_back(original_state_id, new_state_id);
  }
  grammar_->ImplPtr()->per_rule_fsm_new_state_ids[fsm_index] = new_id_mapping;
  return hash_result;
}

std::optional<uint64_t> GrammarFSMHasherImpl::HashSequence(
    const Grammar& grammar, int32_t sequence_id
) {
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  if (sequence_id == -1) {
    return std::nullopt;
  }
  uint64_t hash_result = 0;
  const auto& sequence_expr = grammar->GetGrammarExpr(sequence_id);
  XGRAMMAR_DCHECK(sequence_expr.type == GrammarExprType::kSequence)
      << "GrammarExpr is not a sequence";
  for (const auto& expr_id : sequence_expr) {
    const auto& expr = grammar->GetGrammarExpr(expr_id);
    hash_result = HashCombine(hash_result, static_cast<int32_t>(expr.type));
    switch (expr.type) {
      case (GrammarExprType::kByteString):
      case (GrammarExprType::kCharacterClass):
      case (GrammarExprType::kCharacterClassStar):
      case (GrammarExprType::kEmptyStr): {
        for (const auto& element : expr) {
          hash_result = HashCombine(hash_result, element);
        }
        break;
      }
      case (GrammarExprType::kRuleRef): {
        if (grammar->per_rule_fsm_hashes[expr[0]].has_value()) {
          hash_result = HashCombine(hash_result, grammar->per_rule_fsm_hashes[expr[0]].value());
        } else {
          return std::nullopt;
        }
        break;
      }
      case (GrammarExprType::kRepeat): {
        if (grammar->per_rule_fsm_hashes[expr[0]].has_value()) {
          hash_result = HashCombine(hash_result, grammar->per_rule_fsm_hashes[expr[0]].value());
        } else {
          return std::nullopt;
        }
        hash_result = HashCombine(hash_result, expr[1]);
        hash_result = HashCombine(hash_result, expr[2]);
        break;
      }
      case (GrammarExprType::kSequence):
      case (GrammarExprType::kChoices): {
        return std::nullopt;
      }
      case (GrammarExprType::kTagDispatch): {
        return std::nullopt;
      }
    }
  }
  return hash_result;
}

class RuleLevelCache::Impl {
 public:
  using NodeKey = std::tuple<
      uint64_t /*The hash value of the FSM*/,
      int32_t /* The normalized node id*/,
      int32_t /*The number of states*/,
      int32_t /* The number of edges*/>;
  using NodeType = std::pair<NodeKey, AdaptiveTokenMask>;

  explicit Impl(size_t max_cache_memory_size) : max_cache_memory_size_(max_cache_memory_size) {}

  std::optional<AdaptiveTokenMask> GetCache(
      const uint64_t& fsm_hash,
      int32_t fsm_new_node_id,
      const int32_t& state_cnt,
      const int32_t edge_cnt
  );

  bool AddCache(
      const uint64_t& fsm_hash,
      int32_t fsm_new_node_id,
      const int32_t& state_cnt,
      const int32_t edge_cnt,
      const AdaptiveTokenMask& token_mask
  );

  bool AddCache(
      const uint64_t& fsm_hash,
      int32_t fsm_new_node_id,
      const int32_t& state_cnt,
      const int32_t edge_cnt,
      AdaptiveTokenMask&& token_mask
  );

  void ClearCache();

  friend size_t MemorySize(const Impl* impl) { return impl->current_cache_memory_size_; }

  size_t GetMaxSize() const { return max_cache_memory_size_; }

 private:
  // The cache map: fsm_hash -> fsm_new_node_id -> AdaptiveTokenMask
  std::mutex mutex_;
  const size_t max_cache_memory_size_;
  int64_t current_cache_memory_size_ = 0;
  List<NodeType> cache_list_;
  std::unordered_map<NodeKey, int> cache_;
};

std::optional<AdaptiveTokenMask> RuleLevelCache::GetCache(
    const uint64_t& fsm_hash,
    int32_t fsm_new_node_id,
    const int32_t& state_cnt,
    const int32_t edge_cnt
) {
  return pimpl_->GetCache(fsm_hash, fsm_new_node_id, state_cnt, edge_cnt);
}

bool RuleLevelCache::AddCache(
    const uint64_t& fsm_hash,
    int32_t fsm_new_node_id,
    const int32_t& state_cnt,
    const int32_t edge_cnt,
    const AdaptiveTokenMask& token_mask
) {
  return pimpl_->AddCache(fsm_hash, fsm_new_node_id, state_cnt, edge_cnt, token_mask);
}

bool RuleLevelCache::AddCache(
    const uint64_t& fsm_hash,
    int32_t fsm_new_node_id,
    const int32_t& state_cnt,
    const int32_t edge_cnt,
    AdaptiveTokenMask&& token_mask
) {
  return pimpl_->AddCache(fsm_hash, fsm_new_node_id, state_cnt, edge_cnt, std::move(token_mask));
}

void RuleLevelCache::ClearCache() { pimpl_->ClearCache(); }

size_t RuleLevelCache::GetMaxSize() const { return pimpl_->GetMaxSize(); }

std::optional<AdaptiveTokenMask> RuleLevelCache::Impl::GetCache(
    const uint64_t& fsm_hash,
    int32_t fsm_new_node_id,
    const int32_t& state_cnt,
    const int32_t edge_cnt
) {
  // Find in the cache.
  std::lock_guard<std::mutex> lock(mutex_);
  NodeKey key = std::make_tuple(fsm_hash, fsm_new_node_id, state_cnt, edge_cnt);
  auto it = cache_.find(key);
  if (it == cache_.end()) {
    return std::nullopt;
  }

  // Move the node to the back of the list.
  cache_list_.MoveBack(it->second);
  return List<NodeType>::iterator(it->second, cache_list_)->second;
}

bool RuleLevelCache::Impl::AddCache(
    const uint64_t& fsm_hash,
    int32_t fsm_new_node_id,
    const int32_t& state_cnt,
    const int32_t edge_cnt,
    const AdaptiveTokenMask& token_mask
) {
  return AddCache(fsm_hash, fsm_new_node_id, state_cnt, edge_cnt, AdaptiveTokenMask(token_mask));
}

bool RuleLevelCache::Impl::AddCache(
    const uint64_t& fsm_hash,
    int32_t fsm_new_node_id,
    const int32_t& state_cnt,
    const int32_t edge_cnt,
    AdaptiveTokenMask&& token_mask
) {
  // Check if we can add to the cache.
  std::lock_guard<std::mutex> lock(mutex_);
  NodeKey key = std::make_tuple(fsm_hash, fsm_new_node_id, state_cnt, edge_cnt);
  if (max_cache_memory_size_ != kUnlimitedSize && MemorySize(token_mask) > max_cache_memory_size_) {
    // The token mask is too large to be cached.
    return false;
  }
  if (cache_.find(key) != cache_.end()) {
    // Already exists.
    return false;
  }

  // Evict old entries if needed.
  if (max_cache_memory_size_ != kUnlimitedSize) {
    size_t new_item_size = MemorySize(token_mask);
    while ((current_cache_memory_size_) >
           static_cast<int64_t>(max_cache_memory_size_ - new_item_size)) {
      auto oldest_it = cache_list_.begin();
      if (oldest_it == cache_list_.end()) {
        // This should not happen if the size of the new item is smaller than
        // max_cache_memory_size_, but this is a safeguard.
        break;
      }
      current_cache_memory_size_ -= MemorySize(oldest_it->second);
      cache_.erase(oldest_it->first);
      cache_list_.Erase(oldest_it);
    }
  }

  // Add to the cache.
  auto new_it = cache_list_.PushBack(NodeType(key, std::move(token_mask)));
  current_cache_memory_size_ += MemorySize(new_it->second);
  cache_[key] = new_it.Index();
  return true;
}

RuleLevelCache::RuleLevelCache(size_t max_cache_memory_size)
    : pimpl_(std::make_shared<Impl>(max_cache_memory_size)) {}

void RuleLevelCache::Impl::ClearCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_list_.Clear();
  cache_.clear();
  current_cache_memory_size_ = 0;
}

size_t MemorySize(const RuleLevelCache& manager) { return MemorySize(manager.ImplPtr()); }

/*************************** Forward grammar constructors to their impl ***************************/

Grammar GrammarUnionFunctor::Apply(const std::vector<Grammar>& grammars) {
  return GrammarUnionFunctorImpl().Apply(grammars);
}

Grammar GrammarConcatFunctor::Apply(const std::vector<Grammar>& grammars) {
  return GrammarConcatFunctorImpl().Apply(grammars);
}

int32_t SubGrammarAdder::Apply(GrammarBuilder* builder, const Grammar& sub_grammar) {
  return SubGrammarAdderImpl().ApplyWithBuilder(builder, sub_grammar);
}

/*************************** Forward grammar Normalizers to their impl ***************************/

Grammar GrammarNormalizer::Apply(const Grammar& grammar) {
  return GrammarNormalizerImpl().Apply(grammar);
}

Grammar StructureNormalizer::Apply(const Grammar& grammar) {
  return StructureNormalizerImpl().Apply(grammar);
}

/*************************** Forward grammar optimizers to their impl ***************************/

void GrammarFSMBuilder::Apply(Grammar* grammar) { GrammarFSMBuilderImpl().Apply(grammar); }

void RepetitionNormalizer::Apply(Grammar* grammar) { RepetitionNormalizerImpl().Apply(grammar); }

void GrammarFSMHasher::Apply(Grammar* grammar) { GrammarFSMHasherImpl().Apply(grammar); }

std::optional<uint64_t> GrammarFSMHasher::HashSequence(
    const Grammar& grammar, int32_t sequence_id
) {
  return GrammarFSMHasherImpl().HashSequence(grammar, sequence_id);
}

FSMWithStartEnd GrammarFSMBuilder::RuleRef(const GrammarExpr& expr) {
  return GrammarFSMBuilderImpl::RuleRef(expr);
}

FSMWithStartEnd GrammarFSMBuilder::CharacterClass(const GrammarExpr& expr) {
  return GrammarFSMBuilderImpl::CharacterClass(expr);
}

FSMWithStartEnd GrammarFSMBuilder::ByteString(const GrammarExpr& expr) {
  return GrammarFSMBuilderImpl::ByteString(expr);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilder::Sequence(
    const GrammarExpr& expr, const Grammar& grammar
) {
  return GrammarFSMBuilderImpl::Sequence(expr, grammar);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilder::Choices(
    const GrammarExpr& expr, const Grammar& grammar
) {
  return GrammarFSMBuilderImpl::Choices(expr, grammar);
}

std::optional<FSMWithStartEnd> GrammarFSMBuilder::TagDispatch(
    const Grammar::Impl::TagDispatch& tag_dispatch
) {
  return GrammarFSMBuilderImpl::TagDispatch(tag_dispatch);
}

std::vector<int32_t> AllowEmptyRuleAnalyzer::Apply(const Grammar& grammar) {
  return AllowEmptyRuleAnalyzerImpl().Apply(grammar);
}

Grammar RuleInliner::Apply(const Grammar& grammar) { return RuleInlinerImpl().Apply(grammar); }

Grammar DeadCodeEliminator::Apply(const Grammar& grammar) {
  return DeadCodeEliminatorImpl().Apply(grammar);
}

Grammar LookaheadAssertionAnalyzer::Apply(const Grammar& grammar) {
  return LookaheadAssertionAnalyzerImpl().Apply(grammar);
}

Grammar GrammarOptimizer::Apply(const Grammar& grammar) {
  return GrammarOptimizerImpl::Apply(grammar);
}

Grammar ByteStringFuser::Apply(const Grammar& grammar) {
  return ByteStringFuserImpl().Apply(grammar);
}

Grammar RootRuleRenamer::Apply(const Grammar& grammar) {
  return RootRuleRenamerImpl().Apply(grammar);
}

}  // namespace xgrammar
