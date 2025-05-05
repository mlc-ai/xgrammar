/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_functor.cc
 */

#include "grammar_functor.h"

#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <queue>
#include <set>
#include <unordered_set>
#include <vector>

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
    auto rule_expr = base_grammar_->GetRuleExpr(lookahead_assertion_id);
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
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      builder_.AddEmptyRule(base_grammar_->GetRule(i).name);
    }
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto rule_expr = base_grammar_->GetRuleExpr(rule.body_expr_id);
      cur_rule_name_ = rule.name;
      auto new_body_expr_id = VisitRuleBody(rule_expr);
      builder_.UpdateRuleBody(i, new_body_expr_id);
      builder_.AddLookaheadAssertion(i, VisitLookaheadAssertion(rule.lookahead_assertion_id));
    }
    return builder_.Get(base_grammar_->GetRootRule().name);
  }

 private:
  int32_t VisitLookaheadAssertion(int32_t lookahead_assertion_id) final {
    if (lookahead_assertion_id == -1) {
      return -1;
    }
    auto assertion_expr = base_grammar_->GetRuleExpr(lookahead_assertion_id);
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
      auto choice_expr = base_grammar_->GetRuleExpr(i);
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
      auto element_expr = base_grammar_->GetRuleExpr(i);
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

class StructureNormalizerImpl : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

  Grammar Apply(const Grammar& grammar) final {
    return NestedRuleUnwrapper().Apply(SingleElementExprEliminator().Apply(grammar));
  }
};

class ByteStringFuserImpl : public GrammarMutator {
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
      auto element_expr = base_grammar_->GetRuleExpr(i);
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

class RuleInlinerImpl : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

 private:
  int32_t VisitChoices(const RuleExpr& rule_expr) final {
    std::vector<int32_t> new_choice_ids;
    for (int i : rule_expr) {
      auto choice_expr = base_grammar_->GetRuleExpr(i);
      if (choice_expr.type == RuleExprType::kEmptyStr) {
        new_choice_ids.push_back(VisitExpr(i));
        continue;
      }
      XGRAMMAR_ICHECK(choice_expr.type == RuleExprType::kSequence);
      auto first_element = base_grammar_->GetRuleExpr(choice_expr[0]);
      if (first_element.type != RuleExprType::kRuleRef) {
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
      auto ref_rule_expr = base_grammar_->GetRuleExpr(ref_rule.body_expr_id);

      for (auto ref_choice_id : ref_rule_expr) {
        auto ref_choice_expr = base_grammar_->GetRuleExpr(ref_choice_id);
        XGRAMMAR_ICHECK(ref_choice_expr.type == RuleExprType::kSequence);
        std::vector<int32_t> choice_to_add;
        for (auto ref_element_id : ref_choice_expr) {
          choice_to_add.push_back(VisitExpr(ref_element_id));
        }
        choice_to_add.insert(choice_to_add.end(), other_elements.begin(), other_elements.end());
        new_choice_ids.push_back(builder_.AddSequence(choice_to_add));
      }
    }
    return builder_.AddChoices(new_choice_ids);
  }

  /**
   * The rule should be: a sequence of choices, cannot be empty, cannot refer to other rules
   */
  bool CheckIfRuleCanBeInlined(int32_t rule_id) {
    auto rule = base_grammar_->GetRule(rule_id);
    auto rule_expr = base_grammar_->GetRuleExpr(rule.body_expr_id);
    if (rule_expr.type != RuleExprType::kChoices) {
      return false;
    }
    if (rule_expr.size() == 0) {
      return false;
    }
    for (auto choice_id : rule_expr) {
      auto choice_expr = base_grammar_->GetRuleExpr(choice_id);
      if (choice_expr.type == RuleExprType::kEmptyStr) {
        return false;
      }
      XGRAMMAR_ICHECK(choice_expr.type == RuleExprType::kSequence);
      for (auto element_id : choice_expr) {
        auto element_expr = base_grammar_->GetRuleExpr(element_id);
        if (element_expr.type == RuleExprType::kRuleRef) {
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
    base_grammar_ = grammar;

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
    }

    return std::vector<int32_t>(visited.begin(), visited.end());
  }

  void VisitTagDispatch(const RuleExpr& rule_expr) {
    for (int i = 0; i < rule_expr.size(); i += 2) {
      visit_queue_.push(rule_expr[i + 1]);
    }
  }

  void VisitRuleRef(const RuleExpr& rule_expr) { visit_queue_.push(rule_expr[0]); }

 private:
  std::queue<int32_t> visit_queue_;
};

class DeadCodeEliminatorImpl : public GrammarMutator {
 public:
  using GrammarMutator::Apply;
  using GrammarMutator::GrammarMutator;

  Grammar Apply(const Grammar& grammar) final {
    Init(grammar);
    auto used_rules = UsedRulesAnalyzer().Apply(grammar);
    rule_id_map_.clear();
    for (auto rule_id : used_rules) {
      rule_id_map_[rule_id] = builder_.AddEmptyRule(grammar->GetRule(rule_id).name);
    }
    for (auto rule_id : used_rules) {
      auto rule = grammar->GetRule(rule_id);
      auto new_body_expr_id = VisitExpr(rule.body_expr_id);
      builder_.UpdateRuleBody(rule_id_map_[rule_id], new_body_expr_id);
      builder_.AddLookaheadAssertion(
          rule_id_map_[rule_id], VisitLookaheadAssertion(rule.lookahead_assertion_id)
      );
    }
    XGRAMMAR_CHECK(rule_id_map_.count(grammar->GetRootRuleId()) > 0);
    return builder_.Get(rule_id_map_[grammar->GetRootRuleId()]);
  }

  int32_t VisitTagDispatch(const RuleExpr& rule_expr) final {
    std::vector<std::pair<int32_t, int32_t>> tag_dispatch_list;
    for (int i = 0; i < rule_expr.size(); i += 2) {
      XGRAMMAR_DCHECK(rule_id_map_.count(rule_expr[i + 1]) > 0);
      auto new_rule_id = rule_id_map_[rule_expr[i + 1]];
      tag_dispatch_list.push_back({VisitExpr(rule_expr[i]), new_rule_id});
    }
    return builder_.AddTagDispatch(tag_dispatch_list);
  }

  int32_t VisitRuleRef(const RuleExpr& rule_expr) final {
    XGRAMMAR_DCHECK(rule_id_map_.count(rule_expr[0]) > 0);
    auto new_rule_id = rule_id_map_[rule_expr[0]];
    return builder_.AddRuleRef(new_rule_id);
  }

 private:
  std::unordered_map<int32_t, int32_t> rule_id_map_;
};

class LookaheadAssertionAnalyzerImpl : public GrammarMutator {
 public:
  using GrammarMutator::GrammarMutator;

  Grammar Apply(const Grammar& grammar) final {
    InitWithCopy(grammar);
    auto root_rule = grammar->GetRootRule();
    auto root_rule_expr = base_grammar_->GetRuleExpr(root_rule.body_expr_id);
    if (root_rule_expr.type == RuleExprType::kTagDispatch) {
      return grammar;
    }
    for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
      auto rule = grammar->GetRule(i);
      if (i == grammar->GetRootRuleId() || rule.lookahead_assertion_id != -1) {
        continue;
      }
      auto look_head_assertion_id = DetectLookaheadAssertion(i);
      if (look_head_assertion_id != -1) {
        builder_.AddLookaheadAssertion(i, look_head_assertion_id);
      }
    }
    return builder_.Get(grammar->GetRootRuleId());
  }

  int32_t DetectLookaheadAssertion(int32_t rule_id) {
    std::vector<int32_t> found_sequence;  // Element ids
    bool found = false;
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto rule_expr = base_grammar_->GetRuleExpr(rule.body_expr_id);
      if (rule_expr.type == RuleExprType::kTagDispatch) {
        for (int j = 1; j < rule_expr.size(); j += 2) {
          if (rule_expr[j] == rule_id) {
            return -1;
          }
        }
        continue;
      }
      XGRAMMAR_DCHECK(rule_expr.type == RuleExprType::kChoices);
      for (auto sequence_id : rule_expr) {
        auto sequence_expr = base_grammar_->GetRuleExpr(sequence_id);
        if (sequence_expr.type != RuleExprType::kSequence) {
          continue;
        }
        auto last_element = base_grammar_->GetRuleExpr(sequence_expr.end()[-1]);
        if (last_element.type == RuleExprType::kRuleRef && last_element[0] == rule_id &&
            i != rule_id) {
          return -1;
        }

        for (int j = 0; j < sequence_expr.size() - 1; ++j) {
          auto element_expr = base_grammar_->GetRuleExpr(sequence_expr[j]);
          if (element_expr.type != RuleExprType::kRuleRef || element_expr[0] != rule_id) {
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
    return builder_.AddSequence(found_sequence);
  }
};

/*!
 * \brief A class that normalizes a grammar by applying a series of transformations.
 *
 * The normalizer applies the following transformations in order:
 * 1. SingleElementExprEliminator - Eliminates single element expressions
 * 2. NestedRuleUnwrapper - Unwraps nested rules
 * 3. ByteStringFuser - Fuses consecutive byte strings
 */
class GrammarNormalizerImpl : public GrammarMutator {
 public:
  GrammarNormalizerImpl() = default;

  Grammar Apply(const Grammar& grammar) final {
    std::vector<std::unique_ptr<GrammarMutator>> normalizer_mutators = GetNormalizerList();
    base_grammar_ = grammar;
    for (auto& mutator : normalizer_mutators) {
      base_grammar_ = mutator->Apply(base_grammar_);
    }
    return base_grammar_;
  }

 private:
  // Return the list of all normalizers in the class. The normalizers are applied one by one.
  std::vector<std::unique_ptr<GrammarMutator>> GetNormalizerList() {
    std::vector<std::unique_ptr<GrammarMutator>> normalizer_mutators;
    normalizer_mutators.emplace_back(std::make_unique<StructureNormalizerImpl>());
    normalizer_mutators.emplace_back(std::make_unique<ByteStringFuserImpl>());
    normalizer_mutators.emplace_back(std::make_unique<RuleInlinerImpl>());
    normalizer_mutators.emplace_back(std::make_unique<DeadCodeEliminatorImpl>());
    normalizer_mutators.emplace_back(std::make_unique<LookaheadAssertionAnalyzerImpl>());
    return normalizer_mutators;
  }
};

/*!
 * \brief Base class for grammar mutators that add subgrammars.
 *
 * Provides functionality to visit a subgrammar and add its rules to the builder
 * while maintaining proper rule references and names.
 */
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
    base_grammar_ = grammar;
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

  int32_t VisitRuleRef(const RuleExpr& rule_expr) final {
    return builder_.AddRuleRef(new_rule_ids_names[rule_expr[0]].first);
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

  // Avoid hiding the original Apply(const Grammar&)
  Grammar Apply(const Grammar& grammar) final { XGRAMMAR_LOG(FATAL) << "Should not be called"; }
};

/*!
 * \brief Implementation of grammar concatenation operation.
 *
 * Creates a new grammar that accepts strings that are concatenations of strings
 * from the input grammars in order. The resulting grammar has a new root rule
 * that concatenates the root rules of all input grammars.
 */
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

  // Avoid hiding the original Apply(const Grammar&)
  Grammar Apply(const Grammar& grammar) final { XGRAMMAR_LOG(FATAL) << "Should not be called"; }
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
    base_grammar_ = grammar;
    rule_visit_graph_ = std::vector<std::vector<int32_t>>(base_grammar_->NumRules());
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      auto rule = base_grammar_->GetRule(i);
      auto rule_expr = base_grammar_->GetRuleExpr(rule.body_expr_id);
      cur_rule_id_ = i;
      VisitExpr(rule_expr);
    }
    for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
      std::sort(rule_visit_graph_[i].begin(), rule_visit_graph_[i].end());
      auto end_it = std::unique(rule_visit_graph_[i].begin(), rule_visit_graph_[i].end());
      rule_visit_graph_[i].erase(end_it, rule_visit_graph_[i].end());
    }
    return std::move(rule_visit_graph_);
  }

 private:
  void VisitRuleRef(const RuleExpr& rule_expr) {
    rule_visit_graph_[rule_expr[0]].push_back(cur_rule_id_);
  }

  void VisitTagDispatch(const RuleExpr& rule_expr) {
    for (int i = 1; i < rule_expr.size(); i += 2) {
      rule_visit_graph_[rule_expr[i]].push_back(cur_rule_id_);
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
    base_grammar_ = grammar;

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
      auto rule_expr = base_grammar_->GetRuleExpr(rule.body_expr_id);
      if (rule_expr.type == RuleExprType::kTagDispatch) {
        empty_rule_id_set->insert(i);
        continue;
      }

      XGRAMMAR_DCHECK(rule_expr.type == RuleExprType::kChoices);
      if (base_grammar_->GetRuleExpr(rule_expr[0]).type == RuleExprType::kEmptyStr) {
        empty_rule_id_set->insert(i);
        continue;
      }

      for (auto seq_id : rule_expr) {
        auto seq_expr = base_grammar_->GetRuleExpr(seq_id);
        if (std::all_of(seq_expr.begin(), seq_expr.end(), [&](int32_t i) {
              return base_grammar_->GetRuleExpr(i).type == RuleExprType::kCharacterClassStar;
            })) {
          empty_rule_id_set->insert(i);
          break;
        }
      }
    }
  }

  bool SeqExprIsEpsilon(
      const RuleExpr& seq_expr, const std::unordered_set<int32_t>& empty_rule_id_set
  ) {
    if (seq_expr.type == RuleExprType::kEmptyStr) {
      return true;
    }
    XGRAMMAR_DCHECK(seq_expr.type == RuleExprType::kSequence);

    return std::all_of(seq_expr.begin(), seq_expr.end(), [&](int32_t i) {
      auto element_expr = base_grammar_->GetRuleExpr(i);
      return (element_expr.type == RuleExprType::kRuleRef &&
              empty_rule_id_set.count(element_expr[0])) ||
             element_expr.type == RuleExprType::kCharacterClassStar;
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
        auto rule_expr = base_grammar_->GetRuleExpr(rule.body_expr_id);

        XGRAMMAR_DCHECK(rule_expr.type != RuleExprType::kTagDispatch)
            << "TagDispatch rules should already exist in empty_rule_id_set";

        bool is_epsilon = std::any_of(rule_expr.begin(), rule_expr.end(), [&](int32_t i) {
          auto seq_expr = base_grammar_->GetRuleExpr(i);
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

class StructuralTagGrammarCreatorImpl : public SubGrammarAdder {
 public:
  Grammar Apply(
      const std::vector<std::string>& triggers,
      const std::vector<std::vector<std::pair<StructuralTagItem, Grammar>>>& tag_groups
  ) {
    XGRAMMAR_CHECK(triggers.size() == tag_groups.size())
        << "Number of triggers must match number of tag groups";

    builder_ = GrammarBuilder();
    auto root_rule_id = builder_.AddEmptyRule("root");

    // Create rules for each trigger group
    std::vector<std::pair<int32_t, int32_t>> trigger_rule_pairs;
    trigger_rule_pairs.reserve(triggers.size());
    for (size_t i = 0; i < triggers.size(); i++) {
      // Skip empty trigger groups
      if (tag_groups[i].empty()) {
        continue;
      }

      auto rule_name = "trigger_rule_" + std::to_string(i);
      auto rule_id = builder_.AddEmptyRule(rule_name);

      // Convert trigger string to byte string expr
      auto trigger_expr_id = builder_.AddByteString(triggers[i]);

      // Create choices for each tag in this trigger group
      std::vector<int32_t> choices;
      choices.reserve(tag_groups[i].size());
      for (const auto& [tag, schema_grammar] : tag_groups[i]) {
        // Create sequence: start_suffix + schema + end
        std::vector<int32_t> seq_elements;
        seq_elements.reserve(3);

        // Add begin suffix (everything after trigger)
        XGRAMMAR_DCHECK(tag.begin.size() >= triggers[i].size())
            << "Tag begin must be at least as long as trigger";
        if (tag.begin.size() > triggers[i].size()) {
          seq_elements.push_back(builder_.AddByteString(tag.begin.substr(triggers[i].size())));
        }

        // Create and visit schema grammar for this tag
        auto schema_rule_id = VisitSubGrammar(schema_grammar);
        seq_elements.push_back(builder_.AddRuleRef(schema_rule_id));

        // Add end string
        if (!tag.end.empty()) {
          seq_elements.push_back(builder_.AddByteString(tag.end));
        }

        choices.push_back(builder_.AddSequence(seq_elements));
      }

      builder_.UpdateRuleBody(rule_id, builder_.AddChoices(choices));
      trigger_rule_pairs.emplace_back(trigger_expr_id, rule_id);
    }

    // Create root TagDispatch rule
    std::vector<std::pair<int32_t, int32_t>> tag_dispatch_data;
    tag_dispatch_data.reserve(trigger_rule_pairs.size());
    for (const auto& [trigger_id, rule_id] : trigger_rule_pairs) {
      tag_dispatch_data.emplace_back(trigger_id, rule_id);
    }

    builder_.UpdateRuleBody(root_rule_id, builder_.AddTagDispatch(tag_dispatch_data));
    return builder_.Get(root_rule_id);
  }

  // Avoid hiding the original Apply(const Grammar&)
  Grammar Apply(const Grammar& grammar) final { XGRAMMAR_LOG(FATAL) << "Should not be called"; }
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

std::vector<int32_t> AllowEmptyRuleAnalyzer::Apply(const Grammar& grammar) {
  return AllowEmptyRuleAnalyzerImpl().Apply(grammar);
}

Grammar StructuralTagGrammarCreator::Apply(
    const std::vector<std::string>& triggers,
    const std::vector<std::vector<std::pair<StructuralTagItem, Grammar>>>& tag_groups
) {
  return StructuralTagGrammarCreatorImpl().Apply(triggers, tag_groups);
}

Grammar RuleInliner::Apply(const Grammar& grammar) { return RuleInlinerImpl().Apply(grammar); }

Grammar ByteStringFuser::Apply(const Grammar& grammar) {
  return ByteStringFuserImpl().Apply(grammar);
}

Grammar DeadCodeEliminator::Apply(const Grammar& grammar) {
  return DeadCodeEliminatorImpl().Apply(grammar);
}

Grammar StructureNormalizer::Apply(const Grammar& grammar) {
  return StructureNormalizerImpl().Apply(grammar);
}

Grammar LookaheadAssertionAnalyzer::Apply(const Grammar& grammar) {
  return LookaheadAssertionAnalyzerImpl().Apply(grammar);
}

}  // namespace xgrammar
