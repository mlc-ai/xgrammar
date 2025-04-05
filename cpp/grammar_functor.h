/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_functor.h
 * \brief The header for the simplification of the BNF AST.
 */

#ifndef XGRAMMAR_GRAMMAR_FUNCTOR_H_
#define XGRAMMAR_GRAMMAR_FUNCTOR_H_

#include <xgrammar/xgrammar.h>

#include <string>

#include "grammar_builder.h"
#include "grammar_data_structure.h"

namespace xgrammar {

/*!
 * \brief Base class for visitors and mutators of the BNF grammar.
 * \tparam T The type of the return value of visitor functions. Typical values:
 * - int32_t: the id of the new rule_expr
 * - void: no return value
 * \tparam ReturnType The type of the return value of the transform function Apply(). Typical values
 * are void (for visitor) and Grammar (for mutator).
 */
template <typename T = int32_t, typename ReturnType = Grammar>
class GrammarFunctor {
 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to visit or mutate.
   */
  explicit GrammarFunctor() {}

  /*!
   * \brief Apply the transformation to the grammar, or visit the grammar.
   * \return The transformed grammar, or the visiting result, or void.
   */
  virtual ReturnType Apply(const Grammar& grammar) {
    // The initializer MUST be called at first when overriding the Apply() function.
    Init(grammar);
    if constexpr (std::is_same<T, void>::value) {
      for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
        auto rule = base_grammar_->GetRule(i);
        cur_rule_name_ = rule.name;
        VisitExpr(rule.body_expr_id);
        VisitLookaheadAssertion(rule.lookahead_assertion_id);
      }
      return ReturnType();
    } else if constexpr (std::is_same<T, int32_t>::value &&
                         std::is_same<ReturnType, Grammar>::value) {
      // First add empty rules to ensure the new rule ids the same as the old ones, then update
      // the rule bodies
      for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
        builder_.AddEmptyRule(base_grammar_->GetRule(i).name);
      }
      for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
        auto rule = base_grammar_->GetRule(i);
        cur_rule_name_ = rule.name;
        auto new_body_expr_id = VisitExpr(rule.body_expr_id);
        builder_.UpdateRuleBody(i, new_body_expr_id);
        // Handle lookahead assertion
        builder_.AddLookaheadAssertion(i, VisitLookaheadAssertion(rule.lookahead_assertion_id));
      }
      return builder_.Get(base_grammar_->GetRootRule().name);
    } else {
      return ReturnType();
    }
  }

  /*! \brief Virtual destructor. */
  virtual ~GrammarFunctor() = default;

 protected:
  using Rule = Grammar::Impl::Rule;
  using RuleExpr = Grammar::Impl::RuleExpr;
  using RuleExprType = Grammar::Impl::RuleExprType;

  /*! \brief Initialize the functor. Should be called at the beginning of Apply(). */
  virtual void Init(const Grammar& grammar) {
    base_grammar_ = grammar;
    builder_ = GrammarBuilder();
  }

  virtual void InitWithCopy(const Grammar& grammar) {
    base_grammar_ = grammar;
    builder_ = GrammarBuilder(grammar);
  }

  /*! \brief Visit a lookahead assertion expr referred by id. */
  virtual T VisitLookaheadAssertion(int32_t lookahead_assertion_id) {
    if (lookahead_assertion_id == -1) {
      if constexpr (std::is_same<T, int32_t>::value) {
        return -1;
      } else {
        return T();
      }
    }
    return VisitExpr(lookahead_assertion_id);
  }

  /*! \brief Visit a RuleExpr by id. */
  virtual T VisitExpr(int32_t old_rule_expr_id) {
    return VisitExpr(base_grammar_->GetRuleExpr(old_rule_expr_id));
  }

  /*! \brief Visit a RuleExpr. Dispatch to the corresponding Visit function. */
  virtual T VisitExpr(const RuleExpr& rule_expr) {
    switch (rule_expr.type) {
      case RuleExprType::kSequence:
        return VisitSequence(rule_expr);
      case RuleExprType::kChoices:
        return VisitChoices(rule_expr);
      case RuleExprType::kEmptyStr:
        return VisitEmptyStr(rule_expr);
      case RuleExprType::kByteString:
        return VisitByteString(rule_expr);
      case RuleExprType::kCharacterClass:
        return VisitCharacterClass(rule_expr);
      case RuleExprType::kCharacterClassStar:
        return VisitCharacterClassStar(rule_expr);
      case RuleExprType::kRuleRef:
        return VisitRuleRef(rule_expr);
      case RuleExprType::kTagDispatch:
        return VisitTagDispatch(rule_expr);
      default:
        XGRAMMAR_LOG(FATAL) << "Unexpected sequence type: " << static_cast<int>(rule_expr.type);
    }
  }

  /*! \brief Visit a choices RuleExpr. */
  virtual T VisitChoices(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : rule_expr) {
        VisitExpr(i);
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<int32_t> choice_ids;
      for (int32_t i : rule_expr) {
        choice_ids.push_back(VisitExpr(i));
      }
      return builder_.AddChoices(choice_ids);
    } else {
      return T();
    }
  }

  /*! \brief Visit a sequence RuleExpr. */
  virtual T VisitSequence(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : rule_expr) {
        VisitExpr(i);
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<T> sequence_ids;
      for (int32_t i : rule_expr) {
        sequence_ids.push_back(VisitExpr(i));
      }
      return builder_.AddSequence(sequence_ids);
    } else {
      return T();
    }
  }

  virtual T VisitTagDispatch(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (int i = 0; i < rule_expr.size(); i += 2) {
        VisitExpr(rule_expr[i]);
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<std::pair<int32_t, int32_t>> tag_dispatch_list;
      for (int i = 0; i < rule_expr.size(); i += 2) {
        tag_dispatch_list.push_back({VisitExpr(rule_expr[i]), rule_expr[i + 1]});
      }
      return builder_.AddTagDispatch(tag_dispatch_list);
    } else {
      return T();
    }
  }

  /*! \brief Visit an element RuleExpr, including empty string, character class, and rule ref. */
  virtual T VisitElement(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      return;
    } else if constexpr (std::is_same<T, int32_t>::value) {
      return builder_.AddRuleExpr(rule_expr);
    } else {
      return T();
    }
  }

  /*! \brief Visit an empty string RuleExpr. */
  virtual T VisitEmptyStr(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a character class RuleExpr. */
  virtual T VisitByteString(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a character class RuleExpr. */
  virtual T VisitCharacterClass(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a star quantifier RuleExpr. */
  virtual T VisitCharacterClassStar(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief Visit a rule reference RuleExpr. */
  virtual T VisitRuleRef(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  /*! \brief The grammar to visit or mutate. */
  Grammar base_grammar_;

  /*!
   * \brief The builder to build the new grammar. It is empty when the mutator is constructed, and
   * can be used to build a new grammar in subclasses.
   */
  GrammarBuilder builder_;

  /*! \brief The name of the current rule being visited. */
  std::string cur_rule_name_;
};

/*!
 * \brief Visitor of Grammar.
 * \tparam ReturnType The return type of the Apply() function. Denotes the collected information.
 */
template <typename ReturnType>
using GrammarVisitor = GrammarFunctor<void, ReturnType>;

/*!
 * \brief Mutator of Grammar. The Apply() function returns the updated grammar.
 */
using GrammarMutator = GrammarFunctor<int32_t, Grammar>;

/*************************** Grammar manipulation methods ***************************/
/****** All below methods are implemented as functor to hide the implementation ******/

/*!
 * \brief Normalize a Grammar: expand the nested rules, combine consequent sequences and strings,
 * etc.
 */
class GrammarNormalizer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Find the union of multiple grammars as a new grammar.
 */
class GrammarUnionFunctor {
 public:
  static Grammar Apply(const std::vector<Grammar>& grammars);
};

/*!
 * \brief Find the concatenation of multiple grammars as a new grammar.
 */
class GrammarConcatFunctor {
 public:
  static Grammar Apply(const std::vector<Grammar>& grammars);
};

/*!
 * \brief Analyze the grammar to find the rules that are allowed to be empty.
 */
class AllowEmptyRuleAnalyzer {
 public:
  static std::vector<int32_t> Apply(const Grammar& grammar);
};

/*!
 * \brief Create a grammar that recognizes structural tags based on their triggers. See
 * StructuralTagToGrammar() for more details.
 *
 * \param triggers The trigger strings that identify each tag group
 * \param tag_groups The tags and their schema grammars, grouped by trigger. tag_groups[i][j] is the
 * j-th tag that matches triggers[i], and its corresponding schema grammar.
 * \return A grammar that matches all the tagged patterns.
 */
class StructuralTagGrammarCreator {
 public:
  static Grammar Apply(
      const std::vector<std::string>& triggers,
      const std::vector<std::vector<std::pair<StructuralTagItem, Grammar>>>& tag_groups
  );
};

/*!
 * \brief Normalize the structure of the grammar. It will ensure each rule is a choices of
 * sequences of elements, or a tag dispatch. The expanded context will be a sequence of elements.
 */
class StructureNormalizer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Fuse the byte string elements in the grammar.
 */
class ByteStringFuser {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Inline the rule references in the grammar.
 */
class RuleInliner {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Eliminate the not referenced rules in the grammar.
 */
class DeadCodeEliminator {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Analyze and add lookahead assertions in the grammar.
 */
class LookaheadAssertionAnalyzer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_FUNCTOR_H_
