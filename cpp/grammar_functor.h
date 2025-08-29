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
#include "grammar_impl.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

/*!
 * \brief Base class for visitors and mutators of the BNF grammar.
 * \tparam T The type of the return value of visitor functions. Typical values:
 * - int32_t: the id of the new grammar_expr
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
    InitGrammar(grammar);
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
      InitBuilder();
      // First add empty rules to ensure the new rule ids the same as the old ones, then update
      // the rule bodies
      for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
        builder_->AddEmptyRule(base_grammar_->GetRule(i).name);
      }
      for (int i = 0; i < static_cast<int>(base_grammar_->NumRules()); ++i) {
        auto rule = base_grammar_->GetRule(i);
        cur_rule_name_ = rule.name;
        auto new_body_expr_id = VisitExpr(rule.body_expr_id);
        builder_->UpdateRuleBody(i, new_body_expr_id);
        // Handle lookahead assertion
        builder_->UpdateLookaheadAssertion(i, VisitLookaheadAssertion(rule.lookahead_assertion_id));
      }
      return builder_->Get(base_grammar_->GetRootRule().name);
    } else {
      return ReturnType();
    }
  }

  /*! \brief Virtual destructor. */
  virtual ~GrammarFunctor() = default;

 protected:
  using Rule = Grammar::Impl::Rule;
  using GrammarExpr = Grammar::Impl::GrammarExpr;
  using GrammarExprType = Grammar::Impl::GrammarExprType;

  /*! \brief Initialize the functor. Should be called at the beginning of Apply(). */
  virtual void InitGrammar() {}

  virtual void InitGrammar(const Grammar& grammar) { base_grammar_ = grammar; }

  virtual void InitBuilder() {
    owned_builder_ = GrammarBuilder();
    builder_ = &owned_builder_;
  }

  virtual void InitBuilder(const Grammar& grammar) {
    owned_builder_ = GrammarBuilder(grammar);
    builder_ = &owned_builder_;
  }

  virtual void InitBuilder(GrammarBuilder* builder) { builder_ = builder; }

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

  /*! \brief Visit a GrammarExpr by id. */
  virtual T VisitExpr(int32_t old_grammar_expr_id) {
    return VisitExpr(base_grammar_->GetGrammarExpr(old_grammar_expr_id));
  }

  /*! \brief Visit a GrammarExpr. Dispatch to the corresponding Visit function. */
  virtual T VisitExpr(const GrammarExpr& grammar_expr) {
    switch (grammar_expr.type) {
      case GrammarExprType::kSequence:
        return VisitSequence(grammar_expr);
      case GrammarExprType::kChoices:
        return VisitChoices(grammar_expr);
      case GrammarExprType::kEmptyStr:
        return VisitEmptyStr(grammar_expr);
      case GrammarExprType::kByteString:
        return VisitByteString(grammar_expr);
      case GrammarExprType::kCharacterClass:
        return VisitCharacterClass(grammar_expr);
      case GrammarExprType::kCharacterClassStar:
        return VisitCharacterClassStar(grammar_expr);
      case GrammarExprType::kRuleRef:
        return VisitRuleRef(grammar_expr);
      case GrammarExprType::kTagDispatch:
        return VisitTagDispatch(grammar_expr);
      case GrammarExprType::kRepeat:
        return VisitRepeat(grammar_expr);
      default:
        XGRAMMAR_LOG(FATAL) << "Unexpected sequence type: " << static_cast<int>(grammar_expr.type);
        XGRAMMAR_UNREACHABLE();
    }
  }

  /*! \brief Visit a choices GrammarExpr. */
  virtual T VisitChoices(const GrammarExpr& grammar_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : grammar_expr) {
        VisitExpr(i);
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<int32_t> choice_ids;
      for (int32_t i : grammar_expr) {
        choice_ids.push_back(VisitExpr(i));
      }
      return builder_->AddChoices(choice_ids);
    } else {
      return T();
    }
  }

  /*! \brief Visit a sequence GrammarExpr. */
  virtual T VisitSequence(const GrammarExpr& grammar_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : grammar_expr) {
        VisitExpr(i);
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<T> sequence_ids;
      for (int32_t i : grammar_expr) {
        sequence_ids.push_back(VisitExpr(i));
      }
      return builder_->AddSequence(sequence_ids);
    } else {
      return T();
    }
  }

  virtual T VisitTagDispatch(const GrammarExpr& grammar_expr) {
    if constexpr (std::is_same<T, void>::value) {
      return;
    } else if constexpr (std::is_same<T, int32_t>::value) {
      Grammar::Impl::TagDispatch tag_dispatch = base_grammar_->GetTagDispatch(grammar_expr);
      return builder_->AddTagDispatch(tag_dispatch);
    } else {
      return T();
    }
  }

  /*! \brief Visit an element GrammarExpr, including empty string, character class, and rule ref. */
  virtual T VisitElement(const GrammarExpr& grammar_expr) {
    if constexpr (std::is_same<T, void>::value) {
      return;
    } else if constexpr (std::is_same<T, int32_t>::value) {
      return builder_->AddGrammarExpr(grammar_expr);
    } else {
      return T();
    }
  }

  /*! \brief Visit an empty string GrammarExpr. */
  virtual T VisitEmptyStr(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  /*! \brief Visit a character class GrammarExpr. */
  virtual T VisitByteString(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  /*! \brief Visit a character class GrammarExpr. */
  virtual T VisitCharacterClass(const GrammarExpr& grammar_expr) {
    return VisitElement(grammar_expr);
  }

  /*! \brief Visit a star quantifier GrammarExpr. */
  virtual T VisitCharacterClassStar(const GrammarExpr& grammar_expr) {
    return VisitElement(grammar_expr);
  }

  /*! \brief Visit a rule reference GrammarExpr. */
  virtual T VisitRuleRef(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  /*! \brief Visit a repeat GrammarExpr. */
  virtual T VisitRepeat(const GrammarExpr& grammar_expr) { return VisitElement(grammar_expr); }

  /*! \brief The grammar to visit or mutate. */
  Grammar base_grammar_{NullObj{}};

  /*!
   * \brief The builder to build the new grammar. It is empty when the mutator is constructed, and
   * can be used to build a new grammar in subclasses.
   */
  GrammarBuilder* builder_ = nullptr;

  GrammarBuilder owned_builder_;

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

/*!
 * \brief Build the FSMs of the grammar.
 */
class GrammarFSMBuilder {
  using GrammarExpr = Grammar::Impl::GrammarExpr;

 public:
  static void Apply(Grammar* grammar);
  static FSMWithStartEnd RuleRef(const GrammarExpr& expr);
  static FSMWithStartEnd CharacterClass(const GrammarExpr& expr);
  static FSMWithStartEnd ByteString(const GrammarExpr& expr);
  static std::optional<FSMWithStartEnd> Sequence(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> Choices(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> TagDispatch(const Grammar::Impl::TagDispatch& tag_dispatch);
};
class SubGrammarAdder {
 public:
  static int32_t Apply(GrammarBuilder* builder, const Grammar& sub_grammar);
};

class RepetitionNormalizer {
 public:
  static void Apply(Grammar* grammar);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_FUNCTOR_H_
