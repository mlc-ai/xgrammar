/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_functor.cc
 */

#include "grammar_functor.h"

#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <vector>

#include "grammar_builder.h"
#include "grammar_impl.h"
#include "support/logging.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

using GrammarExpr = Grammar::Impl::GrammarExpr;
using ExprType = Grammar::Impl::GrammarExprType;

/*************************** Impl of grammar functors ***************************/

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

class StructuralTagGrammarCreatorImpl : public GrammarMutator {
 public:
  Grammar Apply(
      const std::vector<std::string>& triggers,
      const std::vector<std::vector<std::pair<StructuralTagItem, Grammar>>>& tag_groups
  ) {
    XGRAMMAR_CHECK(triggers.size() == tag_groups.size())
        << "Number of triggers must match number of tag groups";

    InitGrammar();
    InitBuilder();

    auto root_rule_id = builder_->AddEmptyRule("root");

    Grammar::Impl::TagDispatch tag_dispatch{
        /* tag_rule_pairs = */ {},
        /* stop_eos = */ true,
        /* stop_str = */ {},
        /* loop_after_dispatch = */ true,
    };
    tag_dispatch.tag_rule_pairs.reserve(triggers.size());

    // Create rules for each trigger group
    for (size_t i = 0; i < triggers.size(); i++) {
      // Skip empty trigger groups
      if (tag_groups[i].empty()) {
        continue;
      }

      auto rule_name = "trigger_rule_" + std::to_string(i);
      auto rule_id = builder_->AddEmptyRule(rule_name);

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
          seq_elements.push_back(builder_->AddByteString(tag.begin.substr(triggers[i].size())));
        }

        // Create and visit schema grammar for this tag
        auto schema_rule_id = SubGrammarAdderImpl().ApplyWithBuilder(builder_, schema_grammar);
        seq_elements.push_back(builder_->AddRuleRef(schema_rule_id));

        // Add end string
        if (!tag.end.empty()) {
          seq_elements.push_back(builder_->AddByteString(tag.end));
        }

        choices.push_back(builder_->AddSequence(seq_elements));
      }

      builder_->UpdateRuleBody(rule_id, builder_->AddChoices(choices));
      tag_dispatch.tag_rule_pairs.emplace_back(triggers[i], rule_id);
    }

    // Create root TagDispatch rule
    auto tag_dispatch_id = builder_->AddTagDispatch(tag_dispatch);
    builder_->UpdateRuleBody(root_rule_id, tag_dispatch_id);
    return builder_->Get(root_rule_id);
  }

  // Avoid hiding the original Apply(const Grammar&)
  Grammar Apply(const Grammar& grammar) final {
    XGRAMMAR_LOG(FATAL) << "Should not be called";
    XGRAMMAR_UNREACHABLE();
  }
};

/*************************** Forward grammar functors to their impl ***************************/

Grammar GrammarUnionFunctor::Apply(const std::vector<Grammar>& grammars) {
  return GrammarUnionFunctorImpl().Apply(grammars);
}

Grammar GrammarConcatFunctor::Apply(const std::vector<Grammar>& grammars) {
  return GrammarConcatFunctorImpl().Apply(grammars);
}

Grammar StructuralTagGrammarCreator::Apply(
    const std::vector<std::string>& triggers,
    const std::vector<std::vector<std::pair<StructuralTagItem, Grammar>>>& tag_groups
) {
  return StructuralTagGrammarCreatorImpl().Apply(triggers, tag_groups);
}

int32_t SubGrammarAdder::Apply(GrammarBuilder* builder, const Grammar& sub_grammar) {
  return SubGrammarAdderImpl().ApplyWithBuilder(builder, sub_grammar);
}

}  // namespace xgrammar
