/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/grammar_constructor.cc
 * \brief The implementation for building the BNF AST.
 */
#include "grammar_constructor.h"

#include <xgrammar/grammar.h>

#include <cstdint>
#include <string>

#include "grammar_functor.h"
#include "support/utils.h"

namespace xgrammar {

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
      auto new_root_id_for_grammar = SubGrammarAdder().Apply(builder_, grammar);
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
      auto new_root_id_for_grammar = SubGrammarAdder().Apply(builder_, grammar);
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
        auto schema_rule_id = SubGrammarAdder().Apply(builder_, schema_grammar);
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

class TagDispatchGrammarCreatorImpl : public GrammarMutator {
 public:
  Grammar Apply(
      const std::vector<std::string>& triggers,
      const std::vector<Grammar>& tags,
      bool stop_eos,
      bool loop_after_dispatch,
      std::vector<std::string> stop_strs
  ) {
    InitGrammar();
    InitBuilder();

    auto root_rule_id = builder_->AddEmptyRule("root");

    Grammar::Impl::TagDispatch tag_dispatch{
        /* tag_rule_pairs = */ {},
        /* stop_eos = */ stop_eos,
        /* stop_str = */ stop_strs,
        /* loop_after_dispatch = */ loop_after_dispatch,
    };
    tag_dispatch.tag_rule_pairs.reserve(triggers.size());

    // Create rules for each trigger group
    for (size_t i = 0; i < triggers.size(); i++) {
      auto rule_name = "trigger_rule_" + std::to_string(i);
      auto rule_id = builder_->AddEmptyRule(rule_name);

      // Create choices for each tag in this trigger group
      std::vector<int32_t> choices;
      std::vector<int32_t> seq_elements;
      seq_elements.reserve(1);

      // Create and visit schema grammar for this tag
      auto new_rule_id = SubGrammarAdder().Apply(builder_, tags[i]);
      seq_elements.push_back(builder_->AddRuleRef(new_rule_id));

      choices.push_back(builder_->AddSequence(seq_elements));

      builder_->UpdateRuleBody(rule_id, builder_->AddChoices(choices));
      tag_dispatch.tag_rule_pairs.emplace_back(triggers[i], rule_id);
    }

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

class StarGrammarCreatorImpl : public GrammarMutator {
 public:
  Grammar Apply(const Grammar& grammar) {
    // Initialize the grammar and builder.
    InitGrammar();
    InitBuilder();

    // Add a new empty rule for the root.
    auto root_rule_id = builder_->AddEmptyRule("root");

    // Add the original grammar as a subgrammar.
    auto original_root_rule_id = SubGrammarAdder().Apply(builder_, grammar);

    // Get a rule reference for root_original.
    auto original_root_rule_ref = builder_->AddRuleRef(original_root_rule_id);

    // Get a rule reference for the new root rule.
    auto root_rule_ref = builder_->AddRuleRef(root_rule_id);

    // We get root_original root.
    auto new_root_seq = builder_->AddSequence({original_root_rule_ref, root_rule_ref});

    // root ::= "" | root_original root
    auto new_root_choice = builder_->AddChoices({builder_->AddEmptyStr(), new_root_seq});
    builder_->UpdateRuleBody(root_rule_id, new_root_choice);
    return builder_->Get(root_rule_id);
  }
};

/**************************************** Grammar Functions ***************************************/

Grammar Grammar::Empty() { return Grammar::FromEBNF("root ::= \"\""); }

Grammar Grammar::String(const std::string& str) {
  static const std::unordered_map<char, std::string> kCodepointToEscape = {
      {'\'', "\\\'"},
      {'\"', "\\\""},
      {'\?', "\\?"},
      {'\\', "\\\\"},
      {'\a', "\\a"},
      {'\b', "\\b"},
      {'\f', "\\f"},
      {'\n', "\\n"},
      {'\r', "\\r"},
      {'\t', "\\t"},
      {'\v', "\\v"},
      {'\0', "\\0"},
      {'\x1B', "\\e"}
  };
  std::string ebnf_string = "root ::= \"";
  for (auto ch : str) {
    if (kCodepointToEscape.find(ch) != kCodepointToEscape.end()) {
      ebnf_string += kCodepointToEscape.at(ch);
    } else {
      ebnf_string += ch;
    }
  }
  ebnf_string += "\"";
  return Grammar::FromEBNF(ebnf_string);
}

Grammar Grammar::CharacterClass(const std::string& str) { return Grammar::FromRegex(str); }

Grammar Grammar::TagDispatch(
    const std::vector<std::string>& triggers,
    const std::vector<Grammar>& tags,
    bool stop_eos,
    bool loop_after_dispatch,
    const std::vector<std::string>& stop_strs
) {
  return TagDispatchGrammarCreator::Apply(triggers, tags, stop_eos, loop_after_dispatch, stop_strs);
}

Grammar Grammar::Union(const std::vector<Grammar>& grammars) {
  return GrammarUnionFunctor::Apply(grammars);
}

Grammar Grammar::Concat(const std::vector<Grammar>& grammars) {
  return GrammarConcatFunctor::Apply(grammars);
}

Grammar Grammar::Star(const Grammar& grammar) { return StarGrammarCreator::Apply(grammar); }

Grammar Grammar::Plus(const Grammar& grammar) {
  return Grammar::Concat({grammar, Grammar::Star(grammar)});
}

Grammar Grammar::Optional(const Grammar& grammar) {
  return Grammar::Union({grammar, Grammar::Empty()});
}

/*************************** Forward grammar Constructors to their impl ***************************/

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

Grammar TagDispatchGrammarCreator::Apply(
    const std::vector<std::string>& triggers,
    const std::vector<Grammar>& tags,
    bool stop_eos,
    bool loop_after_dispatch,
    const std::vector<std::string>& stop_strs
) {
  return TagDispatchGrammarCreatorImpl().Apply(
      triggers, tags, stop_eos, loop_after_dispatch, stop_strs
  );
}

Grammar StarGrammarCreator::Apply(const Grammar& grammar) {
  return StarGrammarCreatorImpl().Apply(grammar);
}

}  // namespace xgrammar
