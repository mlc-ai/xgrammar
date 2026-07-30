/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.cc
 */

#include <xgrammar/grammar.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "grammar_functor.h"
#include "grammar_parser.h"
#include "grammar_printer.h"
#include "json_schema_converter.h"
#include "lark_converter.h"
#include "regex_converter.h"
#include "structural_tag.h"
#include "support/json_serializer.h"
#include "support/logging.h"
#include "xgrammar/exception.h"

namespace xgrammar {

struct CompactRuleFSMView {
  int start = 0;
  std::vector<int32_t> ends;
  bool is_dfa = false;
  int edge_num = 0;
  int node_num = 0;

  CompactRuleFSMView() = default;

  explicit CompactRuleFSMView(const CompactFSMWithStartEndWithSize& value)
      : start(value.GetFsm().GetStart()),
        ends(value.GetFsm().GetEnds()),
        is_dfa(value.GetFsm().GetIsDFA()),
        edge_num(value.GetEdgeNum()),
        node_num(value.GetNodeNum()) {}
};

XGRAMMAR_MEMBER_ARRAY(
    CompactRuleFSMView,
    &CompactRuleFSMView::start,
    &CompactRuleFSMView::ends,
    &CompactRuleFSMView::is_dfa,
    &CompactRuleFSMView::edge_num,
    &CompactRuleFSMView::node_num
);

/******************* Grammar::Impl *******************/

std::size_t MemorySize(const Grammar::Impl& impl) {
  /// TODO: Now, we evaluate the memory size of each rule as sizeof(Rule), which counts its
  /// string members as sizeof(std::string), with an assumption that the strings are small.
  /// This should be improved in the future.
  return impl.rules_.size() * sizeof(Grammar::Impl::Rule) +
         impl.suffix_stop_infos_.size() * sizeof(Grammar::Impl::SuffixStopInfo) +
         MemorySize(impl.grammar_expr_data_) + MemorySize(impl.grammar_expr_indptr_) +
         MemorySize(impl.complete_fsm) + MemorySize(impl.per_rule_fsms) +
         MemorySize(impl.allow_empty_rule_ids);
}

picojson::value SerializeJSONValue(const Grammar::Impl& impl) {
  std::vector<std::optional<CompactRuleFSMView>> rule_fsm_views;
  rule_fsm_views.reserve(impl.per_rule_fsms.size());
  for (const auto& rule_fsm : impl.per_rule_fsms) {
    if (rule_fsm.has_value()) {
      rule_fsm_views.emplace_back(CompactRuleFSMView(*rule_fsm));
    } else {
      rule_fsm_views.emplace_back(std::nullopt);
    }
  }

  picojson::object result;
  result["rules"] = AutoSerializeJSONValue(impl.rules_);
  result["suffix_stop_infos"] = AutoSerializeJSONValue(impl.suffix_stop_infos_);
  // Preserve the historical public field labels. The internal vector names
  // are inverted relative to those labels.
  result["grammar_expr_data"] = AutoSerializeJSONValue(impl.grammar_expr_indptr_);
  result["grammar_expr_indptr"] = AutoSerializeJSONValue(impl.grammar_expr_data_);
  result["root_rule_id"] = AutoSerializeJSONValue(impl.root_rule_id_);
  result["complete_fsm"] = AutoSerializeJSONValue(impl.complete_fsm);
  result["per_rule_fsms"] = AutoSerializeJSONValue(rule_fsm_views);
  result["allow_empty_rule_ids"] = AutoSerializeJSONValue(impl.allow_empty_rule_ids);
  result["optimized"] = AutoSerializeJSONValue(impl.optimized);
  return picojson::value(std::move(result));
}

std::optional<SerializationError> DeserializeJSONValue(
    Grammar::Impl* impl, const picojson::value& value, const std::string& type_name
) {
  if (!value.is<picojson::object>()) {
    return ConstructDeserializeError("Expect an object", type_name);
  }
  const auto& object = value.get<picojson::object>();

  auto deserialize_field = [&](const char* name,
                               auto* destination) -> std::optional<SerializationError> {
    const auto it = object.find(name);
    if (it == object.end()) {
      return ConstructDeserializeError("Missing member " + std::string(name), type_name);
    }
    return AutoDeserializeJSONValue(destination, it->second, type_name);
  };

  if (auto error = deserialize_field("rules", &impl->rules_)) {
    return error;
  }
  if (auto error = deserialize_field("suffix_stop_infos", &impl->suffix_stop_infos_)) {
    return error;
  }
  if (auto error = deserialize_field("grammar_expr_data", &impl->grammar_expr_indptr_)) {
    return error;
  }
  if (auto error = deserialize_field("grammar_expr_indptr", &impl->grammar_expr_data_)) {
    return error;
  }
  if (auto error = deserialize_field("root_rule_id", &impl->root_rule_id_)) {
    return error;
  }
  if (auto error = deserialize_field("complete_fsm", &impl->complete_fsm)) {
    return error;
  }
  if (auto error = deserialize_field("allow_empty_rule_ids", &impl->allow_empty_rule_ids)) {
    return error;
  }
  if (auto error = deserialize_field("optimized", &impl->optimized)) {
    return error;
  }

  std::vector<std::optional<CompactRuleFSMView>> rule_fsm_views;
  if (auto error = deserialize_field("per_rule_fsms", &rule_fsm_views)) {
    return error;
  }
  const std::size_t expected_rule_fsm_count = impl->optimized ? impl->rules_.size() : 0;
  if (rule_fsm_views.size() != expected_rule_fsm_count) {
    return ConstructDeserializeError(
        "per_rule_fsms count does not match grammar optimization state", type_name
    );
  }
  if (!rule_fsm_views.empty() && impl->complete_fsm.IsNull()) {
    return ConstructDeserializeError("optimized grammar is missing the complete FSM", type_name);
  }

  const int num_states = rule_fsm_views.empty() ? 0 : impl->complete_fsm.NumStates();
  const std::size_t num_edges = rule_fsm_views.empty() ? 0 : impl->complete_fsm.GetNumEdges();
  impl->per_rule_fsms.clear();
  impl->per_rule_fsms.reserve(rule_fsm_views.size());
  for (auto& rule_fsm : rule_fsm_views) {
    if (!rule_fsm.has_value()) {
      impl->per_rule_fsms.emplace_back(std::nullopt);
      continue;
    }
    if (rule_fsm->start < 0 || rule_fsm->start >= num_states) {
      return ConstructDeserializeError("per-rule FSM start state is out of range", type_name);
    }
    if (rule_fsm->edge_num < 0 || rule_fsm->node_num < 0) {
      return ConstructDeserializeError("per-rule FSM size is negative", type_name);
    }
    if (static_cast<std::size_t>(rule_fsm->edge_num) > num_edges ||
        rule_fsm->node_num > num_states) {
      return ConstructDeserializeError("per-rule FSM size exceeds the complete FSM", type_name);
    }
    if (!std::is_sorted(rule_fsm->ends.begin(), rule_fsm->ends.end()) ||
        std::adjacent_find(rule_fsm->ends.begin(), rule_fsm->ends.end()) != rule_fsm->ends.end()) {
      return ConstructDeserializeError(
          "per-rule FSM end states are not sorted and unique", type_name
      );
    }
    for (const int32_t end : rule_fsm->ends) {
      if (end < 0 || end >= num_states) {
        return ConstructDeserializeError("per-rule FSM end state is out of range", type_name);
      }
    }

    CompactFSMWithStartEnd fsm(
        impl->complete_fsm, rule_fsm->start, std::move(rule_fsm->ends), rule_fsm->is_dfa
    );
    impl->per_rule_fsms.emplace_back(
        CompactFSMWithStartEndWithSize(std::move(fsm), rule_fsm->edge_num, rule_fsm->node_num)
    );
  }
  impl->per_rule_fsm_hashes.clear();
  impl->per_rule_fsm_new_state_ids.clear();

  return std::nullopt;
}

/******************* Grammar *******************/

std::string Grammar::ToString() const { return GrammarPrinter(*this).ToString(); }

Grammar Grammar::FromEBNF(const std::string& ebnf_string, const std::string& root_rule_name) {
  auto grammar = ParseEBNF(ebnf_string, root_rule_name);
  grammar = GrammarNormalizer().Apply(grammar);
  return grammar;
}

Grammar Grammar::FromJSONSchema(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    bool print_converted_ebnf,
    bool any_order
) {
  auto grammar = GrammarNormalizer::Apply(JSONSchemaToGrammar(
      schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, any_order
  ));
  if (print_converted_ebnf) {
    XGRAMMAR_LOG(INFO) << "Converted EBNF: " << grammar.ToString() << std::endl;
  }
  return grammar;
}

Grammar Grammar::FromRegex(const std::string& regex, bool print_converted_ebnf) {
  auto ebnf_string = RegexToEBNF(regex);
  if (print_converted_ebnf) {
    XGRAMMAR_LOG(INFO) << "Converted EBNF: " << ebnf_string << std::endl;
  }
  return FromEBNF(ebnf_string);
}

Grammar Grammar::FromLark(
    const std::string& lark_string,
    const std::optional<TokenizerInfo>& tokenizer_info,
    const std::vector<NamedGrammar>& named_grammars
) {
  return LarkToGrammar(lark_string, tokenizer_info, named_grammars);
}

std::variant<Grammar, StructuralTagError> Grammar::FromStructuralTag(
    const std::string& structural_tag_json, const std::optional<TokenizerInfo>& tokenizer_info
) {
  return StructuralTagToGrammar(structural_tag_json, tokenizer_info).ToVariant();
}

// Optimized json grammar for the speed of the grammar matcher
const std::string kJSONGrammarString = R"(
root ::= (
    "{" [ \n\t]* members_and_embrace |
    "[" [ \n\t]* elements_or_embrace
)
value_non_str ::= (
    "{" [ \n\t]* members_and_embrace |
    "[" [ \n\t]* elements_or_embrace |
    "0" fraction exponent |
    [1-9] [0-9]* fraction exponent |
    "-" [0-9] fraction exponent |
    "-" [1-9] [0-9]* fraction exponent |
    "true" |
    "false" |
    "null"
) (= [ \n\t]* member_suffix_suffix)
members_and_embrace ::= ("\"" characters_and_colon [ \n\t]* members_suffix | "}") (= [ \n\t,}\]])
members_suffix ::= (
    value_non_str [ \n\t]* member_suffix_suffix |
    "\"" characters_and_embrace |
    "\"" characters_and_comma [ \n\t]* "\"" characters_and_colon [ \n\t]* members_suffix
) (= [ \n\t,}\]])
member_suffix_suffix ::= (
    "}" |
    "," [ \n\t]* "\"" characters_and_colon [ \n\t]* members_suffix
) (= [ \n\t,}\]])
elements_or_embrace ::= (
    "{" [ \n\t]* members_and_embrace elements_rest [ \n\t]* "]" |
    "[" [ \n\t]* elements_or_embrace elements_rest [ \n\t]* "]" |
    "\"" characters_item elements_rest [ \n\t]* "]" |
    "0" fraction exponent elements_rest [ \n\t]* "]" |
    [1-9] [0-9]* fraction exponent elements_rest [ \n\t]* "]" |
    "-" "0" fraction exponent elements_rest [ \n\t]* "]" |
    "-" [1-9] [0-9]* fraction exponent elements_rest [ \n\t]* "]" |
    "true" elements_rest [ \n\t]* "]" |
    "false" elements_rest [ \n\t]* "]" |
    "null" elements_rest [ \n\t]* "]" |
    "]"
)
elements ::= (
    "{" [ \n\t]* members_and_embrace elements_rest |
    "[" [ \n\t]* elements_or_embrace elements_rest |
    "\"" characters_item elements_rest |
    "0" fraction exponent elements_rest |
    [1-9] [0-9]* fraction exponent elements_rest |
    "-" [0-9] fraction exponent elements_rest |
    "-" [1-9] [0-9]* fraction exponent elements_rest |
    "true" elements_rest |
    "false" elements_rest |
    "null" elements_rest
)
elements_rest ::= (
    "" |
    [ \n\t]* "," [ \n\t]* elements
)
characters_and_colon ::= (
    "\"" [ \n\t]* ":" |
    [^"\\\x00-\x1F] characters_and_colon |
    "\\" escape characters_and_colon
) (=[ \n\t]* [\"{[0-9tfn-])
characters_and_comma ::= (
    "\"" [ \n\t]* "," |
    [^"\\\x00-\x1F] characters_and_comma |
    "\\" escape characters_and_comma
) (=[ \n\t]* "\"")
characters_and_embrace ::= (
    "\"" [ \n\t]* "}" |
    [^"\\\x00-\x1F] characters_and_embrace |
    "\\" escape characters_and_embrace
) (=[ \n\t]* [},])
characters_item ::= (
    "\"" |
    [^"\\\x00-\x1F] characters_item |
    "\\" escape characters_item
) (= [ \n\t]* [,\]])
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
fraction ::= "" | "." [0-9] [0-9]*
exponent ::= "" |  "e" sign [0-9] [0-9]* | "E" sign [0-9] [0-9]*
sign ::= "" | "+" | "-"
)";

Grammar Grammar::BuiltinJSONGrammar() {
  static const Grammar grammar = FromEBNF(kJSONGrammarString);
  return grammar;
}

Grammar Grammar::Union(const std::vector<Grammar>& grammars) {
  return GrammarUnionFunctor::Apply(grammars);
}

Grammar Grammar::Concat(const std::vector<Grammar>& grammars) {
  return GrammarConcatFunctor::Apply(grammars);
}

std::ostream& operator<<(std::ostream& os, const Grammar& grammar) {
  os << grammar.ToString();
  return os;
}

std::string Grammar::SerializeJSON() const { return AutoSerializeJSON(*this, true); }

std::variant<Grammar, SerializationError> Grammar::DeserializeJSON(const std::string& json_string) {
  Grammar result{NullObj()};
  if (auto err = AutoDeserializeJSON(&result, json_string, true, "Grammar")) {
    return err.value();
  }
  return result;
}

}  // namespace xgrammar
