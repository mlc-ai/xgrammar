/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.cc
 */

#include <xgrammar/grammar.h>

#include <string>

#include "grammar_functor.h"
#include "grammar_parser.h"
#include "grammar_printer.h"
#include "json_schema_converter.h"
#include "regex_converter.h"
#include "structural_tag.h"
#include "support/json_serializer.h"
#include "support/logging.h"
#include "xgrammar/exception.h"

namespace xgrammar {

/******************* Grammar::Impl *******************/

std::size_t MemorySize(const Grammar::Impl& impl) {
  /// TODO: Now, we evaluatve memory size of rule strings as sizeof(std::string),
  /// with an assumption that the string is small.
  /// This should be improved in the future.
  return impl.rules_.size() * sizeof(std::string) + MemorySize(impl.grammar_expr_data_) +
         MemorySize(impl.grammar_expr_indptr_) + MemorySize(impl.complete_fsm) +
         MemorySize(impl.per_rule_fsms) + MemorySize(impl.allow_empty_rule_ids);
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
    bool print_converted_ebnf
) {
  auto ebnf_string =
      JSONSchemaToEBNF(schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt);
  if (print_converted_ebnf) {
    XGRAMMAR_LOG(INFO) << "Converted EBNF: " << ebnf_string << std::endl;
  }
  return FromEBNF(ebnf_string);
}

Grammar Grammar::FromRegex(const std::string& regex, bool print_converted_ebnf) {
  auto ebnf_string = RegexToEBNF(regex);
  if (print_converted_ebnf) {
    XGRAMMAR_LOG(INFO) << "Converted EBNF: " << ebnf_string << std::endl;
  }
  return FromEBNF(ebnf_string);
}

std::variant<Grammar, StructuralTagError> Grammar::FromStructuralTag(
    const std::string& structural_tag_json
) {
  return StructuralTagToGrammar(structural_tag_json).ToVariant();
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
