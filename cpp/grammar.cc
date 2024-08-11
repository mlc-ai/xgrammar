/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.cc
 */

#include <xgrammar/xgrammar.h>

#include "grammar_ast.h"
#include "grammar_functor.h"
#include "grammar_parser.h"
#include "grammar_serializer.h"

namespace xgrammar {

inline BNFGrammar BNFGrammarCtor(const std::string& ebnf_string, const std::string& main_rule) {
  auto grammar = EBNFParser::Parse(ebnf_string, main_rule);
  grammar = BNFGrammarNormalizer().Apply(grammar);
  return grammar;
}

BNFGrammar::BNFGrammar(const std::string& ebnf_string, const std::string& main_rule)
    : BNFGrammar(BNFGrammarCtor(ebnf_string, main_rule)) {}

std::string BNFGrammar::ToString() const { return BNFGrammarPrinter(*this).ToString(); }

std::ostream& operator<<(std::ostream& os, const BNFGrammar& grammar) {
  os << grammar.ToString();
  return os;
}

std::string BNFGrammar::Serialize(bool prettify) const {
  return BNFGrammarJSONSerializer(*this).ToString();
}

BNFGrammar BNFGrammar::Deserialize(const std::string& json_string) {
  return BNFJSONParser::Parse(json_string);
}

// Optimized json grammar for the speed of the grammar state matcher
const std::string kJSONGrammarString = R"(
main ::= (
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
) (= [ \n\t,}\]])
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

BNFGrammar BuiltinGrammar::JSON() {
  static const BNFGrammar grammar(kJSONGrammarString);
  return grammar;
}

BNFGrammar BuiltinGrammar::JSONSchema(
    const std::string& schema,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode
) {
  return BNFGrammar(_JSONSchemaToEBNF(schema, indent, separators, strict_mode));
}

// TVM_REGISTER_GLOBAL("mlc.grammar.BNFGrammarFromEBNFString")
//     .set_body_typed([](String ebnf_string, String main_rule) {
//       return BNFGrammar::FromEBNFString(ebnf_string, main_rule);
//     });

// TVM_REGISTER_GLOBAL("mlc.grammar.BNFGrammarDebugFromEBNFStringNoNormalize")
//     .set_body_typed([](String ebnf_string, String main_rule) {
//       return DebugFromEBNFStringNoNormalize(ebnf_string, main_rule);
//     });

// TVM_REGISTER_GLOBAL("mlc.grammar.BNFGrammarFromSchema").set_body([](TVMArgs args, TVMRetValue*
// rv) {
//   std::optional<int> indent;
//   if (args[1].type_code() != kTVMNullptr) {
//     indent = args[1];
//   } else {
//     indent = std::nullopt;
//   }

//   std::optional<std::pair<std::string, std::string>> separators;
//   if (args[2].type_code() != kTVMNullptr) {
//     Array<String> separators_arr = args[2];
//     XGRAMMAR_CHECK(separators_arr.size() == 2);
//     separators = std::make_pair(separators_arr[0], separators_arr[1]);
//   } else {
//     separators = std::nullopt;
//   }

//   *rv = BNFGrammar::FromSchema(args[0], indent, separators, args[3]);
// });

// TVM_REGISTER_GLOBAL("mlc.grammar.BNFGrammarGetGrammarOfJSON").set_body_typed([]() {
//   return BNFGrammar::GetGrammarOfJSON();
// });

}  // namespace xgrammar
