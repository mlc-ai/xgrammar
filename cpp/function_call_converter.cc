/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/function_call_converter.cc
 * \brief The implementation for converting function calls to Grammars.
 */

#include "function_call_converter.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "support/logging.h"
#include "support/utils.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

class FunctionCallConverterImpl {
 public:
  enum class kParametersType { kNumber = 0, kString = 1, kBoolean = 2, kObject = 3 };

  static Grammar Apply(
      const std::vector<std::string>& args_names,
      const std::vector<std::string>& args_types,
      uint8_t function_type
  );

  static Grammar BuildXmlParameterGrammar(const std::string& arg_name, const std::string& arg_type);

  static Grammar DecorateXmlParameterGrammar(std::vector<Grammar>& grammar);
};

Grammar FunctionCallConverterImpl::Apply(
    const std::vector<std::string>& args_names,
    const std::vector<std::string>& args_types,
    uint8_t function_type
) {
  // Handle the corner cases.
  XGRAMMAR_CHECK(args_types.size() == args_types.size());
  if (args_names.empty()) {
    return Grammar::FromEBNF("root ::= \"\"");
  }

  // Build the single grammars.
  std::vector<Grammar> result_grammar;
  for (int i = 0; i < static_cast<int>(args_names.size()); i++) {
    switch (function_type) {
      case Grammar::kXmlStyleFunctionCall: {
        result_grammar.push_back(BuildXmlParameterGrammar(args_names[i], args_types[i]));
        break;
      }
      default: {
        XGRAMMAR_LOG(FATAL) << "Can't build a function calling grammar for this style!";
      }
    }
  }

  // concat and decorate the grammars.
  switch (function_type) {
    case Grammar::kXmlStyleFunctionCall: {
      return DecorateXmlParameterGrammar(result_grammar);
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Can't build a function calling grammar for this style!";
    }
  }
}

Grammar FunctionCallConverterImpl::BuildXmlParameterGrammar(
    const std::string& arg_name, const std::string& arg_type
) {
  static const std::unordered_map<std::string, kParametersType> raw_string_to_types = {
      {"string", kParametersType::kString},
      {"str", kParametersType::kString},
      {"char", kParametersType::kString},
      {"enum", kParametersType::kString},
      {"text", kParametersType::kString},
      {"varchar", kParametersType::kString},
      {"int", kParametersType::kNumber},
      {"uint", kParametersType::kNumber},
      {"long", kParametersType::kNumber},
      {"short", kParametersType::kNumber},
      {"unsign", kParametersType::kNumber},
      {"float", kParametersType::kNumber},
      {"num", kParametersType::kNumber},
      {"boolean", kParametersType::kBoolean},
      {"bool", kParametersType::kBoolean},
      {"binary", kParametersType::kBoolean},
      {"object", kParametersType::kObject},
      {"dict", kParametersType::kObject}
  };

  // Decode the arg type.
  kParametersType decoded_arg_type = kParametersType::kString;
  if (raw_string_to_types.find(arg_type) != raw_string_to_types.end()) {
    decoded_arg_type = raw_string_to_types.at(arg_type);
  } else {
    // Check if the arg_type is started with some types. Otherwise, use strings.
    for (auto& [raw_string, type] : raw_string_to_types) {
      if (arg_type.size() >= raw_string.size() &&
          arg_type.substr(0, raw_string.size()) == raw_string) {
        decoded_arg_type = type;
        break;
      }
    }
  }
  XGRAMMAR_UNREACHABLE();
}

/*************************** Forward grammar functors to their impl ***************************/

Grammar FunctionCallConverter::Apply(
    const std::vector<std::string>& args_names,
    const std::vector<std::string>& args_types,
    uint8_t function_type
) {
  return FunctionCallConverterImpl::Apply(args_names, args_types, function_type);
}

/*************************** Grammar definitions ***************************/

const std::string kNumberGrammarString = R"(
root ::= sign "0" fraction exponent | sign [1-9] [0-9]* fraction exponent
fraction ::= "" | "." [0-9] [0-9]*
exponent ::= "" |  "e" sign [0-9] [0-9]* | "E" sign [0-9] [0-9]*
sign ::= "" | "+" | "-"
)";

const std::string kStringGrammarString = R"(
root ::= [^\\\x00-\x1F] | [\\] escape
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
)";

const std::string kBooleanGrammarString = R"(
root ::= "true" | "false"
)";

const std::string kObjectGrammarString = R"(
root ::= (
    "{" [ \n\t]* members_and_embrace | "null"
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

}  // namespace xgrammar
