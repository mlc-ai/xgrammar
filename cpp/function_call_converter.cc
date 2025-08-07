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

/*************************** Grammar definitions ***************************/

const std::string kWhiteSpacesString = R"( [ \n\t]* )";

const std::string kNumberGrammarString = R"(
number ::= sign "0" fraction exponent [ \n\t]* "</parameter>" |
    sign [1-9] [0-9]* fraction exponent [ \n\t]* "</parameter>"
fraction ::= "" | "." [0-9] [0-9]*
exponent ::= "" |  "e" sign [0-9] [0-9]* | "E" sign [0-9] [0-9]*
sign ::= "" | "+" | "-"
)";

const std::string kXmlStringGrammarString = R"(
string ::= xml_content [ \n\t]* "</parameter>"
xml_content ::= ([^<>&\\\x00-\x1F] | xml_entity | [\\] escape)*
xml_entity ::= "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
)";

const std::string kBooleanGrammarString = R"(
boolean ::= "true" [ \n\t]* "</parameter>" | "false" [ \n\t]* "</parameter>"
)";

const std::string kXmlObjectGrammarString = R"(
object ::=
    "{" [ \n\t]* members_and_embrace [ \n\t]* "</parameter>" | "null" [ \n\t]* "</parameter>"
xml_entity ::= "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
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
    [^<>&\\\x00-\x1F] characters_and_colon |
    xml_entity characters_and_colon |
    "\\" escape characters_and_colon
) (=[ \n\t]* [\"{[0-9tfn-])
characters_and_comma ::= (
    "\"" [ \n\t]* "," |
    [^<>&\\\x00-\x1F] characters_and_comma |
    xml_entity characters_and_comma |
    "\\" escape characters_and_comma
) (=[ \n\t]* "\"")
characters_and_embrace ::= (
    "\"" [ \n\t]* "}" |
    [^<>&\\\x00-\x1F] characters_and_embrace |
    xml_entity characters_and_embrace |
    "\\" escape characters_and_embrace
) (=[ \n\t]* [},])
characters_item ::= (
    "\"" |
    [^<>&\\\x00-\x1F] characters_item |
    xml_entity characters_item |
    "\\" escape characters_item
) (= [ \n\t]* [,\]])
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
fraction ::= "" | "." [0-9] [0-9]*
exponent ::= "" |  "e" sign [0-9] [0-9]* | "E" sign [0-9] [0-9]*
sign ::= "" | "+" | "-"
)";

/*************************** FunctionCallConverterImpl ***************************/

class FunctionCallConverterImpl {
 public:
  enum class kParametersType : int32_t { kNumber = 0, kString = 1, kBoolean = 2, kObject = 3 };

  static Grammar Apply(
      const std::vector<std::string>& args_names,
      const std::vector<std::string>& args_types,
      uint8_t function_type
  );

  static Grammar BuildXmlParameterGrammar(
      const std::vector<std::string>& args_names, const std::vector<std::string>& args_types
  );

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

  // Build the grammar
  switch (function_type) {
    case Grammar::kXmlStyleFunctionCall: {
      return BuildXmlParameterGrammar(args_names, args_types);
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Can't build a function calling grammar for this style!";
    }
  }
}

using kParametersType = FunctionCallConverterImpl::kParametersType;

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

static const Grammar kXmlStringGrammar = Grammar::FromEBNF(kXmlStringGrammarString, "string");

static const Grammar kNumberGrammar = Grammar::FromEBNF(kNumberGrammarString, "number");

static const Grammar kBooleanGrammar = Grammar::FromEBNF(kBooleanGrammarString, "boolean");

static const Grammar kXmlObjectGrammar = Grammar::FromEBNF(kXmlObjectGrammarString, "object");

Grammar FunctionCallConverterImpl::BuildXmlParameterGrammar(
    const std::vector<std::string>& arg_names, const std::vector<std::string>& arg_types
) {
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
}  // namespace xgrammar
