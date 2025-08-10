/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/function_call_converter.cc
 * \brief The implementation for converting function calls to Grammars.
 */

#include "function_call_converter.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "grammar_builder.h"
#include "grammar_functor.h"
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

class FunctionCallConverterImpl : public GrammarMutator {
 public:
  enum class kParametersType : int32_t { kNumber = 0, kString = 1, kBoolean = 2, kObject = 3 };

  Grammar Apply(
      const std::vector<std::string>& args_names, const std::vector<std::string>& args_types
  );

  Grammar BuildXmlParameterGrammar(
      const std::vector<std::string>& args_names, const std::vector<std::string>& args_types
  );

  // Avoid hiding the original Apply(const Grammar&)
  Grammar Apply(const Grammar& grammar) final {
    XGRAMMAR_LOG(FATAL) << "Should not be called";
    XGRAMMAR_UNREACHABLE();
  }
};

Grammar FunctionCallConverterImpl::Apply(
    const std::vector<std::string>& args_names, const std::vector<std::string>& args_types
) {
  // Handle the corner cases.
  XGRAMMAR_CHECK(args_types.size() == args_types.size());
  if (args_names.empty()) {
    return Grammar::FromEBNF("root ::= \"\"");
  }

  return BuildXmlParameterGrammar(args_names, args_types);
}

using kParametersType = FunctionCallConverterImpl::kParametersType;

Grammar FunctionCallConverterImpl::BuildXmlParameterGrammar(
    const std::vector<std::string>& arg_names, const std::vector<std::string>& arg_types
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

  static const Grammar kXmlStringGrammar = Grammar::FromEBNF(kXmlStringGrammarString, "string");

  static const Grammar kNumberGrammar = Grammar::FromEBNF(kNumberGrammarString, "number");

  static const Grammar kBooleanGrammar = Grammar::FromEBNF(kBooleanGrammarString, "boolean");

  static const Grammar kXmlObjectGrammar = Grammar::FromEBNF(kXmlObjectGrammarString, "object");
  // Initialize the grammar builder.
  InitGrammar();
  InitBuilder();

  // Add the root rule and the xml grammars.
  auto root_rule_id = builder_->AddEmptyRule("root");
  auto object_rule_id = -1;
  auto string_rule_id = -1;
  auto number_rule_id = -1;
  auto boolean_rule_id = -1;

  std::vector<int32_t> parameters_reference_sequence;
  parameters_reference_sequence.reserve(arg_names.size());
  std::vector<GrammarBuilder::CharacterClassElement> whitespace_class;
  GrammarBuilder::CharacterClassElement whitespace_element;
  whitespace_element.lower = ' ';
  whitespace_element.upper = ' ';
  whitespace_class.push_back(whitespace_element);
  whitespace_element.lower = '\n';
  whitespace_element.upper = '\n';
  whitespace_class.push_back(whitespace_element);
  whitespace_element.lower = '\t';
  whitespace_element.upper = '\t';
  whitespace_class.push_back(whitespace_element);

  for (int i = 0; i < static_cast<int>(arg_names.size()); ++i) {
    const auto& arg_name = arg_names[i];
    const auto& arg_type = arg_types[i];
    kParametersType type = kParametersType::kString;  // Default to string type

    // Check the type of the argument.
    if (raw_string_to_types.find(arg_type) != raw_string_to_types.end()) {
      type = raw_string_to_types.at(arg_type);
    } else {
      for (const auto& [raw_string, param_type] : raw_string_to_types) {
        if (arg_type.size() >= raw_string.size() &&
            arg_type.substr(0, raw_string.size()) == raw_string) {
          type = param_type;
          break;
        }
      }
    }

    // Build the prefix.
    std::string prefix = "<parameter=" + arg_name + ">";
    int32_t prefix_id = builder_->AddByteString(prefix);
    int32_t whitespace_id = builder_->AddCharacterClassStar(whitespace_class);

    // Add the reference for the parameter.
    std::vector<int32_t> parameter_sequence;
    parameter_sequence.push_back(prefix_id);
    parameter_sequence.push_back(whitespace_id);
    switch (type) {
      case kParametersType::kString: {
        if (string_rule_id == -1) {
          string_rule_id = SubGrammarAdder().Apply(builder_, kXmlStringGrammar);
        }
        parameter_sequence.push_back(builder_->AddRuleRef(string_rule_id));
        break;
      }
      case kParametersType::kBoolean: {
        if (boolean_rule_id == -1) {
          boolean_rule_id = SubGrammarAdder().Apply(builder_, kBooleanGrammar);
        }
        parameter_sequence.push_back(builder_->AddRuleRef(boolean_rule_id));
        break;
      }
      case kParametersType::kNumber: {
        if (number_rule_id == -1) {
          number_rule_id = SubGrammarAdder().Apply(builder_, kNumberGrammar);
        }
        parameter_sequence.push_back(builder_->AddRuleRef(number_rule_id));
        break;
      }
      case kParametersType::kObject: {
        if (object_rule_id == -1) {
          object_rule_id = SubGrammarAdder().Apply(builder_, kXmlObjectGrammar);
        }
        parameter_sequence.push_back(builder_->AddRuleRef(object_rule_id));
        break;
      }
      default: {
        XGRAMMAR_LOG(FATAL) << "Unsupported parameter type: " << static_cast<int>(type);
      }
    }

    // Add the new rule.
    int32_t parameter_choice_id = builder_->AddSequence(parameter_sequence);
    parameter_choice_id = builder_->AddChoices({parameter_choice_id});
    int32_t parameter_rule_id = builder_->AddRuleWithHint(arg_name, parameter_choice_id);
    parameters_reference_sequence.push_back(builder_->AddRuleRef(parameter_rule_id));
  }

  // Add the root rule with the parameters sequence.
  int32_t parameters_sequence_id = builder_->AddSequence(parameters_reference_sequence);
  builder_->UpdateRuleBody(root_rule_id, builder_->AddChoices({parameters_sequence_id}));
  return builder_->Get(root_rule_id);
}

/*************************** Forward grammar functors to their impl ***************************/

Grammar FunctionCallConverter::Apply(
    const std::vector<std::string>& args_names, const std::vector<std::string>& args_types
) {
  Grammar grammar = FunctionCallConverterImpl().Apply(args_names, args_types);
  return GrammarNormalizer().Apply(grammar);
}
}  // namespace xgrammar
