#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "json_schema_converter.h"
#include "regex_converter.h"

namespace xgrammar {
namespace {

bool Matches(const Grammar& grammar, const std::string& input) {
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);
  GrammarMatcher matcher(compiler.CompileGrammar(grammar));
  return matcher.AcceptString(input) && matcher.IsCompleted();
}

void ExpectDirectGrammarEquivalent(
    const std::string& schema,
    const std::vector<std::string>& inputs,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    bool any_order = false
) {
  Grammar direct = Grammar::FromJSONSchema(
      schema, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, false, any_order
  );
  Grammar parsed = Grammar::FromEBNF(JSONSchemaToEBNF(
      schema,
      any_whitespace,
      indent,
      separators,
      strict_mode,
      max_whitespace_cnt,
      JSONFormat::kJSON,
      any_order
  ));
  for (const auto& input : inputs) {
    EXPECT_EQ(Matches(direct, input), Matches(parsed, input)) << "input: " << input;
  }
}

void ExpectDirectRegexEquivalent(const std::string& regex, const std::vector<std::string>& inputs) {
  Grammar direct = RegexToGrammar(regex);
  Grammar parsed = Grammar::FromEBNF(RegexToEBNF(regex));
  for (const auto& input : inputs) {
    EXPECT_EQ(Matches(direct, input), Matches(parsed, input)) << "input: " << input;
  }
}

TEST(JSONSchemaGrammarConverterTest, BasicTypesAndCompositions) {
  ExpectDirectGrammarEquivalent(
      R"({
        "type": "object",
        "properties": {
          "name": {"type": "string", "minLength": 2, "maxLength": 4},
          "count": {"type": "integer", "minimum": -2, "maximum": 8},
          "choice": {"enum": ["a", "b", 3]},
          "nullable": {"type": ["string", "null"]}
        },
        "required": ["name", "count"],
        "additionalProperties": false
      })",
      {R"({"name":"ab","count":0})",
       R"({"name": "abcd", "count": -2, "choice": "a", "nullable": null})",
       R"({"name":"a","count":0})",
       R"({"count":0,"name":"ab"})",
       R"({"name":"ab","count":9})",
       R"({"name":"ab","count":0,"extra":true})"}
  );
}

TEST(JSONSchemaGrammarConverterTest, ArraysReferencesAndPatterns) {
  ExpectDirectGrammarEquivalent(
      R"({
        "$defs": {
          "entry": {
            "type": "object",
            "properties": {
              "id": {"type": "integer", "multipleOf": 3},
              "code": {"type": "string", "pattern": "^[A-Z]{2}[0-9]+$"}
            },
            "required": ["id", "code"],
            "additionalProperties": false
          }
        },
        "type": "array",
        "prefixItems": [{"const": "header"}],
        "items": {"$ref": "#/$defs/entry"},
        "minItems": 2,
        "maxItems": 3
      })",
      {R"(["header",{"id":3,"code":"AB12"}])",
       R"(["header", {"id": 6, "code": "XY9"}, {"id": -3, "code": "AA0"}])",
       R"(["header"])",
       R"(["header",{"id":4,"code":"AB12"}])",
       R"(["header",{"id":3,"code":"ab12"}])"}
  );
}

TEST(JSONSchemaGrammarConverterTest, PropertyConstraintsAndAdditionalKeys) {
  ExpectDirectGrammarEquivalent(
      R"({
        "type": "object",
        "properties": {
          "fixed": {"type": "boolean"},
          "optional": {"type": "number"}
        },
        "patternProperties": {
          "^x_[a-z]+$": {"type": "integer"}
        },
        "additionalProperties": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 3
      })",
      {R"({"fixed":true})",
       R"({"x_a":2})",
       R"({"other":"value"})",
       R"({"fixed":false,"x_test":3,"other":"value"})",
       R"({})",
       R"({"fixed":true,"x_a":1,"one":"1","two":"2"})"}
  );
}

TEST(JSONSchemaGrammarConverterTest, FormattingOptionsAndAnyOrder) {
  const std::string schema = R"({
    "type": "object",
    "properties": {
      "first": {"type": "string"},
      "second": {"type": "integer"}
    },
    "required": ["first", "second"],
    "additionalProperties": false
  })";
  ExpectDirectGrammarEquivalent(
      schema,
      {R"({"first":"x","second":1})",
       R"({"second":1,"first":"x"})",
       "{\n  \"first\": \"x\",\n  \"second\": 1\n}"},
      false,
      2,
      std::nullopt,
      true,
      std::nullopt,
      true
  );
  ExpectDirectGrammarEquivalent(
      schema,
      {R"({"first":"x","second":1})",
       R"({"first": "x", "second": 1})",
       "{ \"first\":\"x\",\"second\":1 }"},
      true,
      std::nullopt,
      std::nullopt,
      true,
      3
  );
}

TEST(JSONSchemaGrammarConverterTest, RecursiveReference) {
  ExpectDirectGrammarEquivalent(
      R"({
        "type": "object",
        "properties": {
          "value": {"type": "integer"},
          "next": {"anyOf": [{"$ref": "#"}, {"type": "null"}]}
        },
        "required": ["value", "next"],
        "additionalProperties": false
      })",
      {R"({"value":1,"next":null})",
       R"({"value":1,"next":{"value":2,"next":null}})",
       R"({"value":1})",
       R"({"value":1,"next":{"value":"bad","next":null}})"}
  );
}

TEST(JSONSchemaGrammarConverterTest, BuiltinStringFormatUsesDirectRegexGrammar) {
  ExpectDirectGrammarEquivalent(
      R"({"type": "string", "format": "email"})",
      {R"("user@example.com")",
       R"("first.last+tag@example.co.uk")",
       R"("not-an-email")",
       R"("@example.com")"}
  );
}

TEST(RegexGrammarConverterTest, MatchesTextBasedRegexConversion) {
  ExpectDirectRegexEquivalent(
      R"(^(ab|cd?){1,3}$)", {"ab", "c", "cdab", "ababab", "", "abababab", "d"}
  );
  ExpectDirectRegexEquivalent(R"(^[A-Z]\d+\s?x$)", {"A1x", "Z123 x", "a1x", "A x", "A12\nx"});
  ExpectDirectRegexEquivalent(R"(^(|a||bc)$)", {"", "a", "bc", "b", "abc"});
  ExpectDirectRegexEquivalent(R"(^(你好|世界)+$)", {"你好", "世界你好", "", "你", "hello"});
  ExpectDirectRegexEquivalent("", {"", "a", " "});
  ExpectDirectRegexEquivalent(R"(^(?:[a-c\d]|\W)+?$)", {"abc012", "!@", "_", "", "z", "你好"});
  ExpectDirectRegexEquivalent(R"(^\u0041\x42\cC.$)", {"AB\x03x", "AB\x03你", "ABCx", "AB\x03"});
  ExpectDirectRegexEquivalent(R"(^(?<word>ab){2,}$)", {"abab", "ababab", "ab", "abcab"});
}

}  // namespace
}  // namespace xgrammar
