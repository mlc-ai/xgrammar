#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <optional>
#include <string>
#include <vector>

#include "support/logging.h"
#include "test_utils.h"

namespace xgrammar {
namespace {

bool MatchesEntireString(const CompiledGrammar& grammar, const std::string& input) {
  GrammarMatcher matcher(grammar, std::nullopt, /*terminate_without_stop_token=*/true);
  return matcher.AcceptString(input) && matcher.IsTerminated();
}

TEST(JSONTest, BuiltinGrammarAcceptsAllRFCWhitespace) {
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);
  auto compiled = compiler.CompileBuiltinJSONGrammar();

  EXPECT_TRUE(MatchesEntireString(compiled, "{\r\"value\"\r:\r3\r}"));
  EXPECT_TRUE(MatchesEntireString(compiled, "{\r\n\"value\"\r\n:\r\n3\r\n}"));
  EXPECT_TRUE(MatchesEntireString(compiled, "{ \t\r\n\"value\"\n\r\t :\r 3\t\n}"));
}

TEST(JSONTest, JSONSchemaCarriageReturnWhitespaceModes) {
  constexpr const char* kSchema = R"({
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": false
  })";
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  auto flexible = compiler.CompileJSONSchema(kSchema, /*any_whitespace=*/true);
  EXPECT_TRUE(MatchesEntireString(flexible, "{\r\"value\"\r:\r3\r}"));
  EXPECT_TRUE(MatchesEntireString(flexible, "{\r\n\"value\"\r\n:\r\n3\r\n}"));
  EXPECT_TRUE(MatchesEntireString(flexible, "{ \t\r\n\"value\"\n\r\t :\r 3\t\n}"));

  auto fixed = compiler.CompileJSONSchema(kSchema, /*any_whitespace=*/false);
  EXPECT_TRUE(MatchesEntireString(fixed, R"({"value": 3})"));
  EXPECT_FALSE(MatchesEntireString(fixed, "{\r\"value\"\r:\r3\r}"));

  auto capped = compiler.CompileJSONSchema(
      kSchema,
      /*any_whitespace=*/true,
      /*indent=*/std::nullopt,
      /*separators=*/std::nullopt,
      /*strict_mode=*/true,
      /*max_whitespace_cnt=*/2
  );
  EXPECT_TRUE(MatchesEntireString(capped, "{\r\n\"value\"\r:\t3\n\r}"));
  EXPECT_FALSE(MatchesEntireString(capped, "{\r\n\r\"value\": 3}"));
}

TEST(JSONTest, JSONSchemaPropertyNamesAppliesToDeclaredKeys) {
  // propertyNames validates every key of the instance object, including keys declared in
  // `properties` (JSON Schema draft 2020-12, section 6.5.3).
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  // An optional declared key that violates the propertyNames pattern must not be generated.
  constexpr const char* kPatternSchema = R"({
    "type": "object",
    "properties": {"UPPER": {"type": "integer"}},
    "propertyNames": {"pattern": "^[a-z]+$"}
  })";
  auto pattern = compiler.CompileJSONSchema(kPatternSchema);
  EXPECT_FALSE(MatchesEntireString(pattern, R"({"UPPER": 1})"));
  EXPECT_TRUE(MatchesEntireString(pattern, "{}"));

  // A required declared key that violates propertyNames makes the schema unsatisfiable.
  constexpr const char* kMaxLengthSchema = R"({
    "type": "object",
    "properties": {"toolong": {"type": "integer"}},
    "propertyNames": {"maxLength": 3},
    "required": ["toolong"]
  })";
  XGRAMMAR_EXPECT_THROW(
      compiler.CompileJSONSchema(kMaxLengthSchema), LogFatalError, "propertyNames"
  );

  // A declared key outside the propertyNames enum is dropped.
  constexpr const char* kEnumSchema = R"({
    "type": "object",
    "properties": {"bad": {"type": "integer"}, "good": {"type": "integer"}},
    "propertyNames": {"enum": ["good", "other"]}
  })";
  auto enums = compiler.CompileJSONSchema(kEnumSchema);
  EXPECT_FALSE(MatchesEntireString(enums, R"({"bad": 1})"));
  EXPECT_TRUE(MatchesEntireString(enums, R"({"good": 1})"));

  // Compliant declared keys keep working.
  constexpr const char* kOkSchema = R"({
    "type": "object",
    "properties": {"ok": {"type": "integer"}},
    "propertyNames": {"maxLength": 3},
    "required": ["ok"]
  })";
  auto ok = compiler.CompileJSONSchema(kOkSchema);
  EXPECT_TRUE(MatchesEntireString(ok, R"({"ok": 1})"));
}

}  // namespace
}  // namespace xgrammar
