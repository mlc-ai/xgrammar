#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <string>
#include <vector>

namespace xgrammar {
namespace {

bool MatchesEntireString(const CompiledGrammar& grammar, const std::string& input) {
  GrammarMatcher matcher(grammar, std::nullopt, /*terminate_without_stop_token=*/true);
  return matcher.AcceptString(input) && matcher.IsTerminated();
}

// Regression tests for the interaction of propertyNames with
// additionalProperties / unevaluatedProperties JSON Schema keywords.
// Draft 2020-12: a nearby unevaluatedProperties: false must reject every
// property that is not evaluated by properties / patternProperties /
// propertyNames header schemas, even when nothing else evaluates the keys.

TEST(JSONSchemaObjectTest, PropertyNamesWithUnevaluatedPropertiesFalse) {
  constexpr const char* kSchema = R"({
    "type": "object",
    "propertyNames": {"pattern": "^[a-z]+$"},
    "unevaluatedProperties": false
  })";
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  auto compiled = compiler.CompileJSONSchema(kSchema, /*any_whitespace=*/true);
  EXPECT_TRUE(MatchesEntireString(compiled, "{}"))
      << "only an empty object satisfies the schema";
  EXPECT_FALSE(MatchesEntireString(compiled, "{\"score\": 1}"))
      << "propertyNames constrains key shape, it does not evaluate the key against "
         "unevaluatedProperties: false";
}

TEST(JSONSchemaObjectTest, PropertyNamesWithUnevaluatedPropertiesTrue) {
  constexpr const char* kSchema = R"({
    "type": "object",
    "propertyNames": {"pattern": "^[a-z]+$"},
    "unevaluatedProperties": true
  })";
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  auto compiled = compiler.CompileJSONSchema(kSchema, /*any_whitespace=*/true);
  EXPECT_TRUE(MatchesEntireString(compiled, "{}"));
  EXPECT_TRUE(MatchesEntireString(compiled, "{\"score\": 1}"));
  EXPECT_TRUE(MatchesEntireString(compiled, "{\"score\": \"anything\"}"));
}

TEST(JSONSchemaObjectTest, PropertyNamesWithAdditionalPropertiesFalse) {
  constexpr const char* kSchema = R"({
    "type": "object",
    "propertyNames": {"pattern": "^[a-z]+$"},
    "additionalProperties": false
  })";
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  auto compiled = compiler.CompileJSONSchema(kSchema, /*any_whitespace=*/true);
  EXPECT_TRUE(MatchesEntireString(compiled, "{}"));
  EXPECT_FALSE(MatchesEntireString(compiled, "{\"score\": 1}"));
}

TEST(JSONSchemaObjectTest, PropertyNamesWithAdditionalPropertiesTrue) {
  constexpr const char* kSchema = R"({
    "type": "object",
    "propertyNames": {"pattern": "^[a-z]+$"},
    "additionalProperties": true
  })";
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  auto compiled = compiler.CompileJSONSchema(kSchema, /*any_whitespace=*/true);
  EXPECT_TRUE(MatchesEntireString(compiled, "{}"));
  EXPECT_TRUE(MatchesEntireString(compiled, "{\"score\": 1}"));
  EXPECT_TRUE(MatchesEntireString(compiled, "{\"score\": \"anything\"}"));
}

TEST(JSONSchemaObjectTest, PropertyNamesPatternStillAppliesWhenUnevaluatedAllowed) {
  constexpr const char* kSchema = R"({
    "type": "object",
    "propertyNames": {"pattern": "^[a-z]+$"}
  })";
  // Non-strict mode: unknown keys are permitted, but their names must match.
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  auto compiled = compiler.CompileJSONSchema(
      kSchema,
      /*any_whitespace=*/true,
      /*indent=*/std::nullopt,
      /*separators=*/std::nullopt,
      /*strict_mode=*/false
  );
  EXPECT_TRUE(MatchesEntireString(compiled, "{}"));
  EXPECT_TRUE(MatchesEntireString(compiled, "{\"score\": 1}"));
  EXPECT_FALSE(MatchesEntireString(compiled, "{\"SCORE\": 1}"))
      << "the propertyNames pattern must keep constraining key shape";
}

TEST(JSONSchemaObjectTest, PropertyNamesStrictModeDefaultsToEmptyObject) {
  constexpr const char* kSchema = R"({
    "type": "object",
    "propertyNames": {"pattern": "^[a-z]+$"}
  })";
  // Strict mode (the default): keys that nothing evaluates are rejected.
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  auto compiled = compiler.CompileJSONSchema(kSchema, /*any_whitespace=*/true);
  EXPECT_TRUE(MatchesEntireString(compiled, "{}"));
  EXPECT_FALSE(MatchesEntireString(compiled, "{\"score\": 1}"));
}

}  // namespace
}  // namespace xgrammar