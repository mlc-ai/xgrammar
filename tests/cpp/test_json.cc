#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <optional>
#include <string>
#include <vector>

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

TEST(JSONTest, UnicodePatternsRetainJSONStringExclusions) {
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);
  auto grammar = compiler.CompileJSONSchema(
      R"({"type":"string","pattern":"^[é\\s\\S]{1,64}$"})"
  );
  EXPECT_TRUE(MatchesEntireString(grammar, "\"hello é 🙂 world\""));
  EXPECT_FALSE(MatchesEntireString(grammar, "\"hello\"world\""));
  for (char control = 0; control < 0x20; ++control) {
    EXPECT_FALSE(MatchesEntireString(grammar, std::string("\"hello") + control + "world\""));
  }
  auto literal = compiler.CompileJSONSchema(R"({"type":"string","pattern":"^é🙂+$"})");
  EXPECT_TRUE(MatchesEntireString(literal, "\"é🙂🙂\""));
  EXPECT_FALSE(MatchesEntireString(literal, "\"éé\""));
}

TEST(JSONTest, EscapedOnlyPatternsFailBeforeGenerationInsteadOfUsingRawCFG) {
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);
  for (const auto* schema : {
           R"({"type":"string","pattern":"^\\\"$"})",
           R"({"type":"string","pattern":"^\\n$"})",
           R"({"type":"string","pattern":"^\\\\$"})",
           R"({"type":"object","properties":{"text":{"type":"string","pattern":"^\\\"$"}}})",
       }) {
    EXPECT_ANY_THROW(compiler.CompileJSONSchema(schema));
  }
  // The independent raw-regex API is not JSON and must keep accepting these bytes.
  auto raw = compiler.CompileRegex("^\"$");
  EXPECT_TRUE(MatchesEntireString(raw, "\""));
}

}  // namespace
}  // namespace xgrammar
