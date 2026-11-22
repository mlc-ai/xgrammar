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

TEST(JSONTest, NarrowSubMicroFloatRangeNeverEmitsEmptyString) {
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  auto compile_schema = [&](const char* schema) { return compiler.CompileJSONSchema(schema); };

  auto eps_obj = compile_schema(R"({
    "type": "object",
    "properties": {"eps": {"type": "number", "exclusiveMinimum": 0, "maximum": 1e-7}},
    "required": ["eps"],
    "additionalProperties": false
  })");
  EXPECT_FALSE(MatchesEntireString(eps_obj, R"({"eps":})"));
  EXPECT_FALSE(MatchesEntireString(eps_obj, R"({"eps":0})"));
  EXPECT_FALSE(MatchesEntireString(eps_obj, R"({"eps":0.0000002})"));
  EXPECT_TRUE(MatchesEntireString(eps_obj, R"({"eps":0.0000001})"));

  auto open_eps =
      compile_schema(R"({"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1e-7})");
  EXPECT_FALSE(MatchesEntireString(open_eps, ""));
  EXPECT_FALSE(MatchesEntireString(open_eps, "0"));
  EXPECT_FALSE(MatchesEntireString(open_eps, "0.0000001"));
  EXPECT_TRUE(MatchesEntireString(open_eps, "0.00000005"));

  auto narrow_pos = compile_schema(R"({"type":"number","minimum":0.1234567,"maximum":0.1234569})");
  EXPECT_FALSE(MatchesEntireString(narrow_pos, ""));
  EXPECT_FALSE(MatchesEntireString(narrow_pos, "0.1234566"));
  EXPECT_FALSE(MatchesEntireString(narrow_pos, "0.1234570"));
  EXPECT_TRUE(MatchesEntireString(narrow_pos, "0.1234567"));
  EXPECT_TRUE(MatchesEntireString(narrow_pos, "0.1234568"));
  EXPECT_TRUE(MatchesEntireString(narrow_pos, "0.1234569"));

  auto sub_micro_band = compile_schema(R"({"type":"number","minimum":1e-8,"maximum":5e-7})");
  EXPECT_FALSE(MatchesEntireString(sub_micro_band, ""));
  EXPECT_FALSE(MatchesEntireString(sub_micro_band, "0"));
  EXPECT_FALSE(MatchesEntireString(sub_micro_band, "0.0000006"));
  EXPECT_TRUE(MatchesEntireString(sub_micro_band, "0.0000001"));
  EXPECT_TRUE(MatchesEntireString(sub_micro_band, "0.0000005"));

  auto neg_sub_micro = compile_schema(R"({"type":"number","minimum":-5e-7,"maximum":-1e-7})");
  EXPECT_FALSE(MatchesEntireString(neg_sub_micro, ""));
  EXPECT_FALSE(MatchesEntireString(neg_sub_micro, "0"));
  EXPECT_FALSE(MatchesEntireString(neg_sub_micro, "-0.0000006"));
  EXPECT_TRUE(MatchesEntireString(neg_sub_micro, "-0.0000005"));
  EXPECT_TRUE(MatchesEntireString(neg_sub_micro, "-0.0000001"));

  auto excl_unit =
      compile_schema(R"({"type":"number","exclusiveMinimum":1.0,"exclusiveMaximum":1.0000001})");
  EXPECT_FALSE(MatchesEntireString(excl_unit, ""));
  EXPECT_FALSE(MatchesEntireString(excl_unit, "1.0"));
  EXPECT_FALSE(MatchesEntireString(excl_unit, "1.0000001"));
  EXPECT_TRUE(MatchesEntireString(excl_unit, "1.00000005"));

  auto point_sub_micro = compile_schema(R"({"type":"number","minimum":1e-7,"maximum":1e-7})");
  EXPECT_FALSE(MatchesEntireString(point_sub_micro, ""));
  EXPECT_FALSE(MatchesEntireString(point_sub_micro, "0"));
  EXPECT_TRUE(MatchesEntireString(point_sub_micro, "0.0000001"));

  auto ultra_tiny_obj = compile_schema(R"({
    "type": "object",
    "properties": {"x": {"type": "number", "minimum": 1e-50, "maximum": 2e-50}},
    "required": ["x"],
    "additionalProperties": false
  })");
  EXPECT_FALSE(MatchesEntireString(ultra_tiny_obj, R"({"x":})"));

  auto ultra_tiny_anyof = compile_schema(
      R"({"anyOf":[{"type":"number","minimum":1e-50,"maximum":2e-50},{"type":"null"}]})"
  );
  EXPECT_FALSE(MatchesEntireString(ultra_tiny_anyof, ""));
  EXPECT_TRUE(MatchesEntireString(ultra_tiny_anyof, "null"));
}

}  // namespace
}  // namespace xgrammar
