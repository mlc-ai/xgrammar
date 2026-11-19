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

TEST(JSONTest, JSONSchemaNumberMultipleOf) {
  // type:number + multipleOf must be enforced (or rejected at compile time when the
  // combination is unsupported), never silently dropped (issue #996, number row).
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  // Unbounded number with an integer divisor: enforce divisibility.
  constexpr const char* kSchema = R"({"type": "number", "multipleOf": 2})";
  auto compiled = compiler.CompileJSONSchema(kSchema);
  EXPECT_TRUE(MatchesEntireString(compiled, "4"));
  EXPECT_TRUE(MatchesEntireString(compiled, "0"));
  EXPECT_TRUE(MatchesEntireString(compiled, "-6"));
  EXPECT_TRUE(MatchesEntireString(compiled, "4.0"));
  EXPECT_TRUE(MatchesEntireString(compiled, "4.00"));
  EXPECT_FALSE(MatchesEntireString(compiled, "3"));
  EXPECT_FALSE(MatchesEntireString(compiled, "4.5"));
  EXPECT_FALSE(MatchesEntireString(compiled, "-3"));
  EXPECT_FALSE(MatchesEntireString(compiled, "4."));

  // number without multipleOf keeps the full number grammar (incl. exponents).
  constexpr const char* kPlain = R"({"type": "number"})";
  auto plain = compiler.CompileJSONSchema(kPlain);
  EXPECT_TRUE(MatchesEntireString(plain, "3"));
  EXPECT_TRUE(MatchesEntireString(plain, "4.5"));
  EXPECT_TRUE(MatchesEntireString(plain, "1e3"));

  // Unsupported combinations fail closed instead of dropping the constraint.
  constexpr const char* kWithBounds = R"({"type": "number", "multipleOf": 2, "minimum": 0})";
  XGRAMMAR_EXPECT_THROW(compiler.CompileJSONSchema(kWithBounds), LogFatalError, "multipleOf");

  constexpr const char* kFractional = R"({"type": "number", "multipleOf": 0.5})";
  XGRAMMAR_EXPECT_THROW(compiler.CompileJSONSchema(kFractional), LogFatalError, "multipleOf");

  constexpr const char* kTooLarge = R"({"type": "number", "multipleOf": 2000})";
  XGRAMMAR_EXPECT_THROW(compiler.CompileJSONSchema(kTooLarge), LogFatalError, "multipleOf");
}

}  // namespace
}  // namespace xgrammar
