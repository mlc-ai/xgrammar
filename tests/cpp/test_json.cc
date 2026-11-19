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

TEST(JSONTest, JSONSchemaArrayContains) {
  // contains / minContains must check that enough elements match the contains
  // schema, not just raise the item count (issue #968).
  GrammarCompiler compiler(TokenizerInfo(std::vector<std::string>{}), 1, false);

  // Bare contains: at least one element must match.
  constexpr const char* kContains = R"({
    "type": "array", "items": true, "contains": {"type": "string"}
  })";
  auto contains = compiler.CompileJSONSchema(kContains);
  EXPECT_FALSE(MatchesEntireString(contains, "[1,2]"));  // no matching element
  EXPECT_FALSE(MatchesEntireString(contains, "[]"));
  EXPECT_TRUE(MatchesEntireString(contains, R"(["a"])"));
  EXPECT_TRUE(MatchesEntireString(contains, R"([1,"a"])"));

  // Explicit minContains: 2.
  constexpr const char* kMinTwo = R"({
    "type": "array", "items": true,
    "contains": {"type": "string"}, "minContains": 2
  })";
  auto min_two = compiler.CompileJSONSchema(kMinTwo);
  EXPECT_FALSE(MatchesEntireString(min_two, R"(["a"])"));
  EXPECT_FALSE(MatchesEntireString(min_two, R"([1,2])"));
  EXPECT_TRUE(MatchesEntireString(min_two, R"(["a","b"])"));
  EXPECT_TRUE(MatchesEntireString(min_two, R"(["a",1,"b"])"));

  // minItems counts every element, contains counts matching ones.
  constexpr const char* kMinItems = R"({
    "type": "array", "items": true,
    "contains": {"type": "string"}, "minItems": 3
  })";
  auto min_items = compiler.CompileJSONSchema(kMinItems);
  EXPECT_FALSE(MatchesEntireString(min_items, R"(["a","b"])"));  // only 2 elements
  EXPECT_TRUE(MatchesEntireString(min_items, R"(["a","b",1])"));

  // Unsupported combinations fail closed instead of accepting invalid arrays.
  constexpr const char* kMaxContains = R"({
    "type": "array", "items": true,
    "contains": {"type": "string"}, "maxContains": 1
  })";
  XGRAMMAR_EXPECT_THROW(compiler.CompileJSONSchema(kMaxContains), LogFatalError, "contains");

  constexpr const char* kWithPrefix = R"({
    "type": "array", "prefixItems": [{"type": "integer"}],
    "contains": {"type": "string"}
  })";
  XGRAMMAR_EXPECT_THROW(compiler.CompileJSONSchema(kWithPrefix), LogFatalError, "contains");

  // maxContains without contains has no effect (draft 2020-12) and compiles.
  constexpr const char* kOrphanMax = R"({
    "type": "array", "items": true, "maxContains": 1
  })";
  auto orphan_max = compiler.CompileJSONSchema(kOrphanMax);
  EXPECT_TRUE(MatchesEntireString(orphan_max, R"(["a","b"])"));
}

}  // namespace
}  // namespace xgrammar
