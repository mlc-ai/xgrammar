#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

using namespace xgrammar;

TEST(JumpForwardTokens, ConstQueryPreservesAcceptedBoundary) {
  TokenizerInfo tokenizer({"a", "bc", "abc"}, VocabType::RAW);
  GrammarCompiler compiler(tokenizer);
  auto compiled = compiler.CompileGrammar(Grammar::FromEBNF("root ::= \"abc\""));
  GrammarMatcher matcher(compiled);
  ASSERT_TRUE(matcher.AcceptToken(0));
  const GrammarMatcher& query = matcher;
  EXPECT_EQ(query.FindJumpForwardTokens(), std::vector<int32_t>({1}));
  EXPECT_EQ(query.FindJumpForwardTokens(), std::vector<int32_t>({1}));
  ASSERT_TRUE(matcher.AcceptToken(1));
  EXPECT_TRUE(matcher.IsCompleted());
  EXPECT_TRUE(query.FindJumpForwardTokens().empty());
  matcher.Rollback(2);
  EXPECT_EQ(query.FindJumpForwardTokens(), std::vector<int32_t>({2}));
}
