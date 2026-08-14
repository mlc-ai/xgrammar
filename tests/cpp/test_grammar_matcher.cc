#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xgrammar {
namespace {

TEST(GrammarMatcherTest, ValidateTokensUsesActiveStopTokensWithoutMutatingState) {
  TokenizerInfo tokenizer_info(
      {"a", "b", "", "<old-stop>", "<new-stop>"},
      VocabType::RAW,
      std::nullopt,
      std::vector<int32_t>{3}
  );
  GrammarCompiler compiler(tokenizer_info, 1, false);
  auto compiled = compiler.CompileGrammar(R"(root ::= ("a" | "b")*)");
  GrammarMatcher matcher(compiled, std::vector<int>{4});

  EXPECT_TRUE(matcher.IsCompleted());
  EXPECT_FALSE(matcher.IsTerminated());
  EXPECT_EQ(matcher.ValidateTokens({0, 1, 4, 0}), 3);
  EXPECT_EQ(matcher.ValidateTokens({4, 0}), 1);
  EXPECT_EQ(matcher.ValidateTokens({3}), 0);
  EXPECT_EQ(matcher.ValidateTokens({2}), 0);
  EXPECT_EQ(matcher.ValidateTokens({-1}), 0);
  EXPECT_EQ(matcher.ValidateTokens({5}), 0);
  EXPECT_TRUE(matcher.IsCompleted());
  EXPECT_FALSE(matcher.IsTerminated());

  EXPECT_TRUE(matcher.AcceptToken(0));
  EXPECT_TRUE(matcher.AcceptToken(1));
  EXPECT_TRUE(matcher.AcceptToken(4));
  EXPECT_TRUE(matcher.IsTerminated());
  EXPECT_EQ(matcher.ValidateTokens({0}), 0);

  matcher.Rollback(1);
  EXPECT_FALSE(matcher.IsTerminated());
  EXPECT_EQ(matcher.ValidateTokens({4}), 1);
}

TEST(GrammarMatcherTest, ValidateTokensStopsWhenCompletionTerminatesMatcher) {
  TokenizerInfo tokenizer_info({"a", "b", ""});
  GrammarCompiler compiler(tokenizer_info, 1, false);
  auto compiled = compiler.CompileGrammar(R"(root ::= "a" "b")");
  GrammarMatcher matcher(compiled, std::nullopt, /*terminate_without_stop_token=*/true);

  EXPECT_EQ(matcher.ValidateTokens({0, 1, 0}), 2);
  EXPECT_FALSE(matcher.IsCompleted());
  EXPECT_FALSE(matcher.IsTerminated());
  EXPECT_TRUE(matcher.AcceptToken(0));
  EXPECT_TRUE(matcher.AcceptToken(1));
  EXPECT_TRUE(matcher.IsTerminated());
  EXPECT_EQ(matcher.ValidateTokens({0}), 0);
}

TEST(GrammarMatcherTest, EOSExpressionConsumesStopTokenInsideGrammar) {
  TokenizerInfo tokenizer_info(
      {"</s>", "a", "b"}, VocabType::RAW, std::nullopt, std::vector<int32_t>{0}
  );
  GrammarCompiler compiler(tokenizer_info, 1, false);
  auto compiled = compiler.CompileGrammar(
      R"(root ::= item "b"
         item ::= "a"+ ("" | EOS()))"
  );
  GrammarMatcher matcher(compiled);

  EXPECT_TRUE(matcher.AcceptToken(1));
  EXPECT_FALSE(matcher.IsCompleted());
  EXPECT_TRUE(matcher.AcceptToken(0));
  EXPECT_FALSE(matcher.IsCompleted());
  EXPECT_FALSE(matcher.IsTerminated());
  EXPECT_TRUE(matcher.AcceptToken(2));
  EXPECT_TRUE(matcher.IsCompleted());
  EXPECT_FALSE(matcher.IsTerminated());
  EXPECT_TRUE(matcher.AcceptToken(0));
  EXPECT_TRUE(matcher.IsTerminated());

  matcher.Rollback(2);
  EXPECT_FALSE(matcher.IsCompleted());
  EXPECT_FALSE(matcher.IsTerminated());
  EXPECT_TRUE(matcher.AcceptToken(2));
}

}  // namespace
}  // namespace xgrammar
