#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "compiled_grammar_impl.h"

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

TEST(GrammarMatcherTest, MemorySizeIsSafeDuringLazyMetadataInitialization) {
  std::string grammar = "root ::= r0\n";
  constexpr int kRules = 512;
  for (int i = 0; i < kRules; ++i) {
    grammar += "r" + std::to_string(i) + " ::= [ab]";
    if (i + 1 < kRules) {
      grammar += " r" + std::to_string(i + 1);
    }
    grammar += " | \"\"\n";
  }

  TokenizerInfo tokenizer_info({"a", "b", "ab", "ba"});
  GrammarCompiler compiler(
      tokenizer_info,
      /*max_threads=*/1,
      /*cache_enabled=*/false,
      /*max_memory_bytes=*/-1,
      /*enable_dynamic_compilation=*/true
  );
  auto compiled = compiler.CompileGrammar(grammar);

  std::atomic<bool> reader_ready{false};
  std::atomic<bool> start{false};
  std::atomic<bool> done{false};
  std::atomic<bool> observed_zero_size{false};
  std::thread reader([&]() {
    reader_ready.store(true, std::memory_order_release);
    while (!start.load(std::memory_order_acquire)) {
    }
    while (!done.load(std::memory_order_acquire)) {
      if (compiled.MemorySizeBytes() == 0) {
        observed_zero_size.store(true, std::memory_order_relaxed);
      }
    }
  });
  while (!reader_ready.load(std::memory_order_acquire)) {
  }
  start.store(true, std::memory_order_release);
  compiled.ImplPtr()->EnsureRuleLevelMetadata();
  done.store(true, std::memory_order_release);
  reader.join();

  EXPECT_FALSE(observed_zero_size.load(std::memory_order_relaxed));
  EXPECT_EQ(
      compiled.ImplPtr()->rule_level_cacheable.size(), compiled.ImplPtr()->grammar->NumRules()
  );
}

}  // namespace
}  // namespace xgrammar
