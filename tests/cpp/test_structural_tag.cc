#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <optional>
#include <string>
#include <vector>

namespace xgrammar {

TEST(StructuralTagTest, RepeatResolvesNamedTokens) {
  TokenizerInfo tokenizer(
      {"<item>", "</item>", "text", "<stop>"},
      VocabType::RAW,
      std::nullopt,
      std::vector<int32_t>{3}
  );
  GrammarCompiler compiler(tokenizer, 1, false);
  auto compiled = compiler.CompileStructuralTag(R"({
    "type": "structural_tag",
    "format": {
      "type": "repeat", "min": 2, "max": 3,
      "content": {
        "type": "tag",
        "begin": {"type": "token", "token": "<item>"},
        "content": {
          "type": "any_tokens", "exclude_tokens": ["<item>", "</item>"]
        },
        "end": {"type": "token", "token": "</item>"}
      }
    }
  })");

  for (int count : {2, 3}) {
    GrammarMatcher matcher(compiled);
    for (int i = 0; i < count; ++i) {
      if (i < 2) EXPECT_FALSE(matcher.AcceptToken(3));
      ASSERT_TRUE(matcher.AcceptToken(0));
      EXPECT_FALSE(matcher.AcceptToken(0));
      ASSERT_TRUE(matcher.AcceptToken(2));
      ASSERT_TRUE(matcher.AcceptToken(1));
    }
    if (count == 3) EXPECT_FALSE(matcher.AcceptToken(0));
    ASSERT_TRUE(matcher.AcceptToken(3));
    EXPECT_TRUE(matcher.IsTerminated());
  }
}

}  // namespace xgrammar
