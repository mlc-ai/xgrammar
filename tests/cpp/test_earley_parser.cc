/*!
 * \file tests/cpp/test_earley_parser.cc
 * \brief The token mask must admit exactly the tokens AcceptToken accepts.
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace xgrammar;

namespace {

// The ids the next-token mask admits and the ids AcceptToken accepts, after matching prefix.
std::pair<std::vector<int>, std::vector<int>> MaskAndAccepted(
    const CompiledGrammar& compiled, int vocab_size, const std::string& prefix
) {
  GrammarMatcher matcher(compiled);
  EXPECT_TRUE(matcher.AcceptString(prefix));
  std::vector<int32_t> data(GetBitmaskSize(vocab_size), 0);
  int64_t shape = static_cast<int64_t>(data.size());
  DLTensor bitmask{
      data.data(), DLDevice{kDLCPU, 0}, 1, DLDataType{kDLInt, 32, 1}, &shape, nullptr, 0
  };
  matcher.FillNextTokenBitmask(&bitmask);

  std::vector<int> in_mask;
  std::vector<int> accepted;
  for (int id = 0; id < vocab_size; ++id) {
    if ((data[id / 32] >> (id % 32)) & 1) {
      in_mask.push_back(id);
    }
    GrammarMatcher trial(compiled);
    EXPECT_TRUE(trial.AcceptString(prefix));
    if (trial.AcceptToken(id)) {
      accepted.push_back(id);
    }
  }
  return {in_mask, accepted};
}

}  // namespace

TEST(XGrammarEarleyParserTest, CharacterClassBoundsAreCodepointsInLookahead) {
  // A repetition above the unzip threshold gets a unit rule with an exact lookahead of the
  // repeated class, matched through the parser's expression path. Truncated to a byte, the
  // range ‰-⁞ would cover the ASCII bytes 0x30-0x5E, and ​-‧ would cover
  // 0x0B-0x27 (the quote): the mask then admits tokens the matcher rejects.
  const std::vector<std::string> vocab = {
      "\"",
      "a",
      "b",
      "ab",
      "0",
      "a0",
      "<",
      "a<",
      "\\",
      "a\\",
      "a\"",
      "'",
      "a'",
      "\xE2\x80\xB0",   // U+2030
      "a\xE2\x80\xB0",  // a U+2030
      "\xE2\x80\x8B",   // U+200B
      "a\xE2\x80\x8B",  // a U+200B
  };
  const int vocab_size = static_cast<int>(vocab.size());
  TokenizerInfo tokenizer_info(vocab);
  GrammarCompiler compiler(tokenizer_info, /*max_threads=*/1, /*cache_enabled=*/false);

  auto compiled = compiler.CompileGrammar(R"(root ::= "\"" [a-z‰-⁞]{0,200} "\"")");
  auto [in_mask, accepted] = MaskAndAccepted(compiled, vocab_size, "\"a");
  EXPECT_EQ(in_mask, accepted);
  // "a0", "a<" and "a\" hold a byte outside the class; "ab", "a‰" and the closing "a\""
  // are fine.
  EXPECT_EQ(accepted, (std::vector<int>{0, 1, 2, 3, 10, 13, 14}));

  compiled = compiler.CompileGrammar(R"(root ::= "\"" [a-z​-‧]{10,200} "\"")");
  std::tie(in_mask, accepted) = MaskAndAccepted(compiled, vocab_size, "\"a");
  EXPECT_EQ(in_mask, accepted);
  // After one character the string cannot close yet, so no token holding the quote is admitted.
  EXPECT_EQ(accepted, (std::vector<int>{1, 2, 3, 15, 16}));
}
