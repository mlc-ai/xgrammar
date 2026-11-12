/*!
 * Copyright (c) 2024 by Contributors
 * \file tests/cpp/test_character_class_token_summary.cc
 */
#include <gtest/gtest.h>
#include <xgrammar/tokenizer_info.h>

#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "character_class_token_summary.h"
#include "compiled_grammar_impl.h"
#include "tokenizer_info_impl.h"

namespace xgrammar {
namespace {

// Index by original token id so these expectations also check the sorted-index mapping.
using SummaryByTokenId = std::map<int32_t, std::pair<int32_t, bool>>;

SummaryByTokenId Summarize(
    const std::vector<int32_t>& character_class, const std::vector<std::string>& vocabulary
) {
  TokenizerInfo info(vocabulary, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto& sorted_vocab = info.GetSortedDecodedVocab();
  const auto summaries = BuildCharacterClassTokenSummaries(
      character_class,
      sorted_vocab,
      info->GetSortedVocabLCPWithPrevious(),
      info->GetAsciiStringSafeBitset()
  );
  SummaryByTokenId result;
  int32_t previous_index = -1;
  for (const auto& summary : summaries) {
    EXPECT_GT(summary.sorted_vocab_index, previous_index);
    previous_index = summary.sorted_vocab_index;
    result.emplace(
        sorted_vocab.at(summary.sorted_vocab_index).first,
        std::make_pair(summary.locally_consumed_characters, summary.consumed_whole_token)
    );
  }
  return result;
}

TEST(CharacterClassTokenSummary, TokenizerDerivedDataUsesSortedIndicesAndSurvivesSerialization) {
  TokenizerInfo original(
      {"ab",
       "b",
       "a",
       "ab",
       "",
       "<stop>",
       " ",
       "\"",
       "\\",
       "\x1f",
       "\x7f",
       "\x80",
       std::string("\0a", 2),
       " ~",
       "\xc2\x80"},
      VocabType::RAW,
      18,
      std::vector<int32_t>{5}
  );
  auto recovered = TokenizerInfo::DeserializeJSON(original.SerializeJSON());
  ASSERT_TRUE(std::holds_alternative<TokenizerInfo>(recovered));
  for (const auto& info : {original, std::get<TokenizerInfo>(recovered)}) {
    const auto& sorted = info.GetSortedDecodedVocab();
    ASSERT_EQ(sorted.size(), 13u);
    std::vector<std::string> tokens;
    for (const auto& entry : sorted) {
      tokens.push_back(entry.second);
    }
    EXPECT_EQ(
        tokens,
        (std::vector<std::string>{
            std::string("\0a", 2),
            "\x1f",
            " ",
            " ~",
            "\"",
            "\\",
            "a",
            "ab",
            "ab",
            "b",
            "\x7f",
            "\x80",
            "\xc2\x80"
        })
    );
    EXPECT_EQ(
        info->GetSortedVocabLCPWithPrevious(),
        (std::vector<int32_t>{0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0})
    );
    const auto& safe = info->GetAsciiStringSafeBitset();
    ASSERT_EQ(safe.Size(), 13);
    const std::vector<bool> expected_safe = {
        false, false, true, true, false, false, true, true, true, true, false, false, false
    };
    for (int32_t i = 0; i < safe.Size(); ++i) {
      EXPECT_EQ(safe[i], expected_safe[i]) << i;
    }
    EXPECT_EQ(info.GetStopTokenIds(), (std::vector<int32_t>{5}));
    EXPECT_EQ(info.GetSpecialTokenIds(), (std::vector<int32_t>{4, 15, 16, 17}));
  }
}

TEST(CharacterClassTokenSummary, EmptyAndSpecialOnlyVocabulariesHaveNoDerivedEntries) {
  for (const auto& vocabulary : {std::vector<std::string>{}, {"", "<stop>"}}) {
    TokenizerInfo info(
        vocabulary,
        VocabType::RAW,
        std::nullopt,
        vocabulary.empty() ? std::vector<int32_t>{} : std::vector<int32_t>{1}
    );
    EXPECT_TRUE(info->GetSortedVocabLCPWithPrevious().empty());
    EXPECT_EQ(info->GetAsciiStringSafeBitset().Size(), 0);
    EXPECT_TRUE(BuildCharacterClassTokenSummaries(
                    std::vector<int32_t>{0, 'a', 'z'},
                    info.GetSortedDecodedVocab(),
                    info->GetSortedVocabLCPWithPrevious(),
                    info->GetAsciiStringSafeBitset()
    )
                    .empty());
  }
}

TEST(CharacterClassTokenSummary, AsciiPrefixReusePreservesMismatchAndRecoversAtDivergence) {
  // The shared '!' mismatch must carry across both descendants, then reset for "aba".
  // "z!" also inherits a first-byte rejection, while duplicate "aba" reuses its whole prefix.
  const std::vector<std::string> vocabulary = {
      "a", "ab", "ab!", "ab!x", "ab!y", "aba", "aba", "ac", "b", "ba", "z", "z!", "a\x80", "\x80"
  };
  EXPECT_EQ(
      Summarize({0, 'a', 'b'}, vocabulary),
      (SummaryByTokenId{
          {0, {1, true}},
          {1, {2, true}},
          {2, {2, false}},
          {3, {2, false}},
          {4, {2, false}},
          {5, {3, true}},
          {6, {3, true}},
          {7, {1, false}},
          {8, {1, true}},
          {9, {2, true}},
          {12, {1, false}}
      })
  );
  EXPECT_TRUE(Summarize({0}, vocabulary).empty());
}

TEST(CharacterClassTokenSummary, UnicodeCountsCompletedAndPartialCharactersSeparately) {
  const std::vector<std::string> vocabulary = {
      "\xe4\xb8\xad",              // 中
      "\xe4\xb8\xad\xe6\x96\x87",  // 中文
      "\xe4",                      // can become 中
      "\xe4\xb8",                  // can become 中
      "\xe4\xb8\xad\xe6\x96",      // 中 + a prefix of 文
      "\xe4\xb8\xad!",             // complete character followed by a mismatch
      "\xe4\xb8\xad\xff",          // invalid first byte after a complete character
      "\xe4\xb8\xad\xe4!",         // invalid continuation after a complete character
      "\xe4\xb8\xad\xe5",          // partial character outside the class
      "\xe4\xb8\xad\xe4\xb8\x80",  // a complete nonmember 一
      "\xe4!",
      "\xe5",
      "\xff",
      "a",
      "aa"
  };
  EXPECT_EQ(
      Summarize({0, 'a', 'a', 0x4E2D, 0x4E2D, 0x6587, 0x6587}, vocabulary),
      (SummaryByTokenId{
          {0, {1, true}},
          {1, {2, true}},
          {2, {1, true}},
          {3, {1, true}},
          {4, {2, true}},
          {5, {1, false}},
          {6, {1, false}},
          {7, {1, false}},
          {8, {1, false}},
          {9, {1, false}},
          {13, {1, true}},
          {14, {2, true}}
      })
  );
}

TEST(CharacterClassTokenSummary, PartialUtf8BoundsAndNegativeRangeUnions) {
  const std::vector<std::string> vocabulary = {
      "\xc2",
      "\xc2\x80",
      "\xdf\xbf",
      "\xe0\xa0",
      "\xe0\xa0\x80",
      "\xef\xbf\xbf",
      "\xf0\x90\x80",
      "\xf0\x90\x80\x80",
      "\xf4\x8f\xbf",
      "\xf4\x8f\xbf\xbf",
      "\xc0",
      "\xc0\x80",
      "\xe0\x80",
      "\xe0\x80\x80",
      "\xf0\x80\x80",
      "\xf0\x80\x80\x80",
      "\xf4\x90",
      "\xf4\x90\x80\x80",
      "\xf8\x88\x80\x80\x80",
      "\xc2 ",
      "\x80"
  };
  const SummaryByTokenId expected = {
      {0, {1, true}},
      {1, {1, true}},
      {2, {1, true}},
      {3, {1, true}},
      {4, {1, true}},
      {5, {1, true}},
      {6, {1, true}},
      {7, {1, true}},
      {8, {1, true}},
      {9, {1, true}}
  };
  EXPECT_EQ(Summarize({0, 0x80, 0x10FFFF}, vocabulary), expected);
  EXPECT_EQ(Summarize({1, 0, 0x7F}, vocabulary), expected);

  // E4 B8 can become U+4E00..U+4E3F. Adjacent or overlapping exclusions cover it even
  // when their input order differs; leaving one codepoint available keeps the prefix valid.
  const std::vector<std::string> partial = {"\xe4\xb8", "a\xe4\xb8"};
  for (const auto& character_class :
       {std::vector<int32_t>{1, 0x4E20, 0x4E3F, 0x4E00, 0x4E1F},
        std::vector<int32_t>{1, 0x4E00, 0x4E2F, 0x4E20, 0x4E3F}}) {
    EXPECT_EQ(Summarize(character_class, partial), (SummaryByTokenId{{1, {1, false}}}));
  }
  EXPECT_EQ(
      Summarize({1, 0x4E00, 0x4E2C, 0x4E2E, 0x4E3F}, partial),
      (SummaryByTokenId{{0, {1, true}}, {1, {2, true}}})
  );
}

TEST(CharacterClassTokenSummary, PrintableAsciiShortcutDoesNotIncludeQuotesOrBackslashes) {
  const std::vector<std::string> vocabulary = {
      " ~", "plain", "\"", "\\", "\x1f", "\x7f", "a\"", "a\\", "a\x1f", "a\x7f"
  };
  EXPECT_EQ(
      Summarize({1, 0, 0x1F, '"', '"', '\\', '\\'}, vocabulary),
      (SummaryByTokenId{
          {0, {2, true}},
          {1, {5, true}},
          {5, {1, true}},
          {6, {1, false}},
          {7, {1, false}},
          {8, {1, false}},
          {9, {2, true}}
      })
  );
}

TEST(CharacterClassTokenSummary, SurrogateCodepointsFollowExistingGrammarSemantics) {
  // The grammar's codepoint ranges include surrogate values. This preserves the existing
  // parser behavior rather than imposing Unicode scalar-value validation in the summary.
  const std::vector<std::string> vocabulary = {"\xed\xa0", "\xed\xa0\x80", "\xed\x9f\xbf"};
  EXPECT_EQ(
      Summarize({0, 0xD800, 0xDFFF}, vocabulary), (SummaryByTokenId{{0, {1, true}}, {1, {1, true}}})
  );
  EXPECT_EQ(Summarize({1, 0xD800, 0xDFFF}, vocabulary), (SummaryByTokenId{{2, {1, true}}}));
}

TEST(CharacterClassTokenSummary, BitsetMaskOwnsItsAcceptedBitsAndUncertainIndices) {
  DynamicBitset accepted(65);
  accepted.Set(0);
  accepted.Set(31);
  accepted.Set(32);
  accepted.Set(64);
  std::vector<int32_t> uncertain{2, 7};
  AdaptiveTokenMask mask(accepted, uncertain);
  accepted.Reset();
  uncertain.clear();

  EXPECT_EQ(mask.store_type, AdaptiveTokenMask::StoreType::kAcceptedBitset);
  EXPECT_TRUE(mask.accepted_indices.empty());
  EXPECT_TRUE(mask.rejected_indices.empty());
  EXPECT_EQ(mask.uncertain_indices, (std::vector<int32_t>{2, 7}));
  ASSERT_EQ(mask.accepted_bitset.Size(), 65);
  for (int32_t token_id = 0; token_id < 65; ++token_id) {
    EXPECT_EQ(
        mask.accepted_bitset[token_id],
        token_id == 0 || token_id == 31 || token_id == 32 || token_id == 64
    );
  }
}

}  // namespace
}  // namespace xgrammar
