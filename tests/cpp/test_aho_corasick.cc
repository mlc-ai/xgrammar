/*!
 * \file tests/cpp/test_aho_corasick.cc
 * \brief Tests for the byte-oriented Aho-Corasick matcher.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "support/aho_corasick.h"

using namespace xgrammar;

TEST(AhoCorasickTest, FailureTransitionsFindOverlappingSuffix) {
  AhoCorasick matcher({"bcd", "abce"});

  EXPECT_TRUE(matcher.ContainsMatch("abcd"));
  EXPECT_TRUE(matcher.ContainsMatch("zabcdz"));
  EXPECT_FALSE(matcher.ContainsMatch("abc"));
}

TEST(AhoCorasickTest, TerminalOutputsPropagateThroughFailureLinks) {
  AhoCorasick matcher({"bc", "abcd"});

  EXPECT_TRUE(matcher.ContainsMatch("abc"));
  EXPECT_TRUE(matcher.ContainsMatch("zabc"));
  EXPECT_TRUE(matcher.ContainsMatch("abcd"));
  EXPECT_FALSE(matcher.ContainsMatch("ab"));
}

TEST(AhoCorasickTest, HonorsStartOffsetAndBytePatterns) {
  AhoCorasick matcher({"abc", "哈哈"});

  EXPECT_TRUE(matcher.ContainsMatch("abc"));
  EXPECT_FALSE(matcher.ContainsMatch("abc", 1));
  EXPECT_TRUE(matcher.ContainsMatch("xabc", 1));
  EXPECT_TRUE(matcher.ContainsMatch("x哈哈y", 1));
}

TEST(AhoCorasickTest, EmptyPatternMatchesAtAnyValidOffset) {
  AhoCorasick matcher({""});

  EXPECT_TRUE(matcher.ContainsMatch(""));
  EXPECT_TRUE(matcher.ContainsMatch("abc", 1));
  EXPECT_FALSE(matcher.ContainsMatch("abc", 4));
}

TEST(AhoCorasickTest, MatchesNaiveSearchExhaustively) {
  const std::vector<std::string> patterns = {"a", "ab", "bab", "bc", "bca", "c", "caa"};
  AhoCorasick matcher(patterns);

  int32_t num_texts = 1;
  for (int32_t length = 0; length <= 6; ++length) {
    if (length != 0) {
      num_texts *= 3;
    }
    for (int32_t encoded = 0; encoded < num_texts; ++encoded) {
      int32_t value = encoded;
      std::string text(length, 'a');
      for (int32_t index = 0; index < length; ++index) {
        text[index] = static_cast<char>('a' + value % 3);
        value /= 3;
      }
      for (size_t start = 0; start <= text.size() + 1; ++start) {
        bool expected = false;
        for (const auto& pattern : patterns) {
          expected = expected || text.find(pattern, start) != std::string::npos;
        }
        EXPECT_EQ(matcher.ContainsMatch(text, start), expected)
            << "text=" << text << ", start=" << start;
      }
    }
  }
}
