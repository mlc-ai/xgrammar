/*!
 * \file tests/cpp/test_tag_dispatch_slicing.cc
 * \brief Compare tag-dispatch preprocessing with the original substring predicate.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "tag_dispatch_slicing.h"

using namespace xgrammar;

namespace {

void CheckAgainstSubstringSearch(
    const std::vector<std::string>& patterns, const std::vector<std::string>& tokens
) {
  std::vector<std::pair<int32_t, std::string>> vocabulary;
  for (int32_t index = 0; index < static_cast<int32_t>(tokens.size()); ++index) {
    // Deliberately unrelated token IDs: the output must use vocabulary positions.
    vocabulary.emplace_back(1000 - index, tokens[index]);
  }
  auto result = BuildTagDispatchSecondSlicingBitset(vocabulary, patterns);
  for (size_t index = 0; index < tokens.size(); ++index) {
    const auto& token = tokens[index];
    bool expected = token.empty() ||
                    std::none_of(patterns.begin(), patterns.end(), [&](const std::string& pattern) {
                      return token.find(pattern, 1) != std::string::npos;
                    });
    EXPECT_EQ(result[index], expected) << "vocabulary position " << index;
  }
}

}  // namespace

TEST(TagDispatchSlicing, OffsetsLengthsAndEmptyInputs) {
  const std::vector<std::string> tokens = {
      "", "x", "ab", "abc", "xabc", "abcx", "abcabc", "xxabc", "xab", "x<guard>"
  };
  CheckAgainstSubstringSearch({}, tokens);
  CheckAgainstSubstringSearch({""}, tokens);
  CheckAgainstSubstringSearch({"abc", ""}, tokens);
  CheckAgainstSubstringSearch({"abc", "abc", "<guard>"}, tokens);
  CheckAgainstSubstringSearch({"much longer than every token"}, tokens);
  CheckAgainstSubstringSearch({"abc"}, {});
  CheckAgainstSubstringSearch({"a", std::string(10000, 'a')}, tokens);
}

TEST(TagDispatchSlicing, FailureLinksAndSuffixMatches) {
  CheckAgainstSubstringSearch({"abce", "bcd"}, {"zabcd", "abcd", "zabce", "zbcd", "zabc"});
  // At trie node "abc", the suffix "bc" must report a match via the failure link.
  CheckAgainstSubstringSearch({"abcx", "bc"}, {"zabc", "zabcx", "zabcy", "abc", "bc"});
  CheckAgainstSubstringSearch({"a", "aa", "aaa", "baaa"}, {"a", "ba", "aaaa", "bbaaa"});
  CheckAgainstSubstringSearch({"he", "she", "his", "hers"}, {"ushers", "zhis", "zhers"});
}

TEST(TagDispatchSlicing, AllBytesAndEmbeddedNulls) {
  std::vector<std::string> patterns;
  std::vector<std::string> tokens;
  for (int byte = 0; byte < 256; ++byte) {
    auto pattern = std::string(1, static_cast<char>(byte)) + "!";
    patterns.push_back(pattern);
    tokens.push_back(pattern);
    tokens.push_back("z" + pattern);
    tokens.push_back("z" + std::string(1, static_cast<char>(byte)));
  }
  CheckAgainstSubstringSearch(patterns, tokens);
  CheckAgainstSubstringSearch(
      {std::string("\0\xff", 2), "你好"}, {std::string("x\0\xff", 3), "x你好", "你好", "x你"}
  );
}

TEST(TagDispatchSlicing, RandomizedDifferential) {
  std::mt19937 generator(20261001);
  for (int trial = 0; trial < 300; ++trial) {
    int alphabet = trial % 2 == 0 ? 4 : 256;
    auto random_string = [&](size_t length) {
      std::string value;
      for (size_t index = 0; index < length; ++index) {
        value.push_back(static_cast<char>(generator() % alphabet));
      }
      return value;
    };
    std::vector<std::string> patterns;
    for (size_t count = generator() % 50; patterns.size() < count;) {
      patterns.push_back(random_string(1 + generator() % 24));
    }
    if (trial % 31 == 0) {
      patterns.push_back("");
    }
    std::vector<std::string> tokens;
    for (int index = 0; index < 150; ++index) {
      tokens.push_back(random_string(generator() % 80));
    }
    for (const auto& pattern : patterns) {
      tokens.push_back(pattern);
      tokens.push_back("x" + pattern);
      tokens.push_back(random_string(5) + pattern + random_string(3));
    }
    CheckAgainstSubstringSearch(patterns, tokens);
  }
}
