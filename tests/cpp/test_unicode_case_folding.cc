#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "support/unicode_case_folding.h"

using namespace xgrammar;

namespace {

std::vector<TCodepoint> Fold(TCodepoint codepoint) {
  std::vector<TCodepoint> result;
  AppendUnicodeSimpleCaseFold(codepoint, codepoint, &result);
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace

TEST(UnicodeCaseFoldingTest, RepresentativeEquivalenceClasses) {
  EXPECT_EQ(Fold(0x006b), (std::vector<TCodepoint>{0x004b, 0x212a}));
  EXPECT_EQ(Fold(0x03a3), (std::vector<TCodepoint>{0x03c2, 0x03c3}));
  EXPECT_EQ(Fold(0x00df), (std::vector<TCodepoint>{0x1e9e}));
  EXPECT_EQ(Fold(0x1e9e), (std::vector<TCodepoint>{0x00df}));
  EXPECT_TRUE(Fold(0x1f600).empty());
}

TEST(UnicodeCaseFoldingTest, CompleteUnicodeTable) {
  std::vector<TCodepoint> equivalents;
  AppendUnicodeSimpleCaseFold(0, 0x10ffff, &equivalents);
  EXPECT_EQ(equivalents.size(), 3034);
}
