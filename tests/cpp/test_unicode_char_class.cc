/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/test_unicode_char_class.cc
 */
#include <gtest/gtest.h>

#include "support/unicode_char_class.h"

using namespace xgrammar;

TEST(UnicodeCharClassTest, Whitespace) {
  EXPECT_TRUE(IsUnicodeWhitespace(' '));
  EXPECT_TRUE(IsUnicodeWhitespace(0x0085));
  EXPECT_TRUE(IsUnicodeWhitespace(0x2007));
  EXPECT_TRUE(IsUnicodeWhitespace(0x202f));
  EXPECT_TRUE(IsUnicodeWhitespace(0x3000));

  EXPECT_FALSE(IsUnicodeWhitespace(0x0000));
  EXPECT_FALSE(IsUnicodeWhitespace(0x200b));
  EXPECT_FALSE(IsUnicodeWhitespace(0xfeff));
}

TEST(UnicodeCharClassTest, Alphanumeric) {
  EXPECT_TRUE(IsUnicodeAlphanumeric('A'));
  EXPECT_TRUE(IsUnicodeAlphanumeric(0x0345));
  EXPECT_TRUE(IsUnicodeAlphanumeric(0x2167));
  EXPECT_TRUE(IsUnicodeAlphanumeric(0x2460));
  EXPECT_TRUE(IsUnicodeAlphanumeric(0xd55c));

  EXPECT_FALSE(IsUnicodeAlphanumeric('_'));
  EXPECT_FALSE(IsUnicodeAlphanumeric(0x0301));
  EXPECT_FALSE(IsUnicodeAlphanumeric(0x1f642));

  EXPECT_TRUE(IsUnicodeAlphabetic('A'));
  EXPECT_TRUE(IsUnicodeAlphabetic(0x03bb));
  EXPECT_TRUE(IsUnicodeAlphabetic(0x2160));
  EXPECT_FALSE(IsUnicodeAlphabetic(0x00b2));
  EXPECT_FALSE(IsUnicodeAlphabetic('1'));
}

TEST(UnicodeCharClassTest, CompleteTableSizes) {
  int32_t whitespace_count = 0;
  int32_t alphabetic_count = 0;
  int32_t alphanumeric_count = 0;
  for (TCodepoint codepoint = 0; codepoint <= 0x10ffff; ++codepoint) {
    whitespace_count += IsUnicodeWhitespace(codepoint);
    alphabetic_count += IsUnicodeAlphabetic(codepoint);
    alphanumeric_count += IsUnicodeAlphanumeric(codepoint);
  }
  EXPECT_EQ(whitespace_count, 25);
  EXPECT_EQ(alphabetic_count, 142759);
  EXPECT_EQ(alphanumeric_count, 144434);
}
