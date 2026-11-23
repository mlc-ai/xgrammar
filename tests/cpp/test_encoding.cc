#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <string>

#include "support/encoding.h"

using namespace xgrammar;

TEST(XGrammarEncodingTest, EscapeStringPreservesEmbeddedNulBytes) {
  // Embedded NUL bytes are valid in arbitrary byte sequences and must be escaped rather than
  // treated as the end of the string.
  EXPECT_EQ(EscapeString(std::string("a\0b", 3)), "a\\0b");
  EXPECT_EQ(EscapeString(std::string("\0", 1)), "\\0");
  EXPECT_EQ(EscapeString(std::string("\0\0", 2)), "\\0\\0");
}

TEST(XGrammarEncodingTest, EscapeStringPreservesInvalidAndMultibyteBytes) {
  // Invalid UTF-8 bytes are escaped as raw bytes instead of being dropped.
  EXPECT_EQ(EscapeString(std::string("\xFF", 1)), "\\xff");
  EXPECT_EQ(EscapeString(std::string("\x80", 1)), "\\x80");
  // Valid multi-byte UTF-8, an embedded NUL and a trailing invalid byte are all preserved.
  EXPECT_EQ(EscapeString(std::string("a\x00\xFFz", 4)), "a\\0\\xffz");
}

TEST(XGrammarEncodingTest, ByteToLatin1RoundTripsArbitraryBytes) {
  // All byte values, including NUL and non-ASCII bytes, must round-trip through the Latin-1
  // serialization used by the JSON serializer.
  std::string bytes;
  for (int b = 0; b < 256; ++b) {
    bytes.push_back(static_cast<char>(b));
  }
  std::string latin1;
  ByteToLatin1(bytes, &latin1);
  std::string back;
  auto error = Latin1ToBytes(latin1, &back);
  EXPECT_FALSE(error.has_value());
  EXPECT_EQ(back, bytes);
}

TEST(XGrammarEncodingTest, GrammarToStringPreservesNulByteString) {
  // A byte string containing a NUL byte must survive the ToString -> FromEBNF round-trip.
  Grammar grammar = Grammar::FromEBNF("root ::= \"a\\0b\"");
  std::string printed = grammar.ToString();
  EXPECT_EQ(printed, "root ::= ((\"a\\0b\"))\n");

  Grammar reparsed = Grammar::FromEBNF(printed);
  EXPECT_EQ(reparsed.ToString(), printed);
}
