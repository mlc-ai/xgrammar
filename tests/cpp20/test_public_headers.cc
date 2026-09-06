#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

TEST(PublicHeadersTest, FromLarkDefaultArguments) {
  const std::string source = R"(start: "x")";
  const auto expected = xgrammar::Grammar::FromLark(source, std::nullopt, {}).ToString();
  EXPECT_EQ(xgrammar::Grammar::FromLark(source).ToString(), expected);
  EXPECT_EQ(xgrammar::Grammar::FromLark(source, std::nullopt).ToString(), expected);

  auto from_lark = &xgrammar::Grammar::FromLark;
  EXPECT_EQ(from_lark(source, std::nullopt, {}).ToString(), expected);
}
