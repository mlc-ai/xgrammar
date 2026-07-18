#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <string>

#include "test_utils.h"

using namespace xgrammar;

TEST(LarkConverterTest, CoreSyntaxAndImports) {
  auto grammar = Grammar::FromLark(R"(
    %import common.INT
    %import common (WS_INLINE, CNAME)
    %ignore WS_INLINE
    start: item{2,3}
    item: CNAME ":" INT | "none"
  )");
  std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("root"), std::string::npos);
  EXPECT_NE(printed.find("lark_ignore"), std::string::npos);
  EXPECT_NE(printed.find("{2, 3}"), std::string::npos);
}

TEST(LarkConverterTest, InlineAndNestedGrammar) {
  auto grammar = Grammar::FromLark(R"(
    start: "payload=" %json {
      "type": "object",
      "properties": {"x": {"type": "integer"}},
      "required": ["x"],
      "additionalProperties": false
    } ";" %lark {
      start: "yes" | "no"
    }
  )");
  std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("payload="), std::string::npos);
  EXPECT_NE(printed.find("basic_integer"), std::string::npos);
  EXPECT_NE(printed.find("yes"), std::string::npos);
}

TEST(LarkConverterTest, DynamicToolCallLowersToTagDispatch) {
  auto grammar = Grammar::FromLark(R"(
    start: (foo | bar)* tail
    tail: TEXT

    foo_head[lazy]: TEXT "<function"
    foo: foo_head "=foo>" /[a-z]+/ "</function>"

    bar_head[lazy]: TEXT "<function"
    bar: bar_head "=bar>" /[A-Z]+/ "</function>"

    TEXT: /(\n|.)*/
  )");
  std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("TagDispatch"), std::string::npos);
  EXPECT_NE(printed.find("\"<function\""), std::string::npos);
  EXPECT_NE(printed.find("loop_after_dispatch=true"), std::string::npos);
}

TEST(LarkConverterTest, NumericAndNamedSpecialTokens) {
  TokenizerInfo tokenizer_info({"a", "<|tool|>", "b"}, VocabType::RAW, 3, std::vector<int32_t>{});
  auto grammar = Grammar::FromLark("start: <[0,2]> | <|tool|>", tokenizer_info);
  std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("Token(0, 2)"), std::string::npos);
  EXPECT_NE(printed.find("Token(1)"), std::string::npos);
}

TEST(LarkConverterTest, ErrorsContainSourceLocations) {
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("item: \"a\""), XGrammarError, "line 1, column 1.*no start rule"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: missing"), XGrammarError, "line 1, column 8.*unknown name"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: FOO\nFOO: BAR\nBAR: FOO"),
      XGrammarError,
      "circular reference in terminal"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start[capture]: \"a\""),
      XGrammarError,
      "attribute 'capture' is not supported"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: A & B\nA: \"a\"\nB: \"b\""),
      XGrammarError,
      "intersection '&' is not supported"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: /abc/i"),
      XGrammarError,
      "regular-expression flags are not supported"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: <[1-2-3]>"), XGrammarError, "invalid numeric special-token range"
  );
}
