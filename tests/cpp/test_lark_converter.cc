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

TEST(LarkConverterTest, NamedGrammars) {
  auto word = Grammar::FromRegex("[a-z]+");
  auto number = Grammar::FromLark(R"(start: /[0-9]+/)");
  auto grammar = Grammar::FromLark(
      R"(
        start: "<" @word ":" @number ">" %lark {
          start: "/" @word
        }
      )",
      std::nullopt,
      {{"word", word}, {"number", number}}
  );
  std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("[a-z]"), std::string::npos);
  EXPECT_NE(printed.find("[0-9]"), std::string::npos);
}

TEST(LarkConverterTest, NamedGrammarSources) {
  auto grammar = Grammar::FromLark(
      "start: @pair",
      std::nullopt,
      {{"pair", std::string(R"(start: @item ":" @item)")},
       {"item", std::string(R"(start: /[a-z]+/)")}}
  );
  std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("[a-z]"), std::string::npos);
  EXPECT_NE(printed.find("\":\""), std::string::npos);

  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark(
          "start: @left",
          std::nullopt,
          {{"left", std::string("start: @right")}, {"right", std::string("start: @left")}}
      ),
      XGrammarError,
      "circular named grammar reference: @left -> @right -> @left"
  );
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

TEST(LarkConverterTest, StringAndRegexFlags) {
  auto grammar = Grammar::FromLark(R"(
    start: "Ab-1"i /a.b/s
  )");
  std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("root"), std::string::npos);
}

TEST(LarkConverterTest, DynamicRegexSuffixAndSuffixAttribute) {
  auto grammar = Grammar::FromLark(R"(
    start: (foo | bar)* tail
    tail: TEXT

    foo_head[lazy]: /(\n|.)*<foo>/
    foo: foo_head /[a-z]+/ "</foo>"

    bar_head[suffix="<bar>"]: TEXT
    bar: bar_head /[0-9]+/ "</bar>"

    TEXT: /.*/s
  )");
  std::string printed = grammar.ToString();
  EXPECT_NE(printed.find("TagDispatch"), std::string::npos);
  EXPECT_NE(printed.find("\"<foo>\""), std::string::npos);
  EXPECT_NE(printed.find("\"<bar>\""), std::string::npos);
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
      Grammar::FromLark("start[budget=10]: \"a\""),
      XGrammarError,
      "attribute 'budget' is not supported"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start[capture=\"a b\"]: \"a\""),
      XGrammarError,
      "capture name must only contain letters, digits"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: A & B\nA: \"a\"\nB: \"b\""),
      XGrammarError,
      "intersection '&' is not supported"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: /abc/i"),
      XGrammarError,
      "only the regular-expression flag 's' is currently supported"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: \"\\u00c4\"i"),
      XGrammarError,
      "currently support ASCII characters only"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: TOKEN\nTOKEN: %json {}"),
      XGrammarError,
      "%json cannot be used in terminals"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: TOKEN\nTOKEN: %lark { start: \"a\" }"),
      XGrammarError,
      "nested %lark cannot be used in terminals"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: <[1-2-3]>"), XGrammarError, "invalid numeric special-token range"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark("start: @missing"),
      XGrammarError,
      "line 1, column 8.*unknown named grammar '@missing'"
  );
  XGRAMMAR_EXPECT_THROW(
      Grammar::FromLark(
          "start: @item",
          std::nullopt,
          {{"item", Grammar::FromLark(R"(start: "a")")},
           {"item", Grammar::FromLark(R"(start: "b")")}}
      ),
      XGrammarError,
      "Duplicate named grammar 'item'"
  );
}
