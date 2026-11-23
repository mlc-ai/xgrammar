// Regression coverage for length-bounded raw XML tool arguments.
// All schemas and token pieces are synthetic; no model or external fixtures are required.
#include <gtest/gtest.h>
#include <picojson.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace xgrammar {
namespace {

using Json = picojson::value;
using Object = picojson::object;
using Array = picojson::array;

struct XMLFormat {
  const char* style;
  const char* parameter_begin;
  const char* parameter_end;
};

class BoundedXMLTest : public testing::TestWithParam<XMLFormat> {
 protected:
  void SetUp() override {
    vocab_ = {
        "<eos>",
        "<tool_call>probe",
        GetParam().parameter_begin,
        GetParam().parameter_end,
        "</tool_call>",
        "a",
        "b",
        "é",
        "🙂",
        "日本",
        " ",
        "\n",
        "<",
        ">",
        std::string("ab") + GetParam().parameter_end,
        std::string(GetParam().parameter_end) + "</tool_call>",
        "aaaaa",
        "</paramete",
        std::string("\xc3", 1),
        std::string("\xa9", 1)
    };
    compiler_ = std::make_unique<GrammarCompiler>(
        TokenizerInfo(
            vocab_, VocabType::RAW, static_cast<int>(vocab_.size()), std::vector<int32_t>{0}
        ),
        1
    );
  }

  CompiledGrammar CompileSchema(const Json& schema) {
    Object tag{
        {"begin", Json("<tool_call>probe")},
        {"end", Json("</tool_call>")},
        {"content",
         Json(Object{
             {"type", Json("json_schema")},
             {"json_schema", schema},
             {"style", Json(GetParam().style)},
             {"any_order", Json(false)}
         })}
    };
    Object structural{
        {"type", Json("structural_tag")},
        {"format",
         Json(Object{
             {"type", Json("triggered_tags")},
             {"triggers", Json(Array{Json("<tool_call>")})},
             {"tags", Json(Array{Json(tag)})},
             {"stop_after_first", Json(false)}
         })}
    };
    return compiler_->CompileStructuralTag(Json(structural).serialize());
  }

  CompiledGrammar CompileBounded(int lo, int hi, const std::string& variant = "direct") {
    Object bounded{{"type", Json("string")}, {"minLength", Json(static_cast<double>(lo))}};
    if (hi >= 0) bounded["maxLength"] = Json(static_cast<double>(hi));
    Json prop(bounded);
    if (variant == "ref") prop = Json(Object{{"$ref", Json("#/$defs/value")}});
    if (variant == "nullable") {
      prop =
          Json(Object{{"anyOf", Json(Array{Json(bounded), Json(Object{{"type", Json("null")}})})}});
    }
    if (variant == "allOf") prop = Json(Object{{"allOf", Json(Array{Json(bounded)})}});
    if (variant == "recursive") prop = Json(Object{{"$ref", Json("#/$defs/recursive")}});
    Object schema{
        {"type", Json("object")},
        {"properties", Json(Object{{"query", prop}})},
        {"required", Json(Array{Json("query")})},
        {"additionalProperties", Json(false)},
        {"$defs",
         Json(Object{
             {"value", Json(bounded)},
             {"recursive",
              Json(Object{
                  {"anyOf",
                   Json(Array{Json(bounded), Json(Object{{"$ref", Json("#/$defs/recursive")}})})}
              })}
         })}
    };
    return CompileSchema(Json(schema));
  }

  std::string Call(const std::string& value) const {
    return "<tool_call>probe" + std::string(GetParam().parameter_begin) + value +
           GetParam().parameter_end + "</tool_call>";
  }

  std::string Property(const std::string& name, const std::string& value) const {
    const std::string begin = std::string(GetParam().style) == "qwen_xml"
                                  ? "<parameter=" + name + ">"
                                  : "<arg_key>" + name + "</arg_key><arg_value>";
    return begin + value + GetParam().parameter_end;
  }

  std::vector<int32_t> FillMask(GrammarMatcher& matcher) const {
    std::vector<int32_t> words(GetBitmaskSize(static_cast<int>(vocab_.size())));
    int64_t shape[]{1, static_cast<int64_t>(words.size())};
    DLTensor tensor{};
    tensor.data = words.data();
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 2;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = shape;
    matcher.FillNextTokenBitmask(&tensor);
    return words;
  }

  // The public debug API is the available way to observe retained parser states.
  // Compare state counts rather than imposing a machine-dependent time limit.
  static size_t CountParserStates(const GrammarMatcher& matcher) {
    const auto text = matcher._DebugPrintInternalState();
    size_t count = 0;
    for (size_t pos = 0; (pos = text.find("ParserState(", pos)) != std::string::npos; pos += 12) {
      ++count;
    }
    return count;
  }

  std::vector<std::string> vocab_;
  std::unique_ptr<GrammarCompiler> compiler_;
};

TEST_P(BoundedXMLTest, CharacterBoundariesUnicodeWhitespaceAndCompositions) {
  const std::vector<std::pair<std::string, int>> values = {
      {"", 0},
      {"a", 1},
      {"ab", 2},
      {"abcd", 4},
      {"abcde", 5},
      {" ", 1},
      {"     ", 5},
      {" a ", 3},
      {"\n", 1},
      {"é", 1},
      {"日本", 2},
      {"🙂", 1},
      {"a🙂", 2},
      {"<", 1},
      {"a<b", 3},
      {"<x>", 3},
      {"</paramete", 10}
  };
  const std::vector<std::pair<int, int>> bounds = {{0, 0}, {0, 1}, {1, 1}, {1, 4}, {2, 4}, {3, -1}};
  for (const auto& variant : {"direct", "ref", "allOf", "nullable", "recursive"}) {
    for (const auto& [lo, hi] : bounds) {
      const auto grammar = CompileBounded(lo, hi, variant);
      for (const auto& [text, chars] : values) {
        SCOPED_TRACE(
            testing::Message() << "variant=" << variant << " min=" << lo << " max=" << hi
                               << " value=" << testing::PrintToString(text)
        );
        GrammarMatcher matcher(grammar);
        bool wanted = chars >= lo && (hi < 0 || chars <= hi);
        // In Qwen XML a lone newline may also be the template's framing around an empty value.
        if (std::string(GetParam().style) == "qwen_xml" && text == "\n") wanted = lo <= 1;
        EXPECT_EQ(matcher.AcceptString(Call(text)), wanted);
      }
    }
  }
}

TEST_P(BoundedXMLTest, QwenTemplateNewlinesAreNotCounted) {
  if (std::string(GetParam().style) != "qwen_xml") GTEST_SKIP() << "Qwen XML framing only";
  // The Qwen3-Coder template writes "<parameter=k>\n" + value + "\n</parameter>".
  const std::vector<std::pair<std::string, int>> values = {
      {"", 0},
      {"a", 1},
      {"ab", 2},
      {"abcd", 4},
      {"abcde", 5},
      {" a ", 3},
      {"é", 1},
      {"日本", 2},
      {"a\nb", 3}
  };
  const std::vector<std::pair<int, int>> bounds = {{0, 0}, {0, 1}, {1, 1}, {1, 4}, {2, 4}, {3, -1}};
  for (const auto& variant : {"direct", "ref", "allOf", "nullable", "recursive"}) {
    for (const auto& [lo, hi] : bounds) {
      const auto grammar = CompileBounded(lo, hi, variant);
      for (const auto& [text, chars] : values) {
        SCOPED_TRACE(
            testing::Message() << "variant=" << variant << " min=" << lo << " max=" << hi
                               << " value=" << testing::PrintToString(text)
        );
        // maxLength is exact on the framed value. Each optional framing newline can instead be read
        // as part of the value, so minLength has up to two characters of slack.
        bool wanted = false;
        for (int extra = 0; extra <= 2; ++extra) {
          wanted |= chars + extra >= lo && (hi < 0 || chars + extra <= hi);
        }
        if (hi >= 0 && chars > hi) wanted = false;
        GrammarMatcher matcher(grammar);
        EXPECT_EQ(matcher.AcceptString(Call("\n" + text + "\n")), wanted);
      }
    }
  }
}

TEST_P(BoundedXMLTest, NullableNonStringAlternativeKeepsFormattingWhitespace) {
  const auto grammar = CompileBounded(1, 4, "nullable");
  for (const auto& value : {"null", " null ", "\nnull\t"}) {
    SCOPED_TRACE(testing::PrintToString(value));
    GrammarMatcher matcher(grammar);
    EXPECT_TRUE(matcher.AcceptString(Call(value)));
  }
  GrammarMatcher too_long(grammar);
  EXPECT_FALSE(too_long.AcceptString(Call("     ")));
}

TEST_P(BoundedXMLTest, TokenMasksAgreeWithAcceptanceAcrossRepeatedCalls) {
  const auto grammar = CompileBounded(1, 4, "nullable");
  GrammarMatcher matcher(grammar);
  std::vector<int32_t> history;
  for (int call = 0; call < 4; ++call) {
    for (int32_t token : {1, 2, 8, 5, 6, 15}) {
      SCOPED_TRACE(testing::Message() << "step=" << history.size());
      const auto mask = FillMask(matcher);
      for (size_t id = 0; id < vocab_.size(); ++id) {
        SCOPED_TRACE(testing::Message() << "candidate=" << id);
        GrammarMatcher oracle(grammar);
        for (auto previous : history) ASSERT_TRUE(oracle.AcceptToken(previous));
        const bool allowed = (static_cast<uint32_t>(mask[id / 32]) >> (id % 32)) & 1u;
        EXPECT_EQ(allowed, oracle.AcceptToken(static_cast<int32_t>(id)));
      }
      ASSERT_TRUE(matcher.AcceptToken(token));
      history.push_back(token);
    }
  }
}

TEST_P(BoundedXMLTest, FourTokenRollbackRestoresMasks) {
  const auto grammar = CompileBounded(1, 4, "nullable");
  GrammarMatcher matcher(grammar, std::nullopt, false, 4);
  std::vector<int32_t> history;
  for (int call = 0; call < 4; ++call) {
    for (int32_t token : {1, 2, 8, 5, 6, 15}) {
      SCOPED_TRACE(testing::Message() << "step=" << history.size());
      FillMask(matcher);
      ASSERT_TRUE(matcher.AcceptToken(token));
      history.push_back(token);
      const auto expected = FillMask(matcher);
      const size_t n = std::min<size_t>(4, history.size());
      matcher.Rollback(static_cast<int>(n));
      for (size_t i = history.size() - n; i < history.size(); ++i) {
        ASSERT_TRUE(matcher.AcceptToken(history[i]));
      }
      EXPECT_EQ(FillMask(matcher), expected);
    }
  }
}

TEST_P(BoundedXMLTest, RepeatedCallsDoNotAccumulateDelimiterSpanningStates) {
  const auto grammar = CompileBounded(1, 4000);
  GrammarMatcher matcher(grammar);
  const auto call = Call("hello");
  ASSERT_TRUE(matcher.AcceptString(call));
  const auto states_after_one_call = CountParserStates(matcher);
  ASSERT_GT(states_after_one_call, 0u);
  for (int i = 1; i < 100; ++i) ASSERT_TRUE(matcher.AcceptString(call)) << "call=" << i;
  EXPECT_EQ(CountParserStates(matcher), states_after_one_call);
}

TEST_P(BoundedXMLTest, ClosingTagsCannotHideOutOfOrderPropertiesInAString) {
  Object schema{
      {"type", Json("object")},
      {"additionalProperties", Json(false)},
      {"properties",
       Json(Object{
           {"path", Json(Object{{"type", Json("string")}, {"minLength", Json(1.0)}})},
           {"new_text",
            Json(
                Object{{"type", Json("string")}, {"minLength", Json(1.0)}, {"maxLength", Json(4.0)}}
            )},
           {"append", Json(Object{{"type", Json("boolean")}})}
       })},
      {"required", Json(Array{Json("path"), Json("new_text")})}
  };
  const auto grammar = CompileSchema(Json(schema));
  const auto begin = "<tool_call>probe" + Property("path", "x");
  const auto text = Property("new_text", "abcd");
  const auto append = Property("append", "true");
  GrammarMatcher valid(grammar), wrong_order(grammar);
  EXPECT_TRUE(valid.AcceptString(begin + text + append + "</tool_call>"));
  EXPECT_FALSE(wrong_order.AcceptString(begin + append + text + "</tool_call>"));
}

INSTANTIATE_TEST_SUITE_P(
    RawXML,
    BoundedXMLTest,
    testing::Values(
        XMLFormat{"qwen_xml", "<parameter=query>", "</parameter>"},
        XMLFormat{"glm_xml", "<arg_key>query</arg_key><arg_value>", "</arg_value>"}
    ),
    [](const testing::TestParamInfo<XMLFormat>& info) {
      return std::string(info.param.style) == "qwen_xml" ? "Qwen" : "GLM";
    }
);

}  // namespace
}  // namespace xgrammar
