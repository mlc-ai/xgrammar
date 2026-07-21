#include <dlpack/dlpack.h>
#include <gtest/gtest.h>
#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <future>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "dynamic_tag_matcher.h"
#include "grammar_builder.h"
#include "xgrammar/xgrammar.h"

namespace xgrammar {
namespace {

constexpr const char* kNamespace = "]<]minimax[>[";
constexpr int kConstMatcherReadIterations = 1000;
constexpr int kForkedMatcherIterations = 500;

std::string Element(const std::string& name, const std::string& value) {
  return std::string(kNamespace) + "<" + name + ">" + value + kNamespace + "</" + name + ">";
}

DynamicTagMatcher StateAfter(std::string_view text) {
  DynamicTagMatcher matcher(GetMiniMaxM3DynamicTagMatcherConfig());
  if (!matcher.Accept(text)) {
    ADD_FAILURE() << "Invalid dynamic-tag test state prefix: " << std::string(text);
  }
  return matcher;
}

void ClearSerializedDynamicTagElementPrefix(picojson::value* serialized) {
  auto& config = serialized->get<picojson::object>()["dynamic_tag_matcher_config"];
  config.get<picojson::object>()["element_prefix"] = picojson::value(std::string{});
}

TEST(DynamicTagMatcherTest, ConfigValidationReturnsErrors) {
  DynamicTagMatcherConfig config = GetMiniMaxM3DynamicTagMatcherConfig();
  EXPECT_FALSE(ValidateDynamicTagMatcherConfig(config).has_value());

  config.element_prefix.clear();
  EXPECT_EQ(ValidateDynamicTagMatcherConfig(config), "Dynamic tag element_prefix cannot be empty");

  config = GetMiniMaxM3DynamicTagMatcherConfig();
  config.close_marker = "//";
  EXPECT_EQ(
      ValidateDynamicTagMatcherConfig(config),
      "Dynamic tag close_marker must contain exactly one byte"
  );

  config = GetMiniMaxM3DynamicTagMatcherConfig();
  config.tag_suffix.clear();
  EXPECT_EQ(
      ValidateDynamicTagMatcherConfig(config),
      "Dynamic tag tag_suffix must contain exactly one byte"
  );

  config = GetMiniMaxM3DynamicTagMatcherConfig();
  config.attribute_tag_parent_depth = -2;
  EXPECT_EQ(
      ValidateDynamicTagMatcherConfig(config),
      "Dynamic tag attribute_tag_parent_depth must be -1 or non-negative"
  );
}

TEST(DynamicTagMatcherTest, DeserializationReturnsErrorsForInvalidConfig) {
  GrammarBuilder builder(Grammar::FromEBNF("root ::= [^]*"));
  builder.SetDynamicTagMatcherConfig(GetMiniMaxM3DynamicTagMatcherConfig());
  const Grammar grammar = builder.Get();

  picojson::value serialized_grammar;
  ASSERT_TRUE(picojson::parse(serialized_grammar, grammar.SerializeJSON()).empty());
  ClearSerializedDynamicTagElementPrefix(&serialized_grammar);
  EXPECT_TRUE(std::holds_alternative<SerializationError>(
      Grammar::DeserializeJSON(serialized_grammar.serialize())
  ));

  const TokenizerInfo tokenizer_info({}, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const CompiledGrammar compiled =
      GrammarCompiler(tokenizer_info, 4, false).CompileGrammar(grammar);
  picojson::value serialized_compiled;
  ASSERT_TRUE(picojson::parse(serialized_compiled, compiled.SerializeJSON()).empty());
  ClearSerializedDynamicTagElementPrefix(&serialized_compiled);
  EXPECT_TRUE(std::holds_alternative<SerializationError>(
      CompiledGrammar::DeserializeJSON(serialized_compiled.serialize(), tokenizer_info)
  ));
}

TEST(DynamicTagMatcherTest, FastCanAcceptMatchesTransactionalStateMachine) {
  const std::string tool_call = std::string(kNamespace) + "<tool_call>";
  const std::string invoke = tool_call + kNamespace + R"(<invoke name="dynamic">)";
  const std::string runtime = invoke + kNamespace + "<runtime_key>value";
  const std::string long_name = "runtime_key_segment_" + std::string(96, 'x');
  const std::vector<std::string> state_prefixes = {
      "",
      "]",
      "]<]minimax",
      std::string(kNamespace) + "<",
      std::string(kNamespace) + "<runtime",
      std::string(kNamespace) + "<   ",
      tool_call,
      tool_call + kNamespace + "<invoke",
      tool_call + kNamespace + "<invoke ",
      tool_call + kNamespace + R"(<invoke name="dynamic")",
      invoke,
      invoke + kNamespace + "<runtime_key",
      runtime,
      runtime + kNamespace + "<",
      runtime + kNamespace + "</",
      runtime + kNamespace + "</runtime",
      runtime + kNamespace + "</runtime_key",
      invoke + kNamespace + "<" + long_name,
      invoke + Element(long_name, "value") + "]",
      invoke + Element(long_name, "value") + kNamespace + "</invoke>",
  };

  std::vector<std::string> candidates = {
      "",
      "ordinary text",
      ">",
      "/",
      "/runtime_key>",
      "runtime_key",
      "runtime_key>",
      "wrong_key>",
      std::string(kNamespace),
      std::string(kNamespace) + "<",
      std::string(kNamespace) + "<nested>",
      Element("nested", "value"),
      Element("first", "1") + Element("second", "2"),
      std::string("]<]minimax[>[<nested>value") + kNamespace + "</nested>tail",
      std::string("\0\xff", 2),
  };
  constexpr std::string_view kAlphabet = "]<minimax[>/_ invoke=\"toolcaruntimekyb \t\n0123456789";
  std::mt19937 random(0);
  std::uniform_int_distribution<int> length_distribution(0, 64);
  std::uniform_int_distribution<size_t> byte_distribution(0, kAlphabet.size() - 1);
  for (int index = 0; index < 1000; ++index) {
    std::string candidate(length_distribution(random), '\0');
    for (char& byte : candidate) {
      byte = kAlphabet[byte_distribution(random)];
    }
    candidates.push_back(std::move(candidate));
  }

  for (const auto& state_prefix : state_prefixes) {
    const DynamicTagMatcher state = StateAfter(state_prefix);
    for (const auto& candidate : candidates) {
      SCOPED_TRACE("state=" + state_prefix + ", candidate=" + candidate);
      DynamicTagMatcher fast = state;
      DynamicTagMatcher transactional = state;
      EXPECT_EQ(fast.CanAccept(candidate), transactional.Accept(candidate));
      EXPECT_TRUE(fast.HasSameState(state));
    }
  }
}

TEST(DynamicTagMatcherTest, ConstMatcherReadsAreThreadSafe) {
  const std::string name = "runtime key/" + std::string(96, 'x');
  const DynamicTagMatcher shared =
      StateAfter(std::string(kNamespace) + "<" + name + ">value" + kNamespace + "</");
  const std::string valid_compound = name + ">" + Element("nested", "value");
  const std::string invalid_compound =
      name + ">" + std::string(kNamespace) + "<nested>value" + kNamespace + "</wrong>";
  constexpr int kNumThreads = 8;
  std::vector<std::future<bool>> futures;
  futures.reserve(kNumThreads);
  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    futures.push_back(std::async(
        std::launch::async,
        [&shared, &name, &valid_compound, &invalid_compound] {
          for (int iteration = 0; iteration < kConstMatcherReadIterations; ++iteration) {
            if (!shared.CanAccept(name) || shared.CanAccept("wrong name") ||
                !shared.CanAccept(valid_compound) || shared.CanAccept(invalid_compound)) {
              return false;
            }
          }
          return true;
        }
    ));
  }
  for (auto& future : futures) {
    EXPECT_TRUE(future.get());
  }
}

TEST(DynamicTagMatcherTest, RejectsMixedTextAndChildElements) {
  const std::string child = Element("child", "value");
  EXPECT_TRUE(StateAfter(Element("parent", child)).CanTerminate());
  EXPECT_TRUE(StateAfter(Element("parent", child + " \n\t")).CanTerminate());

  DynamicTagMatcher text_then_child(GetMiniMaxM3DynamicTagMatcherConfig());
  EXPECT_FALSE(text_then_child.Accept(Element("parent", "text" + child)));

  DynamicTagMatcher child_then_text(GetMiniMaxM3DynamicTagMatcherConfig());
  EXPECT_FALSE(child_then_text.Accept(Element("parent", child + "text")));
}

TEST(DynamicTagMatcherTest, ForkedMatchersShareOnlyImmutableStateAcrossThreads) {
  const DynamicTagMatcher initial(GetMiniMaxM3DynamicTagMatcherConfig());
  constexpr int kNumThreads = 8;
  std::vector<std::future<bool>> futures;
  futures.reserve(kNumThreads);

  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    futures.push_back(std::async(std::launch::async, [initial, thread_id] {
      const std::string name = "runtime key/" + std::to_string(thread_id) + std::string(96, 'x');
      const std::string valid = Element(name, "value");
      const std::string invalid =
          std::string(kNamespace) + "<" + name + ">value" + kNamespace + "</wrong>";
      for (int iteration = 0; iteration < kForkedMatcherIterations; ++iteration) {
        DynamicTagMatcher valid_matcher = initial;
        if (!valid_matcher.Accept(valid) || !valid_matcher.CanTerminate()) {
          return false;
        }
        DynamicTagMatcher invalid_matcher = initial;
        if (invalid_matcher.Accept(invalid)) {
          return false;
        }
      }
      return true;
    }));
  }

  for (auto& future : futures) {
    EXPECT_TRUE(future.get());
  }
}

TEST(DynamicTagMatcherTest, CompiledGrammarIndexesAndForkScratchAreThreadSafe) {
  GrammarBuilder builder(Grammar::FromEBNF("root ::= [^]*"));
  builder.SetDynamicTagMatcherConfig(GetMiniMaxM3DynamicTagMatcherConfig());
  const std::vector<std::string> vocab = {"alpha_key", "beta_key", "wrong_key", ">"};
  const TokenizerInfo tokenizer_info(vocab, VocabType::RAW, std::nullopt, std::vector<int32_t>{});
  const auto compiled = GrammarCompiler(tokenizer_info, 4, false).CompileGrammar(builder.Get());

  constexpr int kNumMatchers = 16;
  GrammarMatcher initial(compiled, std::nullopt, true);
  std::vector<GrammarMatcher> matchers;
  matchers.reserve(kNumMatchers);
  for (int index = 0; index < kNumMatchers; ++index) {
    const std::string& name = index % 2 == 0 ? vocab[0] : vocab[1];
    auto matcher = initial.Fork();
    ASSERT_TRUE(
        matcher.AcceptString(std::string(kNamespace) + "<" + name + ">value" + kNamespace + "</")
    );
    matchers.push_back(std::move(matcher));
  }

  const int bitmask_size = GetBitmaskSize(tokenizer_info.GetVocabSize());
  std::vector<int32_t> bitmask_data(kNumMatchers * bitmask_size);
  int64_t shape[2] = {kNumMatchers, bitmask_size};
  DLTensor bitmask{};
  bitmask.data = bitmask_data.data();
  bitmask.device = DLDevice{kDLCPU, 0};
  bitmask.ndim = 2;
  bitmask.dtype = GetBitmaskDLType();
  bitmask.shape = shape;

  BatchGrammarMatcher batch_matcher(4);
  for (int iteration = 0; iteration < 50; ++iteration) {
    batch_matcher.BatchFillNextTokenBitmask(&matchers, &bitmask);
  }
  for (int index = 0; index < kNumMatchers; ++index) {
    std::vector<int> rejected;
    _DebugGetMaskedTokensFromBitmask(&rejected, bitmask, tokenizer_info.GetVocabSize(), index);
    const int expected_token_id = index % 2;
    EXPECT_EQ(std::find(rejected.begin(), rejected.end(), expected_token_id), rejected.end());
    EXPECT_NE(std::find(rejected.begin(), rejected.end(), 2), rejected.end());
  }

  std::vector<std::future<bool>> futures;
  futures.reserve(kNumMatchers);
  for (int index = 0; index < kNumMatchers; ++index) {
    futures.push_back(std::async(std::launch::async, [&matchers, bitmask_size, index] {
      std::vector<int32_t> local_data(bitmask_size);
      int64_t local_shape[2] = {1, bitmask_size};
      DLTensor local_bitmask{};
      local_bitmask.data = local_data.data();
      local_bitmask.device = DLDevice{kDLCPU, 0};
      local_bitmask.ndim = 2;
      local_bitmask.dtype = GetBitmaskDLType();
      local_bitmask.shape = local_shape;
      const int token_id = index % 2;
      for (int iteration = 0; iteration < 100; ++iteration) {
        matchers[index].FillNextTokenBitmask(&local_bitmask);
        if (!matchers[index].AcceptToken(token_id)) {
          return false;
        }
        matchers[index].Rollback();
      }
      return true;
    }));
  }
  for (auto& future : futures) {
    EXPECT_TRUE(future.get());
  }
}

}  // namespace
}  // namespace xgrammar
