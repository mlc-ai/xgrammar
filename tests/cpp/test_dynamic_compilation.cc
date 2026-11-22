#include <gtest/gtest.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

#include <atomic>
#include <chrono>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "compiled_grammar_impl.h"
#include "grammar_functor.h"

namespace xgrammar {
namespace {

std::vector<int32_t> Mask(GrammarMatcher* matcher, int vocab_size) {
  int64_t shape[] = {1, GetBitmaskSize(vocab_size)};
  std::vector<int32_t> storage(shape[1], -1);
  DLTensor tensor{};
  tensor.data = storage.data();
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 2;
  tensor.dtype = GetBitmaskDLType();
  tensor.shape = shape;
  matcher->FillNextTokenBitmask(&tensor);
  return storage;
}

TEST(DynamicCompilation, RuleCacheSharedAcrossGrammarsOutlivesCompiler) {
  TokenizerInfo info({"a", "b", "c", "ab", "ac", "x"});
  // A finite limit disables whole-grammar reuse, but rule-level reuse remains enabled.
  std::optional<GrammarCompiler> compiler(std::in_place, info, 1, true, 65536, true);
  auto first = compiler->CompileRegex("ab|ac");
  auto second = compiler->CompileRegex("ab|ac");
  ASSERT_NE(first.ImplPtr(), second.ImplPtr());
  ASSERT_NE(first->rule_level_cache, nullptr);
  ASSERT_NE(second->rule_level_cache, nullptr);
  EXPECT_EQ(first->rule_level_cache->ImplPtr(), second->rule_level_cache->ImplPtr());

  compiler.reset();
  GrammarCompiler eager_compiler(info, 1, false, -1);
  auto eager = eager_compiler.CompileRegex("ab|ac");
  GrammarMatcher expected(eager);
  GrammarMatcher first_matcher(first);
  GrammarMatcher second_matcher(second);
  const auto initial_mask = Mask(&expected, info.GetVocabSize());
  EXPECT_EQ(Mask(&first_matcher, info.GetVocabSize()), initial_mask);
  EXPECT_EQ(Mask(&second_matcher, info.GetVocabSize()), initial_mask);
}

TEST(DynamicCompilation, ConcurrentMasksSerializationAndAccounting) {
  TokenizerInfo info({"a", "b", "c", "ab", "ac", "x"});
  GrammarCompiler eager_compiler(info, 1, false, -1);
  GrammarCompiler compiler(info, 1, true, 65536, true);
  auto eager = eager_compiler.CompileRegex("ab|ac");
  auto dynamic = compiler.CompileRegex("ab|ac");
  GrammarMatcher reference(eager);
  const auto expected = Mask(&reference, info.GetVocabSize());
  ASSERT_TRUE(reference.AcceptToken(0));
  const auto expected_after_a = Mask(&reference, info.GetVocabSize());

  std::atomic<bool> start{false};
  std::vector<std::future<void>> workers;
  for (int worker = 0; worker < 8; ++worker) {
    workers.push_back(std::async(std::launch::async, [&, worker] {
      while (!start.load()) {
        std::this_thread::yield();
      }
      for (int i = 0; i < 20; ++i) {
        if (worker == 0) {
          auto restored = CompiledGrammar::DeserializeJSON(dynamic.SerializeJSON(), info);
          ASSERT_TRUE(std::holds_alternative<CompiledGrammar>(restored));
          GrammarMatcher matcher(std::get<CompiledGrammar>(restored));
          EXPECT_EQ(Mask(&matcher, info.GetVocabSize()), expected);
        } else if (worker == 1) {
          EXPECT_GT(dynamic.MemorySizeBytes(), 0u);
          EXPECT_LE(compiler.GetCacheSizeBytes(), 65536);
          compiler.ClearCache();
        } else {
          GrammarMatcher matcher(dynamic);
          EXPECT_EQ(Mask(&matcher, info.GetVocabSize()), expected);
          ASSERT_TRUE(matcher.AcceptToken(0));
          EXPECT_EQ(Mask(&matcher, info.GetVocabSize()), expected_after_a);
        }
      }
    }));
  }
  start.store(true);
  for (auto& worker : workers) {
    worker.get();
  }
}

// A test-only hook makes cache misses overlap deterministically. Timeouts are failure bounds,
// not synchronization: successful runs advance only after the expected events arrive.
class MaskBuildGate {
 public:
  void Visit(const ParserState&, bool builder) {
    std::unique_lock<std::mutex> lock(mutex);
    if (!builder) {
      ++waiters;
      changed.notify_all();
      return;
    }
    ++builders;
    changed.notify_all();
    if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return released; })) {
      throw std::runtime_error("builder gate timed out");
    }
    if (fail) {
      throw std::runtime_error("injected mask failure");
    }
  }

  bool Wait(int expected_builders, int expected_waiters) {
    std::unique_lock<std::mutex> lock(mutex);
    return changed.wait_for(lock, std::chrono::seconds(10), [&] {
      return builders >= expected_builders && waiters >= expected_waiters;
    });
  }

  void Release(bool should_fail = false) {
    std::lock_guard<std::mutex> lock(mutex);
    fail = should_fail;
    released = true;
    changed.notify_all();
  }

  std::mutex mutex;
  std::condition_variable changed;
  int builders{0};
  int waiters{0};
  bool released{false};
  bool fail{false};
};

TEST(DynamicCompilation, EquivalentColdKeysShareOneBuilder) {
  TokenizerInfo info({"a", "b", "ab", ":", "1", "12", "x"});
  GrammarCompiler eager_compiler(info, 1, false, -1);
  GrammarCompiler compiler(info, 1, false, -1, true);
  auto eager = eager_compiler.CompileRegex("[ab]+:[0-9]+");
  auto dynamic = compiler.CompileRegex("[ab]+:[0-9]+");
  ASSERT_FALSE(eager->adaptive_token_mask_cache.empty());
  auto state = eager->adaptive_token_mask_cache.begin()->first;
  MaskBuildGate gate;
  dynamic->mask_build_test_hook = [&](const ParserState& s, bool builder) {
    gate.Visit(s, builder);
  };
  std::vector<std::future<const AdaptiveTokenMask*>> workers;
  for (int i = 0; i < 8; ++i) {
    auto equivalent = state;
    equivalent.rule_start_pos = i;
    equivalent.repeat_count = i;
    equivalent.partial_codepoint = i;
    workers.push_back(std::async(std::launch::async, [&, equivalent] {
      return &dynamic->GetAdaptiveTokenMask(equivalent);
    }));
  }
  EXPECT_TRUE(gate.Wait(1, 7));
  gate.Release();
  const auto* result = workers.front().get();
  for (size_t i = 1; i < workers.size(); ++i) {
    EXPECT_EQ(workers[i].get(), result);
  }
  EXPECT_EQ(gate.builders, 1);
  EXPECT_EQ(result->Print(info), eager->GetAdaptiveTokenMask(state).Print(info));
  EXPECT_TRUE(dynamic->pending_masks.empty());
  EXPECT_EQ(&dynamic->GetAdaptiveTokenMask(state), result);
  EXPECT_EQ(gate.builders, 1);
}

TEST(DynamicCompilation, DifferentKeysOverlapAndWarmHitsDoNotWait) {
  TokenizerInfo info({"a", "b", "ab", ":", "1", "12", "x"});
  GrammarCompiler eager_compiler(info, 1, false, -1);
  GrammarCompiler compiler(info, 1, false, -1, true);
  auto eager = eager_compiler.CompileRegex("[ab]+:[0-9]+");
  auto dynamic = compiler.CompileRegex("[ab]+:[0-9]+");
  std::vector<ParserState> states;
  for (const auto& entry : eager->adaptive_token_mask_cache) {
    states.push_back(entry.first);
  }
  ASSERT_GE(states.size(), 3u);
  const auto* warm = &dynamic->GetAdaptiveTokenMask(states[0]);
  MaskBuildGate gate;
  dynamic->mask_build_test_hook = [&](const ParserState& s, bool builder) {
    gate.Visit(s, builder);
  };
  auto first =
      std::async(std::launch::async, [&] { return &dynamic->GetAdaptiveTokenMask(states[1]); });
  auto second =
      std::async(std::launch::async, [&] { return &dynamic->GetAdaptiveTokenMask(states[2]); });
  EXPECT_TRUE(gate.Wait(2, 0));
  auto hit =
      std::async(std::launch::async, [&] { return &dynamic->GetAdaptiveTokenMask(states[0]); });
  EXPECT_EQ(hit.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  gate.Release();
  EXPECT_EQ(hit.get(), warm);
  EXPECT_EQ(first.get()->Print(info), eager->GetAdaptiveTokenMask(states[1]).Print(info));
  EXPECT_EQ(second.get()->Print(info), eager->GetAdaptiveTokenMask(states[2]).Print(info));
  EXPECT_EQ(gate.builders, 2);
  EXPECT_TRUE(dynamic->pending_masks.empty());
}

TEST(DynamicCompilation, FailedBuilderReleasesWaitersAndAllowsRetry) {
  TokenizerInfo info({"a", "b", "ab", "ac", "x"});
  GrammarCompiler eager_compiler(info, 1, false, -1);
  GrammarCompiler compiler(info, 1, false, -1, true);
  auto eager = eager_compiler.CompileRegex("ab|ac");
  auto dynamic = compiler.CompileRegex("ab|ac");
  ASSERT_FALSE(eager->adaptive_token_mask_cache.empty());
  const auto state = eager->adaptive_token_mask_cache.begin()->first;
  MaskBuildGate gate;
  dynamic->mask_build_test_hook = [&](const ParserState& s, bool builder) {
    gate.Visit(s, builder);
  };
  auto request = [&] {
    try {
      dynamic->GetAdaptiveTokenMask(state);
      return std::string("unexpected success");
    } catch (const std::runtime_error& error) {
      return std::string(error.what());
    }
  };
  std::vector<std::future<std::string>> workers;
  for (int i = 0; i < 8; ++i) {
    workers.push_back(std::async(std::launch::async, request));
  }
  EXPECT_TRUE(gate.Wait(1, 7));
  // Retain the original record just as a delayed waiter would, across removal and retry.
  std::shared_ptr<CompiledGrammar::Impl::PendingMask> failed;
  {
    std::lock_guard<std::mutex> lock(dynamic->adaptive_token_mask_cache_mutex);
    failed = dynamic->pending_masks.at(state);
  }
  gate.Release(true);
  for (auto& worker : workers) {
    EXPECT_EQ(worker.get(), "injected mask failure");
  }
  EXPECT_EQ(gate.builders, 1);
  EXPECT_TRUE(dynamic->pending_masks.empty());
  EXPECT_TRUE(dynamic->adaptive_token_mask_cache.empty());
  dynamic->mask_build_test_hook = {};
  const auto& retried = dynamic->GetAdaptiveTokenMask(state);
  EXPECT_EQ(retried.Print(info), eager->GetAdaptiveTokenMask(state).Print(info));
  EXPECT_TRUE(failed->done);
  EXPECT_NE(failed->error, nullptr);
  EXPECT_EQ(failed->result, nullptr);
  EXPECT_TRUE(dynamic->pending_masks.empty());
  // Materialization and serialization use the same coordinated lookup path.
  auto restored = CompiledGrammar::DeserializeJSON(dynamic.SerializeJSON(), info);
  ASSERT_TRUE(std::holds_alternative<CompiledGrammar>(restored));
  GrammarMatcher reference(eager);
  GrammarMatcher restored_matcher(std::get<CompiledGrammar>(restored));
  EXPECT_EQ(Mask(&restored_matcher, info.GetVocabSize()), Mask(&reference, info.GetVocabSize()));
  ASSERT_TRUE(reference.AcceptToken(0));
  ASSERT_TRUE(restored_matcher.AcceptToken(0));
  EXPECT_EQ(Mask(&restored_matcher, info.GetVocabSize()), Mask(&reference, info.GetVocabSize()));
}

// Memory inspection must remain safe while missing masks are generated.
TEST(DynamicCompilation, MemorySizeIsSafeDuringMaskGeneration) {
  std::string grammar = "root ::= r0\n";
  constexpr int kRules = 512;
  for (int i = 0; i < kRules; ++i) {
    grammar += "r" + std::to_string(i) + " ::= [ab]";
    if (i + 1 < kRules) {
      grammar += " r" + std::to_string(i + 1);
    }
    grammar += " | \"\"\n";
  }

  TokenizerInfo tokenizer_info({"a", "b", "ab", "ba"});
  GrammarCompiler compiler(
      tokenizer_info,
      /*max_threads=*/1,
      /*cache_enabled=*/false,
      /*max_memory_bytes=*/-1,
      /*enable_dynamic_compilation=*/true
  );
  auto compiled = compiler.CompileGrammar(grammar);
  const auto initial_size = compiled.MemorySizeBytes();
  GrammarMatcher matcher(compiled);

  std::atomic<bool> reader_ready{false};
  std::atomic<bool> start{false};
  std::atomic<bool> done{false};
  std::atomic<bool> observed_zero_size{false};
  std::atomic<int> observations{0};
  std::thread reader([&]() {
    reader_ready.store(true, std::memory_order_release);
    while (!start.load(std::memory_order_acquire)) {
    }
    while (!done.load(std::memory_order_acquire)) {
      if (compiled.MemorySizeBytes() == 0) {
        observed_zero_size.store(true, std::memory_order_relaxed);
      }
      observations.fetch_add(1, std::memory_order_relaxed);
    }
  });
  while (!reader_ready.load(std::memory_order_acquire)) {
  }
  start.store(true, std::memory_order_release);
  while (observations.load(std::memory_order_relaxed) == 0) {
    std::this_thread::yield();
  }
  // Generate missing masks while the reader measures the compiled grammar.
  const auto mask = Mask(&matcher, tokenizer_info.GetVocabSize());
  done.store(true, std::memory_order_release);
  reader.join();

  EXPECT_FALSE(observed_zero_size.load(std::memory_order_relaxed));
  EXPECT_GT(compiled.MemorySizeBytes(), initial_size);
  EXPECT_EQ(mask[0] & 0xf, 0xf);
}

}  // namespace
}  // namespace xgrammar
