#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "support/logging.h"
#include "support/thread_safe_cache.h"

using namespace xgrammar;

static std::atomic_size_t counter{0};

static_assert(
    sizeof(CompiledGrammar) >= sizeof(std::size_t),
    "Our test requires that CompiledGrammar is at least as large as std::size_t"
);

// simulate a CompiledGrammar object
struct MockGrammar {
  std::size_t uuid;
  std::byte padding[sizeof(CompiledGrammar) - sizeof(std::size_t)];
};

using namespace std::chrono_literals;

TEST(XGrammarParallelTest, CacheEfficiency) {
  auto cache = ThreadSafeCache<std::string, MockGrammar>{[](const std::string&) {
    std::this_thread::sleep_for(1s);  // simulate a slow operation
    return MockGrammar{.uuid = counter++, .padding = {}};
  }};
  auto futures = std::vector<std::future<std::size_t>>{};

  static const auto kGroups = 20;
  static const auto kNumThreads = int(std::thread::hardware_concurrency()) * 2;
  static const auto kNumTests = kNumThreads / 2;

  futures.reserve(kNumThreads);
  const auto target = std::chrono::steady_clock::now() + 1s;

  // Whatever the execution order, the cache will only call the constructor for kNumTests times.
  // As a consequence, the sum of the uuids must be equal to the sum of the first kNumTests
  // integers.

  const auto tic = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < kNumThreads; ++i) {
    futures.push_back(std::async(std::launch::async, [&cache, target, i] {
      std::this_thread::sleep_until(target);
      auto sum = std::size_t{0};
      // Test writing to the cache concurrently
      for (auto j = 0; j < kNumTests; ++j) {
        const auto key = std::to_string((j + i) % kNumTests);
        sum += cache.Get(key).uuid;
      }
      // Test reading the same keys again
      for (auto j = 0; j < kNumTests * (kGroups - 1); ++j) {
        const auto key = std::to_string(j % kNumTests);
        sum += cache.Get(key).uuid;
      }
      return sum;
    }));
  }

  // Sum of [0, kNumTests) (I wish i'm not wrong)
  const auto kResult = kNumTests * (kNumTests - 1) / 2;

  for (auto& future : futures) {
    future.wait();
    EXPECT_EQ(future.get(), kResult * kGroups);
  }
  const auto toc = std::chrono::high_resolution_clock::now();
  // Skip the first 2s for preparation
  const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic - 2s).count();
  XGRAMMAR_LOG_INFO << "Duration: " << dur << "ms";
}

// A hook to ensure that the object will not be accessed after its destruction
struct LifeSpanHook {
 private:
  inline static std::unordered_set<const void*> manager{};
  inline static std::mutex mutex{};

  static auto unsafe_construct(const LifeSpanHook* ptr) -> void {
    // insert will return a pair of iterator and bool
    EXPECT_TRUE(manager.insert(ptr).second);
  }
  static auto unsafe_destruct(const LifeSpanHook* ptr) -> void {
    // erase will return 1 if the element is found and removed
    EXPECT_TRUE(manager.erase(ptr));
  }
  static auto unsafe_confirm(const LifeSpanHook* ptr) -> void {
    // ensure that the object is still alive
    EXPECT_TRUE(manager.find(ptr) != manager.end());
  }

 public:
  LifeSpanHook() {
    const auto lock = std::lock_guard{mutex};
    unsafe_construct(this);
  }
  LifeSpanHook(const LifeSpanHook& other) {
    const auto lock = std::lock_guard{mutex};
    unsafe_construct(this);
    unsafe_confirm(&other);
  }
  auto operator=(const LifeSpanHook& other) -> LifeSpanHook& {
    const auto lock = std::lock_guard{mutex};
    unsafe_confirm(this);
    unsafe_confirm(&other);
    return *this;
  }
  ~LifeSpanHook() {
    const auto lock = std::lock_guard{mutex};
    unsafe_destruct(this);
  }
  auto check() const -> void {
    const auto lock = std::lock_guard{mutex};
    unsafe_confirm(this);
  }
};

struct TestObject : LifeSpanHook {
 private:
  std::string name;

 public:
  TestObject() = default;
  TestObject(std::string name) : name(std::move(name)) {}
  auto& operator=(std::string name) {
    this->check();
    this->name = std::move(name);
    return *this;
  }
  operator std::string() const {
    this->check();
    return this->name;
  }
};

TEST(XGrammarParallelTest, CacheCorrectness) {
  auto cache = ThreadSafeCache<std::string, TestObject>{[](const std::string& key) {
    std::this_thread::sleep_for(1s);  // simulate a slow operation
    return key;
  }};

  auto futures = std::vector<std::future<std::string>>{};
  futures.reserve(20);

  for (auto i = 0; i < 20; ++i) {
    futures.push_back(std::async(std::launch::async, [&cache, i] {
      return std::string(cache.Get(std::to_string(i)));
    }));
  }

  // Wait the futures to block
  std::this_thread::sleep_for(100ms);

  cache.Clear();

  for (auto i = 0; i < 20; ++i) {
    EXPECT_EQ(futures[i].get(), std::to_string(i));
  }
}
