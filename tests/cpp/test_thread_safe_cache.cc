#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

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

namespace {

static_assert(
    sizeof(CompiledGrammar) >= sizeof(std::size_t),
    "Our test requires that CompiledGrammar is at least as large as std::size_t"
);

// simulate a CompiledGrammar object
struct MockGrammar {
  std::size_t uuid;
  std::byte padding[sizeof(CompiledGrammar) - sizeof(std::size_t)];
  auto MemorySize() const -> std::size_t { return 1; }
};

struct SizeEstimator {
  template <typename T>
  auto operator()(const T&) const -> std::size_t {
    return 1;
  }
};

using namespace std::chrono_literals;

// This test should cost about 2 * 4 * (log2(64) + 1) * 2s = 112s
TEST(XGrammarParallelTest, CacheContention) {
  const std::size_t kMaxRead = 64;

  XGRAMMAR_LOG_INFO << "Testing the contention performance of the cache";
  for (auto n = 4; n >= 1; --n) {
    for (auto m = kMaxRead; m >= 1; m /= 2) {
    }
  }
}

// This test should cost at most 4 * (log2(64) + 1) * (1s + 4s) = 140s
TEST(XGrammarParallelTest, CacheEviction) {
  const std::size_t kMaxThreads = std::max(std::thread::hardware_concurrency(), 8u) / 2;
  const std::size_t kMaxSize = std::min<std::size_t>(kMaxThreads, 64);

  XGRAMMAR_LOG_INFO << "Testing the eviction performance of the cache";
  for (auto n = 4; n >= 1; --n) {
    for (auto m = kMaxSize; m >= 1; m /= 2) {
    }
  }
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
  auto to_string() const -> std::string {
    this->check();
    return this->name;
  }
  auto MemorySize() const -> std::size_t {
    this->check();
    return 1;
  }
};

struct Computer1 {
  auto operator()(const TestObject& key) const -> TestObject {
    std::this_thread::sleep_for(5s);  // simulate a slow operation
    return TestObject{key};
  }
};

TEST(XGrammarParallelTest, CacheCorrectness) {
  auto cache =
      ThreadSafeLRUCache<std::string, TestObject, Computer1, SizeEstimator>{std::size_t(-1)};

  const auto kNumThreads = int(std::thread::hardware_concurrency()) * 10;
  auto futures = std::vector<std::future<std::string>>{};
  futures.reserve(kNumThreads);

  for (auto i = 0; i < kNumThreads; ++i) {
    futures.push_back(std::async(std::launch::async, [&cache, i] {
      return cache.Get(std::to_string(i)).to_string();
    }));
  }

  // Wait the futures to block
  std::this_thread::sleep_for(100ms);

  cache.Clear();

  for (auto i = 0; i < kNumThreads; ++i) {
    EXPECT_EQ(futures[i].get(), std::to_string(i));
  }
}

}  // namespace
