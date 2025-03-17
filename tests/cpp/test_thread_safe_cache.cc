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

namespace {

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

struct LRUPolicy0 {
  // The interface of the policy
  std::atomic_size_t counter{0};
  const std::size_t max_size;
  LRUPolicy0(std::size_t max_size) : max_size(max_size) {}
  template <typename KeyType>
  auto compute(const KeyType&) -> MockGrammar {
    std::this_thread::sleep_for(1s);  // simulate a slow operation
    MockGrammar g{};
    g.uuid = counter++;
    return g;
  }
  auto should_evict(std::size_t cur_size) -> bool { return cur_size > max_size; }
  static auto size(const MockGrammar&) -> std::size_t { return 1; }
};

auto test_performance(std::size_t max_size, std::size_t num_threads, std::size_t num_tests)
    -> void {
  auto cache = ThreadSafeLRUCache<LRUPolicy0, MockGrammar, std::string>{max_size};
  auto futures = std::vector<std::future<std::size_t>>{};

  static constexpr std::size_t kReadGroup = 20;

  const std::size_t kNumThreads = num_threads;
  const std::size_t kNumTests = num_tests;

  futures.reserve(kNumThreads);
  const auto target = std::chrono::steady_clock::now() + 1s;

  // Whatever the execution order, the cache will only call the constructor for kNumTests times.
  // As a consequence, the sum of the uuids must be equal to the sum of the first kNumTests
  // integers.

  const auto tic = std::chrono::high_resolution_clock::now();
  for (std::size_t i = 0; i < kNumThreads; ++i) {
    futures.push_back(std::async(std::launch::async, [&cache, target, i, kNumTests] {
      std::this_thread::sleep_until(target);
      auto sum = std::size_t{0};
      // Test writing to the cache concurrently
      for (std::size_t j = 0; j < kNumTests; ++j) {
        const auto key = std::to_string((j + i) % kNumTests);
        sum += cache.Get<std::string>(key).uuid;
      }
      // Test reading the same keys again
      for (std::size_t j = 0; j < kNumTests * kReadGroup; ++j) {
        const auto key = std::to_string(j % kNumTests);
        sum += cache.Get<std::string>(key).uuid;
      }
      return sum;
    }));
  }

  // Sum of [0, kNumTests) (I wish i'm not wrong)
  const auto kResult = kNumTests * (kNumTests - 1) / 2;

  for (auto& future : futures) {
    future.wait();
    if (max_size >= kNumTests)  // no eviction in this case
      EXPECT_EQ(future.get(), kResult * kReadGroup);
  }
  const auto toc = std::chrono::high_resolution_clock::now();

  // Skip the first 1s sleeping time, and another 1s for the computation
  const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic - 2s).count();
  auto max_num = max_size == std::size_t(-1) ? "+inf" : std::to_string(max_size);
  XGRAMMAR_LOG_INFO << "  Setting: max_elements=" << max_num << ", num_threads=" << num_threads;
  XGRAMMAR_LOG_INFO << "  Duration: " << dur << "ms";
}

TEST(XGrammarParallelTest, CacheEfficiency) {
  const auto kMaxThreads = std::size_t(std::thread::hardware_concurrency()) * 2;
  const auto kNumTests = kMaxThreads / 2;
  XGRAMMAR_LOG_INFO << "Testing the contention of the cache";
  for (auto n = kMaxThreads; n > 0; n /= 2) {
    test_performance(std::size_t(-1), n, kNumTests);
  }
  XGRAMMAR_LOG_INFO << "Testing the eviction of the cache";
  for (auto n = kMaxThreads; n > 0; n /= 2) {
    test_performance(10, n, kNumTests);
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
};

struct LRUPolicy1 {
  // The interface of the policy
  std::atomic_size_t counter{0};
  template <typename KeyType>
  auto compute(const KeyType& key) -> TestObject {
    std::this_thread::sleep_for(5s);  // simulate a slow operation
    return TestObject{key};
  }
  auto should_evict(std::size_t cur_size) -> bool { return false; }
  static auto size(const MockGrammar&) -> std::size_t { return 1; }
};

TEST(XGrammarParallelTest, CacheCorrectness) {
  auto cache = ThreadSafeLRUCache<LRUPolicy1, TestObject, std::string>{};

  const auto kNumThreads = int(std::thread::hardware_concurrency()) * 10;
  auto futures = std::vector<std::future<std::string>>{};
  futures.reserve(kNumThreads);

  for (auto i = 0; i < kNumThreads; ++i) {
    futures.push_back(std::async(std::launch::async, [&cache, i] {
      return cache.Get<std::string>(std::to_string(i)).to_string();
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
