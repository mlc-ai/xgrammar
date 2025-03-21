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
  auto MemorySize() const -> std::size_t { return 1; }
};

using namespace std::chrono_literals;

struct LRUPolicy0 {
  std::atomic_size_t counter{0};
  template <typename KeyType>
  auto compute(const KeyType&) -> MockGrammar {
    std::this_thread::sleep_for(1s);  // simulate a slow operation
    MockGrammar g{};
    g.uuid = counter++;
    return g;
  }
};

template <bool use_lru>
auto test_contention(std::size_t num_threads, std::size_t num_tests, std::size_t num_reads)
    -> void {
  ASSERT_GE(num_threads, num_tests);
  ASSERT_GE(num_tests, 1);
  auto cache = [=] {
    if constexpr (use_lru) {
      return ThreadSafeLRUCache<LRUPolicy0, MockGrammar, std::string>{num_tests};
    } else {
      static std::atomic_size_t counter{0};
      counter = 0;
      return ThreadSafeCache<std::string, MockGrammar>{
          [&](const std::string& key) {
            std::this_thread::sleep_for(1s);  // simulate a slow operation
            MockGrammar g{};
            g.uuid = counter++;
            return g;
          },
      };
    };
  }();
  auto futures = std::vector<std::future<std::size_t>>{};

  futures.reserve(num_threads);
  const auto target = std::chrono::steady_clock::now() + 1s;

  // Whatever the execution order, the cache will only call the constructor for kNumTests times.
  // As a consequence, the sum of the uuids must be equal to the sum of the first kNumTests
  // integers.

  const auto tic = std::chrono::high_resolution_clock::now();

  for (std::size_t i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [=, &cache] {
      std::this_thread::sleep_until(target);
      auto sum = std::size_t{0};
      // Test writing to the cache concurrently
      for (std::size_t j = 0; j < num_tests; ++j) {
        const auto key = std::to_string((j + i) % num_tests);
        if constexpr (use_lru) {
          sum += cache.template Get<std::string>(key).uuid;
        } else {
          sum += cache.Get(key).uuid;
        }
      }
      // Test reading the same keys again
      for (std::size_t j = 0; j < num_tests * num_reads; ++j) {
        const auto key = std::to_string(j % num_tests);
        if constexpr (use_lru) {
          sum += cache.template Get<std::string>(key).uuid;
        } else {
          sum += cache.Get(key).uuid;
        }
      }
      return sum;
    }));
  }

  // Sum of [0, kNumTests) * (num_reads + 1)
  const auto kExpected = num_tests * (num_tests - 1) / 2 * (num_reads + 1);
  for (auto& future : futures) {
    future.wait();
    EXPECT_EQ(future.get(), kExpected);
  }
  if constexpr (use_lru) {
    EXPECT_EQ(cache.GetPolicy().counter, num_tests);
  }

  const auto toc = std::chrono::high_resolution_clock::now();
  // Skip the first 1s sleeping time, and another 1s for the computation
  const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic - 2s).count();

  XGRAMMAR_LOG_INFO << "  Contention Settings:"
                    << " | Use LRU: " << (use_lru ? " true" : "false")  // start a new line
                    << " | Read: " << num_reads << " | Threads: " << num_threads
                    << " | Elements: " << num_tests << " | Overhead: " << dur << "ms";
}

auto test_eviction(std::size_t num_threads, std::size_t num_inserts, std::size_t max_size) -> void {
  ASSERT_GE(num_inserts, 1);

  auto cache = ThreadSafeLRUCache<LRUPolicy0, MockGrammar, std::string>{max_size};
  auto futures = std::vector<std::future<std::size_t>>{};

  // Whatever the execution order, the cache will call the constructor for kNumTests * kGroup times.
  // And there will be some eviction happening.

  futures.reserve(num_threads);

  const auto target = std::chrono::steady_clock::now() + 1s;
  const auto tic = std::chrono::high_resolution_clock::now();

  for (std::size_t i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [=, &cache] {
      std::this_thread::sleep_until(target);
      auto sum = std::size_t{0};
      // unique key for each thread
      for (std::size_t j = 0; j < num_inserts; ++j) {
        const auto key = std::to_string(j + i * num_inserts);
        sum += cache.Get<std::string>(key).uuid;
      }
      return sum;
    }));
  }

  const auto kNewEntry = num_inserts * num_threads;
  const auto kExpected = kNewEntry * (kNewEntry - 1) / 2;
  auto sum = std::size_t{0};
  for (auto& future : futures) {
    sum += future.get();
  }
  EXPECT_EQ(sum, kExpected);
  EXPECT_EQ(cache.GetPolicy().counter, kNewEntry);

  const auto toc = std::chrono::high_resolution_clock::now();
  // Skip the first 1s sleeping time, and another num_tests * 1s for the computation
  const auto dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic - 1s * (num_inserts + 1))
          .count();
  XGRAMMAR_LOG_INFO << "  Eviction Settings:"
                    << " | Max Size: " << max_size << " | Threads: " << num_threads
                    << " | Elements: " << kNewEntry << " | Overhead: " << dur << "ms";
}

// This test should cost about 2 * 4 * (log2(64) + 1) * 2s = 112s
TEST(XGrammarParallelTest, CacheContention) {
  const std::size_t kMaxTest = std::max(std::thread::hardware_concurrency(), 8u) / 2;
  const std::size_t kMaxRead = 64;

  XGRAMMAR_LOG_INFO << "Testing the contention performance of the cache";
  for (auto n = 4; n >= 1; --n) {
    for (auto m = kMaxRead; m >= 1; m /= 2) {
      test_contention<true>(n * kMaxTest, kMaxTest, m);
      test_contention<false>(n * kMaxTest, kMaxTest, m);
    }
  }
}

// This test should cost at most 4 * (log2(64) + 1) * (1s + 4s) = 140s
TEST(XGrammarParallelTest, CacheEviction) {
  const std::size_t kMaxThreads = std::max(std::thread::hardware_concurrency(), 8u) / 2;
  const std::size_t kInserts = 4;
  const std::size_t kMaxSize = std::min<std::size_t>(kMaxThreads, 64);

  XGRAMMAR_LOG_INFO << "Testing the eviction performance of the cache";
  for (auto n = 4; n >= 1; --n) {
    for (auto m = kMaxSize; m >= 1; m /= 2) {
      test_eviction(n * kMaxThreads, kInserts, m);
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

struct LRUPolicy1 {
  template <typename KeyType>
  auto compute(const KeyType& key) -> TestObject {
    std::this_thread::sleep_for(5s);  // simulate a slow operation
    return TestObject{key};
  }
};

TEST(XGrammarParallelTest, CacheCorrectness) {
  auto cache = ThreadSafeLRUCache<LRUPolicy1, TestObject, std::string>{std::size_t(-1)};

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
