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
  MockGrammar() = default;
  MockGrammar(std::size_t uuid) : uuid(uuid) {}
  auto MemorySize() const -> std::size_t { return 1; }
};

struct SizeEstimator {
  template <typename T>
  auto operator()(const T&) const -> std::size_t {
    return 1;
  }
};

using namespace std::chrono_literals;

struct Computer0 {
  inline static auto counter = std::atomic_size_t{};
  inline static auto sleep_time = 1000ms;
  auto operator()(std::size_t key) const -> MockGrammar {
    std::this_thread::sleep_for(sleep_time);  // simulate a slow operation
    return MockGrammar{counter++};
  }
};

constexpr auto kUnlimited = std::size_t(-1);

TEST(XGrammarParallelTest, CacheContention) {
  XGRAMMAR_LOG_INFO << "Testing the contention performance of the cache (no eviction)";
  constexpr auto kReadGroup = 20;
  const auto kNumThreads = int(std::thread::hardware_concurrency()) * 10;
  Computer0::sleep_time = 10s;

  // never evict
  auto cache = ThreadSafeLRUCache<std::size_t, MockGrammar, Computer0, SizeEstimator>{kUnlimited};

  auto futures = std::vector<std::future<std::size_t>>{};
  futures.reserve(kNumThreads);
  const auto tic = std::chrono::high_resolution_clock::now();
  const auto target = tic + 1s;

  for (int i = 0; i < kNumThreads; ++i) {
    futures.push_back(std::async(std::launch::async, [&] {
      std::this_thread::sleep_until(target);
      auto sum = std::size_t{};
      // write group: they should not compete with each other
      for (int j = 0; j < kNumThreads; ++j) {
        sum += cache.Get((i + j) % kNumThreads).uuid;
      }
      // read group: they should not compete with each other
      for (int k = 0; k < kReadGroup; ++k) {
        for (int j = 0; j < kNumThreads; ++j) {
          sum += cache.Get((i + j) % kNumThreads).uuid;
        }
      }
      return sum;
    }));
  }

  const auto kResult = std::size_t(kNumThreads) * (kNumThreads - 1) / 2 * (1 + kReadGroup);
  for (int i = 0; i < kNumThreads; ++i) EXPECT_EQ(futures[i].get(), kResult);

  const auto toc = std::chrono::high_resolution_clock::now();
  const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

  // remove 1s sleep time and computing sleep time
  const auto overhead = dur - 1s - Computer0::sleep_time;

  XGRAMMAR_LOG_INFO << "(1 write + " << kReadGroup << " reads) "
                    << "* " << kNumThreads << " threads | "
                    << "overhead = " << overhead.count() << "ms";

  // shouldn't exceed 2 times the compute + sleep time
  if (overhead > 1s + Computer0::sleep_time) {
    XGRAMMAR_LOG(FATAL) << "The overhead is too high, maybe the cache holds the lock too long?";
  }
}

TEST(XGrammarParallelTest, CacheEviction) {
  XGRAMMAR_LOG_INFO << "Testing the eviction performance of the cache (always evict)";
  constexpr auto kInsertGroup = 30;
  const auto kNumThreads = int(std::thread::hardware_concurrency()) * 10;
  Computer0::sleep_time = 1s;

  // always evict
  auto cache = ThreadSafeLRUCache<std::size_t, MockGrammar, Computer0, SizeEstimator>{0};

  auto futures = std::vector<std::future<std::size_t>>{};
  futures.reserve(kNumThreads);
  const auto tic = std::chrono::high_resolution_clock::now();
  const auto target = tic + 1s;

  for (int i = 0; i < kNumThreads; ++i) {
    futures.push_back(std::async(std::launch::async, [&] {
      std::this_thread::sleep_until(target);
      auto sum = std::size_t{};
      // each thread writes to a different key
      for (int j = 0; j < kInsertGroup; ++j) {
        sum += cache.Get(i * kInsertGroup + j).uuid;
      }
      return sum;
    }));
  }

  const auto kNumInsert = std::size_t(kNumThreads) * kInsertGroup;
  const auto kResult = kNumInsert * (kNumInsert - 1) / 2;

  auto sum = std::size_t{};
  for (int i = 0; i < kNumThreads; ++i) sum += futures[i].get();
  EXPECT_EQ(sum, kResult);

  const auto toc = std::chrono::high_resolution_clock::now();
  const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

  // remove 1s sleep time and computing sleep time
  const auto overhead = dur - 1s - Computer0::sleep_time * kInsertGroup;

  XGRAMMAR_LOG_INFO << "(" << kInsertGroup << " writes) "
                    << "* " << kNumThreads << " threads | "
                    << "overhead = " << overhead.count() << "ms";

  // shouldn't exceed 2 times the compute + sleep time
  if (overhead > 1s + Computer0::sleep_time * kInsertGroup) {
    XGRAMMAR_LOG(FATAL) << "The overhead is too high, maybe the cache holds the lock too long?";
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
  std::this_thread::sleep_for(1s);

  cache.Clear();

  for (auto i = 0; i < kNumThreads; ++i) {
    EXPECT_EQ(futures[i].get(), std::to_string(i));
  }
}

}  // namespace
