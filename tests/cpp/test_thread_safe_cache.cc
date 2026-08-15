#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "support/logging.h"
#include "support/thread_safe_cache.h"

using namespace xgrammar;

namespace {

// static_assert(
//     sizeof(CompiledGrammar) >= sizeof(std::size_t),
//     "Our test requires that CompiledGrammar is at least as large as std::size_t"
// );

// // simulate a CompiledGrammar object
// struct MockGrammar {
//   std::size_t uuid;
//   std::byte padding[sizeof(CompiledGrammar) - sizeof(std::size_t)];
//   MockGrammar() = default;
//   MockGrammar(std::size_t uuid) : uuid(uuid) {}
// };

// struct SizeEstimator {
//   template <typename T>
//   std::size_t operator()(const T&) const {
//     return 1;
//   }
// };

// using namespace std::chrono_literals;

// struct Computer0 {
//   inline static auto counter = std::atomic_size_t{};
//   inline static constexpr auto kSleepTime = 1000ms;
//   MockGrammar operator()(std::size_t key) const {
//     std::this_thread::sleep_for(kSleepTime);  // simulate a slow operation
//     return MockGrammar{counter++};
//   }
// };

// constexpr auto kUnlimited = std::size_t(-1);
// constexpr auto kOverheadRatio = 0.1;

// TEST(XGrammarParallelTest, CacheContention) {
//   XGRAMMAR_LOG_INFO << "Testing the contention performance of the cache (no eviction)";
//   constexpr auto kReadGroup = 8;
//   const auto kNumThreads = int(std::thread::hardware_concurrency()) * 4;

//   // never evict
//   auto cache = ThreadSafeLRUCache<std::size_t, MockGrammar, Computer0,
//   SizeEstimator>{kUnlimited};

//   auto futures = std::vector<std::future<std::size_t>>{};
//   futures.reserve(kNumThreads);
//   const auto tic = std::chrono::high_resolution_clock::now();
//   const auto target = tic + 1s;

//   for (int i = 0; i < kNumThreads; ++i) {
//     futures.push_back(std::async(std::launch::async, [=, &cache] {
//       std::this_thread::sleep_until(target);
//       auto sum = std::size_t{};
//       // write group: they should not compete with each other
//       for (int j = 0; j < kNumThreads; ++j) {
//         sum += cache.Get((i + j) % kNumThreads).uuid;
//       }
//       // read group: they should not compete with each other
//       for (int k = 0; k < kReadGroup; ++k) {
//         for (int j = 0; j < kNumThreads; ++j) {
//           sum += cache.Get((i + j) % kNumThreads).uuid;
//         }
//       }
//       return sum;
//     }));
//   }

//   const auto kResult = std::size_t(kNumThreads) * (kNumThreads - 1) / 2 * (1 + kReadGroup);
//   for (int i = 0; i < kNumThreads; ++i) EXPECT_EQ(futures[i].get(), kResult);

//   const auto toc = std::chrono::high_resolution_clock::now();
//   const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

//   // remove 1s sleep time and computing sleep time
//   const auto overhead = dur - 1s - Computer0::kSleepTime;

//   XGRAMMAR_LOG_INFO << "(1 write + " << kReadGroup << " reads) "
//                     << "* " << kNumThreads << " threads | "
//                     << "overhead = " << overhead.count() << "ms";

//   if (overhead > kOverheadRatio * kNumThreads * Computer0::kSleepTime + 1s) {
//     XGRAMMAR_LOG(WARNING) << "The overhead is too high, maybe the cache holds the lock too
//     long?";
//   }
// }

// TEST(XGrammarParallelTest, CacheEviction) {
//   XGRAMMAR_LOG_INFO << "Testing the eviction performance of the cache (always evict)";
//   constexpr auto kInsertGroup = 8;
//   const auto kNumThreads = int(std::thread::hardware_concurrency()) * 4;

//   // always evict
//   auto cache = ThreadSafeLRUCache<std::size_t, MockGrammar, Computer0, SizeEstimator>{0};

//   auto futures = std::vector<std::future<std::size_t>>{};
//   futures.reserve(kNumThreads);
//   const auto tic = std::chrono::high_resolution_clock::now();
//   const auto target = tic + 1s;

//   for (int i = 0; i < kNumThreads; ++i) {
//     futures.push_back(std::async(std::launch::async, [=, &cache] {
//       std::this_thread::sleep_until(target);
//       auto sum = std::size_t{};
//       // each thread writes to a different key
//       for (int j = 0; j < kInsertGroup; ++j) {
//         sum += cache.Get(i * kInsertGroup + j).uuid;
//       }
//       return sum;
//     }));
//   }

//   const auto kNumInsert = std::size_t(kNumThreads) * kInsertGroup;
//   const auto kResult = kNumInsert * (kNumInsert - 1) / 2;

//   auto sum = std::size_t{};
//   for (int i = 0; i < kNumThreads; ++i) sum += futures[i].get();
//   EXPECT_EQ(sum, kResult);

//   const auto toc = std::chrono::high_resolution_clock::now();
//   const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

//   // remove 1s sleep time and computing sleep time
//   const auto overhead = dur - 1s - Computer0::kSleepTime * kInsertGroup;

//   XGRAMMAR_LOG_INFO << "(" << kInsertGroup << " writes) "
//                     << "* " << kNumThreads << " threads | "
//                     << "overhead = " << overhead.count() << "ms";

//   // shouldn't exceed compute + sleep time
//   if (overhead > kOverheadRatio * Computer0::kSleepTime * kNumThreads + 1s) {
//     XGRAMMAR_LOG(WARNING) << "The overhead is too high, maybe the cache holds the lock too
//     long?";
//   }
// }

// // A hook to ensure that the object will not be accessed after its destruction
// struct LifeSpanHook {
//  private:
//   inline static std::unordered_set<const void*> manager{};
//   inline static std::mutex mutex{};

//   static void unsafe_construct(const LifeSpanHook* ptr) {
//     // insert will return a pair of iterator and bool
//     EXPECT_TRUE(manager.insert(ptr).second);
//   }
//   static void unsafe_destruct(const LifeSpanHook* ptr) {
//     // erase will return 1 if the element is found and removed
//     EXPECT_TRUE(manager.erase(ptr));
//   }
//   static void unsafe_confirm(const LifeSpanHook* ptr) {
//     // ensure that the object is still alive
//     EXPECT_TRUE(manager.find(ptr) != manager.end());
//   }

//  public:
//   LifeSpanHook() {
//     const auto lock = std::lock_guard{mutex};
//     unsafe_construct(this);
//   }
//   LifeSpanHook(const LifeSpanHook& other) {
//     const auto lock = std::lock_guard{mutex};
//     unsafe_construct(this);
//     unsafe_confirm(&other);
//   }
//   LifeSpanHook& operator=(const LifeSpanHook& other) {
//     const auto lock = std::lock_guard{mutex};
//     unsafe_confirm(this);
//     unsafe_confirm(&other);
//     return *this;
//   }
//   ~LifeSpanHook() {
//     const auto lock = std::lock_guard{mutex};
//     unsafe_destruct(this);
//   }
//   void check() const {
//     const auto lock = std::lock_guard{mutex};
//     unsafe_confirm(this);
//   }
// };

// struct TestObject : LifeSpanHook {
//  private:
//   std::string name;

//  public:
//   TestObject() = default;
//   TestObject(std::string name) : name(std::move(name)) {}
//   TestObject& operator=(std::string name) {
//     this->check();
//     this->name = std::move(name);
//     return *this;
//   }
//   std::string to_string() const {
//     this->check();
//     return this->name;
//   }
//   std::size_t MemorySize() const {
//     this->check();
//     return 1;
//   }
// };

// struct Computer1 {
//   TestObject operator()(const TestObject& key) const {
//     std::this_thread::sleep_for(5s);  // simulate a slow operation
//     return TestObject{key};
//   }
// };

// TEST(XGrammarParallelTest, CacheCorrectness) {
//   auto cache = ThreadSafeLRUCache<std::string, TestObject, Computer1, SizeEstimator>{kUnlimited};

//   const auto kNumThreads = int(std::thread::hardware_concurrency()) * 16;
//   auto futures = std::vector<std::future<std::string>>{};
//   futures.reserve(kNumThreads);

//   for (auto i = 0; i < kNumThreads; ++i) {
//     futures.push_back(std::async(std::launch::async, [&cache, i] {
//       return cache.Get(std::to_string(-i)).to_string();
//     }));
//   }

//   // Wait the futures to block
//   std::this_thread::sleep_for(1s);

//   cache.Clear();

//   for (auto i = 0; i < kNumThreads; ++i) {
//     EXPECT_EQ(futures[i].get(), std::to_string(-i));
//   }
// }

struct CacheValue {
  int payload;
  std::size_t size;
};

struct CacheValueSizeEstimator {
  std::size_t operator()(const CacheValue& value) const { return value.size; }
};

struct CountingCacheComputer {
  std::shared_ptr<std::atomic_size_t> count;

  CacheValue operator()(int key) const {
    ++*count;
    return CacheValue{key, static_cast<std::size_t>(key)};
  }
};

TEST(XGrammarThreadSafeLRUCacheTest, UnlimitedClearResetsMemorySize) {
  auto count = std::make_shared<std::atomic_size_t>(0);
  auto cache = ThreadSafeLRUCache<int, CacheValue, CountingCacheComputer, CacheValueSizeEstimator>{
      ThreadSafeLRUCache<int, CacheValue, CountingCacheComputer, CacheValueSizeEstimator>::
          kUnlimitedSize,
      CountingCacheComputer{count}
  };

  EXPECT_EQ(cache.Get(7).payload, 7);
  EXPECT_EQ(cache.MemorySize(), 7);
  EXPECT_EQ(*count, 1);
  cache.Clear();
  EXPECT_EQ(cache.MemorySize(), 0);

  EXPECT_EQ(cache.Get(7).payload, 7);
  EXPECT_EQ(cache.MemorySize(), 7);
  EXPECT_EQ(*count, 2);
}

TEST(XGrammarThreadSafeLRUCacheTest, LimitedCacheEvictsAfterComputedValueCrossesLimit) {
  auto count = std::make_shared<std::atomic_size_t>(0);
  auto cache = ThreadSafeLRUCache<int, CacheValue, CountingCacheComputer, CacheValueSizeEstimator>{
      5, CountingCacheComputer{count}
  };

  // The size is unknown until computation finishes. The returned value remains usable even when
  // its own cache entry must be evicted to restore the configured limit.
  EXPECT_EQ(cache.Get(7).payload, 7);
  EXPECT_LE(cache.MemorySize(), 5);
  EXPECT_EQ(*count, 1);
  EXPECT_EQ(cache.Get(7).payload, 7);
  EXPECT_LE(cache.MemorySize(), 5);
  EXPECT_EQ(*count, 2);
}

struct BlockingCacheComputer {
  std::shared_ptr<std::promise<void>> started;
  std::shared_future<void> release;

  CacheValue operator()(int key) const {
    started->set_value();
    release.wait();
    return CacheValue{key, 1};
  }
};

TEST(XGrammarThreadSafeLRUCacheTest, UnlimitedClearWaitsForInFlightValue) {
  using namespace std::chrono_literals;

  auto started = std::make_shared<std::promise<void>>();
  auto started_future = started->get_future();
  auto release = std::make_shared<std::promise<void>>();
  auto cache = ThreadSafeLRUCache<int, CacheValue, BlockingCacheComputer, CacheValueSizeEstimator>{
      ThreadSafeLRUCache<int, CacheValue, BlockingCacheComputer, CacheValueSizeEstimator>::
          kUnlimitedSize,
      BlockingCacheComputer{started, release->get_future().share()}
  };

  auto get_future = std::async(std::launch::async, [&cache] { return cache.Get(1).payload; });
  started_future.wait();
  auto clear_future = std::async(std::launch::async, [&cache] { cache.Clear(); });
  EXPECT_EQ(clear_future.wait_for(10ms), std::future_status::timeout);

  release->set_value();
  EXPECT_EQ(get_future.get(), 1);
  clear_future.get();
  EXPECT_EQ(cache.MemorySize(), 0);
}

}  // namespace
