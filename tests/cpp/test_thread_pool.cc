#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <thread>

#include "support/thread_pool.h"
using namespace xgrammar;

TEST(XGrammarThreadPoolTest, FunctionalTest) {
  ThreadPool pool(2);

  const auto tid_map = [&pool] {
    const auto thread_ids = pool.DebugGetThreadIDs();
    std::unordered_map<std::thread::id, int> map;
    for (size_t i = 0; i < thread_ids.size(); ++i) {
      map[thread_ids[i]] = static_cast<int>(i);
    }
    return map;
  }();

  std::thread threads[2];

  // Example 1: Use Execute to submit tasks without return values
  // with a rate limit of 4, meaning at most 4 tasks can be queued at any time.
  const auto start = std::chrono::high_resolution_clock::now();
  for (int j = 0; j < 2; ++j) {
    threads[j] = std::thread([j, &pool, start, &tid_map] {
      auto counter = pool.CreateTaskCounterWithLimit(4);
      if (j == 0) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      for (int i = 0; i < 10; ++i) {
        counter.Submit([i, j, start, &tid_map] {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          const auto now = std::chrono::high_resolution_clock::now();
          const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
          auto os = std::ostringstream{};
          os << "[" << dur.count() << " ms] [job <" << j << ">] ";
          const auto tid = std::this_thread::get_id();
          os << "Execute task " << i << " is running in thread " << tid_map.at(tid) << "\n";
          std::cout << os.str();
        });
        const auto now = std::chrono::high_resolution_clock::now();
        const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
        auto os = std::ostringstream{};
        os << "[" << dur.count() << " ms] [job <" << j << ">] ";
        os << "Submit task " << i << "\n";
        std::cout << os.str();
      }
    });
  }
  threads[0].join();
  threads[1].join();

  // Wait for task to complete
  pool.Join();
}
