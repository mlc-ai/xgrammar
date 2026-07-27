#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>

#include "support/thread_pool.h"
using namespace xgrammar;

TEST(XGramamrThreadPoolTest, FunctionalTest) {
  ThreadPool pool(4);

  // Example 1: Use Submit to submit tasks with return values
  std::vector<std::shared_future<int>> futures;
  for (int i = 0; i < 8; ++i) {
    auto fut = pool.Submit([i] {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      std::cout << "Task " << i << " is running in thread " << std::this_thread::get_id() << "\n";
      return i * i;
    });
    futures.push_back(fut);
  }

  for (auto& fut : futures) {
    int result = fut.get();
    std::cout << "Result: " << result << "\n";
  }

  // Example 2: Use Execute to submit tasks without return values
  for (int i = 0; i < 5; ++i) {
    pool.Execute([i] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      std::cout << "Execute task " << i << " is running in thread " << std::this_thread::get_id()
                << "\n";
    });
  }

  // Wait for task to complete
  pool.Join();
}

TEST(XGramamrThreadPoolTest, WaitKeepsPoolReusable) {
  ThreadPool pool(4);
  std::atomic<int> completed{0};

  for (int round = 0; round < 3; ++round) {
    for (int task = 0; task < 16; ++task) {
      pool.Execute([&completed]() { ++completed; });
    }
    pool.Wait();
    EXPECT_EQ(completed.load(), (round + 1) * 16);
  }

  auto failure = pool.Submit([]() -> int { throw std::runtime_error("expected failure"); });
  EXPECT_THROW(failure.get(), std::runtime_error);
  pool.Wait();

  pool.Execute([]() { throw std::runtime_error("expected execute failure"); });
  EXPECT_THROW(pool.Wait(), std::runtime_error);

  auto success = pool.Submit([]() { return 42; });
  EXPECT_EQ(success.get(), 42);
  pool.Wait();

  pool.Execute([&completed]() { ++completed; });
  EXPECT_NO_THROW(pool.Wait());
  EXPECT_EQ(completed.load(), 49);
}

TEST(XGramamrThreadPoolTest, ParallelForPropagatesWorkerExceptions) {
  EXPECT_THROW(
      ParallelFor(
          0,
          8,
          4,
          [](int index) {
            if (index == 3) {
              throw std::runtime_error("expected parallel-for failure");
            }
          }
      ),
      std::runtime_error
  );
}

// TEST(XGramamrThreadPoolTest, PressureTest) {
//   const size_t num_threads = std::thread::hardware_concurrency();
//   ThreadPool pool(num_threads);

//   const size_t num_tasks = 1000;
//   int counter = 0;
//   std::mutex counter_mutex;

//   auto start_time = std::chrono::high_resolution_clock::now();

//   for (size_t i = 0; i < num_tasks; ++i) {
//     pool.Execute([&counter, &counter_mutex, i]() {
//       std::this_thread::sleep_for(std::chrono::milliseconds(i % 50));
//       std::lock_guard<std::mutex> lock(counter_mutex);
//       counter++;
//     });
//   }

//   pool.Wait();

//   auto end_time = std::chrono::high_resolution_clock::now();

//   EXPECT_EQ(counter, static_cast<int>(num_tasks));

//   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
//   std::cout << "Pressure test completed, time taken: " << duration.count() << " milliseconds.\n";
// }
