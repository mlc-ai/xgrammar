/*!
 * Copyright (c) 2023 by Contributors
 * \file xgrammar/support/thread_pool.h
 * \brief Thread pool.
 */
#ifndef XGRAMMAR_SUPPORT_THREAD_POOL_H_
#define XGRAMMAR_SUPPORT_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "logging.h"

namespace xgrammar {

/*!
 * \brief A thread pool implementation for parallel task execution.
 *
 * ThreadPool manages a pool of worker threads that can execute tasks asynchronously.
 * Tasks are submitted to a queue and executed by available threads from the pool.
 * The pool automatically handles thread synchronization and task distribution.
 */
class ThreadPool {
 public:
  struct TaskCounter {
   public:
    // A dummy callback that does nothing
    inline static constexpr auto kCallback = [] {};
    inline static constexpr auto kNoLimit = std::numeric_limits<std::size_t>::max();

    explicit TaskCounter(ThreadPool& pool) : TaskCounter(pool, GetLimit(pool)) {}
    explicit TaskCounter(ThreadPool& pool, std::size_t limit)
        : m_active(0), m_pool(pool), m_rate_limit(limit) {
      XGRAMMAR_CHECK(m_rate_limit > 0) << "TaskCounter rate limit must be greater than zero.";
      m_pool.active_tasks_++;
    }

    // not copy/moveable
    TaskCounter(const TaskCounter&) = delete;
    TaskCounter(TaskCounter&&) = delete;
    TaskCounter& operator=(const TaskCounter&) = delete;
    TaskCounter& operator=(TaskCounter&&) = delete;

    void WaitUntilComplete() {
      auto lock = std::unique_lock{m_mutex};
      m_cv.wait(lock, [this] { return m_active == 0; });
    }

    template <typename F, typename C = const decltype(kCallback)&>
    void Submit(F&& f, C&& c = kCallback) {
      using ResultType = std::invoke_result_t<F>;
      static_assert(
          std::is_void_v<ResultType> || std::is_invocable_v<C, ResultType>,
          "Callback must be invocable with the result of the task."
      );

      // real task to be executed by the thread pool
      auto fn = std::function{[this, task = std::forward<F>(f), callback = std::forward<C>(c)] {
        if constexpr (std::is_void_v<ResultType>) {
          task();
          {
            const auto lock = std::lock_guard{m_mutex};
            callback();
            m_active -= 1;
          }
          m_cv.notify_all();
        } else {
          auto result = task();
          {
            const auto lock = std::lock_guard{m_mutex};
            callback(std::move(result));
            m_active -= 1;
          }
          m_cv.notify_all();
        }
      }};

      // rate limiting before submitting the task
      {
        auto lock = std::unique_lock{m_mutex};
        m_active += 1;
        m_cv.wait(lock, [this] { return m_active <= m_rate_limit; });
      }

      // emplace the task into the thread pool
      {
        const auto lock = std::lock_guard{m_pool.queue_mutex_};
        m_pool.task_queue_.push(std::move(fn));
      }
      m_pool.queue_condition_.notify_one();
    }

    ~TaskCounter() {
      this->WaitUntilComplete();
      m_pool.active_tasks_--;
    }

   private:
    friend class ThreadPool;

    // default no limit, yet we can still implement rate limiting if needed
    static std::size_t GetLimit([[maybe_unused]] ThreadPool& pool) { return kNoLimit; }

    std::size_t m_active;
    std::condition_variable m_cv;
    std::mutex m_mutex;
    ThreadPool& m_pool;
    const std::size_t m_rate_limit;
  };

  /*!
   * \brief Construct a new thread pool with the specified number of threads.
   * \param num_threads Number of worker threads to create. Defaults to hardware concurrency.
   * \note The pool starts the worker threads immediately upon construction.
   */
  ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
    // Initialize thread pool with num_threads threads
    workers_.resize(num_threads);
    for (auto& worker : workers_) {
      worker = std::thread([this] {
        while (true) {
          std::function<void()> task;
          {
            // Lock queue while waiting for new task
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_condition_.wait(lock, [this] { return shutdown_ || !task_queue_.empty(); });

            // Exit thread if shutdown and queue is empty
            if (shutdown_ && task_queue_.empty()) return;

            // Get task from queue
            task = std::move(task_queue_.front());
            task_queue_.pop();
          }
          task();
        }
      });
    }
  }

  TaskCounter CreateTaskCounter() { return TaskCounter{*this}; }
  TaskCounter CreateTaskCounterWithLimit(std::size_t limit) { return TaskCounter{*this, limit}; }
  std::size_t NumThreads() const { return workers_.size(); }

  /*!
   * \brief Join all threads in the pool.
   *
   * Sets shutdown flag and waits for all threads to complete their current tasks
   * before destroying the pool. Any remaining tasks in the queue will be executed
   * before shutdown completes.
   */
  void Join() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (shutdown_) return;  // Already shut down
      shutdown_ = true;
    }

    queue_condition_.notify_all();  // Wake up all threads so they can exit
    for (std::thread& worker : workers_) {
      if (worker.joinable()) worker.join();  // Wait for thread to finish
    }
  }

  /*!
   * \brief Destructor that ensures graceful shutdown of the thread pool.
   */
  ~ThreadPool() {
    Join();
    XGRAMMAR_CHECK(active_tasks_ == 0) << "ThreadPool destroyed while tasks are still active.";
  }

  // Prevent copying or moving of the thread pool
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  // Debug only function to get thread IDs
  std::vector<std::thread::id> DebugGetThreadIDs() const {
    std::vector<std::thread::id> thread_ids;
    for (const auto& worker : workers_) {
      thread_ids.push_back(worker.get_id());
    }
    return thread_ids;
  }

 private:
  /*! \brief Thread container */
  std::vector<std::thread> workers_;
  /*! \brief Task queue */
  std::queue<std::function<void()>> task_queue_;
  /*! \brief Mutex to protect task queue */
  std::mutex queue_mutex_;
  /*! \brief Condition variable for thread synchronization */
  std::condition_variable queue_condition_;
  /*! \brief Condition variable for task completion */
  std::condition_variable tasks_done_condition_;
  /*! \brief Flag to indicate thread pool shutdown */
  bool shutdown_ = false;
  /*! \brief Number of active tasks */
  std::atomic_size_t active_tasks_{0};
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_THREAD_POOL_H_
