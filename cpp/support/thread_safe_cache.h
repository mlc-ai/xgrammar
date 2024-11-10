/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/thread_safe_cache.h
 * \brief The header for thread-safe caching functionality.
 */
#ifndef XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
#define XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>

namespace xgrammar {

/*!
 * \brief Primary template for ThreadSafeCache
 * \details This class provides thread-safe caching functionality in two forms:
 * 1. Single value cache when only Value template parameter is provided
 * 2. Key-value cache when both Key and Value template parameters are provided
 */
template <typename... Args>
class ThreadSafeCache;

/*!
 * \brief Thread-safe cache for a single computed value
 * \tparam Value The type of value being cached
 * \details Specialization that provides:
 * - Thread-safe access to a single cached value
 * - Lazy computation on first access
 * - Reader-writer locking for concurrent reads
 */
template <typename Value>
class ThreadSafeCache<Value> {
 public:
  /*!
   * \brief Constructs a new single-value cache
   * \param compute The function that computes the cached value
   */
  explicit ThreadSafeCache(std::function<Value()> compute) : compute_(std::move(compute)) {}

  /*!
   * \brief Gets or computes the cached value
   * \return The cached or newly computed value
   */
  Value Get() {
    // First try reading from cache with shared lock
    {
      std::shared_lock<std::shared_mutex> cache_lock(cache_mutex_);
      if (cache_.has_value()) {
        return cache_.value();  // Cache hit
      }
    }

    // Acquire exclusive lock to compute value
    std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);

    // Double-check to prevent redundant computation
    if (cache_.has_value()) {
      return cache_.value();
    }

    Value value = compute_();
    XGRAMMAR_DCHECK(!cache_.has_value());
    cache_ = value;
    return value;
  }

  /*!
   * \brief Clears the cached value
   * This function removes the cached value, so the next call to Get() will recompute it.
   */
  void Clear() {
    std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
    cache_.reset();
  }

 private:
  /*! \brief Optional container holding the cached value */
  std::optional<Value> cache_;
  /*! \brief Function used to compute the value when not cached */
  std::function<Value()> compute_;
  /*! \brief Reader-writer lock protecting access to cache_ */
  std::shared_mutex cache_mutex_;
};

/*!
 * \brief A thread-safe key-value cache with on-demand computation
 * \tparam Key The type of keys used to lookup values. Should be hashable.
 * \tparam Value The type of values stored in the cache
 * \details This cache provides thread-safe access to computed values with the following features:
 * - Lazy computation: Values are only computed when first requested
 * - Thread safety: Uses reader-writer locks for concurrent reads
 * - Parallel computation: Different keys can be computed simultaneously
 * - Double-checked locking: Prevents redundant computation
 */
template <typename Key, typename Value>
class ThreadSafeCache<Key, Value> {
 public:
  /*!
   * \brief Constructs a new thread-safe cache
   * \param compute The function that computes values for uncached keys
   */
  explicit ThreadSafeCache(std::function<Value(const Key&)> compute)
      : compute_(std::move(compute)) {}

  /*!
   * \brief Gets or computes the value for a key
   * \param key The key to lookup
   * \return The cached or newly computed value of the key
   */
  Value Get(const Key& key) {
    // Get or create the per-key mutex
    std::shared_ptr<std::shared_mutex> key_mutex = GetOrCreateMutex(key);

    // First attempt to read from cache_
    {
      std::shared_lock<std::shared_mutex> cache_lock(cache_mutex_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {
        return it->second;  // Cache hit
      }
    }

    // Acquire unique lock on the per-key mutex to compute the value
    std::unique_lock<std::shared_mutex> key_lock(*key_mutex);

    // Double-checked locking
    {
      std::shared_lock<std::shared_mutex> cache_lock(cache_mutex_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {
        return it->second;
      }
    }

    // Compute the value without holding the cache lock
    Value value = compute_(key);

    // Insert the value into cache_
    {
      std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
      XGRAMMAR_DCHECK(cache_.find(key) == cache_.end());
      cache_[key] = value;
    }

    return value;
  }

  /*!
   * \brief Clears all cached values and associated per-key mutexes
   * This function removes all cached key-value pairs, so subsequent calls to Get() will recompute
   * them.
   */
  void Clear() {
    // Acquire locks in the order: global_key_mutex_ -> cache_mutex_
    std::unique_lock<std::mutex> global_key_lock(global_key_mutex_);
    std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
    cache_.clear();
    key_mutexes_.clear();
  }

 private:
  /*!
   * \brief Gets or creates a mutex for the given key
   * \param key The key to get/create a mutex for
   * \return A shared pointer to the mutex for this key
   */
  std::shared_ptr<std::shared_mutex> GetOrCreateMutex(const Key& key) {
    std::unique_lock<std::mutex> lock(global_key_mutex_);
    auto it = key_mutexes_.find(key);
    if (it == key_mutexes_.end()) {
      auto new_mutex = std::make_shared<std::shared_mutex>();
      XGRAMMAR_DCHECK(key_mutexes_.find(key) == key_mutexes_.end());
      key_mutexes_[key] = new_mutex;
      return new_mutex;
    }
    return it->second;
  }

  /*! \brief The cache mapping keys to computed values */
  std::unordered_map<Key, Value> cache_;
  /*! \brief The function used to compute values for uncached keys */
  std::function<Value(const Key&)> compute_;
  /*! \brief Per-key mutexes to allow parallel computation of different keys */
  std::unordered_map<Key, std::shared_ptr<std::shared_mutex>> key_mutexes_;
  /*! \brief Mutex protecting access to key_mutexes_ */
  std::mutex global_key_mutex_;
  /*! \brief Reader-writer lock protecting access to cache_ */
  std::shared_mutex cache_mutex_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
