/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/thread_safe_cache.h
 * \brief The header for thread-safe caching functionality.
 */
#ifndef XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
#define XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_

#include <atomic>
#include <cstddef>
#include <functional>
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
    // Why we need this:
    // - When adding new elements to a unordered_map, the map may be rehashed,
    // - which means all the iterators may be invalidated.
    // - However, cppreference says:
    // - "References and pointers to either key or data stored in the container are only invalidated
    // - by erasing that element, even when the corresponding iterator is invalidated."
    // - (See https://en.cppreference.com/w/cpp/container/unordered_map)
    // - Therefore, we should maintain 2 locks.
    // - When we add something to the cache, we should hold the cache_mutex_.
    // - When we erase something from the cache, we should hold the clear_mutex_.

    auto erase_lock = std::shared_lock(erase_mutex_);

    // First attempt to read from cache_
    {
      auto cache_lock = std::shared_lock(cache_mutex_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {    // Cache hit
        auto& entry = it->second;  // The iterator is invalidated after releasing the lock
        cache_lock.unlock();       // Therefore, we should hold the entry by reference first

        // We should not hold lock here, since this function may be blocking.
        return entry.get(compute_, key);
      }
    }

    // Acquire exclusive lock to compute value
    {
      auto cache_lock = std::unique_lock(cache_mutex_);
      auto& entry = cache_[key];  // Create a new entry
      cache_lock.unlock();        // Release the lock before blocking

      // We should not hold lock here, since this function may be blocking.
      return entry.get(compute_, key);
    }
  }

  /*!
   * \brief Clears all cached values and associated per-key mutexes
   * This function removes all cached key-value pairs, so subsequent calls to Get() will recompute
   * them.
   */
  void Clear() {
    auto erase_lock = std::unique_lock(erase_mutex_);
    cache_.clear();
  }

 private:
  struct Entry {
    Value value;
    std::once_flag flag;
    auto get(const std::function<Value(const Key&)>& f, const Key& key) -> const Value& {
      // block in this lambda until the value is computed
      std::call_once(flag, [&] { value = f(key); });
      return value;
    }
  };

  /*! \brief The cache mapping keys to computed values */
  std::unordered_map<Key, Entry> cache_;
  /*! \brief The function used to compute values for uncached keys */
  std::function<Value(const Key&)> compute_;
  /*! \brief Reader-writer lock protecting access to cache_ */
  std::shared_mutex cache_mutex_;
  /*! \brief Mutex protecting removing elements */
  std::shared_mutex erase_mutex_;
};

template <typename Key, typename Value>
struct DefaultPolicy {
  // function-like member to compute
  std::function<Value(const Key&)> compute;
  std::function<void(std::size_t, bool)> callback;
  std::function<std::size_t(const Value&)> size;
};

template <typename Key, typename Value, typename Policy = DefaultPolicy<Key, Value>>
class ThreadSafeCacheSized {
 public:
  /*!
   * \brief Constructs a new thread-safe cache
   * \param compute The function that computes values for uncached keys
   */
  explicit ThreadSafeCacheSized(Policy policy) : policy_(std::move(policy)) {}

  auto Get(const Key& key) -> Value {
    policy_.callback(current_size_, false);
    auto result = GetAux(key);
    policy_.callback(current_size_, true);
    return result;
  }

  /*!
   * \brief Clears all cached values and associated per-key mutexes
   * This function removes all cached key-value pairs, so subsequent calls to Get() will recompute
   * them.
   */
  auto Clear() -> void {
    auto erase_lock = std::unique_lock(erase_mutex_);
    cache_.clear();
    current_size_ = 0;
  }

 private:
  /*!
   * \brief Gets or computes the value for a key
   * \param key The key to lookup
   * \return The cached or newly computed value of the key
   */
  auto GetAux(const Key& key) -> Value {
    auto erase_lock = std::shared_lock(erase_mutex_);

    // First attempt to read from cache_
    {
      auto cache_lock = std::shared_lock(cache_mutex_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {    // Cache hit
        auto& entry = it->second;  // The iterator is invalidated after releasing the lock
        cache_lock.unlock();       // Therefore, we should hold the entry by reference first

        // We should not hold lock here, since this function may be blocking.
        return entry.get(policy_, key, current_size_);
      }
    }

    // Acquire exclusive lock to compute value
    {
      auto cache_lock = std::unique_lock(cache_mutex_);
      auto& entry = cache_[key];  // Create a new entry
      cache_lock.unlock();        // Release the lock before blocking

      // We should not hold lock here, since this function may be blocking.
      return entry.get(policy_, key, current_size_);
    }
  }

 private:
  struct Entry {
   private:
    Value value;
    std::once_flag flag;

   public:
    auto get(const Policy& p, const Key& key, std::atomic_size_t& size) -> const Value& {
      // block in this lambda until the value is computed
      std::call_once(flag, [&] {
        value = p.compute(key);
        size += Policy::size(value);
      });
      return value;
    }
  };

  /*! \brief The function to call when cache is full */
  const Policy policy_;
  /*! \brief The cache mapping keys to computed values */
  std::unordered_map<Key, Entry> cache_;
  /*! \brief Reader-writer lock protecting access to cache_ */
  std::shared_mutex cache_mutex_;
  /*! \brief Mutex protecting removing elements */
  std::shared_mutex erase_mutex_;
  /*! \brief The current size of the cache */
  std::atomic_size_t current_size_{0};
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
