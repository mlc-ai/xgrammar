/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/thread_safe_cache.h
 * \brief The header for thread-safe caching functionality.
 */
#ifndef XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
#define XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_

#include <atomic>
#include <chrono>  // IWYU pragma: keep
#include <cstddef>
#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

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

namespace details {

template <typename Value, typename... Keys>
class LRUCacheImpl {
 protected:
  struct LRUnode;
  using KeySet = std::variant<const Keys*...>;
  using LRUlist = std::list<LRUnode>;
  using LRUiterator = typename LRUlist::iterator;

  struct Entry {
   public:
    Entry() = default;
    Entry(const Value&) = delete;
    Entry& operator=(const Value&) = delete;

   private:
    Value value;
    LRUiterator iterator;
    friend class LRUCacheImpl;
  };

  struct LRUnode {
    KeySet key;
    Entry& entry;
    LRUnode(KeySet k, Entry& e) : key{k}, entry{e} {}
  };

  template <typename Key>
  auto get_map() -> std::unordered_map<Key, Entry>& {
    static_assert((std::is_same_v<Key, Keys> || ...), "Key type not found in the cache");
    return std::get<std::unordered_map<Key, Entry>>(storage_);
  }

  template <typename Key>
  auto lru_visit(const std::pair<const Key, Entry>& pair) -> const Value& {
    static_assert((std::is_same_v<Key, Keys> || ...), "Key type not found in the cache");
    const auto& entry = pair.second;
    lru_list_.splice(lru_list_.end(), lru_list_, entry.iterator);
    return entry.value;
  }

  template <typename Key>
  auto lru_init(std::pair<const Key, Entry>& pair, const Value& init) -> void {
    static_assert((std::is_same_v<Key, Keys> || ...), "Key type not found in the cache");
    auto& [key, entry] = pair;
    entry.value = init;
    entry.iterator = lru_list_.emplace(lru_list_.end(), KeySet{&key}, entry);
  }

  template <typename Predicate, typename Evict>
  auto lru_evict(const Predicate& predicate, const Evict& evict) -> void {
    if (!predicate())  // short circuit if the predicate is false
      return;

    auto iter = lru_list_.begin();
    if (iter == lru_list_.end()) return;

    auto waiting = LRUlist{};

    do {
      auto& [keyset, entry] = *iter;
      if (evict(entry.value)) {
        std::visit(
            [&](const auto* key) {
              using Key_t = std::decay_t<decltype(*key)>;
              get_map<Key_t>().erase(*key);
            },
            keyset
        );
        iter = lru_list_.erase(iter);
      } else {
        waiting.splice(waiting.end(), lru_list_, iter++);
      }
    } while (predicate() && iter != lru_list_.end());

    // move the waiting list to the end of the lru list
    lru_list_.splice(lru_list_.end(), waiting);
  }

 private:
  std::tuple<std::unordered_map<Keys, Entry>...> storage_;
  LRUlist lru_list_;
};

template <typename Value>
struct SizedValue {
  Value value;
  std::size_t size;
  template <typename Fn>
  SizedValue(Value value, const Fn& size_fn) : value{value}, size{size_fn(value)} {}
};

template <typename Value, typename... Keys>
using LRUCacheSizedImpl = LRUCacheImpl<std::shared_future<SizedValue<Value>>, Keys...>;

template <typename Value>
struct DemoPolicy {
  // The interface of the policy
  template <typename KeyType>
  auto compute(const KeyType&) -> Value;
  auto should_evict(std::size_t) -> bool;
  auto size(const Value&) -> std::size_t;
};

}  // namespace details

template <typename Policy, typename Value, typename... Keys>
class ThreadSafeLRUCache : private Policy, details::LRUCacheSizedImpl<Value, Keys...> {
 private:
  using Impl = details::LRUCacheSizedImpl<Value, Keys...>;
  using Sized = details::SizedValue<Value>;
  using Future = std::shared_future<Sized>;
  using Task = std::packaged_task<Sized()>;
  using typename Impl::Entry;

 public:
  using Policy::Policy;

  auto GetPolicy() -> Policy& { return *this; }

  template <typename Key, typename Tp>
  auto Get(const Tp& key) -> Value {
    static_assert((std::is_same_v<Key, Keys> || ...), "Key type not found in the cache");
    auto future = GetFuture<Key>(key, Impl::template get_map<Key>());
    return future.get().value;
  }

  auto Clear() -> void {
    // Remove all the ready entries.
    const auto lock_map = std::lock_guard{map_mutex_};
    const auto lock_lru = std::lock_guard{lru_mutex_};
    Impl::lru_evict(
        [] { return true; },
        [&](const Future& value) {  // always evict and block until the value is ready
          current_size_ -= value.get().size;
          return true;
        }
    );
  }

 private:
  template <typename Key>
  auto GetFuture(const Key& key, std::unordered_map<Key, Entry>& map) -> Future {
    {
      auto lock_map = std::shared_lock{map_mutex_};
      auto it = map.find(key);
      if (it != map.end()) {
        const auto lock_lru = std::lock_guard{lru_mutex_};
        return Impl::lru_visit(*it);
      }
    }

    auto task = Task{[this, &key] {
      return Sized{
          Policy::template compute<Key>(key),  // compute the value
          [&](const Value& value) {
            auto size = Policy::size(value);
            current_size_ += size;
            return size;
          }
      };
    }};

    {
      auto lock_map = std::unique_lock{map_mutex_};
      auto [it, success] = map.try_emplace(key);
      if (!success) {
        const auto lock_lru = std::lock_guard{lru_mutex_};
        return Impl::lru_visit(*it);
      }

      // in this case, we insert the task, and we need to compute the value
      auto future = task.get_future().share();
      {
        const auto lock_lru = std::lock_guard{lru_mutex_};
        Impl::lru_init(*it, future);
        this->Pop();
      }
      lock_map.unlock();

      // perform the costly computation outside all locks
      task();
      return future;
    }
  }

  auto Pop() -> void {
    Impl::lru_evict(
        [&] { return Policy::should_evict(current_size_); },
        [&](const Future& value) {
          using namespace std::chrono_literals;
          // if not ready, then do not wait and block here
          if (value.wait_for(0s) != std::future_status::ready) return false;
          current_size_ -= value.get().size;
          return true;
        }
    );
  }

 private:
  std::atomic_size_t current_size_{0};
  std::mutex lru_mutex_;
  std::shared_mutex map_mutex_;
};
}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
