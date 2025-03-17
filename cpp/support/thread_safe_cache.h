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

template <typename Key, typename Value>
struct LockedMap {
 private:
  std::unordered_map<Key, Value> map;
  mutable std::shared_mutex mutex;

 public:
  /**
   * \brief Try to insert a key-value pair into the map
   * \param key The key to insert
   * \param exist The function to call if the key already exists (in the lock)
   * \param prepare The function to call to prepare the value (outside the lock)
   * \param init The function to call to initialize the value (in the lock)
   * \param exit The function to finalize the value (outside the lock)
   * \return The result of the exit function or the exist function
   */
  template <typename F0, typename F1, typename F2, typename F3>
  auto try_insert(const Key& key, F0&& exist, F1&& prepare, F2&& init, F3&& exit) {
    {
      auto lock = std::shared_lock(mutex);
      if (auto it = map.find(key); it != map.end()) {
        return std::forward<F0>(exist)(*it);
      }
    }
    auto tmp = std::forward<F1>(prepare)();
    {
      auto lock = std::unique_lock(mutex);
      auto [it, success] = map.try_emplace(key);
      if (success == false) {
        return std::forward<F0>(exist)(*it);
      }
      auto tmp2 = std::forward<F2>(init)(*it, tmp);
      lock.unlock();
      return std::forward<F3>(exit)(tmp, tmp2);
    }
  }

  auto erase(const Key& key) -> void {
    auto lock = std::unique_lock(mutex);
    map.erase(key);
  }
};

template <typename Value, typename... Keys>
class ThreadSafeCacheImpl {
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
    friend class ThreadSafeCacheImpl;
  };

  struct LRUnode {
    KeySet key;
    Entry& entry;
    LRUnode(KeySet k, Entry& e) : key{k}, entry{e} {}
  };

 protected:
  template <typename Key>
  auto get_map() -> LockedMap<Key, Entry>& {
    static_assert((std::is_same_v<Key, Keys> || ...), "Key type not found in the cache");
    return std::get<LockedMap<Key, Entry>>(storage);
  }

  template <typename Key>
  auto lru_visit(const std::pair<const Key, Entry>& pair) -> const Value& {
    static_assert((std::is_same_v<Key, Keys> || ...), "Key type not found in the cache");
    const auto guard = std::lock_guard{lru_mutex};
    /// \attention The dereference of the iterator must be done within the lock
    const auto& entry = pair.second;
    const auto iterator = entry.iterator;
    if (iterator != LRUiterator{}) {  // if not to be evicted
      lru_list.splice(lru_list.end(), lru_list, iterator);
    }
    return entry.value;
  }

  template <typename Key>
  auto lru_init(std::pair<const Key, Entry>& pair, const Value& init) -> void {
    static_assert((std::is_same_v<Key, Keys> || ...), "Key type not found in the cache");
    const auto guard = std::lock_guard{lru_mutex};
    /// \attention This must be done in the lock to ensure the iterator is valid
    auto& [key, entry] = pair;
    entry.value = init;
    entry.iterator = lru_list.emplace(lru_list.end(), KeySet{&key}, entry);
  }

  template <typename Predicate, typename Evict>
  auto lru_evict(const Predicate& predicate, const Evict& evict) -> void {
    if (!predicate())  // short circuit if the predicate is false
      return;

    // only hold the lru lock when doing the eviction
    auto free_list = [&]() -> LRUlist {
      const auto guard = std::lock_guard{lru_mutex};
      auto iter = lru_list.begin();
      if (iter == lru_list.end()) return {};
      auto evicted = LRUlist{};
      auto waiting = LRUlist{};
      do {
        auto old_iter = iter++;
        auto& [_, entry] = *old_iter;
        if (evict(entry.value)) {
          entry.iterator = LRUiterator{};  // to be evicted
          evicted.splice(evicted.end(), lru_list, old_iter);
        } else {
          waiting.splice(waiting.end(), lru_list, old_iter);
        }
      } while (predicate() && iter != lru_list.end());
      // move the waiting list to the end of the lru list
      lru_list.splice(lru_list.end(), waiting);
      return evicted;
    }();

    // free the memory of the map outside the lru lock
    for (auto& [keyset, _] : free_list) {
      std::visit(
          [&](auto* key) {
            using Key_t = std::decay_t<decltype(*key)>;
            auto& locked_map = get_map<Key_t>();
            locked_map.erase(*key);
          },
          keyset
      );
    }
  }

 private:
  std::tuple<LockedMap<Keys, Entry>...> storage;
  LRUlist lru_list;
  std::mutex lru_mutex;
};

template <typename Value>
struct SizedValue {
  Value value;
  std::size_t size;
  template <typename Fn>
  SizedValue(Value value, const Fn& size_fn) : value{value}, size{size_fn(value)} {}
};

template <typename Value, typename... Keys>
using ThreadSafeCacheSized = ThreadSafeCacheImpl<std::shared_future<SizedValue<Value>>, Keys...>;

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
class ThreadSafeLRUCache : private Policy, details::ThreadSafeCacheSized<Value, Keys...> {
 private:
  using Impl = details::ThreadSafeCacheSized<Value, Keys...>;
  using Sized = details::SizedValue<Value>;
  using Future = std::shared_future<Sized>;

 public:
  using Policy::Policy;

  auto GetPolicy() -> Policy& { return *this; }

  template <typename Key, typename Tp>
  auto Get(const Tp& key) -> Value {
    /// \attention We use type_identity_t to force the user to provide the key type
    static_assert((std::is_same_v<Key, Keys> || ...), "Key type not found in the cache");
    auto future = GetFuture<Key>(key);
    Pop();
    return future.get().value;
  }

  auto Clear() -> void {
    // Remove all the ready entries.
    Impl::lru_evict(
        [] { return true; },
        [](const Future& value) {
          using namespace std::chrono_literals;
          return value.wait_for(0s) == std::future_status::ready;
        }
    );
  }

 private:
  template <typename Key>
  auto GetFuture(const Key& key) -> Future {
    using Task = std::packaged_task<Sized()>;
    auto& locked_map = Impl::template get_map<Key>();
    const auto exist = [this](const auto& pair) {
      return this->Impl::lru_visit(pair);  // simply move the entry to the end
    };
    const auto prepare = [&] {
      return Task{[this, &key] {
        return Sized{
            Policy::template compute<Key>(key),  // compute the value
            [&](const Value& value) {
              auto size = Policy::size(value);
              current_size_ += size;
              return size;
            }
        };
      }};
    };
    const auto init = [this](auto& pair, Task& task) {
      auto future = task.get_future().share();
      this->Impl::lru_init(pair, future);
      return future;
    };
    const auto exit = [](Task& task, Future future) {
      task();  // only perform task on first access
      return future;
    };

    return locked_map.try_insert(key, exist, prepare, init, exit);
  }

  auto Pop() -> void {
    Impl::lru_evict(
        [&] { return Policy::should_evict(current_size_); },
        [&](const Future& value) {
          using namespace std::chrono_literals;
          // if not ready, then do not wait and block here
          if (value.wait_for(0s) != std::future_status::ready) return false;
          auto size = value.get().size;
          current_size_ -= size;
          return true;
        }
    );
  }

 private:
  std::atomic_size_t current_size_{0};
};
}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
