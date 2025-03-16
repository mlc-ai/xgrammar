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
    auto result = get_aux(key);
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

  auto PopLast() -> void {
    auto erase_lock = std::unique_lock(erase_mutex_);
    if (lru_.empty()) return;
    const Key& key = lru_.front().get();
    lru_.pop_front();
    auto it = cache_.find(key);
    if (it == cache_.end()) return;
    auto& entry = it->second;
    current_size_ -= entry.get_size();
    cache_.erase(it);
  }

 private:
  /*!
   * \brief Gets or computes the value for a key
   * \param key The key to lookup
   * \return The cached or newly computed value of the key
   */
  auto get_aux(const Key& key) -> Value {
    auto erase_lock = std::shared_lock(erase_mutex_);

    // First attempt to read from cache_
    {
      auto cache_lock = std::shared_lock(cache_mutex_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {    // Cache hit
        auto& entry = it->second;  // The iterator is invalidated after releasing the lock
        lru_visit(entry.get_iterator());
        cache_lock.unlock();  // Therefore, we should hold the entry by reference first

        // We should not hold lock here, since this function may be blocking.
        return entry.get_value(policy_, key, current_size_);
      }
    }

    /// \attention: the cache_lock is dropped here
    /// as a result, another thread may insert the key
    /// before we acquire the lock again
    {
      auto cache_lock = std::unique_lock(cache_mutex_);
      auto [it, success] = cache_.try_emplace(key);
      auto& entry = it->second;
      if (success) {  // current thread inserted the key
        entry.set_iterator(lru_push(it->first));
      } else {
        // if other thread has created this entry,
        // it just happened when we released the cache_mutex_
        // so the new entry should be close to the end of the list.
        // As a result, we don't spare the effort to move it to the end.

        // lru_visit(entry.get_iterator());
      }

      cache_lock.unlock();  // Release the lock before blocking

      // We should not hold lock here, since this function may be blocking.
      return entry.get_value(policy_, key, current_size_);
    }
  }

  using LRU_list = std::list<std::reference_wrapper<const Key>>;
  using iterator = typename LRU_list::iterator;

  auto lru_visit(iterator it) -> void { lru_.splice(lru_.end(), lru_, it); }
  auto lru_push(const Key& key) -> iterator { return lru_.insert(lru_.end(), std::cref(key)); }

  struct Entry {
   private:
    std::once_flag flag;
    Value value;
    std::size_t size;
    iterator it;

   public:
    auto set_iterator(iterator it) -> void { this->it = it; }
    auto get_iterator() const -> iterator { return it; }
    auto get_size() const -> std::size_t { return size; }
    auto get_value(const Policy& p, const Key& key, std::atomic_size_t& sum) -> const Value& {
      // block in this lambda until the value is computed
      std::call_once(flag, [&] {
        value = p.compute(key);
        size = p.size(value);
        sum += size;
      });
      return value;
    }
  };

  /*! \brief The function to call when cache is full */
  const Policy policy_;
  /*! \brief The cache mapping keys to computed values */
  std::unordered_map<Key, Entry> cache_;
  /*! \brief LRU list */
  LRU_list lru_;
  /*! \brief Reader-writer lock protecting access to cache_ */
  std::shared_mutex cache_mutex_;
  /*! \brief Mutex protecting removing elements */
  std::shared_mutex erase_mutex_;
  /*! \brief The current size of the cache */
  std::atomic_size_t current_size_{0};
};

namespace details {

// since C++17 don't have std::type_identity, we implement it here
template <typename T>
struct type_identity {
  using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;

template <typename Key, typename Value>
struct LockedMap {
 private:
  std::unordered_map<Key, Value> map;
  mutable std::shared_mutex mutex;

 public:
  template <typename Fn>
  auto read_access(Fn&& fn) const -> decltype(fn(map)) {
    std::shared_lock lock(mutex);
    return std::forward<Fn>(fn)(map);
  }
  template <typename Fn>
  auto write_access(Fn&& fn) -> decltype(fn(map)) {
    std::unique_lock lock(mutex);
    return std::forward<Fn>(fn)(map);
  }
};

template <typename Value, typename... Keys>
class ThreadSafeCacheImpl {
 private:
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
  // do not hold lock when calling any of these functions

  template <typename Key>
  auto get_cache_lock() -> LockedMap<Key, Entry>& {
    return std::get<LockedMap<Key, Entry>>(storage);
  }

  auto lru_visit(const Entry& entry) -> const Value& {
    const auto guard = std::lock_guard{lru_mutex};
    /** \attention The dereference of the iterator must be done within the lock */
    const auto iterator = entry.iterator;
    if (iterator != LRUiterator{}) lru_list.splice(lru_list.end(), lru_list, iterator);
    return entry.value;
  }

  template <typename Key>
  auto lru_init(std::pair<const Key, Entry>& pair, const Value& init) -> void {
    const auto guard = std::lock_guard{lru_mutex};
    auto& [key, entry] = pair;
    /** \attention This must be done in the lock to ensure the iterator is valid */
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
      do {
        auto& [_, entry] = *(iter++);
        entry.iterator = LRUiterator{};
        evict(entry.value);
      } while (predicate());
      auto result = LRUlist{};
      result.splice(result.begin(), lru_list, lru_list.begin(), iter);
      return result;
    }();

    // free the memory of the map outside the lock
    for (auto& [keyset, _] : free_list) {
      std::visit(
          [&](auto* key) {
            using Key_t = std::decay_t<decltype(*key)>;
            auto& locked_map = get_cache_lock<Key_t>();
            locked_map.write_access([&](auto& map) { map.erase(*key); });
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

}  // namespace details

template <typename Policy, typename Value, typename... Keys>
class ThreadSafeCacheSized2 : private Policy, details::ThreadSafeCacheSized<Value, Keys...> {
 private:
  using Impl = details::ThreadSafeCacheSized<Value, Keys...>;
  using Sized = details::SizedValue<Value>;
  using Future = std::shared_future<Sized>;

 public:
  using Policy::Policy;

  template <typename Key>
  auto Get(const details::type_identity_t<Key>& key) -> Value {
    // We use type_identity_t to force the user to provide the key type
    static_assert((std::is_same_v<Key, Keys> || ...), "Key type not found in the cache");
    auto future = GetFuture<Key>(key);
    Pop();
    return future.get().value;
  }

 private:
  template <typename Key>
  auto GetFuture(const Key& key) -> Future {
    auto& locked_map = Impl::template get_cache_lock<Key>();
    auto future = Future{};

    auto first_hit = locked_map.read_access([&](const auto& map) {
      if (auto it = map.find(key); it != map.end()) {
        future = Impl::lru_visit(it->second);
        return true;
      } else {
        return false;
      }
    });
    if (first_hit) return future;

    auto task = std::packaged_task<Sized()>{[this, &key] {
      return Sized{
          Policy::template compute<Key>(key),  // compute the value
          [&](const Value& value) {
            auto size = Policy::size(value);
            current_size_ += size;
            return size;
          }
      };
    }};
    future = task.get_future().share();

    auto second_hit = locked_map.write_access([&](auto& map) {
      auto [it, success] = map.try_emplace(key);
      if (!success) {
        future = Impl::lru_visit(it->second);
        return true;
      } else {
        Impl::lru_init(*it, future);
        return false;
      }
    });
    if (!second_hit) task();

    return future;
  }

  auto Pop() -> void {
    Impl::lru_evict(
        [this] { return Policy::should_evict(current_size_); },
        [this](const Future& value) { current_size_ -= value.get().size; }
    );
  }

 private:
  std::atomic_size_t current_size_{0};
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_THREAD_SAFE_CACHE_H_
