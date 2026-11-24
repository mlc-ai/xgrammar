/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/json_schema_grammar_cache.h
 * \brief A process-wide cache of the grammars converted from JSON schemas.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_GRAMMAR_CACHE_H_
#define XGRAMMAR_JSON_SCHEMA_GRAMMAR_CACHE_H_

#include <xgrammar/grammar.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace xgrammar {

/*!
 * \brief A thread-safe LRU cache from a JSON schema conversion key to the normalized grammar it
 * produces.
 *
 * \details
 * What it skips. Converting a JSON schema to grammar rules (JSONSchemaToGrammar plus
 * normalization). Every path that converts a schema goes through JSONSchemaToNormalizedGrammar,
 * which consults this cache: Grammar::FromJSONSchema, the json_schema nodes of
 * Grammar::FromStructuralTag, and the GrammarCompiler methods that call them.
 *
 * Lifecycle. There is one instance per process (Global()). It is created on first use and is
 * never destroyed. Entries are added after a successful conversion and removed by LRU eviction,
 * by lowering the budget, or by Grammar::ClearJSONSchemaCache. It belongs to no Grammar or
 * GrammarCompiler, and the compiler's caches (the grammar-level cache of compiled grammars and
 * the rule-level cache of token masks) are separate: GrammarCompiler::ClearCache and
 * cache_limit_bytes do not affect it.
 *
 * Contents. Converting one schema produces several rules that reference each other (the root
 * rule, one rule per sub-schema, `$defs`/`$ref` targets that may be recursive, and helper rules
 * such as `basic_string`). Rule ids cannot outlive the builder they index into, so an entry is the
 * whole converted sub-grammar as a standalone Grammar. Callers splice it into the grammar they
 * are building with SubGrammarAdder, which copies every rule the root reaches and remaps all rule
 * references in one pass, so recursion and shared helper rules stay consistent in the copy. The
 * per-conversion rule caches inside the converters still handle reuse and recursion within one
 * schema; this cache only decides whether that conversion runs at all.
 *
 * A cached Grammar is shared between all readers and must never be modified. Callers that hand
 * the grammar out to users (who may run in-place passes on it) must copy it first.
 *
 * Key. It must encode every input that can change the converted grammar; see
 * JSONSchemaCacheKey. Equal keys must always produce equivalent grammars. Failed conversions are
 * never cached.
 *
 * Memory. It is bounded by an estimate of each entry's size (grammar plus key), with the least
 * recently used entries evicted first. A budget of 0 disables the cache, and -1 removes the bound.
 */
class JSONSchemaGrammarCache {
 public:
  /*! \brief The default memory budget of the process-wide cache, in bytes. */
  static constexpr int64_t kDefaultMaxMemoryBytes = 64LL * 1024 * 1024;
  /*! \brief The budget value meaning "no bound". */
  static constexpr int64_t kUnlimited = -1;

  struct Stats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t evictions = 0;
    int64_t num_entries = 0;
    int64_t memory_bytes = 0;
    int64_t max_memory_bytes = 0;
  };

  explicit JSONSchemaGrammarCache(int64_t max_memory_bytes = kDefaultMaxMemoryBytes)
      : max_memory_bytes_(max_memory_bytes) {}

  JSONSchemaGrammarCache(const JSONSchemaGrammarCache&) = delete;
  JSONSchemaGrammarCache& operator=(const JSONSchemaGrammarCache&) = delete;

  /*! \brief The process-wide instance used by the grammar converters. */
  static JSONSchemaGrammarCache& Global();

  /*! \brief Whether lookups can hit. Callers may skip building a key when this is false. */
  bool Enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_memory_bytes_ != 0;
  }

  /*! \brief Look up a key. Counts a hit or a miss and refreshes the entry's LRU position. */
  std::optional<Grammar> Get(const std::string& key);

  /*! \brief Insert or replace an entry, then evict to fit the budget. */
  void Put(const std::string& key, const Grammar& grammar);

  /*!
   * \brief Return the cached grammar for key, or compute, cache and return it. The computation
   * runs without holding the lock, so two threads that miss the same key at once both compute it;
   * the results are equivalent and the later insert simply replaces the earlier one.
   */
  template <typename Compute>
  Grammar GetOrCompute(const std::string& key, Compute&& compute) {
    if (auto cached = Get(key)) return *std::move(cached);
    Grammar grammar = compute();
    Put(key, grammar);
    return grammar;
  }

  /*! \brief Set the memory budget in bytes (0 disables, -1 unbounded) and evict to fit it. */
  void SetMaxMemoryBytes(int64_t max_memory_bytes);

  int64_t GetMaxMemoryBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_memory_bytes_;
  }

  /*! \brief Remove every entry and reset the hit/miss/eviction counters. */
  void Clear();

  Stats GetStats() const;

  /*! \brief The estimated memory of one entry, in bytes. */
  static std::size_t EstimateEntrySize(const std::string& key, const Grammar& grammar);

 private:
  struct Entry {
    std::string key;
    Grammar grammar;
    std::size_t size;
  };
  using EntryList = std::list<Entry>;

  /*! \brief Evict least recently used entries until the budget holds. Requires the lock. */
  void EvictToFit();

  mutable std::mutex mutex_;
  int64_t max_memory_bytes_;
  // Most recently used at the front.
  EntryList lru_list_;
  // Keys are views into Entry::key; list nodes never move, so the views stay valid.
  std::unordered_map<std::string_view, EntryList::iterator> index_;
  std::size_t memory_bytes_ = 0;
  int64_t hits_ = 0;
  int64_t misses_ = 0;
  int64_t evictions_ = 0;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_GRAMMAR_CACHE_H_
