/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/json_schema_grammar_cache.cc
 */

#include "json_schema_grammar_cache.h"

#include "grammar_impl.h"

namespace xgrammar {

JSONSchemaGrammarCache& JSONSchemaGrammarCache::Global() {
  // Leaked on purpose: converters may run during static destruction of other objects.
  static JSONSchemaGrammarCache* instance = new JSONSchemaGrammarCache();
  return *instance;
}

std::size_t JSONSchemaGrammarCache::EstimateEntrySize(
    const std::string& key, const Grammar& grammar
) {
  // Count what a converted (normalized, not yet optimized) grammar holds: rules with their
  // generated names, and the flat expression storage (two header ints plus data per expression,
  // and one index entry). MemorySize(Grammar::Impl) is not used because it also walks the FSM
  // members, which are unset until the grammar is optimized.
  std::size_t size = sizeof(Entry) + sizeof(Grammar::Impl) + key.size();
  for (int32_t i = 0; i < grammar->NumRules(); ++i) {
    const auto& rule = grammar->GetRule(i);
    size += sizeof(Grammar::Impl::Rule) + rule.name.size() + rule.capture_name.size();
  }
  for (int32_t i = 0; i < grammar->NumGrammarExprs(); ++i) {
    size += (3 + grammar->GetGrammarExpr(i).data_len) * sizeof(int32_t);
  }
  // The index node and the list node.
  size += 4 * sizeof(void*) + sizeof(std::string_view);
  // The counts above use sizes, not capacities, and leave out allocator overhead. On converted
  // tool schemas, process RSS grew about 1.4x the uncorrected sum, so scale by that much to keep
  // the budget close to real memory.
  return size + size * 2 / 5;
}

std::optional<Grammar> JSONSchemaGrammarCache::Get(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_ || max_memory_bytes_ == 0) return std::nullopt;
  auto it = index_.find(std::string_view(key));
  if (it == index_.end()) {
    ++misses_;
    return std::nullopt;
  }
  ++hits_;
  lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
  return it->second->grammar;
}

void JSONSchemaGrammarCache::Put(const std::string& key, const Grammar& grammar) {
  std::size_t size = EstimateEntrySize(key, grammar);
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_ || max_memory_bytes_ == 0) return;
  // An entry larger than the whole budget would only evict everything else and then itself.
  if (max_memory_bytes_ > 0 && size > static_cast<std::size_t>(max_memory_bytes_)) return;
  auto it = index_.find(std::string_view(key));
  if (it != index_.end()) {
    // Another thread computed the same key concurrently. Keep its entry; both are equivalent.
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    return;
  }
  lru_list_.push_front(Entry{key, grammar, size});
  index_.emplace(std::string_view(lru_list_.front().key), lru_list_.begin());
  memory_bytes_ += size;
  EvictToFit();
}

void JSONSchemaGrammarCache::EvictToFit() {
  if (max_memory_bytes_ < 0) return;
  while (!lru_list_.empty() && memory_bytes_ > static_cast<std::size_t>(max_memory_bytes_)) {
    auto& victim = lru_list_.back();
    memory_bytes_ -= victim.size;
    index_.erase(std::string_view(victim.key));
    lru_list_.pop_back();
    ++evictions_;
  }
}

void JSONSchemaGrammarCache::SetMaxMemoryBytes(int64_t max_memory_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  max_memory_bytes_ = max_memory_bytes < 0 ? kUnlimited : max_memory_bytes;
  EvictToFit();
}

void JSONSchemaGrammarCache::SetEnabled(bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_);
  enabled_ = enabled;
  if (!enabled_) {
    index_.clear();
    lru_list_.clear();
    memory_bytes_ = 0;
  }
}

void JSONSchemaGrammarCache::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  index_.clear();
  lru_list_.clear();
  memory_bytes_ = 0;
  hits_ = 0;
  misses_ = 0;
  evictions_ = 0;
}

JSONSchemaGrammarCache::Stats JSONSchemaGrammarCache::GetStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  Stats stats;
  stats.hits = hits_;
  stats.misses = misses_;
  stats.evictions = evictions_;
  stats.num_entries = static_cast<int64_t>(lru_list_.size());
  stats.memory_bytes = static_cast<int64_t>(memory_bytes_);
  stats.max_memory_bytes = max_memory_bytes_;
  return stats;
}

}  // namespace xgrammar
