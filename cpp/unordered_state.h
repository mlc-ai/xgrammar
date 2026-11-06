/*!
 * Copyright (c) 2026 by Contributors
 * \file xgrammar/unordered_state.h
 * \brief Immutable property sets owned by an Earley parser's history.
 */
#ifndef XGRAMMAR_UNORDERED_STATE_H_
#define XGRAMMAR_UNORDERED_STATE_H_

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "support/utils.h"

namespace xgrammar {

class UnorderedStateArena {
 public:
  struct State {
    uint64_t first_word;
    uint64_t hash;
    int32_t rule_id;
    int32_t count;
    int32_t remaining_required;
    int32_t history_row;
    size_t extra_begin;
  };

  using Checkpoint = size_t;

  const State& operator[](int32_t id) const {
    XGRAMMAR_DCHECK(id >= 0 && static_cast<size_t>(id) < states_.size());
    return states_[id];
  }

  bool Contains(int32_t id, int32_t entry) const {
    return id >= 0 && (Word(states_[id], entry / 64) & (uint64_t{1} << (entry % 64)));
  }

  int32_t Insert(
      int32_t previous,
      int32_t rule_id,
      int32_t entry,
      int32_t entry_count,
      int32_t required_count,
      bool required,
      int32_t history_row
  ) {
    State next = previous < 0 ? State{0, HashCombine(rule_id), rule_id, 0, required_count, 0, 0}
                              : states_[previous];
    XGRAMMAR_DCHECK(!Contains(previous, entry));
    const int32_t changed_word = entry / 64;
    const uint64_t bit = uint64_t{1} << (entry % 64);
    const int32_t word_count = (entry_count + 63) / 64;
    auto word = [&](int32_t index) {
      return (previous < 0 ? uint64_t{0} : Word(states_[previous], index)) |
             (index == changed_word ? bit : uint64_t{0});
    };
    // Hash the set, not the path: different derivations of the same keys share an ID.
    next.hash = HashCombine(rule_id, word(0));
    for (int32_t i = 1; i < word_count; ++i) HashCombineBinary(next.hash, word(i));
    ++next.count;
    next.remaining_required -= required;
    auto [begin, end] = index_.equal_range(next.hash);
    for (auto it = begin; it != end; ++it) {
      const auto& candidate = states_[it->second];
      if (candidate.rule_id != rule_id || candidate.count != next.count) continue;
      bool equal = true;
      for (int32_t i = 0; i < word_count; ++i) {
        if (Word(candidate, i) != word(i)) {
          equal = false;
          break;
        }
      }
      if (equal) return it->second;
    }
    next.first_word = word(0);
    next.extra_begin = extra_words_.size();
    for (int32_t i = 1; i < word_count; ++i) extra_words_.push_back(word(i));
    // Retained atomic states can precede a byte-path update of an earlier row.
    next.history_row =
        states_.empty() ? history_row : std::max(history_row, states_.back().history_row);
    int32_t id = states_.size();
    states_.push_back(next);
    index_.emplace(next.hash, id);
    return id;
  }

  Checkpoint Save() const { return states_.size(); }

  void Restore(Checkpoint checkpoint) {
    XGRAMMAR_DCHECK(checkpoint <= states_.size());
    while (states_.size() > checkpoint) {
      auto [begin, end] = index_.equal_range(states_.back().hash);
      for (auto it = begin; it != end; ++it) {
        if (it->second == static_cast<int32_t>(states_.size() - 1)) {
          index_.erase(it);
          break;
        }
      }
      extra_words_.resize(states_.back().extra_begin);
      states_.pop_back();
    }
  }

  void Rewind(int32_t row_count) {
    if (retained_) return;
    auto checkpoint = states_.size();
    while (checkpoint > 0 && states_[checkpoint - 1].history_row >= row_count) --checkpoint;
    Restore(checkpoint);
  }

  void Clear() {
    states_.clear();
    extra_words_.clear();
    index_.clear();
  }

  // Atomic and byte paths temporarily own states outside the parser's history rows.
  class RetainScope {
   public:
    explicit RetainScope(UnorderedStateArena& arena) : arena_(arena), previous_(arena.retained_) {
      arena_.retained_ = true;
    }
    ~RetainScope() { arena_.retained_ = previous_; }
    RetainScope(const RetainScope&) = delete;
    RetainScope& operator=(const RetainScope&) = delete;

   private:
    UnorderedStateArena& arena_;
    bool previous_;
  };

 private:
  uint64_t Word(const State& state, int32_t index) const {
    return index == 0 ? state.first_word : extra_words_[state.extra_begin + index - 1];
  }

  std::vector<State> states_;
  std::vector<uint64_t> extra_words_;
  std::unordered_multimap<uint64_t, int32_t> index_;
  bool retained_ = false;
};

}  // namespace xgrammar
#endif  // XGRAMMAR_UNORDERED_STATE_H_
