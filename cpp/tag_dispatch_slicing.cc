/*!
 * Copyright (c) 2026 by Contributors
 * \file xgrammar/tag_dispatch_slicing.cc
 */
#include "tag_dispatch_slicing.h"

#include <algorithm>
#include <cstddef>
#include <string_view>

namespace xgrammar {
namespace {

// This search automaton is separate from the dispatch FSM: a match here includes excluded
// strings and does not enter a content rule. Sparse byte edges avoid another states-by-256
// transition table when the vocabulary scan only needs to inspect a few long tokens.
class AhoCorasickMatcher {
 public:
  AhoCorasickMatcher(const std::vector<std::string>& patterns, size_t max_length) {
    nodes_.emplace_back();
    for (const auto& pattern : patterns) {
      if (pattern.size() > max_length) {
        continue;
      }
      int32_t state = 0;
      for (unsigned char byte : pattern) {
        int32_t next = FindChild(state, byte);
        if (next == -1) {
          next = static_cast<int32_t>(nodes_.size());
          nodes_.emplace_back();
          edges_.push_back({byte, next, nodes_[state].first_edge});
          nodes_[state].first_edge = static_cast<int32_t>(edges_.size()) - 1;
        }
        state = next;
      }
      nodes_[state].matches = true;
    }

    std::vector<int32_t> pending{0};
    for (size_t head = 0; head < pending.size(); ++head) {
      int32_t state = pending[head];
      for (int32_t edge_id = nodes_[state].first_edge; edge_id != -1;
           edge_id = edges_[edge_id].next_edge) {
        const auto& edge = edges_[edge_id];
        int32_t failure = state == 0 ? 0 : Advance(nodes_[state].failure, edge.byte);
        nodes_[edge.target].failure = failure;
        // A suffix pattern can finish even when this trie node is not itself a pattern end.
        nodes_[edge.target].matches |= nodes_[failure].matches;
        pending.push_back(edge.target);
      }
    }
  }

  bool ContainsAny(std::string_view text) const {
    int32_t state = 0;
    for (unsigned char byte : text) {
      state = Advance(state, byte);
      if (nodes_[state].matches) {
        return true;
      }
    }
    return false;
  }

 private:
  struct Node {
    int32_t first_edge{-1};
    int32_t failure{0};
    bool matches{false};
  };
  struct Edge {
    uint8_t byte;
    int32_t target;
    int32_t next_edge;
  };

  int32_t FindChild(int32_t state, uint8_t byte) const {
    for (int32_t edge = nodes_[state].first_edge; edge != -1; edge = edges_[edge].next_edge) {
      if (edges_[edge].byte == byte) {
        return edges_[edge].target;
      }
    }
    return -1;
  }

  int32_t Advance(int32_t state, uint8_t byte) const {
    while (true) {
      int32_t next = FindChild(state, byte);
      if (next != -1) {
        return next;
      }
      if (state == 0) {
        return 0;
      }
      state = nodes_[state].failure;
    }
  }

  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
};

}  // namespace

DynamicBitset BuildTagDispatchSecondSlicingBitset(
    const std::vector<std::pair<int32_t, std::string>>& sorted_decoded_vocab,
    const std::vector<std::string>& patterns
) {
  DynamicBitset result(static_cast<int32_t>(sorted_decoded_vocab.size()));
  if (patterns.empty()) {
    for (int32_t index = 0; index < static_cast<int32_t>(sorted_decoded_vocab.size()); ++index) {
      result.Set(index);
    }
    return result;
  }
  size_t min_length = patterns.front().size();
  for (const auto& pattern : patterns) {
    min_length = std::min(min_length, pattern.size());
  }
  // string::find("", 1) succeeds for every nonempty token, including a one-byte token.
  // The existing fast path explicitly accepts empty tokens even when a pattern is empty.
  if (min_length == 0) {
    for (int32_t index = 0; index < static_cast<int32_t>(sorted_decoded_vocab.size()); ++index) {
      if (sorted_decoded_vocab[index].second.empty()) {
        result.Set(index);
      }
    }
    return result;
  }

  std::vector<int32_t> candidates;
  size_t max_suffix_length = 0;
  for (int32_t index = 0; index < static_cast<int32_t>(sorted_decoded_vocab.size()); ++index) {
    size_t length = sorted_decoded_vocab[index].second.size();
    if (length <= min_length) {
      result.Set(index);
    } else {
      candidates.push_back(index);
      max_suffix_length = std::max(max_suffix_length, length - 1);
    }
  }
  if (candidates.empty()) {
    return result;
  }

  AhoCorasickMatcher matcher(patterns, max_suffix_length);
  for (int32_t index : candidates) {
    if (!matcher.ContainsAny(std::string_view(sorted_decoded_vocab[index].second).substr(1))) {
      result.Set(index);
    }
  }
  return result;
}

}  // namespace xgrammar
