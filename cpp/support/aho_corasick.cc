/*!
 * Copyright (c) 2026 by Contributors
 * \file xgrammar/support/aho_corasick.cc
 */

#include "aho_corasick.h"

#include <algorithm>
#include <queue>

namespace xgrammar {

AhoCorasick::AhoCorasick(const std::vector<std::string>& patterns) {
  size_t max_num_nodes = 1;
  for (const auto& pattern : patterns) {
    max_num_nodes += pattern.size();
  }
  nodes_.reserve(max_num_nodes);
  nodes_.emplace_back();

  // Build the trie.
  for (const auto& pattern : patterns) {
    int32_t state = 0;
    for (unsigned char byte : pattern) {
      int32_t& next_state = nodes_[state].next[byte];
      if (next_state == -1) {
        next_state = static_cast<int32_t>(nodes_.size());
        nodes_.emplace_back();
      }
      state = next_state;
    }
    nodes_[state].is_terminal = true;
  }

  // Resolve root transitions and seed the breadth-first traversal.
  std::queue<int32_t> queue;
  for (int32_t byte = 0; byte < 256; ++byte) {
    int32_t child = nodes_[0].next[byte];
    if (child == -1) {
      nodes_[0].next[byte] = 0;
    } else {
      nodes_[child].failure = 0;
      queue.push(child);
    }
  }

  // Compute full failure links, propagate terminal outputs through those links, and resolve
  // missing transitions. Propagating terminal outputs is required for cases such as patterns
  // {"bc", "abcd"} and input "abc": the trie state for "abc" is not directly terminal, but its
  // failure state for the suffix "bc" is.
  while (!queue.empty()) {
    int32_t state = queue.front();
    queue.pop();

    const int32_t failure = nodes_[state].failure;
    nodes_[state].is_terminal = nodes_[state].is_terminal || nodes_[failure].is_terminal;
    for (int32_t byte = 0; byte < 256; ++byte) {
      int32_t child = nodes_[state].next[byte];
      if (child == -1) {
        nodes_[state].next[byte] = nodes_[failure].next[byte];
      } else {
        nodes_[child].failure = nodes_[failure].next[byte];
        queue.push(child);
      }
    }
  }
}

bool AhoCorasick::ContainsMatch(std::string_view text, size_t start_offset) const {
  if (start_offset > text.size()) {
    return false;
  }
  int32_t state = 0;
  if (nodes_[state].is_terminal) {
    return true;
  }
  for (size_t index = start_offset; index < text.size(); ++index) {
    state = nodes_[state].next[static_cast<uint8_t>(text[index])];
    if (nodes_[state].is_terminal) {
      return true;
    }
  }
  return false;
}

}  // namespace xgrammar
