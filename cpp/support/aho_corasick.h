/*!
 * Copyright (c) 2026 by Contributors
 * \file xgrammar/support/aho_corasick.h
 * \brief A byte-oriented Aho-Corasick matcher.
 */
#ifndef XGRAMMAR_SUPPORT_AHO_CORASICK_H_
#define XGRAMMAR_SUPPORT_AHO_CORASICK_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace xgrammar {

/*!
 * \brief A byte-oriented Aho-Corasick matcher for finding any of a set of patterns.
 *
 * The transition table is fully resolved when the matcher is constructed. Matching is therefore
 * one table lookup per input byte, with no failure-link loop on the hot path.
 */
class AhoCorasick {
 public:
  explicit AhoCorasick(const std::vector<std::string>& patterns);

  /*!
   * \brief Return whether text contains any pattern at or after start_offset.
   */
  bool ContainsMatch(std::string_view text, size_t start_offset = 0) const;

  int32_t NumStates() const { return static_cast<int32_t>(nodes_.size()); }

 private:
  struct Node {
    Node() { next.fill(-1); }

    std::array<int32_t, 256> next;
    int32_t failure = 0;
    bool is_terminal = false;
  };

  std::vector<Node> nodes_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_AHO_CORASICK_H_
