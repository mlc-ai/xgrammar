#ifndef XGRAMMAR_TOKENIZER_INFO_IMPL_H_
#define XGRAMMAR_TOKENIZER_INFO_IMPL_H_

#include <picojson.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "support/dynamic_bitset.h"
#include "support/reflection.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

class TokenizerInfo::Impl {
 public:
  explicit Impl() = default;

  Impl(
      const std::vector<std::string>& encoded_vocab,
      VocabType vocab_type,
      std::optional<int> vocab_size,
      std::optional<std::vector<int32_t>> stop_token_ids,
      bool add_prefix_space
  );

  VocabType GetVocabType() const { return vocab_type_; }
  bool GetAddPrefixSpace() const { return add_prefix_space_; }
  int GetVocabSize() const { return vocab_size_; }
  const std::vector<std::string>& GetDecodedVocab() { return decoded_vocab_; }
  const std::vector<int32_t>& GetStopTokenIds() const { return stop_token_ids_; }
  const std::vector<int32_t>& GetSpecialTokenIds() const { return special_token_ids_; }
  const std::vector<std::pair<int32_t, std::string>>& GetSortedDecodedVocab() const {
    return sorted_decoded_vocab_;
  }
  const std::vector<int32_t>& GetTrieSubtreeNodesRange() const { return trie_subtree_nodes_range_; }
  const std::vector<int32_t>& GetTokenIdToSortedVocabIndex() const {
    return token_id_to_sorted_vocab_index_;
  }
  const std::vector<int32_t>& GetTokenCharCounts() const;
  int32_t GetMaxTokenChars() const;
  const std::vector<int32_t>& GetJSONStringPlainQuoteTokenIndicesByPrefixCount() const {
    return json_string_plain_quote_token_indices_by_prefix_count_;
  }
  const std::vector<int32_t>& GetJSONStringEscapedQuoteTokenIndices() const {
    return json_string_escaped_quote_token_indices_;
  }
  const std::vector<int32_t>& GetJSONStringEscapedTokenIndices() const {
    return json_string_escaped_token_indices_;
  }
  const DynamicBitset* GetJSONStringPlainPrefixWithinLimitBitset(int32_t limit) const {
    return limit >= 0 &&
                   limit <
                       static_cast<int32_t>(json_string_plain_prefix_within_limit_bitsets_.size())
               ? &json_string_plain_prefix_within_limit_bitsets_[limit]
               : nullptr;
  }
  const std::vector<uint8_t>& GetJSONStringQuoteTokenFlags() const {
    return json_string_quote_token_flags_;
  }
  const std::vector<int32_t>& GetJSONStringPlainPrefixCharCounts() const {
    return json_string_plain_prefix_char_counts_;
  }
  const std::vector<int32_t>& GetTokenIndicesByDescendingCharCount() const {
    return token_indices_by_descending_char_count_;
  }
  const std::vector<int32_t>& GetAsciiStringSafeIndices() const {
    return ascii_string_safe_indices_;
  }
  void BuildTokenCharData();

  std::string DumpMetadata() const;
  picojson::value DumpMetadataValue() const;

  static std::shared_ptr<TokenizerInfo::Impl> FromVocabAndMetadata(
      const std::vector<std::string>& encoded_vocab, const std::string& metadata
  );

  std::optional<std::runtime_error> CheckMetadataMatch(const picojson::value& metadata) const;

  static std::string DetectMetadataFromHF(const std::string& backend_str);

  bool operator==(const Impl& other) const;

 private:
  static bool IsSpecialToken(const std::string& decoded_token);

  /*! \brief The vocabulary type. */
  VocabType vocab_type_;
  /*! \brief The size of the vocabulary. */
  int vocab_size_;
  /*! \brief Whether to add prefix space. */
  bool add_prefix_space_;

  /*! \brief The vocabulary. Special tokens are included. */
  std::vector<std::string> decoded_vocab_;
  /*! \brief All (id, token) pairs sorted in lexicographic order. This sorting is done to
   * maximize prefix reuse during matching. Special tokens and stop tokens are not included. */
  std::vector<std::pair<int32_t, std::string>> sorted_decoded_vocab_;
  /*! \brief A pesudo-trie. trie_subtree_nodes_range[i] stores how many nodes there are in the
   * subtree. */
  std::vector<int32_t> trie_subtree_nodes_range_;
  /*! \brief The stop tokens. When the GrammarMatcher can reach the end of the grammar,
   * stop tokens can be accepted. */
  std::vector<int32_t> stop_token_ids_;
  /*! \brief The special tokens. These tokens are ignored (masked out) during the grammar-guided
   * generation. */
  std::vector<int32_t> special_token_ids_;
  /*! \brief Reverse mapping: token_id -> index in sorted_decoded_vocab_. -1 if not present. */
  std::vector<int32_t> token_id_to_sorted_vocab_index_;
  /*! \brief Unicode codepoint counts for the sorted decoded vocabulary. */
  int32_t max_token_chars_ = 0;
  std::vector<int32_t> token_char_counts_;
  /*! \brief Plain quote-token indices ordered by the character count before the first quote. */
  std::vector<int32_t> json_string_plain_quote_token_indices_by_prefix_count_;
  /*! \brief Quote-token indices containing a backslash and requiring conservative scanning. */
  std::vector<int32_t> json_string_escaped_quote_token_indices_;
  /*! \brief All token indices containing a backslash and requiring decoded-length scanning. */
  std::vector<int32_t> json_string_escaped_token_indices_;
  /*! \brief Plain-token ID masks whose pre-quote decoded length is at most a small limit. */
  std::vector<DynamicBitset> json_string_plain_prefix_within_limit_bitsets_;
  /*! \brief Whether each sorted-vocabulary token contains a JSON quote byte. */
  std::vector<uint8_t> json_string_quote_token_flags_;
  /*! \brief Character count before the first quote, or -1 when an escape requires scanning. */
  std::vector<int32_t> json_string_plain_prefix_char_counts_;
  /*! \brief Sorted-vocabulary indices ordered by descending Unicode codepoint count. */
  std::vector<int32_t> token_indices_by_descending_char_count_;
  /*! \brief Sorted-vocabulary indices safe inside an unescaped JSON string. */
  std::vector<int32_t> ascii_string_safe_indices_;

  /*!
   * \brief The tokens used to detect stop tokens from the vocabulary.
   *
   * LLaMA2: </s>
   * LLaMA3: <|end_of_text|>, <|eot_id|>
   * Phi-2: <|endoftext|>
   * Gemma: <eos>, <end_of_turn>
   * DeepSeek-V2: <｜end▁of▁sentence｜>
   */
  inline static const std::unordered_set<std::string> DETECTION_STOP_TOKENS = {
      "</s>",
      "<|end_of_text|>",
      "<|eot_id|>",
      "<|endoftext|>",
      "<eos>",
      "<|eos|>",
      "<end_of_turn>",
      "<｜end▁of▁sentence｜>"
  };

  friend struct member_trait<Impl>;
};

XGRAMMAR_MEMBER_TABLE(
    TokenizerInfo::Impl,
    "vocab_type",
    &TokenizerInfo::Impl::vocab_type_,
    "vocab_size",
    &TokenizerInfo::Impl::vocab_size_,
    "add_prefix_space",
    &TokenizerInfo::Impl::add_prefix_space_,
    "stop_token_ids",
    &TokenizerInfo::Impl::stop_token_ids_,
    "special_token_ids",
    &TokenizerInfo::Impl::special_token_ids_,
    "decoded_vocab",
    &TokenizerInfo::Impl::decoded_vocab_,
    "sorted_decoded_vocab",
    &TokenizerInfo::Impl::sorted_decoded_vocab_,
    "trie_subtree_nodes_range",
    &TokenizerInfo::Impl::trie_subtree_nodes_range_
);

}  // namespace xgrammar

#endif  // XGRAMMAR_TOKENIZER_INFO_IMPL_H_
