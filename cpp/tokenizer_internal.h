#ifndef XGRAMMAR_TOKENIZER_INTERNAL_H_
#define XGRAMMAR_TOKENIZER_INTERNAL_H_

#include <picojson.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

class TokenizerInfo::Impl {
 public:
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

  std::string DumpMetadata() const;

  static TokenizerInfo FromVocabAndMetadata(
      const std::vector<std::string>& encoded_vocab, const std::string& metadata
  );

  static std::string DetectMetadataFromHF(const std::string& backend_str);

  picojson::value SerializeToJSON() const;
  static TokenizerInfo DeserializeFromJSON(
      const picojson::value& value, const std::vector<std::string>& encoded_vocab = {}
  );

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
  /*! \brief The stop tokens. When the GrammarMatcher can reach the end of the grammar,
   * stop tokens can be accepted. */
  std::vector<int32_t> stop_token_ids_;
  /*! \brief The special tokens. These tokens are ignored (masked out) during the grammar-guided
   * generation. */
  std::vector<int32_t> special_token_ids_;

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
};

}  // namespace xgrammar

#endif
