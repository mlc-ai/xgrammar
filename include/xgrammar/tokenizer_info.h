/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/tokenizer_info.h
 * \brief The header for the tokenizer info.
 */

#ifndef XGRAMMAR_TOKENIZER_INFO_H_
#define XGRAMMAR_TOKENIZER_INFO_H_

#include <xgrammar/object.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xgrammar {

enum class VocabType : int {
  RAW = 0,
  BYTE_FALLBACK = 1,
  BYTE_LEVEL = 2,
};

class TokenizerInfo {
 public:
  TokenizerInfo(
      const std::vector<std::string>& encoded_vocab,
      VocabType vocab_type = VocabType::RAW,
      std::optional<int> vocab_size = std::nullopt,
      std::optional<std::vector<int32_t>> stop_token_ids = std::nullopt,
      bool add_prefix_space = false
  );

  VocabType GetVocabType() const;
  bool GetAddPrefixSpace() const;
  int GetVocabSize() const;
  const std::vector<std::string>& GetDecodedVocab() const;
  const std::vector<int32_t>& GetStopTokenIds() const;
  const std::vector<int32_t>& GetSpecialTokenIds() const;
  const std::vector<std::pair<int32_t, std::string>>& GetSortedDecodedVocab() const;
  std::string DumpMetadata() const;

  static TokenizerInfo FromVocabAndMetadata(
      const std::vector<std::string>& encoded_vocab, const std::string& metadata
  );

  static std::string DetectMetadataFromHF(const std::string& backend_str);

  XGRAMMAR_DEFINE_PIMPL_METHODS(TokenizerInfo);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_TOKENIZER_INFO_H_
