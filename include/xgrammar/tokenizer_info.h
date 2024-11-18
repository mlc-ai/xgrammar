/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/tokenizer_info.h
 * \brief The header for the tokenizer info.
 */

#ifndef XGRAMMAR_TOKENIZER_INFO_H_
#define XGRAMMAR_TOKENIZER_INFO_H_

#include <xgrammar/object.h>

#include <cstdint>
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
      bool prepend_space_in_tokenization = false
  );
  int GetVocabSize() const;
  VocabType GetVocabType() const;
  bool GetPrependSpaceInTokenization() const;
  const std::vector<std::string>& GetDecodedVocab() const;
  const std::vector<int32_t>& GetStopTokenIds() const;
  const std::vector<int32_t>& GetSpecialTokenIds() const;
  const std::vector<std::pair<int32_t, std::string>>& GetSortedDecodedVocab() const;

  static TokenizerInfo FromHuggingFace(
      const std::vector<std::string>& encoded_vocab, const std::string& backend_str
  );

  std::string DumpMetadata() const;
  static TokenizerInfo FromVocabAndMetadata(
      const std::vector<std::string>& encoded_vocab, const std::string& metadata
  );

  XGRAMMAR_DEFINE_PIMPL_METHODS(TokenizerInfo);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_TOKENIZER_INFO_H_
