/*!
 *  Copyright (c) 2023 by Contributors
 * \file xgrammar/tokenizer_info.cc
 */

#include <picojson.h>
#include <xgrammar/tokenizer_info.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "support/encoding.h"
#include "support/logging.h"

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

  static std::shared_ptr<Impl> FromVocabAndMetadata(
      const std::vector<std::string>& encoded_vocab, const std::string& metadata
  );

  static std::string DetectMetadataFromHF(const std::string& backend_str);

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

/************* Token decoders: ByteFallback and ByteLevel *************/

class TokenDecoder {
 public:
  /*!
   * \brief Post-process a raw token to the actual token with the given post-processing method.
   */
  static std::string DecodeToken(const std::string& token, VocabType vocab_type) {
    // TODO(yixin): Avoid allocating new string in decoder calls
    if (vocab_type == VocabType::BYTE_FALLBACK) {
      return SpaceReplacerDecoder(ByteFallbackDecoder(token));
    } else if (vocab_type == VocabType::BYTE_LEVEL) {
      return ByteLevelDecoder(token);
    } else {
      return token;
    }
  }

 private:
  /*! \brief ByteFallback decoder: transform tokens like <0x1B> to hex char byte 1B */
  static std::string ByteFallbackDecoder(const std::string& token) {
    if (token.length() == 6 && token.substr(0, 3) == "<0x" && token.back() == '>') {
      int byte = 0;
      for (int i = 0; i < 2; ++i) {
        byte *= 16;
        byte += token[3 + i] >= '0' && token[3 + i] <= '9' ? token[3 + i] - '0'
                                                           : token[3 + i] - 'A' + 10;
      }
      XGRAMMAR_CHECK(byte >= 0 && byte < 256);
      return std::string(/*n=*/1, static_cast<char>(byte));
    }
    return token;
  }

  /*! \brief SpaceReplacer decoder: transform "\u2581" back to space */
  static std::string SpaceReplacerDecoder(const std::string& token) {
    // \u2581 is the unicode for "lower one eighth block"
    // UTF8 encoding for \u2581 is 0xE2 0x96 0x81
    std::string result;
    for (int i = 0; i < static_cast<int>(token.size()); ++i) {
      if (i + 2 < static_cast<int>(token.size()) && token[i] == char(0xE2) &&
          token[i + 1] == char(0x96) && token[i + 2] == char(0x81)) {
        result += ' ';
        i += 2;
      } else {
        result += token[i];
      }
    }
    return result;
  }

  /*!
   * \brief ByteLevel decoder: inverses the bytes-to-unicode transformation in the encoding
   * process as in
   * https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59
   */
  static std::string ByteLevelDecoder(const std::string& token) {
    // The inverse map of bytes_to_unicode. -1 means there is no mapping to this unicode.
    static const std::array<int, 324> char_to_byte_map = {
        // clang-format off
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
      69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
      92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
      112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, -1,
      174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
      192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
      210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
      228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
      246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 127, 128,
      129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
      147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 173
        // clang-format on
    };

    auto unicode_codepoints = ParseUTF8(token.c_str(), false);
    if (unicode_codepoints.size() == 1 && unicode_codepoints[0] == kInvalidUTF8) {
      return token;
    }

    std::string decoded;
    decoded.reserve(unicode_codepoints.size());

    for (auto unicode_codepoint : unicode_codepoints) {
      XGRAMMAR_CHECK(unicode_codepoint >= 0);
      if (unicode_codepoint >= static_cast<int>(char_to_byte_map.size()) ||
          char_to_byte_map[unicode_codepoint] == -1) {
        // If there is no mapping, return the original token
        return token;
      }
      decoded += static_cast<char>(char_to_byte_map[unicode_codepoint]);
    }
    return decoded;
  }
};

/************* Metadata detection from huggingface tokenizer.json *************/

class HFTokenizerAnalyzer {
 public:
  /*!
   * \brief Detect the vocabulary type from tokenizer.json.
   * \details Find {"type": "ByteFallback"} or {"type": "ByteLevel"} in "decoder" field of the
   * tokenizer.
   */
  static VocabType DetectVocabType(const picojson::object& hf_tokenizer_obj) {
#define CHECK_AND_WARNING(condition, message)                                                   \
  if (!(condition)) {                                                                           \
    XGRAMMAR_LOG(WARNING) << "Vocab type detection failed: (" #condition                        \
                          << ") is false: " << (message) << " Using RAW VocabType by default."; \
    return VocabType::RAW;                                                                      \
  }

    CHECK_AND_WARNING(
        hf_tokenizer_obj.count("decoder") && hf_tokenizer_obj.at("decoder").is<picojson::object>(),
        "Decoder field is not found in tokenizer.json."
    );

    auto decoder_obj = hf_tokenizer_obj.at("decoder").get<picojson::object>();
    CHECK_AND_WARNING(
        decoder_obj.count("type") && decoder_obj.at("type").is<std::string>(),
        "Type field is not found in decoder field"
    );
    auto type = decoder_obj.at("type").get<std::string>();

    std::vector<picojson::value> decoders;
    if (type == "Sequence") {
      CHECK_AND_WARNING(
          decoder_obj.count("decoders") && decoder_obj.at("decoders").is<picojson::array>(),
          "Decoders field is not found in a Sequence decoder"
      );
      decoders = decoder_obj.at("decoders").get<picojson::array>();
    } else {
      decoders.emplace_back(hf_tokenizer_obj.at("decoder"));
    }

    for (const auto& decoder : decoders) {
      CHECK_AND_WARNING(decoder.is<picojson::object>(), "Decoder is not an object");
      auto decoder_obj = decoder.get<picojson::object>();
      CHECK_AND_WARNING(
          decoder_obj.count("type") && decoder_obj.at("type").is<std::string>(),
          "Type field is not found in decoder field"
      );
      auto type = decoder_obj.at("type").get<std::string>();
      if (type == "ByteLevel") {
        return VocabType::BYTE_LEVEL;
      } else if (type == "ByteFallback") {
        return VocabType::BYTE_FALLBACK;
      }
    }

    // If neither byte_level nor byte_fallback decoder is detected, return RAW.
    return VocabType::RAW;

#undef CHECK_AND_WARNING
  }

  static bool DetectPrependNormalizer(const picojson::object& hf_tokenizer_obj) {
    if (!hf_tokenizer_obj.count("normalizer") ||
        !hf_tokenizer_obj.at("normalizer").is<picojson::object>()) {
      return false;
    }

    const picojson::value& normalizer_value = hf_tokenizer_obj.at("normalizer");
    if (!normalizer_value.is<picojson::object>()) {
      return false;
    }
    const picojson::object& normalizer_obj = normalizer_value.get<picojson::object>();
    if (!normalizer_obj.count("type") || !normalizer_obj.at("type").is<std::string>()) {
      return false;
    }
    auto type = normalizer_obj.at("type").get<std::string>();

    std::vector<picojson::value> normalizers;
    if (type == "Sequence") {
      if (!normalizer_obj.count("normalizers") ||
          !normalizer_obj.at("normalizers").is<picojson::array>()) {
        return false;
      }
      normalizers = normalizer_obj.at("normalizers").get<picojson::array>();
    } else {
      normalizers.emplace_back(normalizer_value);
    }

    for (const auto& normalizer : normalizers) {
      if (!normalizer.is<picojson::object>()) {
        continue;
      }
      auto normalizer_obj = normalizer.get<picojson::object>();
      if (!normalizer_obj.count("type") || !normalizer_obj.at("type").is<std::string>()) {
        continue;
      }
      auto type = normalizer_obj.at("type").get<std::string>();
      if (type == "Prepend" && normalizer_obj.count("prepend") &&
          normalizer_obj.at("prepend").is<std::string>() &&
          normalizer_obj.at("prepend").get<std::string>() == "▁") {
        return true;
      }
    }
    return false;
  }

  static bool DetectMetaspacePreTokenizer(const picojson::object& hf_tokenizer_obj) {
    if (!hf_tokenizer_obj.count("pre_tokenizer") ||
        !hf_tokenizer_obj.at("pre_tokenizer").is<picojson::object>()) {
      return false;
    }
    auto pre_tokenizer_obj = hf_tokenizer_obj.at("pre_tokenizer").get<picojson::object>();
    if (!pre_tokenizer_obj.count("type") || !pre_tokenizer_obj.at("type").is<std::string>()) {
      return false;
    }
    auto type = pre_tokenizer_obj.at("type").get<std::string>();
    if (!pre_tokenizer_obj.count("prepend_scheme") ||
        !pre_tokenizer_obj.at("prepend_scheme").is<std::string>()) {
      return false;
    }
    auto prepend_scheme = pre_tokenizer_obj.at("prepend_scheme").get<std::string>();
    return type == "Metaspace" && (prepend_scheme == "always" || prepend_scheme == "first");
  }

  /*!
   * \brief Detect whether add prefix space from tokenizer.json.
   * \details Find {"type": "Prepend", "prepend": "▁"} in "normalizer" field of the tokenizer, or
   * "pre_tokenizer": {"type": "Metaspace", "prepend_scheme": "always" | "first"} in the tokenizer.
   */
  static bool DetectAddPrefixSpace(const picojson::object& hf_tokenizer_obj) {
    return DetectPrependNormalizer(hf_tokenizer_obj) ||
           DetectMetaspacePreTokenizer(hf_tokenizer_obj);
  }
};

/************* TokenizerInfo::Impl *************/

bool TokenizerInfo::Impl::IsSpecialToken(const std::string& token) { return token == ""; }

TokenizerInfo::Impl::Impl(
    const std::vector<std::string>& encoded_vocab,
    VocabType vocab_type,
    std::optional<int> vocab_size,
    std::optional<std::vector<int32_t>> stop_token_ids,
    bool add_prefix_space
)
    : vocab_type_(vocab_type),
      vocab_size_(vocab_size.value_or(encoded_vocab.size())),
      add_prefix_space_(add_prefix_space) {
  decoded_vocab_.reserve(encoded_vocab.size());
  sorted_decoded_vocab_.reserve(encoded_vocab.size());
  for (int i = 0; i < static_cast<int>(encoded_vocab.size()); ++i) {
    const std::string& token = TokenDecoder::DecodeToken(encoded_vocab[i], vocab_type_);
    decoded_vocab_.push_back(token);
    if ((!stop_token_ids && DETECTION_STOP_TOKENS.count(token)) ||
        (stop_token_ids &&
         std::find(stop_token_ids->begin(), stop_token_ids->end(), i) != stop_token_ids->end())) {
      stop_token_ids_.push_back(i);
    } else if (IsSpecialToken(token)) {
      special_token_ids_.push_back(i);
    } else {
      sorted_decoded_vocab_.push_back({i, token});
    }
  }
  for (int i = encoded_vocab.size(); i < vocab_size_; ++i) {
    special_token_ids_.push_back(i);
  }

  auto f_compare_token = [](const std::pair<int32_t, std::string>& a,
                            const std::pair<int32_t, std::string>& b) {
    return a.second < b.second;
  };
  std::sort(sorted_decoded_vocab_.begin(), sorted_decoded_vocab_.end(), f_compare_token);
}

std::string TokenizerInfo::Impl::DumpMetadata() const {
  picojson::object obj;
  obj["vocab_type"] = picojson::value(static_cast<int64_t>(vocab_type_));
  obj["vocab_size"] = picojson::value(static_cast<int64_t>(vocab_size_));
  obj["add_prefix_space"] = picojson::value(add_prefix_space_);
  picojson::array stop_token_ids_array;
  for (auto id : stop_token_ids_) {
    stop_token_ids_array.push_back(picojson::value(static_cast<int64_t>(id)));
  }
  obj["stop_token_ids"] = picojson::value(stop_token_ids_array);

  return picojson::value(obj).serialize(false);
}

std::shared_ptr<TokenizerInfo::Impl> TokenizerInfo::Impl::FromVocabAndMetadata(
    const std::vector<std::string>& encoded_vocab, const std::string& metadata
) {
  picojson::value v;
  std::string err = picojson::parse(v, metadata);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse metadata: " << err;

  const picojson::object& obj = v.get<picojson::object>();

  XGRAMMAR_CHECK(obj.count("vocab_type") && obj["vocab_type"].is<std::int64_t>())
      << "Missing or invalid 'vocab_type' in metadata";
  int vocab_type_int = static_cast<int>(obj["vocab_type"].get<int64_t>());
  XGRAMMAR_CHECK(vocab_type_int == 0 || vocab_type_int == 1 || vocab_type_int == 2)
      << "Invalid vocab_type in metadata: " << vocab_type_int;
  VocabType vocab_type = static_cast<VocabType>(vocab_type_int);

  XGRAMMAR_CHECK(obj.count("vocab_size") && obj["vocab_size"].is<int64_t>())
      << "Missing or invalid 'vocab_size' in metadata";
  int vocab_size = static_cast<int>(obj["vocab_size"].get<int64_t>());

  XGRAMMAR_CHECK(obj.count("add_prefix_space") && obj["add_prefix_space"].is<bool>())
      << "Missing or invalid 'add_prefix_space' in metadata";
  bool add_prefix_space = obj["add_prefix_space"].get<bool>();

  std::vector<int32_t> stop_token_ids;
  XGRAMMAR_CHECK(obj.count("stop_token_ids") && obj["stop_token_ids"].is<picojson::array>())
      << "Missing or invalid 'stop_token_ids' in metadata";
  for (const auto& id : obj["stop_token_ids"].get<picojson::array>()) {
    XGRAMMAR_CHECK(id.is<int64_t>()) << "Stop token id is not an integer";
    stop_token_ids.push_back(static_cast<int32_t>(id.get<int64_t>()));
  }
  return std::make_shared<Impl>(
      encoded_vocab, vocab_type, vocab_size, stop_token_ids, add_prefix_space
  );
}

std::string TokenizerInfo::Impl::DetectMetadataFromHF(const std::string& backend_str) {
  picojson::value v;
  std::string err = picojson::parse(v, backend_str);
  XGRAMMAR_CHECK(err.empty() && v.is<picojson::object>()) << "Failed to parse JSON object: " << err;
  const picojson::object& obj = v.get<picojson::object>();
  VocabType vocab_type = HFTokenizerAnalyzer::DetectVocabType(obj);
  bool add_prefix_space = HFTokenizerAnalyzer::DetectAddPrefixSpace(obj);

  // Serialize the metadata
  picojson::object metadata_obj;
  metadata_obj["vocab_type"] = picojson::value(static_cast<int64_t>(vocab_type));
  metadata_obj["add_prefix_space"] = picojson::value(add_prefix_space);
  return picojson::value(metadata_obj).serialize(false);
}

/************* TokenizerInfo *************/

TokenizerInfo::TokenizerInfo(
    const std::vector<std::string>& encoded_vocab,
    VocabType vocab_type,
    std::optional<int> vocab_size,
    std::optional<std::vector<int32_t>> stop_token_ids,
    bool add_prefix_space
)
    : pimpl_(std::make_shared<Impl>(
          encoded_vocab, vocab_type, vocab_size, stop_token_ids, add_prefix_space
      )) {}

int TokenizerInfo::GetVocabSize() const { return pimpl_->GetVocabSize(); }
VocabType TokenizerInfo::GetVocabType() const { return pimpl_->GetVocabType(); }
bool TokenizerInfo::GetAddPrefixSpace() const { return pimpl_->GetAddPrefixSpace(); }
const std::vector<std::string>& TokenizerInfo::GetDecodedVocab() const {
  return pimpl_->GetDecodedVocab();
}
const std::vector<int32_t>& TokenizerInfo::GetStopTokenIds() const {
  return pimpl_->GetStopTokenIds();
}
const std::vector<int32_t>& TokenizerInfo::GetSpecialTokenIds() const {
  return pimpl_->GetSpecialTokenIds();
}
const std::vector<std::pair<int32_t, std::string>>& TokenizerInfo::GetSortedDecodedVocab() const {
  return pimpl_->GetSortedDecodedVocab();
}

std::string TokenizerInfo::DumpMetadata() const { return pimpl_->DumpMetadata(); }

TokenizerInfo TokenizerInfo::FromVocabAndMetadata(
    const std::vector<std::string>& encoded_vocab, const std::string& metadata
) {
  return TokenizerInfo(Impl::FromVocabAndMetadata(encoded_vocab, metadata));
}

std::string TokenizerInfo::DetectMetadataFromHF(const std::string& backend_str) {
  return Impl::DetectMetadataFromHF(backend_str);
}

}  // namespace xgrammar
