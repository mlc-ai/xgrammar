/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizer.cc
 */

#include <picojson.h>
#include <xgrammar/xgrammar.h>

#include <array>
#include <chrono>
#include <memory>
#include <unordered_map>

#include "support/encoding.h"
#include "support/logging.h"

namespace xgrammar {

class TokenizerInfo::Impl {
 public:
  Impl(
      const std::vector<std::string>& encoded_vocab,
      VocabType vocab_type,
      bool prepend_space_in_tokenization
  );

  int GetVocabSize() const { return decoded_vocab_.size(); }
  VocabType GetVocabType() const { return vocab_type_; }
  bool GetPrependSpaceInTokenization() const { return prepend_space_in_tokenization_; }
  const std::vector<std::string>& GetDecodedVocab() { return decoded_vocab_; }
  std::string DumpMetadata() const;

 private:
  VocabType vocab_type_;
  bool prepend_space_in_tokenization_;
  std::vector<std::string> decoded_vocab_;
};

/************* Metadata detection from huggingface tokenizer.json *************/

/*!
 * \brief Detect the vocabulary type from tokenizer.json.
 * \details Find {"type": "ByteFallback"} or {"type": "ByteLevel"} in "decoder" field of the
 * tokenizer.
 */
inline VocabType DetectVocabType(const picojson::object& hf_tokenizer_obj) {
#define CHECK_AND_WARNING(condition, message)                                                    \
  if (!(condition)) {                                                                            \
    XGRAMMAR_LOG(WARNING) << "Vocab type detection failed: (" #condition                         \
                          << ") is false: " << (message) << " Using VocabType::RAW as default."; \
    return VocabType::RAW;                                                                       \
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

/*!
 * \brief Detect whether prepend space in tokenization from tokenizer.json.
 * \details Find {"type": "Prepend", "prepend": "▁"} in "normalizer" field of the tokenizer.
 */
inline bool DetectPrependSpaceInTokenization(const picojson::object& hf_tokenizer_obj) {
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

/************* Token decoders: ByteFallback and ByteLevel *************/

/*! \brief ByteFallback decoder: transform tokens like <0x1B> to hex char byte 1B */
inline std::string ByteFallbackDecoder(const std::string& token) {
  if (token.length() == 6 && token.substr(0, 3) == "<0x" && token.back() == '>') {
    int byte = 0;
    for (int i = 0; i < 2; ++i) {
      byte *= 16;
      byte +=
          token[3 + i] >= '0' && token[3 + i] <= '9' ? token[3 + i] - '0' : token[3 + i] - 'A' + 10;
    }
    XGRAMMAR_CHECK(byte >= 0 && byte < 256);
    return std::string(/*n=*/1, static_cast<char>(byte));
  }
  return token;
}

/*! \brief SpaceReplacer decoder: transform "\u2581" back to space */
inline std::string SpaceReplacerDecoder(const std::string& token) {
  // \u2581 is the unicode for "lower one eighth block"
  // UTF8 encoding for \u2581 is 0xE2 0x96 0x81
  std::string result;
  for (size_t i = 0; i < token.size(); ++i) {
    if (i + 2 < token.size() && token[i] == char(0xE2) && token[i + 1] == char(0x96) &&
        token[i + 2] == char(0x81)) {
      result += ' ';
      i += 2;
    } else {
      result += token[i];
    }
  }
  return result;
}

/*!
 * \brief ByteLevel decoder: inverses the bytes-to-unicode transformation in the encoding process
 * as in
 * https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59
 */
inline std::string ByteLevelDecoder(const std::string& token) {
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

/*!
 * \brief Post-process a raw token to the actual token with the given post-processing method.
 */
inline std::string DecodeToken(const std::string& token, VocabType vocab_type) {
  if (vocab_type == VocabType::BYTE_FALLBACK) {
    return SpaceReplacerDecoder(ByteFallbackDecoder(token));
  } else if (vocab_type == VocabType::BYTE_LEVEL) {
    return ByteLevelDecoder(token);
  } else {
    return token;
  }
}

/************* TokenizerInfo *************/

TokenizerInfo::Impl::Impl(
    const std::vector<std::string>& encoded_vocab,
    VocabType vocab_type,
    bool prepend_space_in_tokenization
)
    : vocab_type_(vocab_type), prepend_space_in_tokenization_(prepend_space_in_tokenization) {
  decoded_vocab_.reserve(encoded_vocab.size());
  for (const auto& item : encoded_vocab) {
    decoded_vocab_.emplace_back(DecodeToken(item, vocab_type_));
  }
}

std::string TokenizerInfo::Impl::DumpMetadata() const {
  static const std::string VOCAB_TYPE_NAMES[] = {"RAW", "BYTE_FALLBACK", "BYTE_LEVEL"};
  picojson::object obj;
  obj["vocab_type"] = picojson::value(VOCAB_TYPE_NAMES[static_cast<int>(vocab_type_)]);
  obj["prepend_space_in_tokenization"] = picojson::value(prepend_space_in_tokenization_);
  return picojson::value(obj).serialize(false);
}

TokenizerInfo::TokenizerInfo(
    const std::vector<std::string>& encoded_vocab,
    VocabType vocab_type,
    bool prepend_space_in_tokenization
)
    : pimpl_(std::make_shared<Impl>(encoded_vocab, vocab_type, prepend_space_in_tokenization)) {}

int TokenizerInfo::GetVocabSize() const { return pimpl_->GetVocabSize(); }
VocabType TokenizerInfo::GetVocabType() const { return pimpl_->GetVocabType(); }
bool TokenizerInfo::GetPrependSpaceInTokenization() const {
  return pimpl_->GetPrependSpaceInTokenization();
}
const std::vector<std::string>& TokenizerInfo::GetDecodedVocab() const {
  return pimpl_->GetDecodedVocab();
}
std::string TokenizerInfo::DumpMetadata() const { return pimpl_->DumpMetadata(); }

TokenizerInfo TokenizerInfo::FromVocabAndMetadata(
    const std::vector<std::string>& encoded_vocab, const std::string& metadata
) {
  static const std::unordered_map<std::string, VocabType> VOCAB_TYPE_MAP = {
      {"RAW", VocabType::RAW},
      {"BYTE_FALLBACK", VocabType::BYTE_FALLBACK},
      {"BYTE_LEVEL", VocabType::BYTE_LEVEL},
  };

  picojson::value v;
  std::string err = picojson::parse(v, metadata);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse metadata: " << err;
  const picojson::object& obj = v.get<picojson::object>();
  XGRAMMAR_CHECK(obj.count("vocab_type") && obj["vocab_type"].is<std::string>())
      << "Missing or invalid 'vocab_type' in metadata";
  XGRAMMAR_CHECK(
      obj.count("prepend_space_in_tokenization") && obj["prepend_space_in_tokenization"].is<bool>()
  ) << "Missing or invalid 'prepend_space_in_tokenization' in metadata";

  std::string vocab_type_str = obj["vocab_type"].get<std::string>();
  VocabType vocab_type;
  if (vocab_type_str == "RAW") {
    vocab_type = VocabType::RAW;
  } else if (vocab_type_str == "BYTE_FALLBACK") {
    vocab_type = VocabType::BYTE_FALLBACK;
  } else if (vocab_type_str == "BYTE_LEVEL") {
    vocab_type = VocabType::BYTE_LEVEL;
  } else {
    XGRAMMAR_CHECK(false) << "Invalid vocab_type in metadata: " << vocab_type_str;
  }

  bool prepend_space_in_tokenization = obj["prepend_space_in_tokenization"].get<bool>();
  return TokenizerInfo(encoded_vocab, vocab_type, prepend_space_in_tokenization);
}

TokenizerInfo TokenizerInfo::FromHuggingFace(
    const std::vector<std::string>& encoded_vocab, const std::string& backend_str
) {
  picojson::value v;
  std::string err = picojson::parse(v, backend_str);
  if (!err.empty() || !v.is<picojson::object>()) {
    XGRAMMAR_LOG(WARNING) << "Failed to parse JSON object. " << err;
    return TokenizerInfo(encoded_vocab, VocabType::RAW, false);
  }
  const picojson::object& obj = v.get<picojson::object>();

  VocabType vocab_type = DetectVocabType(obj);
  bool prepend_space_in_tokenization = DetectPrependSpaceInTokenization(obj);
  return TokenizerInfo(encoded_vocab, vocab_type, prepend_space_in_tokenization);
}

}  // namespace xgrammar
