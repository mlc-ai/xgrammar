/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/character_class_token_summary.h
 * \brief Vocabulary summaries for tokens matched by one character class.
 */
#ifndef XGRAMMAR_CHARACTER_CLASS_TOKEN_SUMMARY_H_
#define XGRAMMAR_CHARACTER_CLASS_TOKEN_SUMMARY_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "support/encoding.h"

namespace xgrammar {

struct CharacterClassTokenSummary {
  int32_t sorted_vocab_index;
  int32_t locally_consumed_characters;
  bool consumed_whole_token;
  bool has_completed_character_prefix;
};

template <typename CharacterClass>
std::vector<CharacterClassTokenSummary> BuildCharacterClassTokenSummaries(
    const CharacterClass& character_class,
    const std::vector<std::pair<int32_t, std::string>>& sorted_vocab,
    const std::vector<int32_t>& ascii_string_safe_indices
) {
  const bool is_negative = static_cast<bool>(character_class[0]);
  const auto codepoint_is_in_ranges = [&](TCodepoint codepoint) {
    for (int32_t range_index = 1; range_index < character_class.size(); range_index += 2) {
      if (codepoint >= character_class[range_index] &&
          codepoint <= character_class[range_index + 1]) {
        return true;
      }
    }
    return false;
  };
  const auto partial_codepoint_can_match =
      [&](TCodepoint partial, int32_t remaining_bytes, int32_t total_bytes) {
        if (is_negative) {
          return true;
        }
        static constexpr std::array<TCodepoint, 5> kMinCodepointByUtf8Length = {
            0, 0, 0x80, 0x800, 0x10000
        };
        const TCodepoint raw_min_codepoint = partial << (6 * remaining_bytes);
        const TCodepoint min_codepoint =
            std::max(raw_min_codepoint, kMinCodepointByUtf8Length[total_bytes]);
        const TCodepoint max_codepoint = std::min<TCodepoint>(
            raw_min_codepoint | ((TCodepoint{1} << (6 * remaining_bytes)) - 1), 0x10FFFF
        );
        if (min_codepoint > max_codepoint) {
          return false;
        }
        for (int32_t range_index = 1; range_index < character_class.size(); range_index += 2) {
          if (max_codepoint >= character_class[range_index] &&
              min_codepoint <= character_class[range_index + 1]) {
            return true;
          }
        }
        return false;
      };

  // TokenizerInfo already records tokens made entirely from the printable ASCII bytes that are
  // safe inside a JSON string. When this character class accepts that complete byte alphabet,
  // their summaries are known without decoding the same token bytes again.
  bool accepts_ascii_string_safe_alphabet = true;
  for (TCodepoint codepoint = 0x20; codepoint < 0x7f; ++codepoint) {
    if (codepoint == '"' || codepoint == '\\') {
      continue;
    }
    if (codepoint_is_in_ranges(codepoint) == is_negative) {
      accepts_ascii_string_safe_alphabet = false;
      break;
    }
  }

  std::vector<CharacterClassTokenSummary> summaries;
  summaries.reserve(sorted_vocab.size());
  size_t ascii_string_safe_position = 0;
  for (int32_t sorted_vocab_index = 0;
       sorted_vocab_index < static_cast<int32_t>(sorted_vocab.size());
       ++sorted_vocab_index) {
    const auto& token = sorted_vocab[sorted_vocab_index].second;
    if (accepts_ascii_string_safe_alphabet) {
      while (ascii_string_safe_position < ascii_string_safe_indices.size() &&
             ascii_string_safe_indices[ascii_string_safe_position] < sorted_vocab_index) {
        ++ascii_string_safe_position;
      }
      if (ascii_string_safe_position < ascii_string_safe_indices.size() &&
          ascii_string_safe_indices[ascii_string_safe_position] == sorted_vocab_index) {
        summaries.push_back(CharacterClassTokenSummary{
            sorted_vocab_index, static_cast<int32_t>(token.size()), true, true
        });
        continue;
      }
    }
    int32_t byte_offset = 0;
    int32_t completed_characters = 0;
    bool incomplete_character = false;
    bool mismatch = false;
    while (byte_offset < static_cast<int32_t>(token.size())) {
      auto [valid_first_byte, total_bytes, partial_codepoint] =
          HandleUTF8FirstByte(static_cast<uint8_t>(token[byte_offset]));
      if (!valid_first_byte) {
        mismatch = true;
        break;
      }
      if (total_bytes == 1) {
        if (codepoint_is_in_ranges(partial_codepoint) == is_negative) {
          mismatch = true;
          break;
        }
        ++completed_characters;
        ++byte_offset;
        continue;
      }

      int32_t consumed_bytes = 1;
      while (consumed_bytes < total_bytes &&
             byte_offset + consumed_bytes < static_cast<int32_t>(token.size())) {
        const uint8_t continuation = static_cast<uint8_t>(token[byte_offset + consumed_bytes]);
        if ((continuation & 0xC0) != 0x80) {
          mismatch = true;
          break;
        }
        partial_codepoint = (partial_codepoint << 6) | (continuation & 0x3F);
        ++consumed_bytes;
      }
      if (mismatch) {
        break;
      }
      if (consumed_bytes < total_bytes) {
        incomplete_character = partial_codepoint_can_match(
            partial_codepoint, total_bytes - consumed_bytes, total_bytes
        );
        mismatch = !incomplete_character;
        byte_offset = token.size();
        break;
      }
      static constexpr std::array<TCodepoint, 5> kMinCodepointByUtf8Length = {
          0, 0, 0x80, 0x800, 0x10000
      };
      if (!is_negative && (partial_codepoint < kMinCodepointByUtf8Length[total_bytes] ||
                           partial_codepoint > 0x10FFFF)) {
        mismatch = true;
        break;
      }
      if (codepoint_is_in_ranges(partial_codepoint) == is_negative) {
        mismatch = true;
        break;
      }
      ++completed_characters;
      byte_offset += total_bytes;
    }

    const bool consumed_whole_token =
        byte_offset == static_cast<int32_t>(token.size()) && !mismatch;
    if (consumed_whole_token || completed_characters > 0) {
      summaries.push_back(CharacterClassTokenSummary{
          sorted_vocab_index,
          completed_characters + static_cast<int32_t>(incomplete_character),
          consumed_whole_token,
          completed_characters > 0
      });
    }
  }
  return summaries;
}

}  // namespace xgrammar

#endif  // XGRAMMAR_CHARACTER_CLASS_TOKEN_SUMMARY_H_
