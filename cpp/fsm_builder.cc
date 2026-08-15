/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.cc
 */
#include "fsm_builder.h"

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "fsm.h"
#include "grammar_builder.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "support/unicode_case_folding.h"
#include "support/unicode_char_class.h"
#include "support/unicode_regex_char_class.h"
#include "support/utils.h"

namespace xgrammar {

/******************** Packed UTF-8 range helpers ********************/

uint32_t CodepointToPackedUTF8(uint32_t codepoint) {
  if (codepoint <= 0x7F) {
    // 1-byte sequence (ASCII)
    return codepoint;
  } else if (codepoint <= 0x7FF) {
    // 2-byte sequence: byte0 = 110xxxxx, byte1 = 10xxxxxx
    uint8_t byte0 = 0xC0 | ((codepoint >> 6) & 0x1F);
    uint8_t byte1 = 0x80 | (codepoint & 0x3F);
    return (static_cast<uint32_t>(byte0) << 8) | byte1;
  } else if (codepoint <= 0xFFFF) {
    // 3-byte sequence: byte0 = 1110xxxx, byte1 = 10xxxxxx, byte2 = 10xxxxxx
    uint8_t byte0 = 0xE0 | ((codepoint >> 12) & 0x0F);
    uint8_t byte1 = 0x80 | ((codepoint >> 6) & 0x3F);
    uint8_t byte2 = 0x80 | (codepoint & 0x3F);
    return (static_cast<uint32_t>(byte0) << 16) | (static_cast<uint32_t>(byte1) << 8) | byte2;
  } else {
    // 4-byte sequence: byte0 = 11110xxx, byte1-3 = 10xxxxxx
    uint8_t byte0 = 0xF0 | ((codepoint >> 18) & 0x07);
    uint8_t byte1 = 0x80 | ((codepoint >> 12) & 0x3F);
    uint8_t byte2 = 0x80 | ((codepoint >> 6) & 0x3F);
    uint8_t byte3 = 0x80 | (codepoint & 0x3F);
    return (static_cast<uint32_t>(byte0) << 24) | (static_cast<uint32_t>(byte1) << 16) |
           (static_cast<uint32_t>(byte2) << 8) | byte3;
  }
}

// This function will add a range [min, max] of characters to the FSM, and the length
// of the characters are the same.
static void AddSameLengthCharacterRange(FSM& fsm, int from, int to, uint32_t min, uint32_t max) {
  uint8_t byte_min[4] = {
      static_cast<uint8_t>(min & 0xFF),
      static_cast<uint8_t>(min >> 8),
      static_cast<uint8_t>(min >> 16),
      static_cast<uint8_t>(min >> 24)
  };
  uint8_t byte_max[4] = {
      static_cast<uint8_t>(max & 0xFF),
      static_cast<uint8_t>(max >> 8),
      static_cast<uint8_t>(max >> 16),
      static_cast<uint8_t>(max >> 24)
  };

  // ASCII.
  if (byte_max[1] == 0) {
    fsm.AddEdge(from, to, byte_min[0], byte_max[0]);
    return;
  }

  if (byte_max[3] != 0) {
    // 4-byte unicode.
    if (byte_max[3] == byte_min[3]) {
      int tmp_state = fsm.AddState();
      fsm.AddEdge(from, tmp_state, byte_min[3], byte_max[3]);
      min = (min & 0x00FFFFFF);
      max = (max & 0x00FFFFFF);
      AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
      return;
    }
    if ((min & 0x00FFFFFF) != 0x808080) {
      int tmp_state_min = fsm.AddState();
      fsm.AddEdge(from, tmp_state_min, byte_min[3], byte_min[3]);
      AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FFFFFF), 0x00BFBFBF);
    } else {
      byte_min[3]--;
    }
    if ((max & 0x00FFFFFF) != 0xBFBFBF) {
      int tmp_state_max = fsm.AddState();
      fsm.AddEdge(from, tmp_state_max, byte_max[3], byte_max[3]);
      AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x00808080, (max & 0x00FFFFFF));
    } else {
      byte_max[3]++;
    }
    if (byte_max[3] - byte_min[3] > 1) {
      int tmp_state_mid = fsm.AddState();
      // First byte.
      fsm.AddEdge(from, tmp_state_mid, byte_min[3] + 1, byte_max[3] - 1);
      int tmp_state_mid2 = fsm.AddState();
      // Second byte.
      fsm.AddEdge(tmp_state_mid, tmp_state_mid2, 0x80, 0xBF);
      int tmp_state_mid3 = fsm.AddState();
      // Third byte.
      fsm.AddEdge(tmp_state_mid2, tmp_state_mid3, 0x80, 0xBF);
      // Last byte.
      fsm.AddEdge(tmp_state_mid3, to, 0x80, 0xBF);
    }
    return;
  }
  if (byte_max[2] != 0) {
    // 3 byte unicode.
    if (byte_max[2] == byte_min[2]) {
      int tmp_state = fsm.AddState();
      fsm.AddEdge(from, tmp_state, byte_min[2], byte_max[2]);
      min = (min & 0x00FFFF);
      max = (max & 0x00FFFF);
      AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
      return;
    }
    if ((min & 0x00FFFF) != 0x8080) {
      int tmp_state_min = fsm.AddState();
      fsm.AddEdge(from, tmp_state_min, byte_min[2], byte_min[2]);
      AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FFFF), 0x00BFBF);
    } else {
      byte_min[2]--;
    }
    if ((max & 0x00FFFF) != 0xBFBF) {
      int tmp_state_max = fsm.AddState();
      fsm.AddEdge(from, tmp_state_max, byte_max[2], byte_max[2]);
      AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x008080, (max & 0x00FFFF));
    } else {
      byte_max[2]++;
    }
    if (byte_max[2] - byte_min[2] > 1) {
      int tmp_state_mid = fsm.AddState();
      // First byte.
      fsm.AddEdge(from, tmp_state_mid, byte_min[2] + 1, byte_max[2] - 1);
      int tmp_state_mid2 = fsm.AddState();
      // Second byte.
      fsm.AddEdge(tmp_state_mid, tmp_state_mid2, 0x80, 0xBF);
      // Last byte.
      fsm.AddEdge(tmp_state_mid2, to, 0x80, 0xBF);
    }
    return;
  }

  // 2 byte unicode.
  if (byte_max[1] == byte_min[1]) {
    int tmp_state = fsm.AddState();
    fsm.AddEdge(from, tmp_state, byte_min[1], byte_max[1]);
    min = (min & 0x00FF);
    max = (max & 0x00FF);
    AddSameLengthCharacterRange(fsm, tmp_state, to, min, max);
    return;
  }
  if ((min & 0x00FF) != 0x80) {
    int tmp_state_min = fsm.AddState();
    fsm.AddEdge(from, tmp_state_min, byte_min[1], byte_min[1]);
    AddSameLengthCharacterRange(fsm, tmp_state_min, to, (min & 0x00FF), 0x00BF);
  } else {
    byte_min[1]--;
  }
  if ((max & 0x00FF) != 0xBF) {
    int tmp_state_max = fsm.AddState();
    fsm.AddEdge(from, tmp_state_max, byte_max[1], byte_max[1]);
    AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x0080, (max & 0x00FF));
  } else {
    byte_max[1]++;
  }
  if (byte_max[1] - byte_min[1] > 1) {
    int tmp_state_mid = fsm.AddState();
    // First byte.
    fsm.AddEdge(from, tmp_state_mid, byte_min[1] + 1, byte_max[1] - 1);
    fsm.AddEdge(tmp_state_mid, to, 0x80, 0xBF);
  }
  return;
}

void AddPackedUTF8RangeEdges(FSM& fsm, int from, int to, uint32_t min, uint32_t max) {
  XGRAMMAR_CHECK(min <= max) << "Invalid character range: min (" << min << ") > max (" << max
                             << ")";
  // Ensure max and min are valid unicode value.
  if (max > kMax4BytesUnicode) {
    max = kMax4BytesUnicode;
  } else if (max > kMax3BytesUnicode) {
    if (max < kMin4BytesUnicode) {
      max = kMax3BytesUnicode;
    }
  } else if (max > kMax2BytesUnicode) {
    if (max < kMin3BytesUnicode) {
      max = kMax2BytesUnicode;
    }
  } else if (max < kMin2BytesUnicode && (max > kMax1ByteUnicode)) {
    max = kMax1ByteUnicode;
  }

  if (min > kMax4BytesUnicode) {
    min = kMax4BytesUnicode;
  } else if (min > kMax3BytesUnicode) {
    if (min < kMin4BytesUnicode) {
      min = kMin4BytesUnicode;
    }
  } else if (min > kMax2BytesUnicode) {
    if (min < kMin3BytesUnicode) {
      min = kMin3BytesUnicode;
    }
  } else if (min < kMin2BytesUnicode && (min > kMax1ByteUnicode)) {
    min = kMin2BytesUnicode;
  }

  // Step2. Divide the range into several ranges, which contain characters with different lengths.
  if (max <= kMax1ByteUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, max);
    return;
  }
  if (max <= kMax2BytesUnicode) {
    if (min >= kMin2BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, max);
    } else {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, max);
    }
    return;
  }
  if (max <= kMax3BytesUnicode) {
    if (min >= kMin3BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, max);
    } else if (min >= kMin2BytesUnicode) {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax2BytesUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, max);
    } else {
      AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, kMax2BytesUnicode);
      AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, max);
    }
    return;
  }
  XGRAMMAR_CHECK(max <= kMax4BytesUnicode);
  if (min >= kMin4BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, max);
  } else if (min >= kMin3BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  } else if (min >= kMin2BytesUnicode) {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax2BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  } else {
    AddSameLengthCharacterRange(fsm, from, to, min, kMax1ByteUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin2BytesUnicode, kMax2BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin3BytesUnicode, kMax3BytesUnicode);
    AddSameLengthCharacterRange(fsm, from, to, kMin4BytesUnicode, max);
  }
  return;
}

namespace {

struct InlineRegexFlagSpec {
  size_t end = 0;
  bool scoped = false;
  std::optional<bool> case_insensitive;
  std::optional<bool> dot_matches_newline;
  std::optional<bool> multiline;
  std::optional<bool> unicode;
  std::optional<bool> extended;
  std::optional<bool> crlf;
};

/*! \brief Track nested Rust-style character classes, including literal leading `]` and `-`. */
class RegexCharacterClassTracker {
 public:
  bool InClass() const { return !frames_.empty(); }

  void ConsumeEscape() {
    if (!frames_.empty()) {
      frames_.back().at_start = false;
    }
  }

  void Consume(char character) {
    if (character == '[') {
      if (!frames_.empty()) {
        frames_.back().at_start = false;
      }
      frames_.push_back(Frame{});
      return;
    }
    if (frames_.empty()) {
      return;
    }
    Frame& frame = frames_.back();
    if (frame.at_start) {
      if (character == '^' && frame.allow_negation) {
        frame.allow_negation = false;
        return;
      }
      if (character == '-') {
        frame.allow_negation = false;
        frame.saw_initial_hyphen = true;
        return;
      }
      if (character == ']' && !frame.saw_initial_hyphen) {
        frame.at_start = false;
        return;
      }
      frame.at_start = false;
    }
    if (character == ']') {
      frames_.pop_back();
    }
  }

 private:
  struct Frame {
    bool at_start = true;
    bool allow_negation = true;
    bool saw_initial_hyphen = false;
  };
  std::vector<Frame> frames_;
};

/*! \brief Parse an inline flag directive beginning at `(?`. */
std::optional<InlineRegexFlagSpec> ParseInlineRegexFlagSpec(
    const std::string& pattern, size_t position
) {
  if (position + 2 >= pattern.size() || pattern[position] != '(' || pattern[position + 1] != '?') {
    return std::nullopt;
  }
  InlineRegexFlagSpec result;
  bool enabled = true;
  bool saw_flag = false;
  bool saw_dash = false;
  size_t cursor = position + 2;
  for (; cursor < pattern.size(); ++cursor) {
    char character = pattern[cursor];
    if (character == ':' || character == ')') {
      if (!saw_flag) {
        return std::nullopt;
      }
      result.end = cursor + 1;
      result.scoped = character == ':';
      return result;
    }
    if (character == '-') {
      if (saw_dash) {
        return std::nullopt;
      }
      saw_dash = true;
      enabled = false;
      continue;
    }
    saw_flag = true;
    switch (character) {
      case 'i':
        result.case_insensitive = enabled;
        break;
      case 's':
        result.dot_matches_newline = enabled;
        break;
      case 'm':
        result.multiline = enabled;
        break;
      case 'u':
        result.unicode = enabled;
        break;
      case 'x':
        result.extended = enabled;
        break;
      case 'R':
        result.crlf = enabled;
        break;
      case 'U':
        // Greediness does not change the accepted language.
        break;
      default:
        return std::nullopt;
    }
  }
  return std::nullopt;
}

/*! \brief Return the end of a `(?<name>` or `(?P<name>` prefix, if one starts here. */
std::optional<size_t> ParseNamedCapturePrefixEnd(const std::string& pattern, size_t position) {
  if (position + 3 >= pattern.size() || pattern[position] != '(' || pattern[position + 1] != '?') {
    return std::nullopt;
  }
  size_t name_begin;
  if (pattern[position + 2] == '<') {
    if (pattern[position + 3] == '=' || pattern[position + 3] == '!') {
      return std::nullopt;
    }
    name_begin = position + 3;
  } else if (position + 4 < pattern.size() && pattern[position + 2] == 'P' &&
             pattern[position + 3] == '<') {
    name_begin = position + 4;
  } else {
    return std::nullopt;
  }
  size_t close = pattern.find('>', name_begin);
  return close == std::string::npos ? std::nullopt : std::optional<size_t>(close + 1);
}

/*! \brief Remove whitespace and comments enabled by scoped or top-level inline `x` flags. */
std::string RewriteInlineExtendedRegex(const std::string& pattern) {
  std::string result;
  result.reserve(pattern.size());
  bool extended = false;
  RegexCharacterClassTracker class_tracker;
  std::vector<bool> group_flag_stack;
  for (size_t position = 0; position < pattern.size();) {
    char character = pattern[position];
    if (character == '\\') {
      class_tracker.ConsumeEscape();
      result.push_back(character);
      ++position;
      if (position < pattern.size()) {
        result.push_back(pattern[position++]);
      }
      continue;
    }
    if (extended && character == '#') {
      while (position < pattern.size() && pattern[position] != '\n') {
        ++position;
      }
      continue;
    }
    if (extended) {
      auto [codepoint, num_bytes] = ParseNextUTF8(pattern.c_str() + position);
      if (codepoint != CharHandlingError::kInvalidUTF8 && IsUnicodeWhitespace(codepoint)) {
        position += num_bytes;
        continue;
      }
    }
    bool was_in_character_class = class_tracker.InClass();
    class_tracker.Consume(character);
    if (was_in_character_class || character == '[') {
      result.push_back(character);
      ++position;
      continue;
    }
    if (character == '(') {
      auto flag_spec = ParseInlineRegexFlagSpec(pattern, position);
      if (flag_spec.has_value()) {
        result.append(pattern, position, flag_spec->end - position);
        if (flag_spec->scoped) {
          group_flag_stack.push_back(extended);
        }
        if (flag_spec->extended.has_value()) {
          extended = flag_spec->extended.value();
        }
        position = flag_spec->end;
        continue;
      }
      auto capture_prefix_end = ParseNamedCapturePrefixEnd(pattern, position);
      if (capture_prefix_end.has_value()) {
        group_flag_stack.push_back(extended);
        result.append(pattern, position, capture_prefix_end.value() - position);
        position = capture_prefix_end.value();
        continue;
      }
      group_flag_stack.push_back(extended);
      result.push_back(character);
      ++position;
      continue;
    }
    if (character == ')') {
      result.push_back(character);
      if (!group_flag_stack.empty()) {
        extended = group_flag_stack.back();
        group_flag_stack.pop_back();
      }
      ++position;
      continue;
    }
    result.push_back(character);
    ++position;
  }
  return result;
}

}  // namespace

std::string RewriteRegexExtended(const std::string& pattern, bool extended) {
  if (!extended && pattern.find("(?") == std::string::npos) {
    return pattern;
  }
  return RewriteInlineExtendedRegex(extended ? "(?x)" + pattern : pattern);
}

bool ContainsRegexMultilineLineAnchor(const std::string& pattern, bool multiline) {
  if (!multiline && pattern.find("(?") == std::string::npos) {
    return false;
  }
  RegexCharacterClassTracker class_tracker;
  std::vector<bool> group_flag_stack;
  for (size_t position = 0; position < pattern.size();) {
    char character = pattern[position];
    if (character == '\\') {
      class_tracker.ConsumeEscape();
      position += std::min<size_t>(2, pattern.size() - position);
      continue;
    }
    bool was_in_character_class = class_tracker.InClass();
    class_tracker.Consume(character);
    if (was_in_character_class || character == '[') {
      ++position;
      continue;
    }
    if (character == '(') {
      auto flag_spec = ParseInlineRegexFlagSpec(pattern, position);
      if (flag_spec.has_value()) {
        if (flag_spec->scoped) {
          group_flag_stack.push_back(multiline);
        }
        if (flag_spec->multiline.has_value()) {
          multiline = flag_spec->multiline.value();
        }
        position = flag_spec->end;
        continue;
      }
      auto capture_prefix_end = ParseNamedCapturePrefixEnd(pattern, position);
      group_flag_stack.push_back(multiline);
      position = capture_prefix_end.value_or(position + 1);
      continue;
    }
    if (character == ')') {
      if (!group_flag_stack.empty()) {
        multiline = group_flag_stack.back();
        group_flag_stack.pop_back();
      }
      ++position;
      continue;
    }
    if (multiline && (character == '^' || character == '$')) {
      return true;
    }
    ++position;
  }
  return false;
}

std::string RewriteRegexDots(const std::string& pattern, bool dot_matches_newline, bool crlf) {
  std::string result;
  result.reserve(pattern.size());
  RegexCharacterClassTracker class_tracker;
  std::vector<std::pair<bool, bool>> group_flag_stack;
  for (size_t position = 0; position < pattern.size();) {
    char character = pattern[position];
    if (character == '\\') {
      class_tracker.ConsumeEscape();
      result.push_back(character);
      ++position;
      if (position < pattern.size()) {
        result.push_back(pattern[position++]);
      }
      continue;
    }
    bool was_in_character_class = class_tracker.InClass();
    class_tracker.Consume(character);
    if (was_in_character_class || character == '[') {
      result.push_back(character);
      ++position;
      continue;
    }
    if (character == '(') {
      auto flag_spec = ParseInlineRegexFlagSpec(pattern, position);
      if (flag_spec.has_value()) {
        result.append(pattern, position, flag_spec->end - position);
        if (flag_spec->scoped) {
          group_flag_stack.push_back({dot_matches_newline, crlf});
        }
        if (flag_spec->dot_matches_newline.has_value()) {
          dot_matches_newline = flag_spec->dot_matches_newline.value();
        }
        if (flag_spec->crlf.has_value()) {
          crlf = flag_spec->crlf.value();
        }
        position = flag_spec->end;
        continue;
      }
      auto capture_prefix_end = ParseNamedCapturePrefixEnd(pattern, position);
      if (capture_prefix_end.has_value()) {
        group_flag_stack.push_back({dot_matches_newline, crlf});
        result.append(pattern, position, capture_prefix_end.value() - position);
        position = capture_prefix_end.value();
        continue;
      }
      group_flag_stack.push_back({dot_matches_newline, crlf});
      result.push_back(character);
      ++position;
      continue;
    }
    if (character == ')') {
      result.push_back(character);
      if (!group_flag_stack.empty()) {
        dot_matches_newline = group_flag_stack.back().first;
        crlf = group_flag_stack.back().second;
        group_flag_stack.pop_back();
      }
      ++position;
      continue;
    }
    if (character == '.' && !dot_matches_newline) {
      result += crlf ? "[^\\r\\n]" : "[^\\n]";
      ++position;
      continue;
    }
    result.push_back(character);
    ++position;
  }
  return result;
}

/******************** Codepoint range utilities ********************/

namespace {

constexpr uint32_t kMaxCodepoint = 0x10FFFF;
constexpr uint32_t kMaxByteValue = 0xFF;

/*! \brief Bounded repetitions above this threshold are compiled into a repeat FSM edge (when a
 * GrammarBuilder is available) instead of being physically unrolled. Matches the unroll threshold
 * of the grammar-level RepetitionRangeExpander. */
constexpr int kLargeRepeatThreshold = 128;

/*! \brief JSON-source encoding expands every logical regex atom into raw, escaped, and Unicode
 * spellings. Use a lower unroll threshold there so bounded compound patterns do not create a new
 * full-vocabulary mask for dozens of structurally repeated FSM states. Simple character-class
 * JSON Schema patterns take a dedicated compact path before reaching this builder. */
constexpr int kLargeJSONStringRepeatThreshold = 16;

/*! \brief Hard cap of the estimated state count when a bounded repetition has to be unrolled
 * because no GrammarBuilder is available. */
constexpr int64_t kMaxUnrolledRepeatStates = 100000;

using CodepointRange = std::pair<uint32_t, uint32_t>;

/*! \brief Sort the ranges and merge overlapping or adjacent ones. */
void NormalizeRanges(std::vector<CodepointRange>* ranges) {
  std::sort(ranges->begin(), ranges->end());
  std::vector<CodepointRange> result;
  for (const auto& range : *ranges) {
    if (!result.empty() && range.first <= result.back().second + 1 &&
        range.first >= result.back().first) {
      result.back().second = std::max(result.back().second, range.second);
    } else {
      result.push_back(range);
    }
  }
  *ranges = std::move(result);
}

/*! \brief Complement normalized ranges over the domain [0, universe_max]. */
std::vector<CodepointRange> ComplementRanges(
    const std::vector<CodepointRange>& ranges, uint32_t universe_max
) {
  std::vector<CodepointRange> result;
  uint32_t next = 0;
  for (const auto& range : ranges) {
    if (range.first > next) {
      result.push_back({next, range.first - 1});
    }
    if (range.second >= universe_max) {
      return result;
    }
    next = std::max(next, range.second + 1);
  }
  result.push_back({next, universe_max});
  return result;
}

/*! \brief Remove the surrogate codepoint interval, which is not part of the Unicode scalar-value
 * domain used by regex-syntax. */
void RemoveUnicodeSurrogates(std::vector<CodepointRange>* ranges) {
  std::vector<CodepointRange> result;
  result.reserve(ranges->size() + 1);
  for (const auto& [first, last] : *ranges) {
    if (first < 0xD800) {
      result.push_back({first, std::min<uint32_t>(last, 0xD7FF)});
    }
    if (last > 0xDFFF) {
      result.push_back({std::max<uint32_t>(first, 0xE000), last});
    }
  }
  *ranges = std::move(result);
}

std::vector<CodepointRange> IntersectRanges(
    const std::vector<CodepointRange>& left, const std::vector<CodepointRange>& right
) {
  std::vector<CodepointRange> result;
  size_t left_index = 0;
  size_t right_index = 0;
  while (left_index < left.size() && right_index < right.size()) {
    uint32_t first = std::max(left[left_index].first, right[right_index].first);
    uint32_t last = std::min(left[left_index].second, right[right_index].second);
    if (first <= last) {
      result.push_back({first, last});
    }
    if (left[left_index].second < right[right_index].second) {
      ++left_index;
    } else {
      ++right_index;
    }
  }
  return result;
}

std::vector<CodepointRange> DifferenceRanges(
    const std::vector<CodepointRange>& left,
    const std::vector<CodepointRange>& right,
    uint32_t universe_max
) {
  auto complement = ComplementRanges(right, universe_max);
  return IntersectRanges(left, complement);
}

std::vector<CodepointRange> SymmetricDifferenceRanges(
    const std::vector<CodepointRange>& left,
    const std::vector<CodepointRange>& right,
    uint32_t universe_max
) {
  auto left_only = DifferenceRanges(left, right, universe_max);
  auto right_only = DifferenceRanges(right, left, universe_max);
  left_only.insert(left_only.end(), right_only.begin(), right_only.end());
  NormalizeRanges(&left_only);
  return left_only;
}

bool IsRegexCaptureNameCodepoint(uint32_t codepoint, bool first) {
  if (codepoint == '_') {
    return true;
  }
  if (!first && (codepoint == '.' || codepoint == '[' || codepoint == ']')) {
    return true;
  }
  return IsUnicodeAlphabetic(codepoint) || (!first && IsUnicodeAlphanumeric(codepoint));
}

/*! \brief Append the ASCII case-folded counterparts of every letter contained in the ranges. */
void FoldAsciiCaseRanges(std::vector<CodepointRange>* ranges) {
  size_t original_size = ranges->size();
  for (size_t i = 0; i < original_size; ++i) {
    uint32_t low = (*ranges)[i].first;
    uint32_t high = (*ranges)[i].second;
    uint32_t fold_low = std::max<uint32_t>(low, 'a');
    uint32_t fold_high = std::min<uint32_t>(high, 'z');
    if (fold_low <= fold_high) {
      ranges->push_back({fold_low - ('a' - 'A'), fold_high - ('a' - 'A')});
    }
    fold_low = std::max<uint32_t>(low, 'A');
    fold_high = std::min<uint32_t>(high, 'Z');
    if (fold_low <= fold_high) {
      ranges->push_back({fold_low + ('a' - 'A'), fold_high + ('a' - 'A')});
    }
  }
}

/*! \brief Append simple Unicode case-fold equivalents, or ASCII equivalents in byte mode. */
void FoldCaseRanges(std::vector<CodepointRange>* ranges, bool byte_mode) {
  if (byte_mode) {
    FoldAsciiCaseRanges(ranges);
    return;
  }
  const size_t original_size = ranges->size();
  std::vector<TCodepoint> folded;
  for (size_t i = 0; i < original_size; ++i) {
    AppendUnicodeSimpleCaseFold((*ranges)[i].first, (*ranges)[i].second, &folded);
  }
  ranges->reserve(ranges->size() + folded.size());
  for (TCodepoint codepoint : folded) {
    ranges->push_back({codepoint, codepoint});
  }
}

/*! \brief Add edges from `from` to `to` accepting the UTF-8 encoding of every codepoint in the
 * normalized ranges. Multi-byte characters get intermediate states. */
void AddCodepointRangesToFSM(
    FSM* fsm, int from, int to, const std::vector<CodepointRange>& ranges
) {
  for (const auto& [low, high] : ranges) {
    if (low <= kMax1ByteUnicode) {
      fsm->AddEdge(from, to, low, std::min<uint32_t>(high, kMax1ByteUnicode));
    }
    if (high > kMax1ByteUnicode) {
      uint32_t multi_byte_low = std::max<uint32_t>(low, kMax1ByteUnicode + 1);
      AddPackedUTF8RangeEdges(
          *fsm, from, to, CodepointToPackedUTF8(multi_byte_low), CodepointToPackedUTF8(high)
      );
    }
  }
}

void AddJSONStringHexDigitRange(FSM* fsm, int from, int to, int low, int high) {
  XGRAMMAR_DCHECK(0 <= low && low <= high && high <= 15);
  if (low <= 9) {
    fsm->AddEdge(from, to, '0' + low, '0' + std::min(high, 9));
  }
  if (high >= 10) {
    int letter_low = std::max(low, 10) - 10;
    int letter_high = high - 10;
    fsm->AddEdge(from, to, 'A' + letter_low, 'A' + letter_high);
    fsm->AddEdge(from, to, 'a' + letter_low, 'a' + letter_high);
  }
}

void AddAnyJSONStringHexDigits(FSM* fsm, int from, int to, int digits) {
  int current = from;
  for (int index = 0; index < digits; ++index) {
    int next = index + 1 == digits ? to : fsm->AddState();
    AddJSONStringHexDigitRange(fsm, current, next, 0, 15);
    current = next;
  }
}

/*! \brief Add the fixed-width hexadecimal spellings in the inclusive numeric range. */
void AddJSONStringHexValueRange(
    FSM* fsm, int from, int to, uint32_t low, uint32_t high, int digits
) {
  XGRAMMAR_DCHECK(low <= high && digits >= 1 && digits <= 4);
  const uint32_t place = uint32_t{1} << (4 * (digits - 1));
  if (low == 0 && high == place * 16 - 1) {
    AddAnyJSONStringHexDigits(fsm, from, to, digits);
    return;
  }

  const int low_digit = static_cast<int>(low / place);
  const int high_digit = static_cast<int>(high / place);
  const uint32_t low_suffix = low % place;
  const uint32_t high_suffix = high % place;
  auto add_branch = [&](int digit_low, int digit_high, uint32_t suffix_low, uint32_t suffix_high) {
    if (digits == 1) {
      AddJSONStringHexDigitRange(fsm, from, to, digit_low, digit_high);
      return;
    }
    int next = fsm->AddState();
    AddJSONStringHexDigitRange(fsm, from, next, digit_low, digit_high);
    AddJSONStringHexValueRange(fsm, next, to, suffix_low, suffix_high, digits - 1);
  };

  if (low_digit == high_digit) {
    add_branch(low_digit, high_digit, low_suffix, high_suffix);
    return;
  }
  add_branch(low_digit, low_digit, low_suffix, place - 1);
  if (low_digit + 1 <= high_digit - 1) {
    add_branch(low_digit + 1, high_digit - 1, 0, place - 1);
  }
  add_branch(high_digit, high_digit, 0, high_suffix);
}

void AddJSONStringFixedBytes(FSM* fsm, int from, int to, const std::string& bytes) {
  int current = from;
  for (size_t index = 0; index < bytes.size(); ++index) {
    int next = index + 1 == bytes.size() ? to : fsm->AddState();
    uint8_t byte = static_cast<uint8_t>(bytes[index]);
    fsm->AddEdge(current, next, byte, byte);
    current = next;
  }
}

void AddJSONStringUnicodeEscapeRange(FSM* fsm, int from, int to, uint32_t low, uint32_t high) {
  int after_slash = fsm->AddState();
  int after_u = fsm->AddState();
  fsm->AddEdge(from, after_slash, '\\', '\\');
  fsm->AddEdge(after_slash, after_u, 'u', 'u');
  AddJSONStringHexValueRange(fsm, after_u, to, low, high, 4);
}

/*! \brief Add every valid JSON source spelling of the normalized Unicode scalar ranges. */
void AddJSONStringCodepointRangesToFSM(
    FSM* fsm, int from, int to, const std::vector<CodepointRange>& ranges
) {
  std::vector<CodepointRange> raw_ranges;
  for (const auto& range : ranges) {
    // Plain locals: lambdas cannot capture structured bindings before C++20.
    uint32_t low = range.first;
    uint32_t high = range.second;
    auto add_raw = [&](uint32_t raw_low, uint32_t raw_high) {
      raw_low = std::max(raw_low, low);
      raw_high = std::min(raw_high, high);
      if (raw_low <= raw_high) {
        raw_ranges.push_back({raw_low, raw_high});
      }
    };
    add_raw(0x20, 0x21);
    add_raw(0x23, 0x5B);
    add_raw(0x5D, 0xD7FF);
    add_raw(0xE000, kMaxCodepoint);
  }
  AddCodepointRangesToFSM(fsm, from, to, raw_ranges);

  static constexpr std::array<std::pair<uint32_t, char>, 8> kShortEscapes = {
      std::pair<uint32_t, char>{'"', '"'},
      {'\\', '\\'},
      {'/', '/'},
      {'\b', 'b'},
      {'\f', 'f'},
      {'\n', 'n'},
      {'\r', 'r'},
      {'\t', 't'},
  };
  for (const auto& escape : kShortEscapes) {
    // Plain locals: lambdas cannot capture structured bindings before C++20.
    uint32_t codepoint = escape.first;
    char escaped = escape.second;
    if (std::any_of(ranges.begin(), ranges.end(), [&](const CodepointRange& range) {
          return range.first <= codepoint && codepoint <= range.second;
        })) {
      AddJSONStringFixedBytes(fsm, from, to, std::string{'\\', escaped});
    }
  }

  for (const auto& [low, high] : ranges) {
    uint32_t bmp_low = low;
    uint32_t bmp_high = std::min<uint32_t>(high, 0xFFFF);
    if (bmp_low <= bmp_high) {
      if (bmp_low <= 0xD7FF) {
        AddJSONStringUnicodeEscapeRange(fsm, from, to, bmp_low, std::min(bmp_high, 0xD7FFu));
      }
      if (bmp_high >= 0xE000) {
        AddJSONStringUnicodeEscapeRange(fsm, from, to, std::max(bmp_low, 0xE000u), bmp_high);
      }
    }

    uint32_t scalar_low = std::max<uint32_t>(low, 0x10000);
    uint32_t scalar_high = std::min<uint32_t>(high, kMaxCodepoint);
    if (scalar_low > scalar_high) {
      continue;
    }
    uint32_t offset_low = scalar_low - 0x10000;
    uint32_t offset_high = scalar_high - 0x10000;
    uint32_t high_surrogate_low = 0xD800 + (offset_low >> 10);
    uint32_t high_surrogate_high = 0xD800 + (offset_high >> 10);
    uint32_t low_surrogate_low = 0xDC00 + (offset_low & 0x3FF);
    uint32_t low_surrogate_high = 0xDC00 + (offset_high & 0x3FF);

    auto add_surrogate_branch =
        [&](uint32_t high_low, uint32_t high_high, uint32_t low_low, uint32_t low_high) {
          int between = fsm->AddState();
          AddJSONStringUnicodeEscapeRange(fsm, from, between, high_low, high_high);
          AddJSONStringUnicodeEscapeRange(fsm, between, to, low_low, low_high);
        };
    if (high_surrogate_low == high_surrogate_high) {
      add_surrogate_branch(
          high_surrogate_low, high_surrogate_high, low_surrogate_low, low_surrogate_high
      );
      continue;
    }
    add_surrogate_branch(high_surrogate_low, high_surrogate_low, low_surrogate_low, 0xDFFF);
    if (high_surrogate_low + 1 <= high_surrogate_high - 1) {
      add_surrogate_branch(high_surrogate_low + 1, high_surrogate_high - 1, 0xDC00, 0xDFFF);
    }
    add_surrogate_branch(high_surrogate_high, high_surrogate_high, 0xDC00, low_surrogate_high);
  }
}

/*! \brief One parsed regex escape (or literal): either a single codepoint, or a (possibly
 * negated) set of codepoint ranges for class escapes like \d, \D, \w, \W, \s, \S. */
struct RegexEscapeItem {
  std::vector<CodepointRange> ranges;
  bool negated = false;
  bool is_single = false;
  uint32_t codepoint = 0;
};

/*! \brief Character-class escape semantics used by the regex source language. */
enum class RegexCharacterClassDialect {
  // Rust/Lark-compatible Unicode shorthands.
  kUnicode,
  // ECMA-262 shorthands used by JSON Schema patterns.
  kECMAScript,
};

std::vector<CodepointRange> RegexWhitespaceRanges(RegexCharacterClassDialect character_class_dialect
) {
  if (character_class_dialect == RegexCharacterClassDialect::kECMAScript) {
    // ECMA-262 WhiteSpace and LineTerminator code points. Unlike the Unicode White_Space
    // property this includes U+FEFF and excludes U+0085.
    return {
        {0x0009, 0x000D},
        {0x0020, 0x0020},
        {0x00A0, 0x00A0},
        {0x1680, 0x1680},
        {0x2000, 0x200A},
        {0x2028, 0x2029},
        {0x202F, 0x202F},
        {0x205F, 0x205F},
        {0x3000, 0x3000},
        {0xFEFF, 0xFEFF},
    };
  }
  return {
      {0x0009, 0x000D},
      {0x0020, 0x0020},
      {0x0085, 0x0085},
      {0x00A0, 0x00A0},
      {0x1680, 0x1680},
      {0x2000, 0x200A},
      {0x2028, 0x2029},
      {0x202F, 0x202F},
      {0x205F, 0x205F},
      {0x3000, 0x3000},
  };
}

/*!
 * \brief Parse the escape sequence starting at regex[*pos] == '\\'. On success, *pos is advanced
 * past the escape sequence.
 * \param in_class Whether the escape appears inside a character class ([...]). Inside a class,
 * \b means the backspace character instead of a word boundary assertion.
 */
Result<RegexEscapeItem> ParseByteRegexEscape(const std::string& regex, size_t* pos, bool in_class) {
  XGRAMMAR_DCHECK(regex[*pos] == '\\');
  if (*pos + 1 >= regex.size()) {
    return ResultErr("unfinished byte escape");
  }
  char escaped = regex[*pos + 1];
  *pos += 2;
  RegexEscapeItem item;
  auto single = [&](uint32_t byte) {
    item.is_single = true;
    item.codepoint = byte;
    return ResultOk(std::move(item));
  };
  switch (escaped) {
    case 'd':
      item.ranges = {{'0', '9'}};
      return ResultOk(std::move(item));
    case 'D':
      item.ranges = {{'0', '9'}};
      item.negated = true;
      return ResultOk(std::move(item));
    case 'w':
      item.ranges = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}};
      return ResultOk(std::move(item));
    case 'W':
      item.ranges = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}};
      item.negated = true;
      return ResultOk(std::move(item));
    case 's':
      item.ranges = {{0x09, 0x0D}, {' ', ' '}};
      return ResultOk(std::move(item));
    case 'S':
      item.ranges = {{0x09, 0x0D}, {' ', ' '}};
      item.negated = true;
      return ResultOk(std::move(item));
    case 'x': {
      if (*pos < regex.size() && regex[*pos] == '{') {
        return ResultErr("Unicode character escapes are not available in byte regular expressions");
      }
      if (*pos + 2 > regex.size() || HexCharToInt(regex[*pos]) < 0 ||
          HexCharToInt(regex[*pos + 1]) < 0) {
        return ResultErr("\\x escape must contain exactly two hexadecimal digits");
      }
      uint32_t byte =
          static_cast<uint32_t>(HexCharToInt(regex[*pos]) * 16 + HexCharToInt(regex[*pos + 1]));
      *pos += 2;
      return single(byte);
    }
    case 'p':
    case 'P':
      return ResultErr("Unicode property escapes \\p and \\P are not supported");
    case 'u':
    case 'U':
      return ResultErr("Unicode character escapes are not available in byte regular expressions");
    case 'b':
    case 'B':
      if (!in_class) {
        return ResultErr("word-boundary assertions are not supported");
      }
      return single('\b');
    case 'a':
      return single('\a');
    case 'e':
      return single(0x1B);
    case 'f':
      return single('\f');
    case 'n':
      return single('\n');
    case 'r':
      return single('\r');
    case 't':
      return single('\t');
    case 'v':
      return single('\v');
    case '0':
      return single('\0');
    case '\\':
    case '/':
    case '.':
    case '*':
    case '+':
    case '?':
    case '(':
    case ')':
    case '[':
    case ']':
    case '{':
    case '}':
    case '|':
    case '^':
    case '$':
    case '-':
      return single(static_cast<uint8_t>(escaped));
    default:
      if ((escaped >= '1' && escaped <= '9') || escaped == 'k') {
        return ResultErr("backreferences are not supported");
      }
      return ResultErr(std::string("unrecognized byte escape '\\") + escaped + "'");
  }
}

Result<RegexEscapeItem> ParseCodepointRegexEscape(
    const std::string& regex,
    size_t* pos,
    bool in_class,
    RegexCharacterClassDialect character_class_dialect
) {
  XGRAMMAR_DCHECK(regex[*pos] == '\\');
  if (*pos + 1 >= regex.size()) {
    return ResultErr("Regex ends with a trailing backslash");
  }
  char escaped = regex[*pos + 1];
  *pos += 2;
  RegexEscapeItem item;
  auto single = [&](uint32_t codepoint) {
    item.is_single = true;
    item.codepoint = codepoint;
    return ResultOk(std::move(item));
  };
  switch (escaped) {
    case 'n':
      return single('\n');
    case 't':
      return single('\t');
    case 'r':
      return single('\r');
    case 'f':
      return single('\f');
    case 'v':
      return single('\v');
    case 'a':
      return single('\a');
    case '0':
      return single(0);
    case 'b':
      if (in_class) {
        return single(0x08);
      }
      return ResultErr("Word boundary assertion \\b is not supported in regex");
    case 'B':
      return ResultErr("Word boundary assertion \\B is not supported in regex");
    case 'p':
    case 'P':
      return ResultErr("Unicode property escapes \\p and \\P are not supported");
    case 'k':
      return ResultErr("Backreference \\k is not supported in regex");
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return ResultErr("Backreference \\" + std::string(1, escaped) + " is not supported in regex");
    case 'd':
      if (character_class_dialect == RegexCharacterClassDialect::kECMAScript) {
        item.ranges = {{'0', '9'}};
      } else {
        AppendUnicodeRegexDecimalRanges(&item.ranges);
      }
      return ResultOk(std::move(item));
    case 'D':
      if (character_class_dialect == RegexCharacterClassDialect::kECMAScript) {
        item.ranges = {{'0', '9'}};
      } else {
        AppendUnicodeRegexDecimalRanges(&item.ranges);
      }
      item.negated = true;
      return ResultOk(std::move(item));
    case 'w':
      if (character_class_dialect == RegexCharacterClassDialect::kECMAScript) {
        item.ranges = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}};
      } else {
        AppendUnicodeRegexWordRanges(&item.ranges);
      }
      return ResultOk(std::move(item));
    case 'W':
      if (character_class_dialect == RegexCharacterClassDialect::kECMAScript) {
        item.ranges = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}};
      } else {
        AppendUnicodeRegexWordRanges(&item.ranges);
      }
      item.negated = true;
      return ResultOk(std::move(item));
    case 's':
      item.ranges = RegexWhitespaceRanges(character_class_dialect);
      return ResultOk(std::move(item));
    case 'S':
      item.ranges = RegexWhitespaceRanges(character_class_dialect);
      item.negated = true;
      return ResultOk(std::move(item));
    case 'x':
    case 'u':
    case 'U': {
      if (*pos < regex.size() && regex[*pos] == '{') {
        size_t close = regex.find('}', *pos + 1);
        if (close == std::string::npos || close == *pos + 1 || close > *pos + 7) {
          return ResultErr(
              "\\" + std::string(1, escaped) +
              "{...} must contain one to six hexadecimal digits in regex"
          );
        }
        uint32_t codepoint = 0;
        for (size_t i = *pos + 1; i < close; ++i) {
          int digit = HexCharToInt(regex[i]);
          if (digit < 0) {
            return ResultErr(
                "\\" + std::string(1, escaped) +
                "{...} must contain one to six hexadecimal digits in regex"
            );
          }
          codepoint = codepoint * 16 + digit;
        }
        if (codepoint > kMaxCodepoint || (0xD800 <= codepoint && codepoint <= 0xDFFF)) {
          return ResultErr("\\" + std::string(1, escaped) + "{...} is not a Unicode scalar value");
        }
        *pos = close + 1;
        return single(codepoint);
      }
      size_t digit_count = escaped == 'x' ? 2 : (escaped == 'u' ? 4 : 8);
      if (*pos + digit_count > regex.size()) {
        return ResultErr(
            "\\" + std::string(1, escaped) + " must be followed by " + std::to_string(digit_count) +
            " hexadecimal digits in regex"
        );
      }
      uint32_t codepoint = 0;
      for (size_t i = *pos; i < *pos + digit_count; ++i) {
        int digit = HexCharToInt(regex[i]);
        if (digit < 0) {
          return ResultErr(
              "\\" + std::string(1, escaped) + " must be followed by " +
              std::to_string(digit_count) + " hexadecimal digits in regex"
          );
        }
        codepoint = codepoint * 16 + digit;
      }
      if (codepoint > kMaxCodepoint || (0xD800 <= codepoint && codepoint <= 0xDFFF)) {
        return ResultErr("escaped value is not a Unicode scalar value");
      }
      *pos += digit_count;
      return single(codepoint);
    }
    case 'c': {
      if (*pos >= regex.size() || !std::isalpha(static_cast<unsigned char>(regex[*pos]))) {
        return ResultErr("\\c must be followed by a letter in regex");
      }
      uint32_t codepoint = static_cast<unsigned char>(regex[*pos]) & 0x1F;
      *pos += 1;
      return single(codepoint);
    }
    default: {
      if (static_cast<unsigned char>(escaped) >= 0x80) {
        // Multi-byte UTF-8 character after the backslash: match it literally.
        *pos -= 1;
        auto [codepoint, num_bytes] = ParseNextUTF8(regex.c_str() + *pos);
        if (codepoint == CharHandlingError::kInvalidUTF8 || *pos + num_bytes > regex.size()) {
          return ResultErr("Invalid UTF-8 in regex escape sequence");
        }
        *pos += num_bytes;
        return single(static_cast<uint32_t>(codepoint));
      }
      if (std::isalnum(static_cast<unsigned char>(escaped)) || escaped == '<' || escaped == '>') {
        XGRAMMAR_LOG(WARNING) << "Escape sequence \\" << escaped
                              << " is not recognized in regex; matching the character literally";
      }
      return single(static_cast<unsigned char>(escaped));
    }
  }
}

Result<RegexEscapeItem> ParseRegexEscape(
    const std::string& regex,
    size_t* pos,
    bool in_class,
    bool byte_mode,
    RegexCharacterClassDialect character_class_dialect
) {
  return byte_mode ? ParseByteRegexEscape(regex, pos, in_class)
                   : ParseCodepointRegexEscape(regex, pos, in_class, character_class_dialect);
}

enum class CharacterClassSetOp { kIntersection, kDifference, kSymmetricDifference };

struct CharacterClassUnit {
  std::vector<CodepointRange> ranges;
  bool is_single = false;
  uint32_t codepoint = 0;
};

std::optional<std::vector<CodepointRange>> ASCIICharacterClassRanges(const std::string& name) {
  if (name == "alnum") return {{{'0', '9'}, {'A', 'Z'}, {'a', 'z'}}};
  if (name == "alpha") return {{{'A', 'Z'}, {'a', 'z'}}};
  if (name == "ascii") return {{{0x00, 0x7F}}};
  if (name == "blank") return {{{'\t', '\t'}, {' ', ' '}}};
  if (name == "cntrl") return {{{0x00, 0x1F}, {0x7F, 0x7F}}};
  if (name == "digit") return {{{'0', '9'}}};
  if (name == "graph") return {{{'!', '~'}}};
  if (name == "lower") return {{{'a', 'z'}}};
  if (name == "print") return {{{' ', '~'}}};
  if (name == "punct") {
    return {{{'!', '/'}, {':', '@'}, {'[', '`'}, {'{', '~'}}};
  }
  if (name == "space") return {{{0x09, 0x0D}, {' ', ' '}}};
  if (name == "upper") return {{{'A', 'Z'}}};
  if (name == "word") return {{{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}}};
  if (name == "xdigit") return {{{'0', '9'}, {'A', 'F'}, {'a', 'f'}}};
  return std::nullopt;
}

Result<std::vector<CodepointRange>> ParseBracketedCharacterClass(
    const std::string& regex,
    size_t* pos,
    bool case_insensitive,
    bool byte_mode,
    RegexCharacterClassDialect character_class_dialect
);

Result<CharacterClassUnit> ParseCharacterClassUnit(
    const std::string& regex,
    size_t* pos,
    bool case_insensitive,
    bool byte_mode,
    RegexCharacterClassDialect character_class_dialect
) {
  XGRAMMAR_DCHECK(*pos < regex.size());
  if (regex[*pos] == '[') {
    if (*pos + 1 < regex.size() && regex[*pos + 1] == ':') {
      size_t name_start = *pos + 2;
      bool negated = name_start < regex.size() && regex[name_start] == '^';
      if (negated) ++name_start;
      size_t close = regex.find(":]", name_start);
      if (close != std::string::npos) {
        std::string name = regex.substr(name_start, close - name_start);
        auto ranges = ASCIICharacterClassRanges(name);
        if (ranges.has_value()) {
          *pos = close + 2;
          if (case_insensitive) {
            FoldCaseRanges(&ranges.value(), byte_mode);
          }
          NormalizeRanges(&ranges.value());
          if (negated) {
            ranges = ComplementRanges(ranges.value(), byte_mode ? kMaxByteValue : kMaxCodepoint);
          }
          if (!byte_mode) {
            RemoveUnicodeSurrogates(&ranges.value());
          }
          return ResultOk(CharacterClassUnit{std::move(ranges.value()), false, 0});
        }
      }
      // regex-syntax treats a malformed or unknown `[:name:]` spelling as an ordinary nested
      // class instead of reporting a special POSIX-class error.
    }
    auto nested = ParseBracketedCharacterClass(
        regex, pos, case_insensitive, byte_mode, character_class_dialect
    );
    if (nested.IsErr()) {
      return ResultErr(std::move(nested).UnwrapErr());
    }
    return ResultOk(CharacterClassUnit{std::move(nested).Unwrap(), false, 0});
  }

  RegexEscapeItem item;
  if (regex[*pos] == '\\') {
    auto parsed =
        ParseRegexEscape(regex, pos, /*in_class=*/true, byte_mode, character_class_dialect);
    if (parsed.IsErr()) {
      return ResultErr(std::move(parsed).UnwrapErr());
    }
    item = std::move(parsed).Unwrap();
  } else if (byte_mode) {
    uint8_t byte = static_cast<uint8_t>(regex[*pos]);
    if (byte >= 0x80) {
      return ResultErr("non-ASCII characters are not available in byte character classes; use \\xHH"
      );
    }
    ++*pos;
    item.is_single = true;
    item.codepoint = byte;
  } else {
    auto [codepoint, num_bytes] = ParseNextUTF8(regex.c_str() + *pos);
    if (codepoint == CharHandlingError::kInvalidUTF8 || *pos + num_bytes > regex.size()) {
      return ResultErr("Invalid UTF-8 in regex character class");
    }
    *pos += num_bytes;
    item.is_single = true;
    item.codepoint = static_cast<uint32_t>(codepoint);
  }

  if (item.is_single) {
    return ResultOk(CharacterClassUnit{{{item.codepoint, item.codepoint}}, true, item.codepoint});
  }
  if (case_insensitive) {
    FoldCaseRanges(&item.ranges, byte_mode);
  }
  NormalizeRanges(&item.ranges);
  if (item.negated) {
    item.ranges = ComplementRanges(item.ranges, byte_mode ? kMaxByteValue : kMaxCodepoint);
  }
  if (!byte_mode) {
    RemoveUnicodeSurrogates(&item.ranges);
  }
  return ResultOk(CharacterClassUnit{std::move(item.ranges), false, 0});
}

Result<std::vector<CodepointRange>> ParseBracketedCharacterClass(
    const std::string& regex,
    size_t* pos,
    bool case_insensitive,
    bool byte_mode,
    RegexCharacterClassDialect character_class_dialect
) {
  XGRAMMAR_DCHECK(*pos < regex.size() && regex[*pos] == '[');
  ++*pos;
  bool negated = *pos < regex.size() && regex[*pos] == '^';
  if (negated) ++*pos;

  std::vector<CodepointRange> accumulated;
  std::vector<CodepointRange> operand;
  std::optional<CharacterClassSetOp> pending_op;
  bool saw_unit = false;
  const uint32_t universe_max = byte_mode ? kMaxByteValue : kMaxCodepoint;

  // At the opening of a class, regex-syntax treats any number of '-' characters literally.
  while (*pos < regex.size() && regex[*pos] == '-') {
    operand.push_back({'-', '-'});
    saw_unit = true;
    ++*pos;
  }
  // A closing bracket is literal only when it is the first item (after an optional '^').
  if (!saw_unit && *pos < regex.size() && regex[*pos] == ']') {
    operand.push_back({']', ']'});
    saw_unit = true;
    ++*pos;
  }

  auto finish_operand = [&]() -> Result<bool> {
    if (!saw_unit) {
      return ResultErr("character-class set operator is missing an operand");
    }
    if (case_insensitive) {
      FoldCaseRanges(&operand, byte_mode);
    }
    NormalizeRanges(&operand);
    if (!pending_op.has_value()) {
      accumulated = std::move(operand);
    } else {
      switch (pending_op.value()) {
        case CharacterClassSetOp::kIntersection:
          accumulated = IntersectRanges(accumulated, operand);
          break;
        case CharacterClassSetOp::kDifference:
          accumulated = DifferenceRanges(accumulated, operand, universe_max);
          break;
        case CharacterClassSetOp::kSymmetricDifference:
          accumulated = SymmetricDifferenceRanges(accumulated, operand, universe_max);
          break;
      }
    }
    operand.clear();
    return ResultOk(true);
  };

  while (*pos < regex.size()) {
    if (regex[*pos] == ']') {
      if (!saw_unit) {
        if (pending_op.has_value()) {
          return ResultErr("character-class set operator is missing an operand");
        }
        return ResultErr("Empty character class is not allowed in regex");
      }
      auto finished = finish_operand();
      if (finished.IsErr()) {
        return ResultErr(std::move(finished).UnwrapErr());
      }
      ++*pos;
      if (case_insensitive) {
        FoldCaseRanges(&accumulated, byte_mode);
      }
      NormalizeRanges(&accumulated);
      if (negated) {
        accumulated = ComplementRanges(accumulated, universe_max);
      }
      if (!byte_mode) {
        RemoveUnicodeSurrogates(&accumulated);
      }
      return ResultOk(std::move(accumulated));
    }

    std::optional<CharacterClassSetOp> next_op;
    if (*pos + 1 < regex.size()) {
      std::string_view candidate(regex.data() + *pos, 2);
      if (candidate == "&&") next_op = CharacterClassSetOp::kIntersection;
      if (candidate == "--") next_op = CharacterClassSetOp::kDifference;
      if (candidate == "~~") next_op = CharacterClassSetOp::kSymmetricDifference;
    }
    if (next_op.has_value()) {
      auto finished = finish_operand();
      if (finished.IsErr()) {
        return ResultErr(std::move(finished).UnwrapErr());
      }
      pending_op = next_op;
      *pos += 2;
      saw_unit = false;
      continue;
    }

    auto unit_result =
        ParseCharacterClassUnit(regex, pos, case_insensitive, byte_mode, character_class_dialect);
    if (unit_result.IsErr()) {
      return ResultErr(std::move(unit_result).UnwrapErr());
    }
    auto unit = std::move(unit_result).Unwrap();
    saw_unit = true;
    bool is_range = unit.is_single && *pos < regex.size() && regex[*pos] == '-' &&
                    *pos + 1 < regex.size() && regex[*pos + 1] != '-' && regex[*pos + 1] != ']';
    if (is_range) {
      ++*pos;
      auto high_result =
          ParseCharacterClassUnit(regex, pos, case_insensitive, byte_mode, character_class_dialect);
      if (high_result.IsErr()) {
        return ResultErr(std::move(high_result).UnwrapErr());
      }
      auto high = std::move(high_result).Unwrap();
      if (!high.is_single) {
        return ResultErr(
            byte_mode ? "character-class range endpoint must be a single byte"
                      : "character-class range endpoint must be a single character"
        );
      }
      if (high.codepoint < unit.codepoint) {
        return ResultErr("character-class range lower bound exceeds its upper bound");
      }
      operand.push_back({unit.codepoint, high.codepoint});
    } else {
      operand.insert(operand.end(), unit.ranges.begin(), unit.ranges.end());
    }
  }
  return ResultErr("Unclosed '[' in regular expression");
}

/*!
 * \brief Parse a character class leaf "[...]" into the final set of accepted codepoint ranges.
 */
Result<std::vector<CodepointRange>> ParseCharacterClassLeaf(
    const std::string& regex,
    bool case_insensitive,
    bool byte_mode,
    RegexCharacterClassDialect character_class_dialect
) {
  size_t pos = 0;
  auto result = ParseBracketedCharacterClass(
      regex, &pos, case_insensitive, byte_mode, character_class_dialect
  );
  if (result.IsErr()) {
    return result;
  }
  if (pos != regex.size()) {
    return ResultErr("unexpected characters after regex character class");
  }
  return result;
}

}  // namespace

/******************** RegexIR ********************/

class RegexIR {
 public:
  struct Leaf;

  struct Symbol;

  struct Union;

  struct Bracket;

  struct Repeat;

  struct RuleRefNode;

  struct RepeatSubrule;

  static constexpr int kRepeatNoUpperBound = -1;

  using State = std::variant<Leaf, Symbol, Union, Bracket, Repeat, RuleRefNode, RepeatSubrule>;

  // This struct is used to store one atom of the regex: the empty string (regex == ""), a
  // character class (regex == "[...]"), or a short sequence of literal characters / escapes.
  struct Leaf {
    std::string regex;
    bool case_insensitive = false;
    bool byte_mode = false;
  };

  // This struct is used to store the symbol in regex, i.e.
  // +, *, ?
  enum class RegexSymbol {
    star,
    plus,
    optional,
  };

  struct Bracket {
    std::vector<State> states;
  };

  struct Symbol {
    RegexSymbol symbol;
    std::vector<State> state;
  };

  // This struct is used to represent a union symbol.
  struct Union {
    std::vector<State> states;
  };

  struct Repeat {
    std::vector<State> states;
    int lower_bound = 0;
    int upper_bound = 0;
  };

  // A reference to a grammar rule, compiled into a kRuleRef FSM edge.
  struct RuleRefNode {
    int32_t rule_id;
  };

  // A bounded repetition of a grammar rule, compiled into a kRepeatRef FSM edge. The referenced
  // rule holds the repeated sub-pattern; the Earley parser executes the repetition with a
  // counter at runtime, so no FSM unrolling happens.
  struct RepeatSubrule {
    int32_t rule_id;
    int lower_bound = 0;
    int upper_bound = 0;
  };

  // The top-level sequence of the regex.
  std::vector<State> states;

  // Whether characters are interpreted as raw bytes instead of Unicode codepoints.
  bool byte_mode = false;

  // Whether logical codepoints are matched through their valid JSON source spellings.
  bool json_string = false;

  /*!
    \brief Constructs a NFA from the regex IR.
  */
  Result<FSMWithStartEnd> Build() const;

  /*! \brief Validate all leaves without expanding repetition or composition nodes. */
  Result<bool> Validate() const;

  /*! \brief Whether the parsed regex contains a repetition retained by a GrammarBuilder. */
  bool HasLargeRepeat() const;

  /*!
    \brief the visit function for the variant.
  */
  Result<FSMWithStartEnd> visit(const Leaf& state) const;

  Result<FSMWithStartEnd> visit(const Symbol& state) const;

  Result<FSMWithStartEnd> visit(const Union& state) const;

  Result<FSMWithStartEnd> visit(const Bracket& state) const;

  Result<FSMWithStartEnd> visit(const Repeat& state) const;

  Result<FSMWithStartEnd> visit(const RuleRefNode& state) const;

  Result<FSMWithStartEnd> visit(const RepeatSubrule& state) const;

  /*! \brief Whether the IR node can match the empty string. Purely syntactic; no FSM is built. */
  static bool IsNullable(const State& state);

  /*! \brief Whether a sequence of IR nodes can match the empty string. */
  static bool IsNullableSequence(const std::vector<State>& states);

  /*!
   * \brief Check repeat in regex. i.e {...} and {...,...}
   * \param regex The regex string.
   * \param start The start position of the repeat. i.e. regex[start] == '{'.
   * After the function, start will be the position of '}'.
   * \return The repeat range.
   */
  static Result<std::pair<int, int>> CheckRepeat(const std::string& regex, int& start);

 private:
  /*!
   * \brief Construct a FSM from a regex leaf.
   * \details The leaf is the empty string, a character class like [a-c0-9], or a sequence of
   * literal characters / escapes like "ab\n". Any symbols like "a|b" or "a*b" are not supported.
   * \param regex The regex string.
   * \return The FSM with start and end states.
   */
  Result<FSMWithStartEnd> BuildLeafFSMFromRegex(
      const std::string& regex, bool case_insensitive, bool leaf_byte_mode
  ) const;

  /*!
   * \brief Add the transition(s) accepting a single codepoint (with case folding when requested)
   * from `current` to a new state, and return the new state.
   */
  int AddSingleCodepoint(
      FSMWithStartEnd& result,
      int current,
      uint32_t codepoint,
      bool case_insensitive,
      bool leaf_byte_mode
  ) const;

  Result<bool> ValidateState(const State& state) const;

  static bool HasLargeRepeatState(const State& state, int threshold);
};

Result<std::pair<int, int>> RegexIR::CheckRepeat(const std::string& regex, int& start) {
  // 10^9 fits in an int; longer counts would overflow.
  constexpr size_t kMaxRepeatDigits = 9;
  if (start < 0 || static_cast<size_t>(start) >= regex.size() || regex[start] != '{') {
    return ResultErr("Invalid repetition: expected '{'");
  }
  int lower_bound = 0;
  int upper_bound = RegexIR::kRepeatNoUpperBound;
  std::string num_str;
  XGRAMMAR_DCHECK(regex[start] == '{');
  start++;
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  while (static_cast<size_t>(start) < regex.size() && std::isdigit(regex[start])) {
    num_str += regex[start];
    start++;
  }
  if (num_str.empty()) {
    return ResultErr("Invalid repetition count: expected a number after '{'");
  }
  if (num_str.size() > kMaxRepeatDigits) {
    return ResultErr("Invalid repetition count: the count " + num_str + " is too large");
  }
  lower_bound = std::stoi(num_str);
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  // The format is {n}
  if (static_cast<size_t>(start) >= regex.size()) {
    return ResultErr("Invalid repetition count: expected ',' or '}' after the lower bound");
  }
  if (regex[start] == '}') {
    upper_bound = lower_bound;
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  if (regex[start] != ',') {
    return ResultErr("Invalid repetition count: expected ',' or '}' after the lower bound");
  }
  XGRAMMAR_DCHECK(regex[start] == ',');
  start++;
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  // The format is {n,}
  if (static_cast<size_t>(start) >= regex.size()) {
    return ResultErr("Invalid repetition count: expected a number or '}' after ','");
  }
  if (regex[start] == '}') {
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  num_str.clear();
  while (static_cast<size_t>(start) < regex.size() && std::isdigit(regex[start])) {
    num_str += regex[start];
    start++;
  }
  if (num_str.empty()) {
    return ResultErr("Invalid repetition count: expected a number or '}' after ','");
  }
  if (num_str.size() > kMaxRepeatDigits) {
    return ResultErr("Invalid repetition count: the count " + num_str + " is too large");
  }
  upper_bound = std::stoi(num_str);
  if (upper_bound < lower_bound) {
    return ResultErr(
        "Invalid repetition count: the lower bound " + std::to_string(lower_bound) +
        " is larger than the upper bound " + std::to_string(upper_bound)
    );
  }
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  if (static_cast<size_t>(start) >= regex.size() || regex[start] != '}') {
    return ResultErr("Invalid repetition count: expected '}' after the upper bound");
  }
  XGRAMMAR_DCHECK(regex[start] == '}');
  return ResultOk(std::make_pair(lower_bound, upper_bound));
}

bool RegexIR::IsNullableSequence(const std::vector<State>& states) {
  return std::all_of(states.begin(), states.end(), [](const State& state) {
    return IsNullable(state);
  });
}

bool RegexIR::IsNullable(const State& state) {
  return std::visit(
      [](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Leaf>) {
          return node.regex.empty();
        } else if constexpr (std::is_same_v<T, Symbol>) {
          return node.symbol != RegexSymbol::plus || IsNullableSequence(node.state);
        } else if constexpr (std::is_same_v<T, Union>) {
          return std::any_of(node.states.begin(), node.states.end(), [](const State& child) {
            return IsNullable(child);
          });
        } else if constexpr (std::is_same_v<T, Bracket>) {
          return IsNullableSequence(node.states);
        } else if constexpr (std::is_same_v<T, Repeat>) {
          return node.lower_bound == 0 || IsNullableSequence(node.states);
        } else if constexpr (std::is_same_v<T, RuleRefNode>) {
          return false;
        } else {
          static_assert(std::is_same_v<T, RepeatSubrule>);
          return node.lower_bound == 0;
        }
      },
      state
  );
}

Result<bool> RegexIR::ValidateState(const State& state) const {
  return std::visit(
      [&](const auto& node) -> Result<bool> {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Leaf>) {
          auto leaf = BuildLeafFSMFromRegex(node.regex, node.case_insensitive, node.byte_mode);
          if (leaf.IsErr()) {
            return ResultErr(std::move(leaf).UnwrapErr());
          }
          return ResultOk(true);
        } else if constexpr (std::is_same_v<T, Symbol>) {
          for (const auto& child : node.state) {
            auto validation = ValidateState(child);
            if (validation.IsErr()) {
              return validation;
            }
          }
          return ResultOk(true);
        } else if constexpr (std::is_same_v<T, Union> || std::is_same_v<T, Bracket> ||
                             std::is_same_v<T, Repeat>) {
          for (const auto& child : node.states) {
            auto validation = ValidateState(child);
            if (validation.IsErr()) {
              return validation;
            }
          }
          return ResultOk(true);
        } else {
          return ResultOk(true);
        }
      },
      state
  );
}

Result<bool> RegexIR::Validate() const {
  for (const auto& state : states) {
    auto validation = ValidateState(state);
    if (validation.IsErr()) {
      return validation;
    }
  }
  return ResultOk(true);
}

bool RegexIR::HasLargeRepeatState(const State& state, int threshold) {
  return std::visit(
      [threshold](const auto& node) -> bool {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Repeat>) {
          const bool is_large = node.upper_bound == kRepeatNoUpperBound
                                    ? node.lower_bound > threshold
                                    : node.upper_bound > threshold;
          return is_large || std::any_of(
                                 node.states.begin(),
                                 node.states.end(),
                                 [threshold](const State& child) {
                                   return HasLargeRepeatState(child, threshold);
                                 }
                             );
        } else if constexpr (std::is_same_v<T, Symbol>) {
          return std::any_of(node.state.begin(), node.state.end(), [threshold](const State& child) {
            return HasLargeRepeatState(child, threshold);
          });
        } else if constexpr (std::is_same_v<T, Union> || std::is_same_v<T, Bracket>) {
          return std::any_of(
              node.states.begin(),
              node.states.end(),
              [threshold](const State& child) { return HasLargeRepeatState(child, threshold); }
          );
        } else {
          return false;
        }
      },
      state
  );
}

bool RegexIR::HasLargeRepeat() const {
  const int threshold = json_string ? kLargeJSONStringRepeatThreshold : kLargeRepeatThreshold;
  return std::any_of(states.begin(), states.end(), [threshold](const State& state) {
    return HasLargeRepeatState(state, threshold);
  });
}

Result<FSMWithStartEnd> RegexIR::Build() const {
  if (states.empty()) {
    FSM empty_fsm(1);
    FSMWithStartEnd result(empty_fsm, 0, {0}, false);
    return ResultOk(std::move(result));
  }
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& state : states) {
    auto visited = std::visit([&](auto&& arg) { return visit(arg); }, state);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.size() > 1) {
    return ResultOk(FSMWithStartEnd::Concat(fsm_list));
  } else {
    // If there is only one FSM, return it directly.
    return ResultOk(std::move(fsm_list[0]));
  }
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Leaf& state) const {
  return BuildLeafFSMFromRegex(state.regex, state.case_insensitive, state.byte_mode);
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Union& state) const {
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : state.states) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.size() <= 1) {
    return ResultErr("Internal error: a union node in the regex IR has fewer than two branches");
  }
  return ResultOk(FSMWithStartEnd::Union(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Symbol& state) const {
  if (state.state.size() != 1) {
    return ResultErr("Internal error: a quantifier node in the regex IR must hold exactly one child"
    );
  }
  Result<FSMWithStartEnd> child_result =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, state.state[0]);
  if (child_result.IsErr()) {
    return child_result;
  }
  auto child = std::move(child_result).Unwrap();

  switch (state.symbol) {
    case RegexIR::RegexSymbol::plus: {
      return ResultOk(child.Plus());
    }
    case RegexIR::RegexSymbol::star: {
      return ResultOk(child.Star());
    }
    case RegexIR::RegexSymbol::optional: {
      return ResultOk(child.Optional());
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Unknown regex symbol: " << static_cast<int>(state.symbol);
      XGRAMMAR_UNREACHABLE();
    }
  }
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Bracket& state) const {
  if (state.states.empty()) {
    // An empty group or an empty union branch matches the empty string.
    FSM empty_fsm(1);
    return ResultOk(FSMWithStartEnd(empty_fsm, 0, {0}, false));
  }
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : state.states) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  return ResultOk(FSMWithStartEnd::Concat(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::RuleRefNode& state) const {
  FSM fsm(2);
  fsm.AddRuleEdge(0, 1, state.rule_id);
  return ResultOk(FSMWithStartEnd(fsm, 0, {1}, false));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::RepeatSubrule& state) const {
  // The state holding the repeat edge is padded with epsilon transitions on both sides, so that
  // FSM compositions (Concat / Star / Optional / ...) never add another outgoing edge to it: a
  // state with a kRepeatRef edge must have no other outgoing edges.
  FSM fsm(4);
  fsm.AddEpsilonEdge(0, 1);
  fsm.AddRepeatEdge(1, 2, state.rule_id, state.lower_bound, state.upper_bound);
  fsm.AddEpsilonEdge(2, 3);
  return ResultOk(FSMWithStartEnd(fsm, 0, {3}, false));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Repeat& state) const {
  if (state.states.size() != 1) {
    return ResultErr("Internal error: a repetition node in the regex IR must hold exactly one child"
    );
  }
  bool has_upper_bound = state.upper_bound != RegexIR::kRepeatNoUpperBound;
  if (has_upper_bound && state.upper_bound == 0) {
    // {0} / {0,0}: matches exactly the empty string. The general path below cannot express
    // this: it starts from one copy of the child whose end states stay accepting.
    FSM empty_fsm(1);
    return ResultOk(FSMWithStartEnd(empty_fsm, 0, {0}, false));
  }
  Result<FSMWithStartEnd> child_result =
      std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, state.states[0]);
  if (child_result.IsErr()) {
    return child_result;
  }
  FSMWithStartEnd child = std::move(child_result).Unwrap();

  // Guard against FSM state explosion when the repetition has to be physically unrolled. When
  // a GrammarBuilder is available, large repetitions are compiled into repeat edges instead and
  // never reach this point.
  int64_t num_copies = has_upper_bound ? state.upper_bound : std::max(state.lower_bound, 1);
  if (static_cast<int64_t>(child.NumStates()) * num_copies > kMaxUnrolledRepeatStates) {
    return ResultErr(
        "The bounded repetition {" + std::to_string(state.lower_bound) + "," +
        (has_upper_bound ? std::to_string(state.upper_bound) : "") +
        "} is too large to compile into a FSM"
    );
  }

  FSMWithStartEnd result = child.Copy();
  std::unordered_set<int> new_ends;

  if (state.lower_bound <= 1 && (!has_upper_bound || state.upper_bound >= 1)) {
    // A single copy is accepting when the lower bound is at most 1.
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        new_ends.insert(end);
      }
    }
  }

  // Add a fresh accepting start state so that zero repetitions match. A fresh state is
  // required: making the original start accepting would also accept strings that merely
  // loop back to the start inside the first copy.
  auto allow_zero_repetitions = [](FSMWithStartEnd* fsm) {
    int new_start = fsm->AddState();
    fsm->GetFsm().AddEpsilonEdge(new_start, fsm->GetStart());
    fsm->SetStartState(new_start);
    fsm->AddEndState(new_start);
  };

  // Handling {n,}
  if (!has_upper_bound) {
    for (int i = 2; i < state.lower_bound; i++) {
      result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    }
    int end_state_of_lower_bound_fsm = -1;
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        end_state_of_lower_bound_fsm = end;
        break;
      }
    }
    XGRAMMAR_DCHECK(end_state_of_lower_bound_fsm != -1)
        << "No end state found in the lower bound FSM.";
    result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    for (int end = 0; end < result.NumStates(); ++end) {
      if (result.IsEndState(end)) {
        result.GetFsm().AddEpsilonEdge(end, end_state_of_lower_bound_fsm);
      }
    }
    for (const auto& end : new_ends) {
      result.AddEndState(end);
    }
    if (state.lower_bound == 0) {
      allow_zero_repetitions(&result);
    }
    return ResultOk(std::move(result));
  }
  // Handling {n, m} or {n}
  for (int i = 2; i <= state.upper_bound; i++) {
    result = FSMWithStartEnd::Concat(std::vector<FSMWithStartEnd>{result, child});
    if (i >= state.lower_bound) {
      for (int end = 0; end < result.NumStates(); ++end) {
        if (result.IsEndState(end)) {
          new_ends.insert(end);
        }
      }
    }
  }
  for (const auto& end : new_ends) {
    result.AddEndState(end);
  }
  if (state.lower_bound == 0) {
    allow_zero_repetitions(&result);
  }
  return ResultOk(std::move(result));
}

int RegexIR::AddSingleCodepoint(
    FSMWithStartEnd& result,
    int current,
    uint32_t codepoint,
    bool case_insensitive,
    bool leaf_byte_mode
) const {
  int next = result.AddState();
  if (case_insensitive && !leaf_byte_mode) {
    std::vector<CodepointRange> ranges = {{codepoint, codepoint}};
    FoldCaseRanges(&ranges, /*byte_mode=*/false);
    NormalizeRanges(&ranges);
    if (json_string) {
      AddJSONStringCodepointRangesToFSM(&result.GetFsm(), current, next, ranges);
    } else {
      AddCodepointRangesToFSM(&result.GetFsm(), current, next, ranges);
    }
    return next;
  }
  if (json_string) {
    AddJSONStringCodepointRangesToFSM(&result.GetFsm(), current, next, {{codepoint, codepoint}});
    return next;
  }
  if (leaf_byte_mode || codepoint <= kMax1ByteUnicode) {
    XGRAMMAR_DCHECK(!leaf_byte_mode || codepoint <= kMaxByteValue);
    result.GetFsm().AddEdge(current, next, codepoint, codepoint);
    if (case_insensitive) {
      if (codepoint >= 'a' && codepoint <= 'z') {
        uint32_t upper = codepoint - ('a' - 'A');
        result.GetFsm().AddEdge(current, next, upper, upper);
      } else if (codepoint >= 'A' && codepoint <= 'Z') {
        uint32_t lower = codepoint + ('a' - 'A');
        result.GetFsm().AddEdge(current, next, lower, lower);
      }
    }
    return next;
  }
  std::string utf8_bytes = CharToUTF8(static_cast<TCodepoint>(codepoint));
  int state = current;
  for (size_t i = 0; i < utf8_bytes.size(); ++i) {
    int target = (i + 1 == utf8_bytes.size()) ? next : result.AddState();
    uint8_t byte = static_cast<uint8_t>(utf8_bytes[i]);
    result.GetFsm().AddEdge(state, target, byte, byte);
    state = target;
  }
  return next;
}

Result<FSMWithStartEnd> RegexIR::BuildLeafFSMFromRegex(
    const std::string& regex, bool case_insensitive, bool leaf_byte_mode
) const {
  const auto character_class_dialect =
      json_string ? RegexCharacterClassDialect::kECMAScript : RegexCharacterClassDialect::kUnicode;
  FSM initial_fsm(1);
  FSMWithStartEnd result(initial_fsm, 0, {}, false);
  if (regex.empty()) {
    // The empty leaf matches the empty string.
    result.AddEndState(0);
    return ResultOk(std::move(result));
  }
  if (regex[0] == '[') {
    // Character class.
    auto ranges_result =
        ParseCharacterClassLeaf(regex, case_insensitive, leaf_byte_mode, character_class_dialect);
    if (ranges_result.IsErr()) {
      return ResultErr(std::move(ranges_result).UnwrapErr());
    }
    auto ranges = std::move(ranges_result).Unwrap();
    int end_state = result.AddState();
    if (leaf_byte_mode && !byte_mode &&
        std::any_of(ranges.begin(), ranges.end(), [](const CodepointRange& range) {
          return range.second > 0x7F;
        })) {
      return ResultErr("pattern can match invalid UTF-8");
    }
    if (leaf_byte_mode) {
      for (const auto& [low, high] : ranges) {
        result.GetFsm().AddEdge(0, end_state, low, high);
      }
    } else if (json_string) {
      AddJSONStringCodepointRangesToFSM(&result.GetFsm(), 0, end_state, ranges);
    } else {
      AddCodepointRangesToFSM(&result.GetFsm(), 0, end_state, ranges);
    }
    result.AddEndState(end_state);
    return ResultOk(std::move(result));
  }
  // Sequence of literal characters, '.' and escapes.
  int current = 0;
  size_t pos = 0;
  while (pos < regex.size()) {
    if (regex[pos] == '.') {
      ++pos;
      int next = result.AddState();
      if (leaf_byte_mode && !byte_mode) {
        return ResultErr("pattern can match invalid UTF-8");
      }
      if (leaf_byte_mode) {
        result.GetFsm().AddEdge(current, next, 0, kMaxByteValue);
      } else if (json_string) {
        AddJSONStringCodepointRangesToFSM(&result.GetFsm(), current, next, {{0, kMaxCodepoint}});
      } else {
        AddCodepointRangesToFSM(&result.GetFsm(), current, next, {{0, kMaxCodepoint}});
      }
      current = next;
      continue;
    }
    if (regex[pos] == '\\') {
      auto item_result = ParseRegexEscape(
          regex, &pos, /*in_class=*/false, leaf_byte_mode, character_class_dialect
      );
      if (item_result.IsErr()) {
        return ResultErr(std::move(item_result).UnwrapErr());
      }
      auto item = std::move(item_result).Unwrap();
      if (item.is_single) {
        if (leaf_byte_mode && !byte_mode && item.codepoint > 0x7F) {
          return ResultErr("pattern can match invalid UTF-8");
        }
        current =
            AddSingleCodepoint(result, current, item.codepoint, case_insensitive, leaf_byte_mode);
      } else {
        auto ranges = std::move(item.ranges);
        if (case_insensitive) {
          FoldCaseRanges(&ranges, leaf_byte_mode);
        }
        NormalizeRanges(&ranges);
        if (item.negated) {
          ranges = ComplementRanges(ranges, leaf_byte_mode ? kMaxByteValue : kMaxCodepoint);
        }
        if (!leaf_byte_mode) {
          RemoveUnicodeSurrogates(&ranges);
        }
        if (leaf_byte_mode && !byte_mode &&
            std::any_of(ranges.begin(), ranges.end(), [](const CodepointRange& range) {
              return range.second > 0x7F;
            })) {
          return ResultErr("pattern can match invalid UTF-8");
        }
        int next = result.AddState();
        if (leaf_byte_mode) {
          for (const auto& [low, high] : ranges) {
            result.GetFsm().AddEdge(current, next, low, high);
          }
        } else if (json_string) {
          AddJSONStringCodepointRangesToFSM(&result.GetFsm(), current, next, ranges);
        } else {
          AddCodepointRangesToFSM(&result.GetFsm(), current, next, ranges);
        }
        current = next;
      }
      continue;
    }
    if (leaf_byte_mode) {
      uint8_t byte = static_cast<uint8_t>(regex[pos]);
      if (!byte_mode && byte > 0x7F) {
        return ResultErr("pattern can match invalid UTF-8");
      }
      current = AddSingleCodepoint(result, current, byte, case_insensitive, leaf_byte_mode);
      ++pos;
      continue;
    }
    auto [codepoint, num_bytes] = ParseNextUTF8(regex.c_str() + pos);
    if (codepoint == CharHandlingError::kInvalidUTF8 || pos + num_bytes > regex.size()) {
      if (json_string) {
        return ResultErr("Invalid UTF-8 in regex matched as decoded JSON string contents");
      }
      // Be permissive with non-UTF-8 patterns: match the raw byte.
      int next = result.AddState();
      uint8_t byte = static_cast<uint8_t>(regex[pos]);
      result.GetFsm().AddEdge(current, next, byte, byte);
      current = next;
      ++pos;
      continue;
    }
    current = AddSingleCodepoint(
        result, current, static_cast<uint32_t>(codepoint), case_insensitive, leaf_byte_mode
    );
    pos += num_bytes;
  }
  result.AddEndState(current);
  return ResultOk(std::move(result));
}

/******************** RegexFSMBuilder ********************/

namespace {

/*! \brief One entry of the regex parsing stack: an IR node or a marker character ('(' or '|'),
 * together with the source span [span_begin, span_end) of the atom in the regex string. */
struct RegexStackEntry {
  std::variant<RegexIR::State, char> item;
  int span_begin = 0;
  int span_end = 0;
  bool inherited_case_insensitive = false;
  bool inherited_multiline = false;
  bool inherited_unicode = true;
};

/*! \brief Skip a character class starting at regex[pos] == '['. Returns the position right after
 * the closing ']', or std::string::npos if the class is not closed. */
size_t SkipCharacterClass(const std::string& regex, size_t pos) {
  XGRAMMAR_DCHECK(regex[pos] == '[');
  struct Frame {
    bool at_start = true;
    bool allow_negation = true;
    bool saw_initial_hyphen = false;
  };
  std::vector<Frame> frames;
  while (pos < regex.size()) {
    if (regex[pos] == '\\') {
      if (!frames.empty()) {
        frames.back().at_start = false;
      }
      pos += 2;
      continue;
    }
    if (regex[pos] == '[') {
      if (!frames.empty()) {
        frames.back().at_start = false;
      }
      frames.push_back(Frame{});
      ++pos;
      continue;
    }
    XGRAMMAR_DCHECK(!frames.empty());
    if (frames.back().at_start) {
      if (regex[pos] == '^' && frames.back().allow_negation) {
        frames.back().allow_negation = false;
        ++pos;
        continue;
      }
      if (regex[pos] == '-') {
        frames.back().allow_negation = false;
        frames.back().saw_initial_hyphen = true;
        ++pos;
        continue;
      }
      if (regex[pos] == ']' && !frames.back().saw_initial_hyphen) {
        frames.back().at_start = false;
        ++pos;
        continue;
      }
      frames.back().at_start = false;
    }
    if (regex[pos] == ']') {
      frames.pop_back();
      ++pos;
      if (frames.empty()) {
        return pos;
      }
      continue;
    }
    ++pos;
  }
  return std::string::npos;
}

/*!
 * \brief Parse a regex string into a RegexIR.
 * \param regex_with_flags The regex, including any top-level or scoped inline flags.
 * \param builder If not null, large bounded repetitions are compiled into subrules added through
 * this builder; see RegexFSMBuilder::Build.
 * \param rule_hint Name hint for the created subrules.
 */
Result<RegexIR> ParseRegexToIR(
    const std::string& regex_with_flags,
    GrammarBuilder* builder,
    const std::string& rule_hint,
    bool byte_mode,
    bool json_string = false
) {
  RegexIR ir;
  ir.byte_mode = byte_mode;
  ir.json_string = json_string;
  std::string rewritten_regex = RewriteRegexExtended(regex_with_flags);
  const std::string& regex = rewritten_regex;
  bool case_insensitive = false;
  bool multiline = false;
  bool unicode = !byte_mode;
  const auto character_class_dialect =
      json_string ? RegexCharacterClassDialect::kECMAScript : RegexCharacterClassDialect::kUnicode;

  // Mirror the error format of the RegexConverter path: a 1-based position in the pattern.
  auto error_at = [&](int pos, const std::string& message) {
    return "Regex parsing error at position " + std::to_string(pos + 1) + ": " + message;
  };

  std::vector<RegexStackEntry> stack;
  std::unordered_set<std::string> capture_names;
  int size = static_cast<int>(regex.size());
  for (int i = 0; i < size; i++) {
    char current_char = regex[i];
    if (current_char == '\\' && i + 1 < size && (regex[i + 1] == 'A' || regex[i + 1] == 'z')) {
      bool is_start = regex[i + 1] == 'A';
      bool at_branch_boundary =
          is_start ? (stack.empty() || std::holds_alternative<char>(stack.back().item))
                   : (i + 2 == size || regex[i + 2] == ')' || regex[i + 2] == '|');
      if (!at_branch_boundary) {
        if (byte_mode) {
          return ResultErr(error_at(
              i,
              is_start ? "start anchor is only allowed at the beginning of the pattern"
                       : "end anchor is only allowed at the end of the pattern"
          ));
        }
        XGRAMMAR_LOG(WARNING) << "Anchor '\\" << regex[i + 1]
                              << "' in the middle of regex is ignored: " << regex;
      }
      ++i;
      continue;
    }
    // Handle anchors.
    if (current_char == '^' || current_char == '$') {
      if (multiline) {
        return ResultErr(error_at(i, "line anchors in multiline mode are not supported"));
      }
      bool at_branch_boundary =
          current_char == '^' ? (stack.empty() || std::holds_alternative<char>(stack.back().item))
                              : (i + 1 == size || regex[i + 1] == ')' || regex[i + 1] == '|');
      if (!at_branch_boundary) {
        if (byte_mode) {
          return ResultErr(error_at(
              i,
              current_char == '^' ? "start anchor is only allowed at the beginning of the pattern"
                                  : "end anchor is only allowed at the end of the pattern"
          ));
        }
        XGRAMMAR_LOG(WARNING) << "Anchor '" << current_char
                              << "' in the middle of regex is ignored: " << regex;
      }
      continue;
    }
    // Handle the character class.
    if (current_char == '[') {
      size_t class_end = SkipCharacterClass(regex, i);
      if (class_end == std::string::npos) {
        return ResultErr(error_at(i, "Unclosed '['"));
      }
      size_t content_begin = i + 1;
      if (content_begin < regex.size() && regex[content_begin] == '^') {
        ++content_begin;
      }
      if (content_begin + 1 == class_end) {
        return ResultErr(error_at(i, "Empty character class is not allowed in regex"));
      }
      RegexIR::Leaf leaf;
      leaf.regex = regex.substr(i, class_end - i);
      leaf.case_insensitive = case_insensitive;
      leaf.byte_mode = !unicode;
      if (leaf.byte_mode) {
        auto validation = ParseCharacterClassLeaf(
            leaf.regex, case_insensitive, leaf.byte_mode, character_class_dialect
        );
        if (validation.IsErr()) {
          return ResultErr(error_at(i, std::move(validation).UnwrapErr().what()));
        }
      }
      stack.push_back({leaf, i, static_cast<int>(class_end), case_insensitive, multiline, unicode});
      i = static_cast<int>(class_end) - 1;
      continue;
    }
    if (current_char == ']') {
      return ResultErr(error_at(i, "Unmatched ']'"));
    }
    // Handle quantifiers.
    if (current_char == '+' || current_char == '*' || current_char == '?') {
      if (stack.empty() || std::holds_alternative<char>(stack.back().item)) {
        return ResultErr(
            error_at(i, std::string("There is nothing to repeat before '") + current_char + "'")
        );
      }
      RegexStackEntry atom = std::move(stack.back());
      stack.pop_back();
      RegexIR::Symbol symbol;
      symbol.state.push_back(std::move(std::get<RegexIR::State>(atom.item)));
      switch (current_char) {
        case '+': {
          symbol.symbol = RegexIR::RegexSymbol::plus;
          break;
        }
        case '*': {
          symbol.symbol = RegexIR::RegexSymbol::star;
          break;
        }
        case '?': {
          symbol.symbol = RegexIR::RegexSymbol::optional;
          break;
        }
      }
      // Skip the non-greedy modifier: greedy and non-greedy quantifiers accept the same
      // language, and the Earley parser explores all derivations anyway.
      if (i + 1 < size && regex[i + 1] == '?') {
        i++;
      }
      if (i + 1 < size && (regex[i + 1] == '*' || regex[i + 1] == '+' || regex[i + 1] == '?' ||
                           regex[i + 1] == '{')) {
        return ResultErr(error_at(i + 1, "Two consecutive repetition modifiers are not allowed"));
      }
      stack.push_back(
          {std::move(symbol),
           atom.span_begin,
           i + 1,
           atom.inherited_case_insensitive,
           atom.inherited_multiline,
           atom.inherited_unicode}
      );
      continue;
    }
    // Handle groups and alternation.
    if (current_char == '(') {
      if (i + 1 < size && regex[i + 1] == '?') {
        if (i + 2 >= size) {
          return ResultErr(error_at(i, "Group modifier is not finished"));
        }
        auto flag_spec = ParseInlineRegexFlagSpec(regex, i);
        if (flag_spec.has_value()) {
          bool parent_case_insensitive = case_insensitive;
          bool parent_multiline = multiline;
          bool parent_unicode = unicode;
          if (flag_spec->case_insensitive.has_value()) {
            case_insensitive = flag_spec->case_insensitive.value();
          }
          if (flag_spec->multiline.has_value()) {
            multiline = flag_spec->multiline.value();
          }
          if (flag_spec->unicode.has_value()) {
            unicode = flag_spec->unicode.value();
          }
          if (flag_spec->scoped) {
            stack.push_back(
                {'(',
                 i,
                 static_cast<int>(flag_spec->end),
                 parent_case_insensitive,
                 parent_multiline,
                 parent_unicode}
            );
          }
          i = static_cast<int>(flag_spec->end) - 1;
          continue;
        }
        char modifier = regex[i + 2];
        if (modifier == ':') {
          stack.push_back({'(', i, i + 1, case_insensitive, multiline, unicode});
          i += 2;
          continue;
        }
        if (modifier == '=' || modifier == '!') {
          if (byte_mode) {
            return ResultErr(error_at(i, "lookaround assertions are not supported"));
          }
          // Skip the whole lookahead group and treat it as the empty string.
          XGRAMMAR_LOG(WARNING) << "Lookahead assertion is not supported and is ignored in regex: "
                                << regex;
          int depth = 1;
          size_t j = i + 3;
          while (j < regex.size() && depth > 0) {
            if (regex[j] == '\\') {
              j += 2;
              continue;
            }
            if (regex[j] == '(') {
              auto capture_prefix_end = ParseNamedCapturePrefixEnd(regex, j);
              if (capture_prefix_end.has_value()) {
                ++depth;
                j = capture_prefix_end.value();
                continue;
              }
            }
            if (regex[j] == '[') {
              size_t class_begin = j;
              j = SkipCharacterClass(regex, j);
              if (j == std::string::npos) {
                return ResultErr(error_at(static_cast<int>(class_begin), "Unclosed '['"));
              }
              continue;
            }
            if (regex[j] == '(') {
              depth++;
            } else if (regex[j] == ')') {
              depth--;
            }
            j++;
          }
          if (depth != 0) {
            return ResultErr(error_at(i, "The parenthesis is not closed"));
          }
          stack.push_back(
              {RegexIR::Leaf{"", case_insensitive, !unicode},
               i,
               static_cast<int>(j),
               case_insensitive,
               multiline,
               unicode}
          );
          i = static_cast<int>(j) - 1;
          continue;
        }
        if (modifier == '<' || (modifier == 'P' && i + 3 < size && regex[i + 3] == '<')) {
          size_t name_begin = (modifier == '<') ? i + 3 : i + 4;
          if (name_begin < regex.size() && (regex[name_begin] == '=' || regex[name_begin] == '!')) {
            return ResultErr(error_at(
                i,
                byte_mode ? "lookaround assertions are not supported"
                          : "Lookbehind assertion is not supported in regex"
            ));
          }
          size_t j = name_begin;
          bool first_name_codepoint = true;
          while (j < regex.size() && regex[j] != '>') {
            auto [name_codepoint, name_bytes] = ParseNextUTF8(regex.c_str() + j);
            if (name_codepoint == CharHandlingError::kInvalidUTF8 ||
                j + name_bytes > regex.size() ||
                !IsRegexCaptureNameCodepoint(name_codepoint, first_name_codepoint)) {
              return ResultErr(error_at(i, "Invalid named capturing group"));
            }
            first_name_codepoint = false;
            j += name_bytes;
          }
          if (j == name_begin || j >= regex.size() || regex[j] != '>') {
            return ResultErr(error_at(i, "Invalid named capturing group"));
          }
          std::string capture_name = regex.substr(name_begin, j - name_begin);
          if (!capture_names.insert(capture_name).second) {
            return ResultErr(error_at(i, "Duplicate named capturing group: " + capture_name));
          }
          // Ignore the group name and compile the content as a normal group.
          stack.push_back({'(', i, i + 1, case_insensitive, multiline, unicode});
          i = static_cast<int>(j);
          continue;
        }
        return ResultErr(error_at(
            i,
            byte_mode ? "inline regular-expression flags are not supported"
                      : "Unsupported group modifier '(?" + std::string(1, modifier) + "'"
        ));
      }
      stack.push_back({'(', i, i + 1, case_insensitive, multiline, unicode});
      continue;
    }
    if (current_char == '|') {
      stack.push_back({'|', i, i + 1});
      continue;
    }
    if (current_char == ')') {
      std::vector<RegexStackEntry> popped;
      bool paired = false;
      bool unioned = false;
      int group_begin = 0;
      bool group_inherited_case_insensitive = false;
      bool group_inherited_multiline = false;
      bool group_inherited_unicode = !byte_mode;
      while (!stack.empty()) {
        RegexStackEntry entry = std::move(stack.back());
        stack.pop_back();
        if (std::holds_alternative<char>(entry.item)) {
          char marker = std::get<char>(entry.item);
          if (marker == '(') {
            paired = true;
            group_begin = entry.span_begin;
            group_inherited_case_insensitive = entry.inherited_case_insensitive;
            group_inherited_multiline = entry.inherited_multiline;
            group_inherited_unicode = entry.inherited_unicode;
            break;
          }
          XGRAMMAR_DCHECK(marker == '|');
          unioned = true;
        }
        popped.push_back(std::move(entry));
      }
      if (!paired) {
        return ResultErr(error_at(i, "Unmatched ')'"));
      }
      case_insensitive = group_inherited_case_insensitive;
      multiline = group_inherited_multiline;
      unicode = group_inherited_unicode;
      // `popped` stores the group content from right to left.
      if (!unioned) {
        RegexIR::Bracket bracket;
        for (auto it = popped.rbegin(); it != popped.rend(); ++it) {
          bracket.states.push_back(std::move(std::get<RegexIR::State>(it->item)));
        }
        // An empty bracket (e.g. "()") matches the empty string.
        stack.push_back(
            {std::move(bracket),
             group_begin,
             i + 1,
             group_inherited_case_insensitive,
             group_inherited_multiline,
             group_inherited_unicode}
        );
      } else {
        RegexIR::Union union_state;
        RegexIR::Bracket bracket;
        for (auto it = popped.rbegin(); it != popped.rend(); ++it) {
          if (std::holds_alternative<char>(it->item)) {
            XGRAMMAR_DCHECK(std::get<char>(it->item) == '|');
            // An empty bracket represents an empty alternative, e.g. "(a|)".
            union_state.states.push_back(std::move(bracket));
            bracket = RegexIR::Bracket();
            continue;
          }
          bracket.states.push_back(std::move(std::get<RegexIR::State>(it->item)));
        }
        union_state.states.push_back(std::move(bracket));
        stack.push_back(
            {std::move(union_state),
             group_begin,
             i + 1,
             group_inherited_case_insensitive,
             group_inherited_multiline,
             group_inherited_unicode}
        );
      }
      continue;
    }
    // Handle repetitions.
    if (current_char == '{') {
      if (stack.empty() || std::holds_alternative<char>(stack.back().item)) {
        return ResultErr(error_at(i, "There is nothing to repeat before the repetition"));
      }
      RegexStackEntry atom = std::move(stack.back());
      stack.pop_back();
      int repeat_begin = i;
      auto bounds_result = RegexIR::CheckRepeat(regex, i);
      if (bounds_result.IsErr()) {
        return ResultErr(error_at(repeat_begin, std::move(bounds_result).UnwrapErr().what()));
      }
      auto [lower_bound, upper_bound] = std::move(bounds_result).Unwrap();
      // Skip the non-greedy modifier.
      if (i + 1 < size && regex[i + 1] == '?') {
        i++;
      }
      if (i + 1 < size && (regex[i + 1] == '*' || regex[i + 1] == '+' || regex[i + 1] == '?' ||
                           regex[i + 1] == '{')) {
        return ResultErr(error_at(i + 1, "Two consecutive repetition modifiers are not allowed"));
      }
      const int repeat_threshold =
          json_string ? kLargeJSONStringRepeatThreshold : kLargeRepeatThreshold;
      bool is_large_repeat = upper_bound == RegexIR::kRepeatNoUpperBound
                                 ? lower_bound > repeat_threshold
                                 : upper_bound > repeat_threshold;
      if (is_large_repeat && builder != nullptr) {
        // Compile the repeated sub-pattern into a new rule (with a kRegex body), and represent
        // the repetition as a kRepeatRef FSM edge. The Earley parser executes it with a counter
        // at runtime, so the FSM is not unrolled.
        const auto& atom_state = std::get<RegexIR::State>(atom.item);
        std::string inner_regex = regex.substr(atom.span_begin, atom.span_end - atom.span_begin);
        if (atom.inherited_unicode != !byte_mode) {
          inner_regex = atom.inherited_unicode ? "(?u)" + inner_regex : "(?-u)" + inner_regex;
        }
        if (atom.inherited_case_insensitive) {
          inner_regex = "(?i)" + inner_regex;
        }
        if (RegexIR::IsNullable(atom_state)) {
          // Mirror RepetitionNormalizer: when the repeated element can match the empty string,
          // the repetition count cannot be enforced, so the lower bound is relaxed to 0.
          lower_bound = 0;
        }
        std::string name_hint = (rule_hint.empty() ? "regex" : rule_hint) + "_repeat";
        int32_t inner_rule_id = builder->AddRuleWithHint(
            name_hint, builder->AddRegex(inner_regex, json_string, /*byte_mode=*/byte_mode)
        );
        builder->UpdateLookaheadExact(inner_rule_id, true);
        if (upper_bound == RegexIR::kRepeatNoUpperBound) {
          // {n,} == {n}{0,}: a repeat edge for the mandatory part, then a starred rule
          // reference.
          RegexIR::Symbol star_symbol;
          star_symbol.symbol = RegexIR::RegexSymbol::star;
          star_symbol.state.push_back(RegexIR::RuleRefNode{inner_rule_id});
          if (lower_bound > 0) {
            RegexIR::Bracket bracket;
            bracket.states.push_back(RegexIR::RepeatSubrule{inner_rule_id, lower_bound, lower_bound}
            );
            bracket.states.push_back(std::move(star_symbol));
            stack.push_back(
                {std::move(bracket),
                 atom.span_begin,
                 i + 1,
                 atom.inherited_case_insensitive,
                 atom.inherited_multiline,
                 atom.inherited_unicode}
            );
          } else {
            stack.push_back(
                {std::move(star_symbol),
                 atom.span_begin,
                 i + 1,
                 atom.inherited_case_insensitive,
                 atom.inherited_multiline,
                 atom.inherited_unicode}
            );
          }
        } else {
          stack.push_back(
              {RegexIR::RepeatSubrule{inner_rule_id, lower_bound, upper_bound},
               atom.span_begin,
               i + 1,
               atom.inherited_case_insensitive,
               atom.inherited_multiline,
               atom.inherited_unicode}
          );
        }
      } else {
        RegexIR::Repeat repeat;
        repeat.lower_bound = lower_bound;
        repeat.upper_bound = upper_bound;
        repeat.states.push_back(std::move(std::get<RegexIR::State>(atom.item)));
        stack.push_back(
            {std::move(repeat),
             atom.span_begin,
             i + 1,
             atom.inherited_case_insensitive,
             atom.inherited_multiline,
             atom.inherited_unicode}
        );
      }
      continue;
    }
    // Handle literal characters and escapes. Each leaf holds exactly one codepoint or escape
    // sequence, so that a following quantifier applies to the whole character.
    RegexIR::Leaf leaf;
    if (current_char == '\\') {
      size_t escape_end = i;
      auto escape_result = ParseRegexEscape(
          regex, &escape_end, /*in_class=*/false, !unicode, character_class_dialect
      );
      if (escape_result.IsErr()) {
        return ResultErr(error_at(i, std::move(escape_result).UnwrapErr().what()));
      }
      leaf.regex = regex.substr(i, escape_end - i);
      leaf.case_insensitive = case_insensitive;
      leaf.byte_mode = !unicode;
      stack.push_back(
          {std::move(leaf), i, static_cast<int>(escape_end), case_insensitive, multiline, unicode}
      );
      i = static_cast<int>(escape_end) - 1;
      continue;
    }
    int num_bytes = 1;
    if (unicode) {
      auto [codepoint, parsed_bytes] = ParseNextUTF8(regex.c_str() + i);
      num_bytes = parsed_bytes;
      if (codepoint == CharHandlingError::kInvalidUTF8 || i + num_bytes > size) {
        num_bytes = 1;
      }
    }
    leaf.regex = regex.substr(i, num_bytes);
    leaf.case_insensitive = case_insensitive;
    leaf.byte_mode = !unicode;
    stack.push_back({std::move(leaf), i, i + num_bytes, case_insensitive, multiline, unicode});
    i += num_bytes - 1;
    continue;
  }

  // Assemble the top-level sequence / union. `stack` stores the content from left to right.
  std::vector<RegexIR::State> segment;
  std::vector<std::vector<RegexIR::State>> union_segments;
  bool unioned = false;
  for (auto& entry : stack) {
    if (std::holds_alternative<char>(entry.item)) {
      char marker = std::get<char>(entry.item);
      if (marker == '|') {
        union_segments.push_back(std::move(segment));
        segment.clear();
        unioned = true;
        continue;
      }
      return ResultErr(error_at(entry.span_begin, "The parenthesis is not closed"));
    }
    segment.push_back(std::move(std::get<RegexIR::State>(entry.item)));
  }
  if (!unioned) {
    ir.states = std::move(segment);
  } else {
    union_segments.push_back(std::move(segment));
    RegexIR::Union union_state;
    for (auto& branch : union_segments) {
      RegexIR::Bracket bracket;
      bracket.states = std::move(branch);
      union_state.states.push_back(std::move(bracket));
    }
    ir.states.push_back(std::move(union_state));
  }
  return ResultOk(std::move(ir));
}

}  // namespace

Result<FSMWithStartEnd> RegexFSMBuilder::Build(
    const std::string& regex, GrammarBuilder* builder, const std::string& rule_hint, bool byte_mode
) {
  auto ir_result = ParseRegexToIR(regex, builder, rule_hint, byte_mode);
  if (ir_result.IsErr()) {
    return ResultErr(std::move(ir_result).UnwrapErr());
  }
  return std::move(ir_result).Unwrap().Build();
}

Result<bool> RegexFSMBuilder::MatchesEmpty(const std::string& regex, bool byte_mode) {
  auto ir_result = ParseRegexToIR(regex, nullptr, "", byte_mode);
  if (ir_result.IsErr()) {
    return ResultErr(std::move(ir_result).UnwrapErr());
  }
  auto ir = std::move(ir_result).Unwrap();
  auto validation = ir.Validate();
  if (validation.IsErr()) {
    return validation;
  }
  return ResultOk(RegexIR::IsNullableSequence(ir.states));
}

Result<bool> RegexFSMBuilder::CanDeferLargeRepeat(
    const std::string& regex, bool json_string, bool byte_mode
) {
  if (json_string && byte_mode) {
    return ResultErr("json_string and byte_mode cannot be enabled together");
  }
  auto ir_result = ParseRegexToIR(regex, nullptr, "", byte_mode, json_string);
  if (ir_result.IsErr()) {
    return ResultErr(std::move(ir_result).UnwrapErr());
  }
  auto ir = std::move(ir_result).Unwrap();
  auto validation = ir.Validate();
  if (validation.IsErr()) {
    return validation;
  }
  if (!ir.HasLargeRepeat()) {
    return ResultOk(false);
  }

  // Confirm that the real GrammarBuilder-backed representation can also be assembled. This
  // keeps deferral specific to errors that the compact repeat path actually resolves.
  GrammarBuilder probe_builder;
  auto probe = json_string ? BuildForJSONString(regex, &probe_builder, "regex_probe")
                           : Build(regex, &probe_builder, "regex_probe", byte_mode);
  return ResultOk(probe.IsOk());
}

Result<FSMWithStartEnd> RegexFSMBuilder::BuildWithForbiddenChars(
    const std::string& regex,
    const std::bitset<256>& forbidden_chars,
    GrammarBuilder* builder,
    const std::string& rule_hint,
    bool byte_mode
) {
  auto build_result = Build(regex, builder, rule_hint, byte_mode);
  if (build_result.IsErr() || forbidden_chars.none()) {
    return build_result;
  }
  auto fsm_wse = std::move(build_result).Unwrap();
  const auto& fsm = fsm_wse.GetFsm();
  FSM new_fsm(fsm_wse.NumStates());
  for (int state = 0; state < fsm_wse.NumStates(); ++state) {
    for (const auto& edge : fsm.GetEdges(state)) {
      if (!edge.IsCharRange()) {
        new_fsm.AddEdge(state, edge.target, edge.min, edge.max);
        continue;
      }
      // Split the character range into the maximal sub-ranges of allowed characters.
      int range_start = -1;
      for (int c = edge.min; c <= edge.max + 1; ++c) {
        if (c <= edge.max && !forbidden_chars[c]) {
          if (range_start == -1) {
            range_start = c;
          }
        } else if (range_start != -1) {
          new_fsm.AddEdge(state, edge.target, range_start, c - 1);
          range_start = -1;
        }
      }
    }
  }
  return ResultOk(FSMWithStartEnd(new_fsm, fsm_wse.GetStart(), fsm_wse.GetEnds()));
}

Result<FSMWithStartEnd> RegexFSMBuilder::BuildForJSONString(
    const std::string& regex, GrammarBuilder* builder, const std::string& rule_hint
) {
  auto ir_result =
      ParseRegexToIR(regex, builder, rule_hint, /*byte_mode=*/false, /*json_string=*/true);
  if (ir_result.IsErr()) {
    return ResultErr(std::move(ir_result).UnwrapErr());
  }
  return std::move(ir_result).Unwrap().Build();
}

class TrieFSMBuilderImpl {
 public:
  TrieFSMBuilderImpl() = default;
  std::optional<FSMWithStartEnd> Build(
      const std::vector<std::string>& patterns,
      const std::vector<std::string>& excluded_patterns,
      std::vector<int32_t>* end_states,
      bool allow_overlap,
      bool add_back_edges
  );
  void AddBackEdges(FSM* fsm, int start, const std::unordered_set<int>& ends);
};

std::optional<FSMWithStartEnd> TrieFSMBuilderImpl::Build(
    const std::vector<std::string>& patterns,
    const std::vector<std::string>& excluded_patterns,
    std::vector<int32_t>* end_states,
    bool allow_overlap,
    bool add_back_edges
) {
  FSM fsm(1);
  int start = 0;
  std::unordered_set<int> ends;

  if (end_states) {
    end_states->clear();
  }

  for (const auto& pattern : patterns) {
    // Check for empty patterns
    if (!allow_overlap && pattern.empty()) {
      return std::nullopt;
    }

    int current_state = 0;
    for (const auto& ch : pattern) {
      int32_t ch_int32 = static_cast<int32_t>(static_cast<uint8_t>(ch));
      int next_state = fsm.GetNextState(current_state, ch_int32);
      if (next_state == FSM::kNoNextState) {
        next_state = fsm.AddState();
        fsm.AddEdge(current_state, next_state, ch_int32, ch_int32);
      }
      current_state = next_state;
      if (!allow_overlap && ends.count(current_state) > 0) {
        return std::nullopt;
      }
    }
    if (!allow_overlap && fsm.GetEdges(current_state).size() > 0) {
      return std::nullopt;
    }
    ends.insert(current_state);
    if (end_states) {
      end_states->push_back(current_state);
    }
  }

  std::unordered_set<int32_t> dead_state_set;

  if (add_back_edges) {
    // Build trie for excluded patterns.
    for (const auto& excluded_pattern : excluded_patterns) {
      if (!allow_overlap && excluded_pattern.empty()) {
        return std::nullopt;
      }

      int current_state = 0;
      for (const auto& ch : excluded_pattern) {
        int32_t ch_int32 = static_cast<int32_t>(static_cast<uint8_t>(ch));
        int next_state = fsm.GetNextState(current_state, ch_int32);
        if (next_state == FSM::kNoNextState) {
          next_state = fsm.AddState();
          fsm.AddEdge(current_state, next_state, ch_int32, ch_int32);
        }
        current_state = next_state;
        if (!allow_overlap && ends.count(current_state) > 0) {
          return std::nullopt;
        }
      }
      if (!allow_overlap && fsm.GetEdges(current_state).size() > 0) {
        return std::nullopt;
      }

      ends.insert(current_state);
      dead_state_set.insert(current_state);
    }

    // Add back edges.
    AddBackEdges(&fsm, start, ends);

    // Remove the edges to excluded end states.
    if (dead_state_set.size() != 0) {
      for (int state = 0; state < fsm.NumStates(); state++) {
        std::vector<FSMEdge>& edges = fsm.GetEdges(state);
        std::vector<FSMEdge> new_edges;
        for (const auto& edge : edges) {
          if (dead_state_set.count(edge.target) == 0) {
            new_edges.push_back(edge);
          }
        }
        edges = std::move(new_edges);
      }
    }
  } else if (excluded_patterns.size() > 0) {
    XGRAMMAR_LOG(WARNING) << "Excluded patterns are ignored when back edges are not added.";
  }

  return FSMWithStartEnd(fsm, start, std::vector<int32_t>(ends.begin(), ends.end()));
}

void TrieFSMBuilderImpl::AddBackEdges(FSM* fsm, int start, const std::unordered_set<int>& ends) {
  // Build an Aho-Corasick automaton by adding back edges.
  // When matching on the trie fails at state u on byte b, the matcher must resume from
  // the longest proper suffix of u's prefix that is still a path in the trie (the
  // failure state), and retry b from there. Falling back only to the start state (or to
  // the start state's direct children) loses matches whose start lies inside an
  // already-followed branch of another pattern. Example: patterns {"bcd", "abce"} on
  // input "abcd" -- after following "abc" of the "abce" branch, 'd' must transition to
  // the "bcd" end state via the failure state "bc", not back to the start state.

  int num_states = fsm->NumStates();

  // Step 1. Record the BFS order of the trie (a tree at this point), so that shallower
  // states are always processed first.
  std::vector<int> bfs_order;
  bfs_order.reserve(num_states);
  bfs_order.push_back(start);
  for (size_t head = 0; head < bfs_order.size(); head++) {
    for (const auto& edge : fsm->GetEdges(bfs_order[head])) {
      XGRAMMAR_DCHECK(edge.min == edge.max && edge.min >= 0 && edge.min <= 255);
      bfs_order.push_back(edge.target);
    }
  }
  XGRAMMAR_DCHECK(static_cast<int>(bfs_order.size()) == num_states);

  // Step 2. Compute the failure link and the fully resolved transition table with the
  // textbook O(num_states * 256) dynamic program: delta[u][b] is the trie child when it
  // exists, and delta[fail[u]][b] otherwise -- fail[u] is strictly shallower than u, so
  // its row is already final when u is processed in BFS order.
  std::vector<int> fail(num_states, start);
  std::vector<std::array<int, 256>> delta(num_states);
  for (auto& row : delta) {
    row.fill(FSM::kNoNextState);
  }
  for (int u = 0; u < num_states; u++) {
    for (const auto& edge : fsm->GetEdges(u)) {
      delta[u][edge.min] = edge.target;
    }
  }
  for (int u : bfs_order) {
    for (int byte = 0; byte < 256; byte++) {
      // Entries of deeper states are untouched so far, so a non-empty entry here is
      // exactly a trie child of u.
      int child = delta[u][byte];
      int fallback = (u == start) ? start : delta[fail[u]][byte];
      if (child == FSM::kNoNextState) {
        delta[u][byte] = fallback;
      } else {
        fail[child] = fallback;
      }
    }
  }

  // Step 3. Overwrite the edges of every non-end state with its resolved row,
  // compressing consecutive bytes with the same target into range edges.
  for (int u = 0; u < num_states; u++) {
    if (u != start && ends.count(u) > 0) {
      continue;
    }
    const auto& row = delta[u];
    std::vector<FSMEdge> new_edges;
    for (int byte = 0; byte < 256;) {
      int target = row[byte];
      int range_end = byte;
      while (range_end + 1 < 256 && row[range_end + 1] == target) {
        range_end++;
      }
      new_edges.push_back(FSMEdge(byte, range_end, target));
      byte = range_end + 1;
    }
    fsm->GetEdges(u) = std::move(new_edges);
  }
}

std::optional<FSMWithStartEnd> TrieFSMBuilder::Build(
    const std::vector<std::string>& patterns,
    const std::vector<std::string>& exclude_patterns,
    std::vector<int32_t>* end_states,
    bool allow_overlap,
    bool add_back_edges
) {
  return TrieFSMBuilderImpl().Build(
      patterns, exclude_patterns, end_states, allow_overlap, add_back_edges
  );
}

}  // namespace xgrammar
