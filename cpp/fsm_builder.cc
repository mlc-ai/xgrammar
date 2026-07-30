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
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "fsm.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

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

void AddSameLengthCharacterRange(FSM& fsm, int from, int to, uint32_t min, uint32_t max) {
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
      AddSameLengthCharacterRange(fsm, tmp_state_max, to, 0x0080, (max & 0x00FFFF));
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

// This function will add a range [min, max] of unicode characters to the FSM.
void AddCharacterRange(FSM& fsm, int from, int to, uint32_t min, uint32_t max) {
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

// The largest value of the character universe for each regex mode: raw byte values in byte
// mode, Unicode codepoints otherwise.
constexpr uint32_t kMaxByteValue = 0xFF;
constexpr uint32_t kMaxCodepoint = 0x10FFFF;

// One inclusive range of allowed characters: byte values in byte mode, codepoints otherwise.
struct CharRange {
  uint32_t lower;
  uint32_t upper;
};

// Sort the ranges and merge the overlapping or adjacent ones.
std::vector<CharRange> NormalizeRanges(std::vector<CharRange> ranges) {
  std::sort(ranges.begin(), ranges.end(), [](const CharRange& lhs, const CharRange& rhs) {
    return lhs.lower < rhs.lower || (lhs.lower == rhs.lower && lhs.upper < rhs.upper);
  });
  std::vector<CharRange> result;
  for (const auto& range : ranges) {
    if (!result.empty() && range.lower <= result.back().upper + 1) {
      result.back().upper = std::max(result.back().upper, range.upper);
    } else {
      result.push_back(range);
    }
  }
  return result;
}

// The complement of the ranges over the universe [0, universe_max].
std::vector<CharRange> ComplementRanges(std::vector<CharRange> ranges, uint32_t universe_max) {
  auto sorted = NormalizeRanges(std::move(ranges));
  std::vector<CharRange> result;
  uint32_t next = 0;
  for (const auto& range : sorted) {
    if (range.lower > next) {
      result.push_back({next, range.lower - 1});
    }
    if (range.upper >= universe_max) {
      return result;
    }
    next = range.upper + 1;
  }
  result.push_back({next, universe_max});
  return result;
}

// The result of parsing one escape sequence or one literal character: either a single
// character/byte, or a set of ranges (for class escapes like \d and \S).
struct RegexCharUnit {
  bool is_set = false;
  uint32_t value = 0;
  std::vector<CharRange> ranges;
};

RegexCharUnit MakeSingle(uint32_t value) {
  RegexCharUnit unit;
  unit.value = value;
  return unit;
}

RegexCharUnit MakeSet(std::vector<CharRange> ranges, bool negated, uint32_t universe_max) {
  RegexCharUnit unit;
  unit.is_set = true;
  unit.ranges = negated ? ComplementRanges(std::move(ranges), universe_max)
                        : NormalizeRanges(std::move(ranges));
  return unit;
}

// ASCII definitions of the shorthand classes.
const std::vector<CharRange> kDigitRanges = {{'0', '9'}};
const std::vector<CharRange> kWordRanges = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}};
// Byte mode uses the ASCII definition [\t-\r ] of whitespace.
const std::vector<CharRange> kByteSpaceRanges = {{'\t', '\r'}, {' ', ' '}};
// Codepoint mode follows RegexToEBNF: [\f\n\r\t\v\u0020\u00a0].
const std::vector<CharRange> kCodepointSpaceRanges = {{'\t', '\r'}, {' ', ' '}, {0xA0, 0xA0}};

// Parse the escape sequence at regex[*pos] (regex[*pos] must be '\\') in byte mode and advance
// *pos past it. The supported dialect and the error messages follow the byte-oriented regexes
// of the Lark allow_invalid_utf8 option.
Result<RegexCharUnit> ParseByteRegexEscape(const std::string& regex, size_t* pos, bool in_class) {
  if (*pos + 1 >= regex.size()) {
    return ResultErr("unfinished byte escape");
  }
  char escaped = regex[*pos + 1];
  *pos += 2;
  switch (escaped) {
    case 'd':
      return ResultOk(MakeSet(kDigitRanges, false, kMaxByteValue));
    case 'D':
      return ResultOk(MakeSet(kDigitRanges, true, kMaxByteValue));
    case 'w':
      return ResultOk(MakeSet(kWordRanges, false, kMaxByteValue));
    case 'W':
      return ResultOk(MakeSet(kWordRanges, true, kMaxByteValue));
    case 's':
      return ResultOk(MakeSet(kByteSpaceRanges, false, kMaxByteValue));
    case 'S':
      return ResultOk(MakeSet(kByteSpaceRanges, true, kMaxByteValue));
    case 'x': {
      if (*pos < regex.size() && regex[*pos] == '{') {
        return ResultErr("Unicode character escapes are not available in byte regular expressions");
      }
      if (*pos + 2 > regex.size() || HexCharToInt(regex[*pos]) < 0 ||
          HexCharToInt(regex[*pos + 1]) < 0) {
        return ResultErr("\\x escape must contain exactly two hexadecimal digits");
      }
      uint32_t value =
          static_cast<uint32_t>(HexCharToInt(regex[*pos]) * 16 + HexCharToInt(regex[*pos + 1]));
      *pos += 2;
      return ResultOk(MakeSingle(value));
    }
    case 'p':
    case 'P':
      return ResultErr("Unicode character classes are not available in byte regular expressions");
    case 'u':
    case 'U':
      return ResultErr("Unicode character escapes are not available in byte regular expressions");
    case 'b':
    case 'B':
      if (!in_class) {
        return ResultErr("word-boundary assertions are not supported");
      }
      return ResultOk(MakeSingle('\b'));
    case 'a':
      return ResultOk(MakeSingle('\a'));
    case 'f':
      return ResultOk(MakeSingle('\f'));
    case 'n':
      return ResultOk(MakeSingle('\n'));
    case 'r':
      return ResultOk(MakeSingle('\r'));
    case 't':
      return ResultOk(MakeSingle('\t'));
    case 'v':
      return ResultOk(MakeSingle('\v'));
    case '0':
      return ResultOk(MakeSingle('\0'));
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
      return ResultOk(MakeSingle(static_cast<uint8_t>(escaped)));
    default:
      if ((escaped >= '1' && escaped <= '9') || escaped == 'k') {
        return ResultErr("backreferences are not supported");
      }
      return ResultErr(std::string("unrecognized byte escape '\\") + escaped + "'");
  }
}

// Parse the escape sequence at regex[*pos] (regex[*pos] must be '\\') in codepoint mode and
// advance *pos past it. The supported dialect follows RegexToEBNF.
Result<RegexCharUnit> ParseCodepointRegexEscape(
    const std::string& regex, size_t* pos, bool in_class
) {
  // The same punctuation escapes as RegexToEBNF.
  static const std::unordered_map<char, TCodepoint> kCustomEscapeMap = {
      // clang-format off
      {'^', '^'}, {'$', '$'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'}, {'\\', '\\'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}, {'-', '-'}  // clang-format on
  };
  if (*pos + 1 >= regex.size()) {
    return ResultErr("Escape sequence is not finished.");
  }
  char escaped = regex[*pos + 1];
  switch (escaped) {
    case 'd':
      *pos += 2;
      return ResultOk(MakeSet(kDigitRanges, false, kMaxCodepoint));
    case 'D':
      *pos += 2;
      return ResultOk(MakeSet(kDigitRanges, true, kMaxCodepoint));
    case 'w':
      *pos += 2;
      return ResultOk(MakeSet(kWordRanges, false, kMaxCodepoint));
    case 'W':
      *pos += 2;
      return ResultOk(MakeSet(kWordRanges, true, kMaxCodepoint));
    case 's':
      *pos += 2;
      return ResultOk(MakeSet(kCodepointSpaceRanges, false, kMaxCodepoint));
    case 'S':
      *pos += 2;
      return ResultOk(MakeSet(kCodepointSpaceRanges, true, kMaxCodepoint));
    case 'p':
    case 'P':
      return ResultErr("Unicode character class escape sequence is not supported yet.");
    default:
      break;
  }
  if ((escaped >= '1' && escaped <= '9') || escaped == 'k') {
    return ResultErr("Backreference is not supported yet.");
  }
  if (!in_class && (escaped == 'b' || escaped == 'B')) {
    return ResultErr("Word boundary is not supported yet.");
  }
  if (escaped == 'u' && *pos + 2 < regex.size() && regex[*pos + 2] == '{') {
    size_t hex_start = *pos + 3;
    size_t hex_end = hex_start;
    TCodepoint value = 0;
    while (hex_end < regex.size() && HexCharToInt(regex[hex_end]) != -1 && hex_end - hex_start < 7
    ) {
      value = value * 16 + HexCharToInt(regex[hex_end]);
      ++hex_end;
    }
    if (hex_end == hex_start || hex_end - hex_start > 6 || hex_end >= regex.size() ||
        regex[hex_end] != '}' || value > static_cast<TCodepoint>(kMaxCodepoint)) {
      return ResultErr("Invalid Unicode escape sequence.");
    }
    *pos = hex_end + 1;
    return ResultOk(MakeSingle(static_cast<uint32_t>(value)));
  }
  if (escaped == 'c') {
    if (*pos + 2 >= regex.size() || !std::isalpha(static_cast<unsigned char>(regex[*pos + 2]))) {
      return ResultErr("Invalid control character escape sequence.");
    }
    uint32_t value = static_cast<unsigned char>(regex[*pos + 2]) % 32;
    *pos += 3;
    return ResultOk(MakeSingle(value));
  }
  if (static_cast<unsigned char>(escaped) >= 0x80) {
    // An escaped non-ASCII character matches the character itself.
    auto [codepoint, consumed] = ParseNextUTF8(regex.c_str() + *pos + 1);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      return ResultErr("the regex pattern is not a valid UTF-8 string");
    }
    *pos += 1 + consumed;
    return ResultOk(MakeSingle(static_cast<uint32_t>(codepoint)));
  }
  auto [codepoint, consumed] = ParseNextEscaped(regex.c_str() + *pos, kCustomEscapeMap);
  if (codepoint != CharHandlingError::kInvalidEscape) {
    if (codepoint < 0 || codepoint > static_cast<TCodepoint>(kMaxCodepoint)) {
      return ResultErr("Invalid escaped codepoint.");
    }
    *pos += consumed;
    return ResultOk(MakeSingle(static_cast<uint32_t>(codepoint)));
  }
  XGRAMMAR_LOG(WARNING) << "Escape sequence '\\" << escaped
                        << "' is not recognized. The character itself will be matched";
  *pos += 2;
  return ResultOk(MakeSingle(static_cast<unsigned char>(escaped)));
}

Result<RegexCharUnit> ParseRegexEscape(
    const std::string& regex, size_t* pos, bool byte_mode, bool in_class
) {
  return byte_mode ? ParseByteRegexEscape(regex, pos, in_class)
                   : ParseCodepointRegexEscape(regex, pos, in_class);
}

// Parse one literal character at regex[*pos] and advance *pos: a single byte in byte mode, or a
// full UTF-8 character otherwise.
Result<RegexCharUnit> ParseRegexLiteralChar(
    const std::string& regex, size_t* pos, bool byte_mode, bool in_class
) {
  unsigned char byte = static_cast<unsigned char>(regex[*pos]);
  if (byte_mode) {
    if (in_class && byte >= 0x80) {
      return ResultErr("non-ASCII characters are not available in byte character classes; use \\xHH"
      );
    }
    ++*pos;
    return ResultOk(MakeSingle(byte));
  }
  auto [codepoint, consumed] = ParseNextUTF8(regex.c_str() + *pos);
  if (codepoint == CharHandlingError::kInvalidUTF8) {
    return ResultErr("the regex pattern is not a valid UTF-8 string");
  }
  *pos += consumed;
  return ResultOk(MakeSingle(static_cast<uint32_t>(codepoint)));
}

class RegexIR {
 public:
  struct Leaf;

  struct Symbol;

  struct Union;

  struct Bracket;

  struct Repeat;

  static constexpr int kRepeatNoUpperBound = -1;

  using State = std::variant<Leaf, Symbol, Union, Bracket, Repeat>;

  // A leaf atom: either an exact byte sequence, or a single character constrained to a set of
  // ranges (a character class, '.', or a class escape like \d).
  struct Leaf {
    bool is_char_set = false;
    std::string literal;
    std::vector<CharRange> ranges;
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

  // The top-level concatenation of the regex.
  std::vector<State> states;

  // Whether the regex is matched over raw bytes instead of Unicode characters.
  bool byte_mode = false;

  /*!
    \brief Constructs a NFA from the regex IR.
  */
  Result<FSMWithStartEnd> Build() const;

  /*!
    \brief the visit function for the variant.
  */
  Result<FSMWithStartEnd> visit(const Leaf& state) const;

  Result<FSMWithStartEnd> visit(const Symbol& state) const;

  Result<FSMWithStartEnd> visit(const Union& state) const;

  Result<FSMWithStartEnd> visit(const Bracket& state) const;

  Result<FSMWithStartEnd> visit(const Repeat& state) const;

  /*!
   * \brief Check repeat in regex. i.e {...} and {...,...}
   * \param regex The regex string.
   * \param start The start position of the repeat. i.e. regex[start] == '{'.
   * After the function, start will be the position of '}'.
   * \return The repeat range.
   */
  static Result<std::pair<int, int>> CheckRepeat(const std::string& regex, int& start);
};

Result<std::pair<int, int>> RegexIR::CheckRepeat(const std::string& regex, int& start) {
  if (regex[start] != '{') {
    return ResultErr("expected '{' at the start of a repetition range");
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
    return ResultErr("expected a decimal repetition count");
  }
  lower_bound = std::stoi(num_str);
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  // The format is {n}
  if (regex[start] == '}') {
    upper_bound = lower_bound;
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  if (regex[start] != ',') {
    return ResultErr("expected ',' or '}' in repetition range");
  }
  XGRAMMAR_DCHECK(regex[start] == ',');
  start++;
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  // The format is {n,}
  if (regex[start] == '}') {
    return ResultOk(std::make_pair(lower_bound, upper_bound));
  }
  num_str.clear();
  while (static_cast<size_t>(start) < regex.size() && std::isdigit(regex[start])) {
    num_str += regex[start];
    start++;
  }
  if (num_str.empty()) {
    return ResultErr("expected a decimal repetition count");
  }
  upper_bound = std::stoi(num_str);
  if (upper_bound < lower_bound) {
    return ResultErr("repetition upper bound is smaller than its lower bound");
  }
  while (static_cast<size_t>(start) < regex.size() && regex[start] == ' ') {
    start++;
  }
  if (regex[start] != '}') {
    return ResultErr("expected '}' after repetition range");
  }
  XGRAMMAR_DCHECK(regex[start] == '}');
  return ResultOk(std::make_pair(lower_bound, upper_bound));
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
  if (!state.is_char_set) {
    // An exact byte sequence is lowered to a chain of single-byte edges.
    FSM fsm(1);
    int current = 0;
    for (char c : state.literal) {
      int next = fsm.AddState();
      fsm.AddEdge(current, next, static_cast<uint8_t>(c), static_cast<uint8_t>(c));
      current = next;
    }
    return ResultOk(FSMWithStartEnd(fsm, 0, {current}, false));
  }
  // A character set is lowered range by range. In byte mode every range is one byte edge; in
  // codepoint mode every range is expanded to UTF-8 byte edges, possibly through intermediate
  // states. An empty set (e.g. the complement of the full universe) yields an FSM that never
  // matches.
  FSM fsm(2);
  for (const auto& range : state.ranges) {
    if (byte_mode) {
      fsm.AddEdge(0, 1, static_cast<int32_t>(range.lower), static_cast<int32_t>(range.upper));
    } else {
      AddCharacterRange(
          fsm, 0, 1, CodepointToPackedUTF8(range.lower), CodepointToPackedUTF8(range.upper)
      );
    }
  }
  return ResultOk(FSMWithStartEnd(fsm, 0, {1}, false));
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
    return ResultErr("Invalid union");
  }
  return ResultOk(FSMWithStartEnd::Union(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Symbol& state) const {
  if (state.state.size() != 1) {
    return ResultErr("Invalid symbol");
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
  std::vector<FSMWithStartEnd> fsm_list;
  for (const auto& child : state.states) {
    auto visited = std::visit([&](auto&& arg) { return RegexIR::visit(arg); }, child);
    if (visited.IsErr()) {
      return visited;
    }
    fsm_list.push_back(std::move(visited).Unwrap());
  }
  if (fsm_list.empty()) {
    // An empty alternative like the branches of (|a|) matches exactly the empty string.
    FSM fsm(1);
    return ResultOk(FSMWithStartEnd(fsm, 0, {0}, false));
  }
  return ResultOk(FSMWithStartEnd::Concat(fsm_list));
}

Result<FSMWithStartEnd> RegexIR::visit(const RegexIR::Repeat& state) const {
  if (state.states.size() != 1) {
    return ResultErr("Invalid repeat");
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

// Parse a character class starting at regex[*pos] (which must be '[') and advance *pos past the
// closing ']'. The class is parsed into ranges; a negated class is complemented over the
// universe of the mode before it is returned.
Result<RegexIR::Leaf> ParseRegexCharacterClass(
    const std::string& regex, size_t* pos, bool byte_mode
) {
  XGRAMMAR_DCHECK(regex[*pos] == '[');
  ++*pos;
  bool negated = false;
  if (*pos < regex.size() && regex[*pos] == '^') {
    negated = true;
    ++*pos;
  }
  std::vector<CharRange> ranges;
  bool has_item = false;

  auto parse_unit = [&]() -> Result<RegexCharUnit> {
    if (regex[*pos] == '\\') {
      return ParseRegexEscape(regex, pos, byte_mode, /*in_class=*/true);
    }
    return ParseRegexLiteralChar(regex, pos, byte_mode, /*in_class=*/true);
  };

  while (*pos < regex.size() && regex[*pos] != ']') {
    if (byte_mode) {
      char current = regex[*pos];
      char next = *pos + 1 < regex.size() ? regex[*pos + 1] : '\0';
      if ((current == '&' && next == '&') || (current == '-' && next == '-') ||
          (current == '~' && next == '~') || current == '[') {
        return ResultErr("byte character-class set operations are not supported");
      }
    }
    auto lower_result = parse_unit();
    if (lower_result.IsErr()) {
      return ResultErr(std::move(lower_result).UnwrapErr());
    }
    auto lower = std::move(lower_result).Unwrap();
    has_item = true;
    if (!lower.is_set && *pos + 1 < regex.size() && regex[*pos] == '-' && regex[*pos + 1] != ']') {
      ++*pos;
      auto upper_result = parse_unit();
      if (upper_result.IsErr()) {
        return ResultErr(std::move(upper_result).UnwrapErr());
      }
      auto upper = std::move(upper_result).Unwrap();
      if (upper.is_set) {
        return ResultErr(
            byte_mode ? "character-class range endpoint must be a single byte"
                      : "character-class range endpoint must be a single character"
        );
      }
      if (lower.value > upper.value) {
        return ResultErr("character-class range start exceeds its end");
      }
      ranges.push_back({lower.value, upper.value});
    } else if (lower.is_set) {
      ranges.insert(ranges.end(), lower.ranges.begin(), lower.ranges.end());
    } else {
      ranges.push_back({lower.value, lower.value});
    }
  }
  if (*pos >= regex.size()) {
    return ResultErr("unclosed '['");
  }
  ++*pos;  // Consume ']'.
  if (!has_item) {
    return ResultErr("empty character class is not allowed");
  }
  uint32_t universe_max = byte_mode ? kMaxByteValue : kMaxCodepoint;
  RegexIR::Leaf leaf;
  leaf.is_char_set = true;
  leaf.ranges = negated ? ComplementRanges(std::move(ranges), universe_max)
                        : NormalizeRanges(std::move(ranges));
  return ResultOk(std::move(leaf));
}

}  // namespace

Result<FSMWithStartEnd> RegexFSMBuilder::Build(const std::string& regex, bool byte_mode) {
  RegexIR ir;
  ir.byte_mode = byte_mode;
  using IRState = std::variant<RegexIR::State, char>;
  // We use a stack to store the states.
  std::stack<IRState> stack;

  auto push_state = [&](RegexIR::State state) { stack.push(std::move(state)); };

  size_t i = 0;
  while (i < regex.size()) {
    char c = regex[i];
    // Anchors are only recognized at the pattern boundaries and match the empty string there.
    if (c == '^') {
      if (i == 0) {
        ++i;
        continue;
      }
      if (byte_mode) {
        return ResultErr("start anchor is only allowed at the beginning of the pattern");
      }
      RegexIR::Leaf leaf;
      leaf.literal = "^";
      push_state(std::move(leaf));
      ++i;
      continue;
    }
    if (c == '$') {
      if (i + 1 == regex.size()) {
        ++i;
        continue;
      }
      if (byte_mode) {
        return ResultErr("end anchor is only allowed at the end of the pattern");
      }
      RegexIR::Leaf leaf;
      leaf.literal = "$";
      push_state(std::move(leaf));
      ++i;
      continue;
    }
    if (c == '[') {
      auto leaf_result = ParseRegexCharacterClass(regex, &i, byte_mode);
      if (leaf_result.IsErr()) {
        return ResultErr(std::move(leaf_result).UnwrapErr());
      }
      push_state(std::move(leaf_result).Unwrap());
      continue;
    }
    if (c == '+' || c == '*' || c == '?') {
      if (stack.empty() || std::holds_alternative<char>(stack.top())) {
        return ResultErr("repetition modifier has no preceding atom");
      }
      auto state = stack.top();
      stack.pop();
      auto child = std::get<RegexIR::State>(state);
      RegexIR::Symbol symbol;
      symbol.state.push_back(child);
      switch (c) {
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
      stack.push(symbol);
      ++i;
      // Greedy and lazy repetitions have the same language, so a trailing '?' is skipped.
      if (i < regex.size() && regex[i] == '?') {
        ++i;
      }
      if (i < regex.size() &&
          (regex[i] == '*' || regex[i] == '+' || regex[i] == '?' || regex[i] == '{')) {
        return ResultErr("two consecutive repetition modifiers are not allowed");
      }
      continue;
    }
    if (c == '|') {
      stack.push('|');
      ++i;
      continue;
    }
    if (c == '(') {
      stack.push('(');
      ++i;
      if (i >= regex.size() || regex[i] != '?') {
        continue;
      }
      // Group modifiers: regex[i] == '?'.
      if (i + 1 < regex.size() && regex[i + 1] == ':') {
        // Non-capturing group.
        i += 2;
        continue;
      }
      // Named capture groups (?<name>...) and (?P<name>...) are treated as plain groups.
      size_t name_pos = 0;
      if (i + 1 < regex.size() && regex[i + 1] == '<' &&
          (i + 2 >= regex.size() || (regex[i + 2] != '=' && regex[i + 2] != '!'))) {
        name_pos = i + 2;
      } else if (i + 2 < regex.size() && regex[i + 1] == 'P' && regex[i + 2] == '<') {
        name_pos = i + 3;
      }
      if (name_pos != 0) {
        size_t name_end = name_pos;
        while (name_end < regex.size() &&
               (std::isalnum(static_cast<unsigned char>(regex[name_end])) || regex[name_end] == '_')
        ) {
          ++name_end;
        }
        if (name_end == name_pos || name_end >= regex.size() || regex[name_end] != '>') {
          return ResultErr("invalid capture group name");
        }
        i = name_end + 1;
        continue;
      }
      if (byte_mode) {
        bool is_lookaround =
            (i + 1 < regex.size() && (regex[i + 1] == '=' || regex[i + 1] == '!')) ||
            (i + 2 < regex.size() && regex[i + 1] == '<' &&
             (regex[i + 2] == '=' || regex[i + 2] == '!'));
        if (is_lookaround) {
          return ResultErr("lookaround assertions are not supported");
        }
        return ResultErr("inline regular-expression flags are not supported");
      }
      if (i + 1 < regex.size() && (regex[i + 1] == '=' || regex[i + 1] == '!')) {
        i += 2;
        // TODO(Linzhang Li): Handling the lookahead. Currently treated as a plain group.
        continue;
      }
      continue;
    }
    if (c == ')') {
      std::stack<IRState> states;
      bool paired = false;
      bool unioned = false;
      while ((!stack.empty()) && (!paired)) {
        auto state = stack.top();
        stack.pop();
        if (std::holds_alternative<char>(state)) {
          char top_char = std::get<char>(state);
          if (top_char == '(') {
            paired = true;
            break;
          }
          if (top_char == '|') {
            unioned = true;
          }
          states.push(state);
        } else {
          states.push(state);
        }
      }
      if (!paired) {
        return ResultErr("unmatched ')'");
      }
      ++i;
      if (states.empty()) {
        continue;
      }
      if (!unioned) {
        RegexIR::Bracket bracket;
        while (!states.empty()) {
          auto state = states.top();
          states.pop();
          auto child = std::get<RegexIR::State>(state);
          bracket.states.push_back(child);
        }
        stack.push(bracket);
      } else {
        RegexIR::Union union_state;
        RegexIR::Bracket bracket;
        while (!states.empty()) {
          auto state = states.top();
          states.pop();
          if (std::holds_alternative<char>(state)) {
            char top_char = std::get<char>(state);
            if (top_char == '|') {
              union_state.states.push_back(bracket);
              bracket.states.clear();
              continue;
            }
            return ResultErr("unmatched ')'");
          }
          auto child = std::get<RegexIR::State>(state);
          bracket.states.push_back(child);
        }
        union_state.states.push_back(bracket);
        stack.push(union_state);
      }
      continue;
    }
    if (c == '{') {
      if (stack.empty() || std::holds_alternative<char>(stack.top())) {
        return ResultErr("repetition modifier has no preceding atom");
      }
      auto state = stack.top();
      stack.pop();
      int repeat_pos = static_cast<int>(i);
      auto bounds_result = RegexIR::CheckRepeat(regex, repeat_pos);
      if (bounds_result.IsErr()) {
        return ResultErr(std::move(bounds_result).UnwrapErr());
      }
      auto bounds = std::move(bounds_result).Unwrap();
      auto child = std::get<RegexIR::State>(state);
      RegexIR::Repeat repeat;
      repeat.lower_bound = bounds.first;
      repeat.upper_bound = bounds.second;
      repeat.states.push_back(child);
      stack.push(repeat);
      i = static_cast<size_t>(repeat_pos) + 1;
      // Greedy and lazy repetitions have the same language, so a trailing '?' is skipped.
      if (i < regex.size() && regex[i] == '?') {
        ++i;
      }
      if (i < regex.size() &&
          (regex[i] == '*' || regex[i] == '+' || regex[i] == '?' || regex[i] == '{')) {
        return ResultErr("two consecutive repetition modifiers are not allowed");
      }
      continue;
    }
    if (c == '\\') {
      auto unit_result = ParseRegexEscape(regex, &i, byte_mode, /*in_class=*/false);
      if (unit_result.IsErr()) {
        return ResultErr(std::move(unit_result).UnwrapErr());
      }
      auto unit = std::move(unit_result).Unwrap();
      RegexIR::Leaf leaf;
      if (unit.is_set) {
        leaf.is_char_set = true;
        leaf.ranges = std::move(unit.ranges);
      } else {
        leaf.literal =
            byte_mode ? std::string(1, static_cast<char>(unit.value)) : CharToUTF8(unit.value);
      }
      push_state(std::move(leaf));
      continue;
    }
    if (c == '.') {
      // '.' matches any single byte in byte mode and any single character otherwise. The Lark
      // converter rewrites '.' to [^\n] beforehand unless the dot-all flag is set.
      RegexIR::Leaf leaf;
      leaf.is_char_set = true;
      leaf.ranges = {{0, byte_mode ? kMaxByteValue : kMaxCodepoint}};
      push_state(std::move(leaf));
      ++i;
      continue;
    }
    // A literal character: one byte in byte mode, one full UTF-8 character otherwise.
    RegexIR::Leaf leaf;
    if (byte_mode) {
      leaf.literal = std::string(1, c);
      ++i;
    } else {
      auto [codepoint, consumed] = ParseNextUTF8(regex.c_str() + i);
      if (codepoint == CharHandlingError::kInvalidUTF8) {
        return ResultErr("the regex pattern is not a valid UTF-8 string");
      }
      leaf.literal = regex.substr(i, consumed);
      i += consumed;
    }
    push_state(std::move(leaf));
    continue;
  }
  std::vector<RegexIR::State> res_states;
  std::vector<decltype(res_states)> union_state_list;
  bool unioned = false;
  while (!stack.empty()) {
    if (std::holds_alternative<char>(stack.top())) {
      char top_char = std::get<char>(stack.top());
      if (top_char == '|') {
        union_state_list.push_back(res_states);
        res_states.clear();
        unioned = true;
        stack.pop();
        continue;
      }
      return ResultErr("unclosed '('");
    }
    auto state = stack.top();
    stack.pop();
    auto child = std::get<RegexIR::State>(state);
    res_states.push_back(std::move(child));
  }
  if (!unioned) {
    for (auto it = res_states.rbegin(); it != res_states.rend(); ++it) {
      ir.states.push_back(std::move(*it));
    }
  } else {
    union_state_list.push_back(res_states);
    RegexIR::Union union_state;
    for (auto it = union_state_list.begin(); it != union_state_list.end(); ++it) {
      RegexIR::Bracket bracket;
      for (auto state = it->rbegin(); state != it->rend(); ++state) {
        bracket.states.push_back(std::move(*state));
      }
      union_state.states.push_back(std::move(bracket));
    }
    ir.states.push_back(std::move(union_state));
  }
  return ir.Build();
}

Result<FSMWithStartEnd> RegexFSMBuilder::BuildWithForbiddenChars(
    const std::string& regex, const std::bitset<256>& forbidden_chars
) {
  auto build_result = Build(regex);
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
