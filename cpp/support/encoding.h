/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/support/encoding.h
 * \brief Encoding and decoding from/to UTF-8 and escape sequence to/from codepoints.
 */
#ifndef XGRAMMAR_SUPPORT_ENCODING_H_
#define XGRAMMAR_SUPPORT_ENCODING_H_
// TODO(yixin): enhance performance

#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace xgrammar {

/*! \brief Represents a unicode codepoint. */
using TCodepoint = int32_t;

/*!
 * \brief Handle the utf-8 first byte.
 * \returns (is_valid, total_number_of_bytes, initial_codepoint).
 */
std::tuple<bool, int, TCodepoint> HandleUTF8FirstByte(uint8_t byte);

/*!
 * \brief Print a codepoint to a UTF-8 string.
 * \param codepoint The codepoint.
 * \return The UTF-8 string.
 */
std::string PrintAsUTF8(TCodepoint codepoint);

/*!
 * \brief Print a codepoint to a escaped string. If the codepoint is not printable, it will be
 * escaped. By default the function support escape sequences in C ("\n", "\t", "\u0123"). User can
 * specify more escape sequences using additional_escape_map.
 * \param codepoint The codepoint.
 * \param additional_escape_map A map from codepoint to escape sequence. If the codepoint is in the
 * map, it will be escaped using the corresponding escape sequence. e.g. {{'-', "\\-"}}. \return The
 * printable string.
 */
std::string PrintAsEscapedUTF8(
    TCodepoint codepoint,
    const std::unordered_map<TCodepoint, std::string>& additional_escape_map = {}
);

/*!
 * \brief Print the given char to a escaped string that can be printed.
 * \return The escaped string.
 */
std::string PrintAsEscapedUTF8(uint8_t raw_char);

/*!
 * \brief Print the given string to a escaped string that can be printed.
 * \return The escaped string.
 */
std::string PrintAsEscapedUTF8(std::string raw_str);

inline int HexCharToInt(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  } else if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  } else if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  } else {
    return -1;
  }
}

/*!
 * \brief Represents an error when handling characters. Will be returned as a special TCodepoint
 * value.
 */
enum CharHandlingError : TCodepoint {
  /*! \brief The UTF-8 string is invalid. */
  kInvalidUTF8 = -10,
  /*! \brief The escape sequence is invalid. */
  kInvalidEscape = -11,
};

/*!
 * \brief Parse all codepoints in a UTF-8 string.
 * \param utf8 The UTF-8 string.
 * \return All codepoints. If the UTF-8 string is invalid, and the error policy is
 * kReturnInvalid, the function returns {CharHandlingError::kInvalidUTF8}.
 */
std::vector<TCodepoint> ParseUTF8(const char* utf8, bool return_byte_on_error = false);

template <typename CharType>
std::pair<TCodepoint, int32_t> HandleEscape(
    const CharType* data, const std::unordered_map<char, TCodepoint>& additional_escape_map = {}
);

/*!
 * \brief Parse the first codepoint from a UTF-8 string. Also checks escape sequences and converts
 * the escaped char to its original value.
 * \param utf8 The UTF-8 string or the escape sequence.
 * \param additional_escape_map A map from escape sequence to codepoint. If the escape sequence is
 * in the map, it will be converted to the corresponding codepoint. e.g. {{"\\-", '-'}}.
 * \return The codepoint and the new pointer. If the UTF-8 string or the escape sequence is
 * invalid, and the error policy is kReturnInvalid, the function returns
 * (CharHandlingError::kInvalidUTF8, input char pointer).
 */
std::pair<TCodepoint, int32_t> ParseNextUTF8OrEscaped(
    const char* utf8, const std::unordered_map<char, TCodepoint>& additional_escape_map = {}
);

// Template implementation
template <typename CharType>
std::pair<TCodepoint, int32_t> HandleEscape(
    const CharType* data, const std::unordered_map<char, TCodepoint>& additional_escape_map
) {
  static const std::unordered_map<char, TCodepoint> kEscapeToCodepoint = {
      {'\'', '\''},
      {'\"', '\"'},
      {'\?', '\?'},
      {'\\', '\\'},
      {'a', '\a'},
      {'b', '\b'},
      {'f', '\f'},
      {'n', '\n'},
      {'r', '\r'},
      {'t', '\t'},
      {'v', '\v'},
      {'0', '\0'},
      {'e', '\x1B'}
  };
  if (data[0] != '\\') {
    return {CharHandlingError::kInvalidEscape, 0};
  }
  if (auto it = additional_escape_map.find(static_cast<char>(data[1]));
      it != additional_escape_map.end()) {
    return {it->second, 2};
  }
  if (auto it = kEscapeToCodepoint.find(static_cast<char>(data[1]));
      it != kEscapeToCodepoint.end()) {
    return {it->second, 2};
  }

  if (data[1] == 'x') {
    // arbitrary length hex
    int len = 0;
    TCodepoint codepoint = 0;
    int32_t digit;
    while ((digit = HexCharToInt(data[2 + len])) != -1) {
      codepoint = codepoint * 16 + digit;
      ++len;
    }
    if (len == 0) {
      return {CharHandlingError::kInvalidEscape, 0};
    }
    return {codepoint, len + 2};
  } else if (data[1] == 'u' || data[1] == 'U') {
    // 4- or 8-digit hex
    int len = data[1] == 'u' ? 4 : 8;
    TCodepoint codepoint = 0;

    for (int i = 0; i < len; ++i) {
      auto digit = HexCharToInt(data[i + 2]);
      if (digit == -1) {
        return {CharHandlingError::kInvalidEscape, 0};
      }
      codepoint = codepoint * 16 + digit;
    }
    return {codepoint, len + 2};
  } else {
    return {CharHandlingError::kInvalidEscape, 0};
  }
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_ENCODING_H_
