/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.h
 */
#ifndef XGRAMMAR_FSM_BUILDER_H_
#define XGRAMMAR_FSM_BUILDER_H_

#include <bitset>
#include <cstdint>
#include <string>
#include <vector>

#include "fsm.h"
#include "support/utils.h"

namespace xgrammar {

/*!
 * \brief Boundaries of the packed UTF-8 representation used by AddCharacterRange. The packed
 * format stores the UTF-8 bytes of one character as: (byte0 << 24) | (byte1 << 16) |
 * (byte2 << 8) | byte3, aligned to the low end, so it is monotonic in the codepoint.
 */
inline constexpr uint32_t kMax1ByteUnicode = 0x7F;
inline constexpr uint32_t kMin2BytesUnicode = 0xC080;
inline constexpr uint32_t kMax2BytesUnicode = 0xDFBF;
inline constexpr uint32_t kMin3BytesUnicode = 0xE08080;
inline constexpr uint32_t kMax3BytesUnicode = 0xEFBFBF;
inline constexpr uint32_t kMin4BytesUnicode = 0xF0808080;
inline constexpr uint32_t kMax4BytesUnicode = 0xF7BFBFBF;

/*! \brief Convert a Unicode codepoint to the packed UTF-8 format used by AddCharacterRange. */
uint32_t CodepointToPackedUTF8(uint32_t codepoint);

/*!
 * \brief Add transitions from `from` to `to` accepting every character in the packed UTF-8 range
 * [min, max], where min and max encode characters of the same UTF-8 length. Multi-byte characters
 * are lowered to chains of byte edges through freshly added intermediate states.
 */
void AddSameLengthCharacterRange(FSM& fsm, int from, int to, uint32_t min, uint32_t max);

/*!
 * \brief Add transitions from `from` to `to` accepting every character in the packed UTF-8 range
 * [min, max]. The range is split by UTF-8 encoded length and each part is lowered with
 * AddSameLengthCharacterRange.
 */
void AddCharacterRange(FSM& fsm, int from, int to, uint32_t min, uint32_t max);

/*!
 * \brief A builder that converts a regex string to a FSM.
 */
class RegexFSMBuilder {
 public:
  /*!
   * \brief Converts a regex string to a FSM.
   * \param regex The regex string.
   * \param byte_mode Whether the regex is matched over raw bytes (0-255) instead of Unicode
   * characters. In byte mode, \xHH escapes denote single bytes, negated character classes
   * complement within the 256 byte values, and Unicode-specific constructs are rejected.
   * \return The FSM with start and end states.
   */
  static Result<FSMWithStartEnd> Build(const std::string& regex, bool byte_mode = false);

  /*!
   * \brief Converts a regex string to a FSM, then removes the forbidden characters from every
   * character transition. The result accepts the intersection of the regex language and the
   * set of strings that contain no forbidden character. The result language may be empty.
   * \param regex The regex string.
   * \param forbidden_chars The forbidden characters.
   * \return The FSM with start and end states.
   */
  static Result<FSMWithStartEnd> BuildWithForbiddenChars(
      const std::string& regex, const std::bitset<256>& forbidden_chars
  );
};

/*!
 * \brief A builder that converts a list of patterns to a trie-based FSM.
 */
class TrieFSMBuilder {
 public:
  /*!
   * \brief Build a trie-based FSM from a list of patterns.
   * \param patterns The patterns to be built.
   * \param excluded_patterns The patterns to be excluded.
   * \param end_states The end states of the FSM. This is the terminal state of each pattern and
   * the order follows the order of patterns.
   * \param allow_overlap Whether to allow overlap between patterns (one being a prefix of the
   * other). It does not allow empty patterns either. If false and there is overlap, will return
   * std::nullopt.
   * \param add_back_edges Whether to add back edges to the FSM. This complements the trie to an
   * Aho-Corasick automaton.
   * \return If success, the FSM with start and end states. Otherwise, std::nullopt.
   */
  static std::optional<FSMWithStartEnd> Build(
      const std::vector<std::string>& patterns,
      const std::vector<std::string>& excluded_patterns,
      std::vector<int32_t>* end_states = nullptr,
      bool allow_overlap = true,
      bool add_back_edges = false
  );
};

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_BUILDER_H_
