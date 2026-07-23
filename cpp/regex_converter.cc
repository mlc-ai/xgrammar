/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/regex_converter.cc
 */
#include "regex_converter.h"

#include <cctype>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

/*!
 * \brief Convert a regex to EBNF.
 * \details The implementation refers to the regex described in
 * https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions
 */
class RegexConverter {
 public:
  explicit RegexConverter(const std::string& regex) : regex_(regex) {
    if (!regex.empty()) {
      regex_codepoints_ = ParseUTF8(regex_.c_str(), false);
      if (regex_codepoints_[0] == kInvalidUTF8) {
        XGRAMMAR_LOG(FATAL) << "The regex is not a valid UTF-8 string.";
        XGRAMMAR_UNREACHABLE();
      }
    }
    regex_codepoints_.push_back(0);  // Add a null terminator
  }
  std::string Convert();

 private:
  /**
   * \brief Add a segment string to the result EBNF string. It especially adds a space if needed
   * and add_space is true.
   */
  void AddEBNFSegment(const std::string& element);

  [[noreturn]] void RaiseError(const std::string& message);
  void RaiseWarning(const std::string& message);

  std::string HandleCharacterClass();
  std::string HandleRepetitionRange();
  std::string HandleCharEscape();
  std::string HandleEscape();
  std::string HandleEscapeInCharClass();
  /**
   * \brief Handle group modifier. The general format is "(?" + modifier + content + ")". E.g.
   * "(?:abc)" is a non-capturing group.
   */
  void HandleGroupModifier();

  std::string regex_;
  std::vector<TCodepoint> regex_codepoints_;
  TCodepoint* start_;
  TCodepoint* current_;
  TCodepoint* end_;
  std::string result_ebnf_;
  int parenthesis_level_ = 0;
};

class RegexGrammarConverter {
 public:
  explicit RegexGrammarConverter(const std::string& regex) {
    if (!regex.empty()) {
      codepoints_ = ParseUTF8(regex.c_str(), false);
      if (codepoints_[0] == kInvalidUTF8) {
        XGRAMMAR_LOG(FATAL) << "The regex is not a valid UTF-8 string.";
      }
    }
    codepoints_.push_back(0);
  }

  Grammar Convert() {
    start_ = codepoints_.data();
    current_ = start_;
    end_ = start_ + codepoints_.size() - 1;

    int32_t root_rule_id = builder_.AddEmptyRule("root");
    int32_t body_expr_id = ParseAlternation();
    if (current_ != end_) {
      RaiseError("Unmatched ')'");
    }
    builder_.UpdateRuleBody(root_rule_id, body_expr_id);
    return GrammarNormalizer::Apply(builder_.Get(root_rule_id));
  }

 private:
  struct ClassAtom {
    std::vector<GrammarBuilder::CharacterClassElement> elements;
    std::optional<TCodepoint> single_character;
  };

  static bool IsDigit(TCodepoint codepoint) { return codepoint >= '0' && codepoint <= '9'; }

  static bool IsAlpha(TCodepoint codepoint) {
    return (codepoint >= 'a' && codepoint <= 'z') || (codepoint >= 'A' && codepoint <= 'Z');
  }

  [[noreturn]] void RaiseError(const std::string& message) {
    XGRAMMAR_LOG(FATAL) << "Regex parsing error at position " << current_ - start_ + 1 << ": "
                        << message;
    XGRAMMAR_UNREACHABLE();
  }

  void RaiseWarning(const std::string& message) {
    XGRAMMAR_LOG(WARNING) << "Regex parsing warning at position " << current_ - start_ + 1 << ": "
                          << message;
  }

  int32_t Empty() {
    if (!empty_expr_id_.has_value()) {
      empty_expr_id_ = builder_.AddEmptyStr();
    }
    return *empty_expr_id_;
  }

  int32_t Sequence(const std::vector<int32_t>& elements) {
    if (elements.empty()) {
      return Empty();
    }
    if (elements.size() == 1) {
      return elements[0];
    }
    return builder_.AddSequence(elements);
  }

  int32_t Choice(const std::vector<int32_t>& choices) {
    if (choices.empty()) {
      return Empty();
    }
    if (choices.size() == 1) {
      return choices[0];
    }
    return builder_.AddChoices(choices);
  }

  int32_t ByteString(TCodepoint codepoint) { return builder_.AddByteString(CharToUTF8(codepoint)); }

  int32_t Repeat(int32_t expr_id, int32_t min_count, int32_t max_count) {
    if (min_count == 0 && max_count == 0) {
      return Empty();
    }
    if (min_count == 1 && max_count == 1) {
      return expr_id;
    }
    if (min_count == 0 && max_count == 1) {
      return Choice({Empty(), expr_id});
    }
    if (min_count == 0 && max_count == -1) {
      auto expr = builder_.GetGrammarExpr(expr_id);
      if (expr.type == GrammarBuilder::GrammarExprType::kCharacterClass) {
        std::vector<int32_t> data(expr.begin(), expr.end());
        return builder_.AddGrammarExpr(
            {GrammarBuilder::GrammarExprType::kCharacterClassStar,
             data.data(),
             static_cast<int32_t>(data.size())}
        );
      }
    }
    return builder_.AddRepeatFromExpr("root_regex", expr_id, min_count, max_count);
  }

  int32_t ParseAlternation() {
    std::vector<int32_t> choices;
    choices.push_back(ParseSequence());
    while (current_ != end_ && *current_ == '|') {
      ++current_;
      choices.push_back(ParseSequence());
    }
    return Choice(choices);
  }

  int32_t ParseSequence() {
    std::vector<int32_t> elements;
    while (current_ != end_ && *current_ != '|' && *current_ != ')') {
      if (*current_ == '^') {
        if (current_ != start_) {
          RaiseWarning(
              "'^' should be at the start of the regex, but found in the middle. It is ignored."
          );
        }
        ++current_;
        continue;
      }
      if (*current_ == '$') {
        if (current_ != end_ - 1 && current_[1] != '|' && current_[1] != ')') {
          RaiseWarning(
              "'$' should be at the end of the regex, but found in the middle. It is ignored."
          );
        }
        ++current_;
        continue;
      }
      elements.push_back(ParseTerm());
    }
    return Sequence(elements);
  }

  int32_t ParseTerm() {
    int32_t atom = ParseAtom();
    if (current_ == end_) {
      return atom;
    }

    int32_t min_count;
    int32_t max_count;
    bool has_quantifier = true;
    if (*current_ == '*') {
      min_count = 0;
      max_count = -1;
      ++current_;
    } else if (*current_ == '+') {
      min_count = 1;
      max_count = -1;
      ++current_;
    } else if (*current_ == '?') {
      min_count = 0;
      max_count = 1;
      ++current_;
    } else if (*current_ == '{') {
      std::tie(min_count, max_count) = ParseRepetitionRange();
    } else {
      has_quantifier = false;
    }
    if (!has_quantifier) {
      return atom;
    }

    if (current_ != end_ && *current_ == '?') {
      ++current_;
    }
    if (current_ != end_ &&
        (*current_ == '{' || *current_ == '*' || *current_ == '+' || *current_ == '?')) {
      RaiseError("Two consecutive repetition modifiers are not allowed.");
    }
    return Repeat(atom, min_count, max_count);
  }

  std::pair<int32_t, int32_t> ParseRepetitionRange() {
    ++current_;
    if (current_ == end_ || !IsDigit(*current_)) {
      RaiseError("Invalid repetition count.");
    }
    int64_t lower = 0;
    while (current_ != end_ && IsDigit(*current_)) {
      lower = lower * 10 + (*current_ - '0');
      if (lower > std::numeric_limits<int32_t>::max()) {
        RaiseError("Repetition count is too large.");
      }
      ++current_;
    }
    if (current_ == end_ || (*current_ != ',' && *current_ != '}')) {
      RaiseError("Invalid repetition count.");
    }
    if (*current_ == '}') {
      ++current_;
      return {static_cast<int32_t>(lower), static_cast<int32_t>(lower)};
    }

    ++current_;
    if (current_ != end_ && *current_ == '}') {
      ++current_;
      return {static_cast<int32_t>(lower), -1};
    }
    if (current_ == end_ || !IsDigit(*current_)) {
      RaiseError("Invalid repetition count.");
    }
    int64_t upper = 0;
    while (current_ != end_ && IsDigit(*current_)) {
      upper = upper * 10 + (*current_ - '0');
      if (upper > std::numeric_limits<int32_t>::max()) {
        RaiseError("Repetition count is too large.");
      }
      ++current_;
    }
    if (current_ == end_ || *current_ != '}' || upper < lower) {
      RaiseError("Invalid repetition count.");
    }
    ++current_;
    return {static_cast<int32_t>(lower), static_cast<int32_t>(upper)};
  }

  int32_t ParseAtom() {
    if (current_ == end_) {
      RaiseError("Expected a regular expression atom.");
    }
    if (*current_ == '[') {
      return ParseCharacterClass();
    }
    if (*current_ == '(') {
      ++current_;
      if (current_ != end_ && *current_ == '?') {
        ++current_;
        ParseGroupModifier();
      }
      int32_t group = ParseAlternation();
      if (current_ == end_ || *current_ != ')') {
        RaiseError("The parenthesis is not closed.");
      }
      ++current_;
      return group;
    }
    if (*current_ == '\\') {
      return ParseEscape();
    }
    if (*current_ == '.') {
      ++current_;
      return builder_.AddCharacterClass({{0, 0x10ffff}});
    }
    if (*current_ == '*' || *current_ == '+' || *current_ == '?' || *current_ == '{') {
      RaiseError("Repetition modifier has no preceding expression.");
    }
    TCodepoint codepoint = *current_;
    ++current_;
    return ByteString(codepoint);
  }

  void ParseGroupModifier() {
    if (current_ == end_) {
      RaiseError("Group modifier is not finished.");
    }
    if (*current_ == ':') {
      ++current_;
      return;
    }
    if (*current_ == '=' || *current_ == '!') {
      RaiseError("Lookahead is not supported yet.");
    }
    if (*current_ == '<' && current_ + 1 != end_ && (current_[1] == '=' || current_[1] == '!')) {
      RaiseError("Lookbehind is not supported yet.");
    }
    if (*current_ == '<') {
      ++current_;
      while (current_ != end_ && IsAlpha(*current_)) {
        ++current_;
      }
      if (current_ == end_ || *current_ != '>') {
        RaiseError("Invalid named capturing group.");
      }
      ++current_;
      return;
    }
    RaiseError("Group modifier flag is not supported yet.");
  }

  TCodepoint ParseEscapedCharacter() {
    static const std::unordered_map<char, TCodepoint> kEscapeMap = {
        {'^', '^'},
        {'$', '$'},
        {'.', '.'},
        {'*', '*'},
        {'+', '+'},
        {'?', '?'},
        {'\\', '\\'},
        {'(', '('},
        {')', ')'},
        {'[', '['},
        {']', ']'},
        {'{', '{'},
        {'}', '}'},
        {'|', '|'},
        {'/', '/'},
        {'-', '-'}
    };
    if (end_ - current_ < 2 || (current_[1] == 'u' && end_ - current_ < 5) ||
        (current_[1] == 'x' && end_ - current_ < 4) ||
        (current_[1] == 'c' && end_ - current_ < 3)) {
      RaiseError("Escape sequence is not finished.");
    }
    auto [codepoint, length] = ParseNextEscaped(current_, kEscapeMap);
    if (codepoint != CharHandlingError::kInvalidEscape) {
      current_ += length;
      return codepoint;
    }
    if (current_[1] == 'u' && current_[2] == '{') {
      current_ += 3;
      int digits = 0;
      TCodepoint value = 0;
      while (current_ != end_ && HexCharToInt(*current_) != -1 && digits <= 6) {
        value = value * 16 + HexCharToInt(*current_);
        ++current_;
        ++digits;
      }
      if (digits == 0 || digits > 6 || current_ == end_ || *current_ != '}') {
        RaiseError("Invalid Unicode escape sequence.");
      }
      ++current_;
      return value;
    }
    if (current_[1] == 'c') {
      current_ += 2;
      if (current_ == end_ || !IsAlpha(*current_)) {
        RaiseError("Invalid control character escape sequence.");
      }
      TCodepoint value = *current_ % 32;
      ++current_;
      return value;
    }

    RaiseWarning(
        "Escape sequence '\\" + CharToUTF8(current_[1]) +
        "' is not recognized. The character itself will be matched"
    );
    current_ += 2;
    return current_[-1];
  }

  int32_t ParseEscape() {
    if (end_ - current_ < 2) {
      RaiseError("Escape sequence is not finished.");
    }
    TCodepoint escaped = current_[1];
    if (escaped == 'd' || escaped == 'D') {
      current_ += 2;
      return builder_.AddCharacterClass({{'0', '9'}}, escaped == 'D');
    }
    if (escaped == 'w' || escaped == 'W') {
      current_ += 2;
      return builder_.AddCharacterClass(
          {{'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {'_', '_'}}, escaped == 'W'
      );
    }
    if (escaped == 's' || escaped == 'S') {
      current_ += 2;
      return builder_.AddCharacterClass(
          {{'\f', '\f'},
           {'\n', '\n'},
           {'\r', '\r'},
           {'\t', '\t'},
           {'\v', '\v'},
           {0x20, 0x20},
           {0xa0, 0xa0}},
          escaped == 'S'
      );
    }
    if ((escaped >= '1' && escaped <= '9') || escaped == 'k') {
      RaiseError("Backreference is not supported yet.");
    }
    if (escaped == 'p' || escaped == 'P') {
      RaiseError("Unicode character class escape sequence is not supported yet.");
    }
    if (escaped == 'b' || escaped == 'B') {
      RaiseError("Word boundary is not supported yet.");
    }
    return ByteString(ParseEscapedCharacter());
  }

  ClassAtom ParseClassAtom() {
    if (current_ == end_ || *current_ == ']') {
      RaiseError("Invalid character class element.");
    }
    if (*current_ != '\\') {
      TCodepoint codepoint = *current_;
      ++current_;
      return {{{codepoint, codepoint}}, codepoint};
    }
    if (end_ - current_ < 2) {
      RaiseError("Escape sequence is not finished.");
    }

    TCodepoint escaped = current_[1];
    if (escaped == 'd') {
      current_ += 2;
      return {{{'0', '9'}}, std::nullopt};
    }
    if (escaped == 'D') {
      current_ += 2;
      return {{{0, 0x2f}, {0x3a, 0x10ffff}}, std::nullopt};
    }
    if (escaped == 'w') {
      current_ += 2;
      return {{{'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {'_', '_'}}, std::nullopt};
    }
    if (escaped == 'W') {
      current_ += 2;
      return {
          {{0, 0x2f}, {0x3a, 0x40}, {0x5b, 0x5e}, {0x60, 0x60}, {0x7b, 0x10ffff}}, std::nullopt
      };
    }
    if (escaped == 's') {
      current_ += 2;
      return {
          {{'\f', '\f'},
           {'\n', '\n'},
           {'\r', '\r'},
           {'\t', '\t'},
           {'\v', '\v'},
           {0x20, 0x20},
           {0xa0, 0xa0}},
          std::nullopt
      };
    }
    if (escaped == 'S') {
      current_ += 2;
      return {{{0, 0x08}, {0x0e, 0x1f}, {0x21, 0x9f}, {0xa1, 0x10ffff}}, std::nullopt};
    }
    TCodepoint codepoint = ParseEscapedCharacter();
    return {{{codepoint, codepoint}}, codepoint};
  }

  int32_t ParseCharacterClass() {
    ++current_;
    bool is_negative = false;
    if (current_ != end_ && *current_ == '^') {
      is_negative = true;
      ++current_;
    }
    if (current_ == end_ || *current_ == ']') {
      RaiseError("Empty character class is not allowed in regex.");
    }

    std::vector<GrammarBuilder::CharacterClassElement> elements;
    while (current_ != end_ && *current_ != ']') {
      ClassAtom first = ParseClassAtom();
      if (first.single_character.has_value() && current_ != end_ && *current_ == '-' &&
          current_ + 1 != end_ && current_[1] != ']') {
        ++current_;
        ClassAtom second = ParseClassAtom();
        if (!second.single_character.has_value() ||
            *first.single_character > *second.single_character) {
          RaiseError("Invalid character class range.");
        }
        elements.push_back({*first.single_character, *second.single_character});
      } else {
        elements.insert(elements.end(), first.elements.begin(), first.elements.end());
      }
    }
    if (current_ == end_) {
      RaiseError("Unclosed '['");
    }
    ++current_;
    return builder_.AddCharacterClass(elements, is_negative);
  }

  std::vector<TCodepoint> codepoints_;
  TCodepoint* start_;
  TCodepoint* current_;
  TCodepoint* end_;
  GrammarBuilder builder_;
  std::optional<int32_t> empty_expr_id_;
};

void RegexConverter::AddEBNFSegment(const std::string& element) {
  if (!result_ebnf_.empty()) {
    result_ebnf_ += ' ';
  }
  result_ebnf_ += element;
}

void RegexConverter::RaiseError(const std::string& message) {
  XGRAMMAR_LOG(FATAL) << "Regex parsing error at position " << current_ - start_ + 1 << ": "
                      << message;
  XGRAMMAR_UNREACHABLE();
}

void RegexConverter::RaiseWarning(const std::string& message) {
  XGRAMMAR_LOG(WARNING) << "Regex parsing warning at position " << current_ - start_ + 1 << ": "
                        << message;
}

std::string RegexConverter::HandleCharacterClass() {
  std::string char_class = "[";
  ++current_;
  if (*current_ == ']') {
    RaiseError("Empty character class is not allowed in regex.");
  }
  while (*current_ != ']' && current_ != end_) {
    if (*current_ == '\\') {
      char_class += HandleEscapeInCharClass();
    } else {
      char_class += CharToUTF8(*current_);
      ++current_;
    }
  }
  if (current_ == end_) {
    RaiseError("Unclosed '['");
  }
  char_class += ']';
  ++current_;
  return char_class;
}

// {x}: Match exactly x occurrences of the preceding regular expression.
// {x,}
// {x,y}
std::string RegexConverter::HandleRepetitionRange() {
  std::string result = "{";
  ++current_;
  if (!isdigit(*current_)) {
    RaiseError("Invalid repetition count.");
  }
  while (isdigit(*current_)) {
    result += static_cast<char>(*current_);
    ++current_;
  }
  if (*current_ != ',' && *current_ != '}') {
    RaiseError("Invalid repetition count.");
  }
  result += static_cast<char>(*current_);
  ++current_;
  if (current_[-1] == '}') {
    // Matches {x}
    return result;
  }
  if (!isdigit(*current_) && *current_ != '}') {
    RaiseError("Invalid repetition count.");
  }
  while (isdigit(*current_)) {
    result += static_cast<char>(*current_);
    ++current_;
  }
  if (*current_ != '}') {
    RaiseError("Invalid repetition count.");
  }
  result += '}';
  ++current_;
  return result;
}

std::string RegexConverter::HandleCharEscape() {
  // clang-format off
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {
      {'^', '^'}, {'$', '$'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'}, {'\\', '\\'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}, {'-', '-'}
  };
  // clang-format on
  if (end_ - current_ < 2 || (current_[1] == 'u' && end_ - current_ < 5) ||
      (current_[1] == 'x' && end_ - current_ < 4) || (current_[1] == 'c' && end_ - current_ < 3)) {
    RaiseError("Escape sequence is not finished.");
  }
  auto [codepoint, len] = ParseNextEscaped(current_, CUSTOM_ESCAPE_MAP);
  if (codepoint != CharHandlingError::kInvalidEscape) {
    current_ += len;
    return EscapeString(codepoint);
  } else if (current_[1] == 'u' && current_[2] == '{') {
    current_ += 3;
    int len = 0;
    TCodepoint value = 0;
    while (HexCharToInt(current_[len]) != -1 && len <= 6) {
      value = value * 16 + HexCharToInt(current_[len]);
      ++len;
    }
    if (len == 0 || len > 6 || current_[len] != '}') {
      RaiseError("Invalid Unicode escape sequence.");
    }
    current_ += len + 1;
    return EscapeString(value);
  } else if (current_[1] == 'c') {
    current_ += 2;
    if (!std::isalpha(*current_)) {
      RaiseError("Invalid control character escape sequence.");
    }
    ++current_;
    return EscapeString((*(current_ - 1)) % 32);
  } else {
    RaiseWarning(
        "Escape sequence '\\" + EscapeString(current_[1]) +
        "' is not recognized. The character itself will be matched"
    );
    current_ += 2;
    return EscapeString(current_[-1]);
  }
}

std::string RegexConverter::HandleEscapeInCharClass() {
  if (end_ - current_ < 2) {
    RaiseError("Escape sequence is not finished.");
  }
  if (current_[1] == 'd') {
    current_ += 2;
    return "0-9";
  } else if (current_[1] == 'D') {
    current_ += 2;
    return R"(\x00-\x2F\x3A-\U0010FFFF)";
  } else if (current_[1] == 'w') {
    current_ += 2;
    return "a-zA-Z0-9_";
  } else if (current_[1] == 'W') {
    current_ += 2;
    return R"(\x00-\x2F\x3A-\x40\x5B-\x5E\x60\x7B-\U0010FFFF)";
  } else if (current_[1] == 's') {
    current_ += 2;
    return R"(\f\n\r\t\v\u0020\u00a0)";
  } else if (current_[1] == 'S') {
    current_ += 2;
    return R"(\x00-\x08\x0E-\x1F\x21-\x9F\xA1-\U0010FFFF)";
  } else {
    auto res = HandleCharEscape();
    if (res == "]" || res == "-") {
      return "\\" + res;
    } else {
      return res;
    }
  }
}

std::string RegexConverter::HandleEscape() {
  // clang-format off
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {
      {'^', '^'}, {'$', '$'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'}, {'\\', '\\'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}
  };
  // clang-format on
  if (end_ - current_ < 2) {
    RaiseError("Escape sequence is not finished.");
  }
  if (current_[1] == 'd') {
    current_ += 2;
    return "[0-9]";
  } else if (current_[1] == 'D') {
    current_ += 2;
    return "[^0-9]";
  } else if (current_[1] == 'w') {
    current_ += 2;
    return "[a-zA-Z0-9_]";
  } else if (current_[1] == 'W') {
    current_ += 2;
    return "[^a-zA-Z0-9_]";
  } else if (current_[1] == 's') {
    current_ += 2;
    return R"([\f\n\r\t\v\u0020\u00a0])";
  } else if (current_[1] == 'S') {
    current_ += 2;
    return R"([^\f\n\r\t\v\u0020\u00a0])";
  } else if ((current_[1] >= '1' && current_[1] <= '9') || current_[1] == 'k') {
    RaiseError("Backreference is not supported yet.");
  } else if (current_[1] == 'p' || current_[1] == 'P') {
    RaiseError("Unicode character class escape sequence is not supported yet.");
  } else if (current_[1] == 'b' || current_[1] == 'B') {
    RaiseError("Word boundary is not supported yet.");
  } else {
    return "\"" + HandleCharEscape() + "\"";
  }
}

void RegexConverter::HandleGroupModifier() {
  if (current_ == end_) {
    RaiseError("Group modifier is not finished.");
  }
  if (*current_ == ':') {
    // Non-capturing group.
    ++current_;
  } else if (*current_ == '=' || *current_ == '!') {
    // Positive or negative lookahead.
    RaiseError("Lookahead is not supported yet.");
  } else if (*current_ == '<' && current_ + 1 != end_ &&
             (current_[1] == '=' || current_[1] == '!')) {
    // Positive or negative lookbehind.
    RaiseError("Lookbehind is not supported yet.");
  } else if (*current_ == '<') {
    ++current_;
    while (current_ != end_ && isalpha(*current_)) {
      ++current_;
    }
    if (current_ == end_ || *current_ != '>') {
      RaiseError("Invalid named capturing group.");
    }
    // Just ignore the named of the group.
    ++current_;
  } else {
    // Group modifier flag.
    RaiseError("Group modifier flag is not supported yet.");
  }
}

std::string RegexConverter::Convert() {
  start_ = regex_codepoints_.data();
  current_ = start_;
  end_ = start_ + regex_codepoints_.size() - 1;
  bool is_empty = true;
  while (current_ != end_) {
    if (*current_ == '^') {
      if (current_ != start_) {
        RaiseWarning(
            "'^' should be at the start of the regex, but found in the middle. It is ignored."
        );
      }
      ++current_;
    } else if (*current_ == '$') {
      if (current_ != end_ - 1) {
        RaiseWarning(
            "'$' should be at the end of the regex, but found in the middle. It is ignored."
        );
      }
      ++current_;
    } else if (*current_ == '[') {
      is_empty = false;
      AddEBNFSegment(HandleCharacterClass());
    } else if (*current_ == '(') {
      is_empty = false;
      ++current_;
      ++parenthesis_level_;
      AddEBNFSegment("(");
      if (current_ != end_ && *current_ == '?') {
        ++current_;
        HandleGroupModifier();
      }
    } else if (*current_ == ')') {
      is_empty = false;
      if (parenthesis_level_ == 0) {
        RaiseError("Unmatched ')'");
      }
      // Empty alternative before ')' (e.g. "(a|)" or "(a|$)"): emit "" so it isn't a bare '|'.
      if (!result_ebnf_.empty() && result_ebnf_.back() == '|') {
        AddEBNFSegment("\"\"");
      }
      --parenthesis_level_;
      AddEBNFSegment(")");
      ++current_;
    } else if (*current_ == '*' || *current_ == '+' || *current_ == '?') {
      is_empty = false;
      result_ebnf_ += static_cast<char>(*current_);
      ++current_;
      if (current_ != end_ && *current_ == '?') {
        // Ignore the non-greedy modifier because our grammar handles all repetition numbers
        // non-deterministically.
        ++current_;
      }
      if (current_ != end_ &&
          (*current_ == '{' || *current_ == '*' || *current_ == '+' || *current_ == '?')) {
        RaiseError("Two consecutive repetition modifiers are not allowed.");
      }
    } else if (*current_ == '{') {
      is_empty = false;
      result_ebnf_ += HandleRepetitionRange();
      if (current_ != end_ && *current_ == '?') {
        // Still ignore the non-greedy modifier.
        ++current_;
      }
      if (current_ != end_ &&
          (*current_ == '{' || *current_ == '*' || *current_ == '+' || *current_ == '?')) {
        RaiseError("Two consecutive repetition modifiers are not allowed.");
      }
    } else if (*current_ == '|') {
      is_empty = false;
      // Empty alternative before '|': emit "" so there's no bare '|' on the left.
      // Covers leading ("^$|abc"), consecutive ("a||b") and group-start ("(|a)") cases.
      if (result_ebnf_.empty() || result_ebnf_.back() == '|' || result_ebnf_.back() == '(') {
        AddEBNFSegment("\"\"");
      }
      AddEBNFSegment("|");
      ++current_;
    } else if (*current_ == '\\') {
      is_empty = false;
      AddEBNFSegment(HandleEscape());
    } else if (*current_ == '.') {
      is_empty = false;
      AddEBNFSegment(R"([\u0000-\U0010FFFF])");
      ++current_;
    } else {
      is_empty = false;
      // Non-special characters are matched literally.
      AddEBNFSegment("\"" + EscapeString(*current_) + "\"");
      ++current_;
    }
  }
  if (parenthesis_level_ != 0) {
    RaiseError("The parenthesis is not closed.");
  }
  // Trailing empty alternative, e.g. "abc|": emit "" so it doesn't end with a bare '|'.
  if (!result_ebnf_.empty() && result_ebnf_.back() == '|') {
    AddEBNFSegment("\"\"");
  }
  if (is_empty) {
    AddEBNFSegment("\"\"");
  }
  return result_ebnf_;
}

std::string RegexToEBNF(const std::string& regex, bool with_rule_name) {
  RegexConverter converter(regex);
  if (with_rule_name) {
    return "root ::= " + converter.Convert() + "\n";
  } else {
    return converter.Convert();
  }
}

Grammar RegexToGrammar(const std::string& regex) { return RegexGrammarConverter(regex).Convert(); }

}  // namespace xgrammar
