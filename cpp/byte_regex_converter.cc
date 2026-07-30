/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/byte_regex_converter.cc
 */

#include "byte_regex_converter.h"

#include <bitset>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "grammar_builder.h"
#include "support/encoding.h"
#include "support/logging.h"

namespace xgrammar {
namespace {

class ByteRegexConverter {
 public:
  ByteRegexConverter(GrammarBuilder* builder, const std::string& pattern, std::string rule_hint)
      : builder_(builder), pattern_(pattern), rule_hint_(std::move(rule_hint)) {}

  int32_t Convert() {
    int32_t result = ParseAlternation();
    if (!AtEnd()) {
      if (Peek() == ')') {
        RaiseError("unmatched ')'");
      }
      RaiseError("unexpected character");
    }
    return result;
  }

 private:
  struct Atom {
    int32_t expr_id;
    bool quantifiable = true;
  };

  struct ClassUnit {
    std::bitset<256> bytes;
    bool is_singleton = false;
    uint8_t singleton = 0;
  };

  [[noreturn]] void RaiseError(const std::string& message) const {
    throw XGrammarError(
        "byte regular-expression parsing error at byte " + std::to_string(position_ + 1) + ": " +
        message
    );
  }

  bool AtEnd() const { return position_ == pattern_.size(); }

  char Peek(size_t offset = 0) const {
    return position_ + offset < pattern_.size() ? pattern_[position_ + offset] : '\0';
  }

  char Consume() {
    if (AtEnd()) {
      RaiseError("unexpected end of pattern");
    }
    return pattern_[position_++];
  }

  bool Match(char expected) {
    if (Peek() != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  int32_t MakeSequence(std::vector<int32_t> elements) {
    if (elements.empty()) {
      return builder_->AddEmptyStr();
    }
    if (elements.size() == 1) {
      return elements[0];
    }
    return builder_->AddSequence(elements);
  }

  int32_t MakeChoice(std::vector<int32_t> choices) {
    if (choices.size() == 1) {
      return choices[0];
    }
    return builder_->AddChoices(choices);
  }

  int32_t MakeByte(uint8_t byte) {
    return builder_->AddByteString(std::string(1, static_cast<char>(byte)));
  }

  int32_t MakeByteSet(const std::bitset<256>& bytes) {
    std::vector<int32_t> choices;
    choices.reserve(bytes.count());
    for (int byte = 0; byte < 256; ++byte) {
      if (bytes[byte]) {
        choices.push_back(MakeByte(static_cast<uint8_t>(byte)));
      }
    }
    if (choices.empty()) {
      int32_t empty_rule = builder_->AddEmptyRuleWithHint(rule_hint_ + "_byte_regex_empty");
      builder_->UpdateRuleBody(empty_rule, builder_->AddRuleRef(empty_rule));
      return builder_->AddRuleRef(empty_rule);
    }
    return MakeChoice(std::move(choices));
  }

  int32_t ParseAlternation() {
    std::vector<int32_t> choices;
    choices.push_back(ParseConcatenation());
    while (Match('|')) {
      choices.push_back(ParseConcatenation());
    }
    return MakeChoice(std::move(choices));
  }

  int32_t ParseConcatenation() {
    std::vector<int32_t> elements;
    while (!AtEnd() && Peek() != '|' && Peek() != ')') {
      elements.push_back(ParseRepetition());
    }
    return MakeSequence(std::move(elements));
  }

  int32_t ParseRepetition() {
    Atom atom = ParseAtom();
    int32_t min_repeat = 1;
    int32_t max_repeat = 1;
    bool has_quantifier = true;
    if (Match('*')) {
      min_repeat = 0;
      max_repeat = -1;
    } else if (Match('+')) {
      min_repeat = 1;
      max_repeat = -1;
    } else if (Match('?')) {
      min_repeat = 0;
      max_repeat = 1;
    } else if (Match('{')) {
      std::tie(min_repeat, max_repeat) = ParseRepetitionRange();
    } else {
      has_quantifier = false;
    }

    if (!has_quantifier) {
      return atom.expr_id;
    }
    if (!atom.quantifiable) {
      RaiseError("anchors cannot be repeated");
    }
    if (Match('?')) {
      // Matching is nondeterministic, so greedy and lazy repetitions have the same language.
    }
    if (Peek() == '*' || Peek() == '+' || Peek() == '?' || Peek() == '{') {
      RaiseError("two consecutive repetition modifiers are not allowed");
    }
    return builder_->AddRepeatFromExpr(
        rule_hint_ + "_byte_regex_repeat", atom.expr_id, min_repeat, max_repeat
    );
  }

  std::pair<int32_t, int32_t> ParseRepetitionRange() {
    int32_t lower = ParseRepetitionCount();
    if (Match('}')) {
      return {lower, lower};
    }
    if (!Match(',')) {
      RaiseError("expected ',' or '}' in repetition range");
    }
    if (Match('}')) {
      return {lower, -1};
    }
    int32_t upper = ParseRepetitionCount();
    if (!Match('}')) {
      RaiseError("expected '}' after repetition range");
    }
    if (upper < lower) {
      RaiseError("repetition upper bound is smaller than its lower bound");
    }
    return {lower, upper};
  }

  int32_t ParseRepetitionCount() {
    if (!std::isdigit(static_cast<unsigned char>(Peek()))) {
      RaiseError("expected a decimal repetition count");
    }
    int64_t value = 0;
    while (std::isdigit(static_cast<unsigned char>(Peek()))) {
      value = value * 10 + (Consume() - '0');
      if (value > std::numeric_limits<int32_t>::max()) {
        RaiseError("repetition count is too large");
      }
    }
    return static_cast<int32_t>(value);
  }

  Atom ParseAtom() {
    if (AtEnd()) {
      RaiseError("expected a regular-expression atom");
    }
    size_t atom_position = position_;
    char current = Consume();
    switch (current) {
      case '(':
        return {ParseGroup(), true};
      case '[':
        return {ParseCharacterClass(), true};
      case '.': {
        std::bitset<256> all_bytes;
        all_bytes.set();
        return {MakeByteSet(all_bytes), true};
      }
      case '\\':
        return {MakeClassUnit(ParseEscape(false)), true};
      case '^':
        if (atom_position != 0) {
          RaiseError("start anchor is only allowed at the beginning of the pattern");
        }
        return {builder_->AddEmptyStr(), false};
      case '$':
        if (!AtEnd()) {
          RaiseError("end anchor is only allowed at the end of the pattern");
        }
        return {builder_->AddEmptyStr(), false};
      case '*':
      case '+':
      case '?':
      case '{':
        RaiseError("repetition modifier has no preceding atom");
      default:
        return {MakeByte(static_cast<uint8_t>(current)), true};
    }
  }

  int32_t ParseGroup() {
    if (Match('?')) {
      if (Match(':')) {
        // Non-capturing group.
      } else if (Peek() == '<' && Peek(1) != '=' && Peek(1) != '!') {
        Consume();
        ParseCaptureName();
      } else if (Peek() == 'P' && Peek(1) == '<') {
        Consume();
        Consume();
        ParseCaptureName();
      } else if (Peek() == '=' || Peek() == '!' ||
                 (Peek() == '<' && (Peek(1) == '=' || Peek(1) == '!'))) {
        RaiseError("lookaround assertions are not supported");
      } else {
        RaiseError("inline regular-expression flags are not supported");
      }
    }

    int32_t result = ParseAlternation();
    if (!Match(')')) {
      RaiseError("unclosed '('");
    }
    return result;
  }

  void ParseCaptureName() {
    size_t name_start = position_;
    while (std::isalnum(static_cast<unsigned char>(Peek())) || Peek() == '_') {
      ++position_;
    }
    if (position_ == name_start || !Match('>')) {
      RaiseError("invalid capture group name");
    }
  }

  int32_t ParseCharacterClass() {
    bool negated = Match('^');
    std::bitset<256> bytes;
    bool has_item = false;
    while (!AtEnd() && Peek() != ']') {
      if ((Peek() == '&' && Peek(1) == '&') || (Peek() == '-' && Peek(1) == '-') ||
          (Peek() == '~' && Peek(1) == '~') || Peek() == '[') {
        RaiseError("byte character-class set operations are not supported");
      }
      ClassUnit lower = ParseClassUnit();
      has_item = true;
      if (lower.is_singleton && Peek() == '-' && Peek(1) != ']' && Peek(1) != '\0') {
        Consume();
        ClassUnit upper = ParseClassUnit();
        if (!upper.is_singleton) {
          RaiseError("character-class range endpoint must be a single byte");
        }
        if (lower.singleton > upper.singleton) {
          RaiseError("character-class range start exceeds its end");
        }
        for (int byte = lower.singleton; byte <= upper.singleton; ++byte) {
          bytes.set(byte);
        }
      } else {
        bytes |= lower.bytes;
      }
    }
    if (!Match(']')) {
      RaiseError("unclosed '['");
    }
    if (!has_item) {
      RaiseError("empty byte character class is not allowed");
    }
    if (negated) {
      bytes.flip();
    }
    return MakeByteSet(bytes);
  }

  ClassUnit ParseClassUnit() {
    if (AtEnd()) {
      RaiseError("unclosed '['");
    }
    if (Match('\\')) {
      return ParseEscape(true);
    }
    unsigned char byte = static_cast<unsigned char>(Consume());
    if (byte >= 0x80) {
      RaiseError("non-ASCII characters are not available in byte character classes; use \\xHH");
    }
    return Singleton(byte);
  }

  ClassUnit ParseEscape(bool in_character_class) {
    if (AtEnd()) {
      RaiseError("unfinished byte escape");
    }
    char escaped = Consume();
    switch (escaped) {
      case 'd':
        return Range('0', '9');
      case 'D':
        return Complement(Range('0', '9'));
      case 'w': {
        ClassUnit result = Range('0', '9');
        result.bytes |= Range('A', 'Z').bytes;
        result.bytes |= Range('a', 'z').bytes;
        result.bytes.set('_');
        result.is_singleton = false;
        return result;
      }
      case 'W': {
        ClassUnit word = Range('0', '9');
        word.bytes |= Range('A', 'Z').bytes;
        word.bytes |= Range('a', 'z').bytes;
        word.bytes.set('_');
        return Complement(word);
      }
      case 's': {
        ClassUnit whitespace;
        whitespace.bytes.set(' ');
        for (int byte = '\t'; byte <= '\r'; ++byte) {
          whitespace.bytes.set(byte);
        }
        return whitespace;
      }
      case 'S': {
        ClassUnit whitespace;
        whitespace.bytes.set(' ');
        for (int byte = '\t'; byte <= '\r'; ++byte) {
          whitespace.bytes.set(byte);
        }
        return Complement(whitespace);
      }
      case 'x':
        if (Peek() == '{') {
          RaiseError("Unicode character escapes are not available in byte regular expressions");
        }
        return Singleton(ParseHexByte());
      case 'p':
      case 'P':
        RaiseError("Unicode character classes are not available in byte regular expressions");
      case 'u':
      case 'U':
        RaiseError("Unicode character escapes are not available in byte regular expressions");
      case 'b':
      case 'B':
        if (!in_character_class) {
          RaiseError("word-boundary assertions are not supported");
        }
        return Singleton('\b');
      case 'a':
        return Singleton('\a');
      case 'f':
        return Singleton('\f');
      case 'n':
        return Singleton('\n');
      case 'r':
        return Singleton('\r');
      case 't':
        return Singleton('\t');
      case 'v':
        return Singleton('\v');
      case '0':
        return Singleton('\0');
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
        return Singleton(static_cast<uint8_t>(escaped));
      default:
        if ((escaped >= '1' && escaped <= '9') || escaped == 'k') {
          RaiseError("backreferences are not supported");
        }
        RaiseError(std::string("unrecognized byte escape '\\") + escaped + "'");
    }
  }

  uint8_t ParseHexByte() {
    if (position_ + 2 > pattern_.size()) {
      RaiseError("\\x escape must contain exactly two hexadecimal digits");
    }
    int high = HexCharToInt(Peek());
    int low = HexCharToInt(Peek(1));
    if (high < 0 || low < 0) {
      RaiseError("\\x escape must contain exactly two hexadecimal digits");
    }
    position_ += 2;
    return static_cast<uint8_t>((high << 4) | low);
  }

  int32_t MakeClassUnit(const ClassUnit& unit) {
    return unit.is_singleton ? MakeByte(unit.singleton) : MakeByteSet(unit.bytes);
  }

  static ClassUnit Singleton(uint8_t byte) {
    ClassUnit result;
    result.bytes.set(byte);
    result.is_singleton = true;
    result.singleton = byte;
    return result;
  }

  static ClassUnit Range(uint8_t lower, uint8_t upper) {
    ClassUnit result;
    for (int byte = lower; byte <= upper; ++byte) {
      result.bytes.set(byte);
    }
    result.is_singleton = lower == upper;
    result.singleton = lower;
    return result;
  }

  static ClassUnit Complement(ClassUnit unit) {
    unit.bytes.flip();
    unit.is_singleton = false;
    return unit;
  }

  GrammarBuilder* builder_;
  const std::string& pattern_;
  std::string rule_hint_;
  size_t position_ = 0;
};

}  // namespace

int32_t ByteRegexToGrammarExpr(
    GrammarBuilder* builder, const std::string& pattern, const std::string& rule_hint
) {
  return ByteRegexConverter(builder, pattern, rule_hint).Convert();
}

}  // namespace xgrammar
