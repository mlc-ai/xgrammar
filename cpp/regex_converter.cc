/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter.cc
 */
#include <picojson.h>
#include <xgrammar/xgrammar.h>

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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
  RegexConverter(const std::string& regex) : regex_(regex) {
    regex_codepoints_ = ParseUTF8(regex_.c_str(), false);
    if (regex_codepoints_[0] == kInvalidUTF8) {
      XGRAMMAR_LOG(ERROR) << "The regex is not a valid UTF-8 string.";
      XGRAMMAR_UNREACHABLE();
    }
    regex_codepoints_.push_back(0);  // Add a null terminator
  }
  std::string Convert();

 private:
  void AddEBNFSegment(const std::string& element, bool add_space = true);
  [[noreturn]] void RaiseError(const std::string& message);
  void RaiseWarning(const std::string& message);

  std::string HandleCharacterClass();
  std::string HandleRepetitionRange();
  std::string HandleCharEscape();
  std::string HandleEscape();
  std::string regex_;
  std::vector<TCodepoint> regex_codepoints_;
  TCodepoint* start_;
  TCodepoint* current_;
  TCodepoint* end_;
  std::string result_ebnf_;
  int parenthesis_level_ = 0;
};

void RegexConverter::AddEBNFSegment(const std::string& element, bool add_space) {
  if (!result_ebnf_.empty() && add_space) {
    result_ebnf_ += ' ';
  }
  result_ebnf_ += element;
}

void RegexConverter::RaiseError(const std::string& message) {
  XGRAMMAR_LOG(ERROR) << "Regex parsing error at position " << current_ - start_ + 1 << ": "
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
  while ((*current_ != ']' || *(current_ - 1) != '\\') && current_ != end_) {
    if (*current_ == '\\') {
      char_class += HandleCharEscape();
    } else {
      char_class += PrintAsUTF8(*current_);
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

std::string RegexConverter::HandleRepetitionRange() {
  std::string result = "{";
  if (!isdigit(*current_) && *current_ != ',') {
    RaiseError("Invalid repetition count.");
  }
  while (isdigit(*current_)) {
    result += static_cast<char>(*current_);
    ++current_;
  }
  if (*current_ != ',') {
    RaiseError("Invalid repetition count.");
  }
  result += ',';
  ++current_;
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
  return result_ebnf_;
}

std::string RegexConverter::HandleCharEscape() {
  // clang-format off
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {
      {'^', '^'}, {'$', '$'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'}, {'\\', '\\'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}
  };
  // clang-format on
  if (end_ - current_ < 2 || (current_[1] == 'u' && end_ - current_ < 5) ||
      (current_[1] == 'x' && end_ - current_ < 4) || (current_[1] == 'c' && end_ - current_ < 3)) {
    RaiseError("Escape sequence is not finished.");
  }
  auto [codepoint, len] = xgrammar::HandleEscape(current_, CUSTOM_ESCAPE_MAP);
  if (codepoint != CharHandlingError::kInvalidEscape) {
    current_ += len;
    return PrintAsEscapedUTF8(codepoint);
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
    return PrintAsEscapedUTF8(value);
  } else if (current_[1] == 'c') {
    current_ += 2;
    if (!std::isalpha(*current_)) {
      RaiseError("Invalid control character escape sequence.");
    }
    ++current_;
    return PrintAsEscapedUTF8((*current_) % 32);
  } else {
    RaiseWarning(
        "Escape sequence '\\" + PrintAsEscapedUTF8(current_[1]) +
        "' is not recognized. The character itself will be matched"
    );
    current_ += 2;
    return PrintAsEscapedUTF8(current_[1]);
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
  } else if (current_[1] == 's') {
    current_ += 2;
    return "[a-zA-Z0-9_]";
  } else if (current_[1] == 'S') {
    current_ += 2;
    return "[^a-zA-Z0-9_]";
  } else if (current_[1] == 'w') {
    current_ += 2;
    return R"([\f\n\r\t\v\u0020\u00a0])";
  } else if (current_[1] == 'W') {
    current_ += 2;
    return R"([^[\f\n\r\t\v\u0020\u00a0])";
  } else if ((current_[1] >= '0' && current_[1] <= '9') || current_[1] == 'k') {
    RaiseError("Backreference is not supported yet.");
  } else if (current_[1] == 'p' || current_[1] == 'P') {
    RaiseError("Unicode character class escape sequence is not supported yet.");
  } else if (current_[1] == 'b' || current_[1] == 'B') {
    RaiseError("Word boundary is not supported yet.");
  } else {
    return "\"" + HandleCharEscape() + "\"";
  }
}

std::string RegexConverter::Convert() {
  start_ = regex_codepoints_.data();
  current_ = start_;
  end_ = start_ + regex_codepoints_.size() - 1;
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
      AddEBNFSegment(HandleCharacterClass());
    } else if (*current_ == '(') {
      if (current_ != end_ - 1 && current_[1] == '?') {
        RaiseError(
            "Assertions, named capturing groups and non-capturing groups are not supported yet."
        );
      }
      ++parenthesis_level_;
      AddEBNFSegment("(");
    } else if (*current_ == ')') {
      --parenthesis_level_;
      AddEBNFSegment(")");
    } else if (*current_ == '*' || *current_ == '+' || *current_ == '?') {
      result_ebnf_ += static_cast<char>(*current_);
      ++current_;
    } else if (*current_ == '{') {
      HandleRepetitionRange();
    } else if (*current_ == '|') {
      AddEBNFSegment("|");
      ++current_;
    } else if (*current_ == '\\') {
      AddEBNFSegment(HandleEscape());
    } else if (*current_ == '.') {
      AddEBNFSegment("[\\u0000-\\\U0010FFFF]");
    } else {
      // Non-special characters are matched literally.
      AddEBNFSegment("\"" + PrintAsEscapedUTF8(*current_) + "\"");
      ++current_;
    }
  }
  return "main ::= " + result_ebnf_;
}

std::string BuiltinGrammar::_RegexToEBNF(const std::string& regex) {
  RegexConverter converter(regex);
  return converter.Convert();
}

}  // namespace xgrammar
