/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/regex_converter.cc
 */
#include "regex_converter.h"

#include <xgrammar/xgrammar.h>

#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
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
class RegexConverter::Impl {
 public:
  explicit Impl(const std::string& regex) : regex_(regex) {
    regex_codepoints_ = ParseUTF8(regex_.c_str(), false);
    if (regex_codepoints_[0] == kInvalidUTF8) {
      XGRAMMAR_LOG(FATAL) << "The regex is not a valid UTF-8 string.";
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
  std::string HandleEscapeInCharClass();
  std::string regex_;
  std::vector<TCodepoint> regex_codepoints_;
  TCodepoint* start_;
  TCodepoint* current_;
  TCodepoint* end_;
  std::string result_ebnf_;
  int parenthesis_level_ = 0;
};

void RegexConverter::Impl::AddEBNFSegment(const std::string& element, bool add_space) {
  if (!result_ebnf_.empty() && add_space) {
    result_ebnf_ += ' ';
  }
  result_ebnf_ += element;
}

void RegexConverter::Impl::RaiseError(const std::string& message) {
  XGRAMMAR_LOG(FATAL) << "Regex parsing error at position " << current_ - start_ + 1 << ": "
                      << message;
  XGRAMMAR_UNREACHABLE();
}

void RegexConverter::Impl::RaiseWarning(const std::string& message) {
  XGRAMMAR_LOG(WARNING) << "Regex parsing warning at position " << current_ - start_ + 1 << ": "
                        << message;
}

std::string RegexConverter::Impl::HandleCharacterClass() {
  std::string char_class = "[";
  ++current_;
  while (*current_ != ']' && current_ != end_) {
    if (*current_ == '\\') {
      char_class += HandleEscapeInCharClass();
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
// {x}: Match exactly x occurrences of the preceding regular expression.
// {x,}
// {x,y}
std::string RegexConverter::Impl::HandleRepetitionRange() {
  std::string result = "{";
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

std::string RegexConverter::Impl::HandleCharEscape() {
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
    return PrintAsEscapedUTF8((*(current_ - 1)) % 32);
  } else {
    RaiseWarning(
        "Escape sequence '\\" + PrintAsEscapedUTF8(current_[1]) +
        "' is not recognized. The character itself will be matched"
    );
    current_ += 2;
    return PrintAsEscapedUTF8(current_[-1]);
  }
}

std::string RegexConverter::Impl::HandleEscapeInCharClass() {
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

std::string RegexConverter::Impl::HandleEscape() {
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
    return R"([^[\f\n\r\t\v\u0020\u00a0])";
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

std::string RegexConverter::Impl::Convert() {
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
      ++current_;
    } else if (*current_ == ')') {
      --parenthesis_level_;
      AddEBNFSegment(")");
      ++current_;
    } else if (*current_ == '*' || *current_ == '+' || *current_ == '?') {
      result_ebnf_ += static_cast<char>(*current_);
      ++current_;
    } else if (*current_ == '{') {
      result_ebnf_ += HandleRepetitionRange();
    } else if (*current_ == '|') {
      AddEBNFSegment("|");
      ++current_;
    } else if (*current_ == '\\') {
      AddEBNFSegment(HandleEscape());
    } else if (*current_ == '.') {
      AddEBNFSegment(R"([\u0000-\U0010FFFF])");
      ++current_;
    } else {
      // Non-special characters are matched literally.
      AddEBNFSegment("\"" + PrintAsEscapedUTF8(*current_) + "\"");
      ++current_;
    }
  }
  if (parenthesis_level_ != 0) {
    RaiseError("The paranthesis is not closed.");
  }
  return result_ebnf_;
}

RegexConverter::RegexConverter(const std::string& regex) : pimpl_(std::make_shared<Impl>(regex)) {}

std::string RegexConverter::Convert() { return pimpl_->Convert(); }

std::string BuiltinGrammar::_RegexToEBNF(const std::string& regex) {
  RegexConverter converter(regex);
  return "root ::= " + converter.Convert() + "\n";
}

}  // namespace xgrammar
