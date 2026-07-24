/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/lark_converter.cc
 */

#include "lark_converter.h"

#include <picojson.h>
#include <xgrammar/exception.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "support/encoding.h"
#include "support/logging.h"

namespace xgrammar {
namespace {

struct Location {
  int line = 1;
  int column = 1;
};

[[noreturn]] void RaiseLarkError(
    const std::string& source, const Location& location, const std::string& message
) {
  size_t line_start = 0;
  int current_line = 1;
  while (current_line < location.line && line_start < source.size()) {
    size_t newline = source.find('\n', line_start);
    if (newline == std::string::npos) {
      line_start = source.size();
      break;
    }
    line_start = newline + 1;
    ++current_line;
  }
  size_t line_end = source.find('\n', line_start);
  if (line_end == std::string::npos) {
    line_end = source.size();
  }
  std::string line_text = source.substr(line_start, line_end - line_start);
  std::ostringstream os;
  os << "Lark error at line " << location.line << ", column " << location.column << ": " << message;
  if (!line_text.empty()) {
    os << "\n" << line_text << "\n" << std::string(std::max(0, location.column - 1), ' ') << "^";
  }
  throw XGrammarError(os.str());
}

enum class TokenType {
  kName,
  kString,
  kRegex,
  kNumber,
  kSpecialToken,
  kGrammarRef,
  kJson,
  kRegexExt,
  kGrammarOptions,
  kImport,
  kIgnore,
  kLark,
  kIf,
  kUnsupportedDirective,
  kColon,
  kDoubleColon,
  kComma,
  kDot,
  kDotDot,
  kArrow,
  kEquals,
  kLParen,
  kRParen,
  kLBracket,
  kRBracket,
  kLBrace,
  kRBrace,
  kPipe,
  kAnd,
  kTilde,
  kQuestion,
  kStar,
  kPlus,
  kNewline,
  kEnd,
};

struct Token {
  TokenType type;
  std::string text;
  std::string flags;
  Location location;
};

class LarkLexer {
 public:
  explicit LarkLexer(const std::string& source) : source_(source) {}

  std::vector<Token> Tokenize() {
    std::vector<Token> result;
    while (position_ < source_.size()) {
      char c = source_[position_];
      if (c == ' ' || c == '\t' || c == '\f') {
        Advance();
        continue;
      }
      if (c == '\r' || c == '\n') {
        Location location = CurrentLocation();
        if (c == '\r') {
          Advance();
          if (position_ < source_.size() && source_[position_] == '\n') {
            Advance();
          }
        } else {
          Advance();
        }
        result.push_back({TokenType::kNewline, "\n", "", location});
        continue;
      }
      if (c == '#') {
        SkipComment();
        continue;
      }
      if (c == '/' && PeekChar(1) == '/') {
        SkipComment();
        continue;
      }

      Location location = CurrentLocation();
      switch (c) {
        case ':':
          if (PeekChar(1) == ':') {
            result.push_back(SimpleToken(TokenType::kDoubleColon, 2));
          } else {
            result.push_back(SimpleToken(TokenType::kColon, 1));
          }
          break;
        case ',':
          result.push_back(SimpleToken(TokenType::kComma, 1));
          break;
        case '.':
          if (PeekChar(1) == '.') {
            result.push_back(SimpleToken(TokenType::kDotDot, 2));
          } else {
            result.push_back(SimpleToken(TokenType::kDot, 1));
          }
          break;
        case '-':
          if (PeekChar(1) == '>') {
            result.push_back(SimpleToken(TokenType::kArrow, 2));
          } else if (std::isdigit(static_cast<unsigned char>(PeekChar(1)))) {
            result.push_back(LexNumber());
          } else {
            RaiseLarkError(source_, location, "unexpected '-' character");
          }
          break;
        case '+':
          if (std::isdigit(static_cast<unsigned char>(PeekChar(1)))) {
            result.push_back(LexNumber());
          } else {
            result.push_back(SimpleToken(TokenType::kPlus, 1));
          }
          break;
        case '=':
          result.push_back(SimpleToken(TokenType::kEquals, 1));
          break;
        case '(':
          result.push_back(SimpleToken(TokenType::kLParen, 1));
          break;
        case ')':
          result.push_back(SimpleToken(TokenType::kRParen, 1));
          break;
        case '[':
          result.push_back(SimpleToken(TokenType::kLBracket, 1));
          break;
        case ']':
          result.push_back(SimpleToken(TokenType::kRBracket, 1));
          break;
        case '{':
          result.push_back(SimpleToken(TokenType::kLBrace, 1));
          break;
        case '}':
          result.push_back(SimpleToken(TokenType::kRBrace, 1));
          break;
        case '|':
          result.push_back(SimpleToken(TokenType::kPipe, 1));
          break;
        case '&':
          result.push_back(SimpleToken(TokenType::kAnd, 1));
          break;
        case '~':
          result.push_back(SimpleToken(TokenType::kTilde, 1));
          break;
        case '?':
          if (std::isalpha(static_cast<unsigned char>(PeekChar(1))) || PeekChar(1) == '_') {
            result.push_back(LexName());
          } else {
            result.push_back(SimpleToken(TokenType::kQuestion, 1));
          }
          break;
        case '*':
          result.push_back(SimpleToken(TokenType::kStar, 1));
          break;
        case '!':
          if (std::isalpha(static_cast<unsigned char>(PeekChar(1))) || PeekChar(1) == '_') {
            result.push_back(LexName());
          } else {
            RaiseLarkError(source_, location, "unexpected '!' character");
          }
          break;
        case '"':
          result.push_back(LexString());
          break;
        case '/':
          result.push_back(LexRegex());
          break;
        case '<':
          result.push_back(LexSpecialToken());
          break;
        case '@':
          result.push_back(LexGrammarRef());
          break;
        case '%':
          result.push_back(LexDirective());
          break;
        default:
          if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            result.push_back(LexName());
          } else if (std::isdigit(static_cast<unsigned char>(c))) {
            result.push_back(LexNumber());
          } else {
            RaiseLarkError(source_, location, std::string("unexpected character '") + c + "'");
          }
      }
    }
    result.push_back({TokenType::kEnd, "", "", CurrentLocation()});
    return result;
  }

 private:
  char PeekChar(size_t offset) const {
    size_t index = position_ + offset;
    return index < source_.size() ? source_[index] : '\0';
  }

  Location CurrentLocation() const { return {line_, column_}; }

  void Advance() {
    if (position_ >= source_.size()) {
      return;
    }
    char c = source_[position_++];
    if (c == '\n') {
      ++line_;
      column_ = 1;
    } else {
      ++column_;
    }
  }

  void AdvanceTo(size_t end_position) {
    while (position_ < end_position) {
      Advance();
    }
  }

  Token SimpleToken(TokenType type, size_t length) {
    Location location = CurrentLocation();
    std::string text = source_.substr(position_, length);
    AdvanceTo(position_ + length);
    return {type, std::move(text), "", location};
  }

  void SkipComment() {
    while (position_ < source_.size() && source_[position_] != '\n' && source_[position_] != '\r') {
      Advance();
    }
  }

  Token LexName() {
    Location location = CurrentLocation();
    size_t start = position_;
    if (source_[position_] == '!' || source_[position_] == '?') {
      Advance();
    }
    while (position_ < source_.size()) {
      char c = source_[position_];
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
        break;
      }
      Advance();
    }
    return {TokenType::kName, source_.substr(start, position_ - start), "", location};
  }

  Token LexNumber() {
    Location location = CurrentLocation();
    size_t start = position_;
    if (source_[position_] == '+' || source_[position_] == '-') {
      Advance();
    }
    while (std::isdigit(static_cast<unsigned char>(PeekChar(0)))) {
      Advance();
    }
    if (PeekChar(0) == '.' && PeekChar(1) != '.') {
      Advance();
      while (std::isdigit(static_cast<unsigned char>(PeekChar(0)))) {
        Advance();
      }
    }
    if (PeekChar(0) == 'e' || PeekChar(0) == 'E') {
      Advance();
      if (PeekChar(0) == '+' || PeekChar(0) == '-') {
        Advance();
      }
      while (std::isdigit(static_cast<unsigned char>(PeekChar(0)))) {
        Advance();
      }
    }
    return {TokenType::kNumber, source_.substr(start, position_ - start), "", location};
  }

  Token LexString() {
    Location location = CurrentLocation();
    size_t start = position_;
    Advance();
    bool escaped = false;
    while (position_ < source_.size()) {
      char c = source_[position_];
      if (!escaped && c == '"') {
        Advance();
        if (PeekChar(0) == 'i') {
          Advance();
        }
        return {TokenType::kString, source_.substr(start, position_ - start), "", location};
      }
      if (c == '\n' || c == '\r') {
        RaiseLarkError(source_, location, "unterminated string literal");
      }
      if (!escaped && c == '\\') {
        escaped = true;
      } else {
        escaped = false;
      }
      Advance();
    }
    RaiseLarkError(source_, location, "unterminated string literal");
  }

  Token LexRegex() {
    Location location = CurrentLocation();
    Advance();
    size_t pattern_start = position_;
    bool escaped = false;
    while (position_ < source_.size()) {
      char c = source_[position_];
      if (!escaped && c == '/') {
        std::string pattern = source_.substr(pattern_start, position_ - pattern_start);
        Advance();
        size_t flags_start = position_;
        while (std::isalpha(static_cast<unsigned char>(PeekChar(0)))) {
          Advance();
        }
        return {
            TokenType::kRegex,
            std::move(pattern),
            source_.substr(flags_start, position_ - flags_start),
            location
        };
      }
      if (!escaped && c == '\\') {
        escaped = true;
      } else {
        escaped = false;
      }
      Advance();
    }
    RaiseLarkError(source_, location, "unterminated regular expression");
  }

  Token LexSpecialToken() {
    Location location = CurrentLocation();
    size_t start = position_;
    Advance();
    while (position_ < source_.size() && source_[position_] != '>') {
      char c = source_[position_];
      if (std::isspace(static_cast<unsigned char>(c)) || c == '<') {
        RaiseLarkError(source_, location, "invalid special token");
      }
      Advance();
    }
    if (position_ == source_.size()) {
      RaiseLarkError(source_, location, "unterminated special token");
    }
    Advance();
    return {TokenType::kSpecialToken, source_.substr(start, position_ - start), "", location};
  }

  Token LexGrammarRef() {
    Location location = CurrentLocation();
    size_t start = position_;
    Advance();
    while (std::isalnum(static_cast<unsigned char>(PeekChar(0))) || PeekChar(0) == '_' ||
           PeekChar(0) == '-') {
      Advance();
    }
    if (position_ == start + 1) {
      RaiseLarkError(source_, location, "empty grammar reference");
    }
    return {TokenType::kGrammarRef, source_.substr(start, position_ - start), "", location};
  }

  Token LexDirective() {
    Location location = CurrentLocation();
    size_t start = position_;
    Advance();
    while (std::isalpha(static_cast<unsigned char>(PeekChar(0))) || PeekChar(0) == '_') {
      Advance();
    }
    std::string directive = source_.substr(start, position_ - start);
    if (directive == "%json") {
      return LexJSONValue(TokenType::kJson, location, directive);
    }
    if (directive == "%regex") {
      return LexJSONValue(TokenType::kRegexExt, location, directive);
    }
    if (directive == "%grammar_options") {
      return LexJSONValue(TokenType::kGrammarOptions, location, directive);
    }
    if (directive == "%import") {
      return {TokenType::kImport, directive, "", location};
    }
    if (directive == "%ignore") {
      return {TokenType::kIgnore, directive, "", location};
    }
    if (directive == "%lark") {
      return {TokenType::kLark, directive, "", location};
    }
    if (directive == "%if") {
      return {TokenType::kIf, directive, "", location};
    }
    return {TokenType::kUnsupportedDirective, directive, "", location};
  }

  Token LexJSONValue(TokenType type, const Location& location, const std::string& directive) {
    while (position_ < source_.size() &&
           std::isspace(static_cast<unsigned char>(source_[position_]))) {
      Advance();
    }
    auto begin = source_.begin() + static_cast<std::ptrdiff_t>(position_);
    auto end = source_.end();
    picojson::value value;
    std::string error;
    auto parsed_end = picojson::parse(value, begin, end, &error);
    if (!error.empty() || parsed_end == begin) {
      RaiseLarkError(
          source_, location, "failed to parse JSON value after " + directive + ": " + error
      );
    }
    size_t new_position = static_cast<size_t>(parsed_end - source_.begin());
    AdvanceTo(new_position);
    return {type, value.serialize(), "", location};
  }

  const std::string& source_;
  size_t position_ = 0;
  int line_ = 1;
  int column_ = 1;
};

struct Document;

struct Node {
  enum class Kind {
    kSequence,
    kChoice,
    kRepeat,
    kString,
    kRegex,
    kRange,
    kName,
    kJson,
    kRegexExt,
    kNestedLark,
    kSpecialToken,
    kGrammarRef,
    kNot,
  };

  Kind kind = Kind::kSequence;
  Location location;
  std::string text;
  std::string text2;
  std::string flags;
  int32_t min_repeat = 0;
  int32_t max_repeat = 0;
  std::vector<Node> children;
  std::shared_ptr<Document> nested;
};

struct Definition {
  std::string name;
  bool is_terminal = false;
  bool lazy = false;
  std::optional<std::string> suffix;
  Location suffix_location;
  std::optional<int32_t> max_tokens;
  Location max_tokens_location;
  std::optional<std::string> capture_name;
  Location capture_location;
  Node body;
  Location location;
};

struct Import {
  std::string path;
  std::string local_name;
  Location location;
};

struct Document {
  std::vector<Definition> definitions;
  std::vector<Node> ignores;
  std::vector<Import> imports;
  std::vector<std::pair<picojson::value, Location>> options;
};

class LarkParser {
 public:
  LarkParser(const std::string& source, std::vector<Token> tokens)
      : source_(source), tokens_(std::move(tokens)) {}

  Document Parse() { return ParseDocument(false); }

 private:
  const Token& Peek(size_t offset = 0) const {
    size_t index = std::min(position_ + offset, tokens_.size() - 1);
    return tokens_[index];
  }

  bool Match(TokenType type) {
    if (Peek().type != type) {
      return false;
    }
    ++position_;
    return true;
  }

  Token Consume(TokenType type, const std::string& message) {
    if (Peek().type != type) {
      RaiseLarkError(source_, Peek().location, message);
    }
    return tokens_[position_++];
  }

  void ConsumeNewlines() {
    while (Match(TokenType::kNewline)) {
    }
  }

  Document ParseDocument(bool stop_at_rbrace) {
    Document document;
    ConsumeNewlines();
    while (Peek().type != TokenType::kEnd && !(stop_at_rbrace && Peek().type == TokenType::kRBrace)
    ) {
      switch (Peek().type) {
        case TokenType::kImport:
          ParseImport(&document);
          break;
        case TokenType::kIgnore:
          ParseIgnore(&document);
          break;
        case TokenType::kGrammarOptions:
          ParseOptions(&document);
          break;
        case TokenType::kUnsupportedDirective:
          RaiseLarkError(
              source_, Peek().location, "directive " + Peek().text + " is not supported"
          );
        default:
          document.definitions.push_back(ParseDefinition());
          break;
      }
      if (Peek().type != TokenType::kNewline && Peek().type != TokenType::kEnd &&
          !(stop_at_rbrace && Peek().type == TokenType::kRBrace)) {
        RaiseLarkError(source_, Peek().location, "expected end of grammar item");
      }
      ConsumeNewlines();
    }
    return document;
  }

  static bool IsTerminalName(const std::string& raw_name) {
    size_t index = 0;
    if (!raw_name.empty() && (raw_name[0] == '!' || raw_name[0] == '?')) {
      index = 1;
    }
    if (index < raw_name.size() && raw_name[index] == '_') {
      ++index;
    }
    return index < raw_name.size() && std::isupper(static_cast<unsigned char>(raw_name[index]));
  }

  static std::string NormalizeRuleName(std::string name) {
    if (!name.empty() && (name[0] == '!' || name[0] == '?')) {
      name.erase(name.begin());
    }
    return name;
  }

  void ParseImport(Document* document) {
    Location location = Consume(TokenType::kImport, "expected %import").location;
    Token first = Consume(TokenType::kName, "expected import path");
    std::string path = first.text;
    while (Match(TokenType::kDot)) {
      path += "." + Consume(TokenType::kName, "expected name after '.'").text;
    }

    if (Match(TokenType::kLParen)) {
      do {
        Token name = Consume(TokenType::kName, "expected imported terminal name");
        document->imports.push_back({path + "." + name.text, name.text, location});
      } while (Match(TokenType::kComma));
      Consume(TokenType::kRParen, "expected ')' after import list");
      return;
    }

    std::string local_name = path.substr(path.find_last_of('.') + 1);
    if (Match(TokenType::kArrow)) {
      local_name = Consume(TokenType::kName, "expected import alias").text;
    }
    document->imports.push_back({path, local_name, location});
  }

  void ParseIgnore(Document* document) {
    Consume(TokenType::kIgnore, "expected %ignore");
    document->ignores.push_back(ParseChoice());
  }

  void ParseOptions(Document* document) {
    Token token = Consume(TokenType::kGrammarOptions, "expected %grammar_options");
    picojson::value value;
    std::string error = picojson::parse(value, token.text);
    if (!error.empty()) {
      RaiseLarkError(source_, token.location, "invalid %grammar_options value: " + error);
    }
    document->options.push_back({std::move(value), token.location});
  }

  Definition ParseDefinition() {
    Token name_token = Consume(TokenType::kName, "expected rule or terminal name");
    Definition result;
    result.name = NormalizeRuleName(name_token.text);
    result.is_terminal = IsTerminalName(name_token.text);
    result.location = name_token.location;

    if (Peek().type == TokenType::kLBracket) {
      if (result.is_terminal) {
        RaiseLarkError(source_, Peek().location, "attributes are only supported on rules");
      }
      ParseAttributes(&result);
    }
    if (Peek().type == TokenType::kDot) {
      RaiseLarkError(source_, Peek().location, "rule and terminal priorities are not supported");
    }
    if (Peek().type == TokenType::kDoubleColon) {
      RaiseLarkError(source_, Peek().location, "parametric grammar is not supported");
    }
    if (Peek().type == TokenType::kLBrace) {
      RaiseLarkError(source_, Peek().location, "Lark templates are not supported");
    }
    Consume(TokenType::kColon, "expected ':' after rule name");
    result.body = ParseChoice();
    return result;
  }

  void ParseAttributes(Definition* definition) {
    Consume(TokenType::kLBracket, "expected '['");
    while (Peek().type != TokenType::kRBracket) {
      Token key = Consume(TokenType::kName, "expected rule attribute");
      if (key.text == "lazy" && Peek().type != TokenType::kEquals) {
        definition->lazy = true;
      } else if (key.text == "max_tokens") {
        Consume(TokenType::kEquals, "expected '=' after max_tokens attribute");
        Location value_location = Peek().location;
        int32_t value = ParseInteger();
        if (value <= 0) {
          RaiseLarkError(source_, value_location, "max_tokens must be positive");
        }
        if (value > 1'000'000) {
          RaiseLarkError(source_, value_location, "max_tokens is too large");
        }
        if (definition->max_tokens.has_value()) {
          RaiseLarkError(source_, key.location, "max_tokens attribute is specified more than once");
        }
        definition->max_tokens = value;
        definition->max_tokens_location = key.location;
      } else if (key.text == "capture") {
        std::string capture_name;
        Location capture_location = key.location;
        if (Match(TokenType::kEquals)) {
          Token name_token = Consume(TokenType::kString, "expected string literal after capture=");
          Node name_node = ParseStringNode(name_token);
          if (!name_node.flags.empty()) {
            RaiseLarkError(
                source_, name_node.location, "case-insensitive flags are not supported on capture"
            );
          }
          capture_name = std::move(name_node.text);
          capture_location = name_node.location;
        } else {
          capture_name = definition->name;
        }
        if (capture_name.empty()) {
          RaiseLarkError(source_, capture_location, "capture name must not be empty");
        }
        for (char c : capture_name) {
          bool valid = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                       c == '_' || c == '-' || c == '.';
          if (!valid) {
            RaiseLarkError(
                source_,
                capture_location,
                "capture name must only contain letters, digits, '_', '-' and '.'"
            );
          }
        }
        if (definition->capture_name.has_value()) {
          RaiseLarkError(source_, key.location, "capture attribute is specified more than once");
        }
        definition->capture_name = std::move(capture_name);
        definition->capture_location = capture_location;
      } else if (key.text == "suffix") {
        Consume(TokenType::kEquals, "expected '=' after suffix attribute");
        Token suffix_token = Consume(TokenType::kString, "expected string literal after suffix=");
        Node suffix = ParseStringNode(suffix_token);
        if (!suffix.flags.empty()) {
          RaiseLarkError(
              source_, suffix.location, "case-insensitive flags are not supported on suffix"
          );
        }
        if (suffix.text.empty()) {
          RaiseLarkError(source_, suffix.location, "suffix must not be empty");
        }
        if (definition->suffix.has_value()) {
          RaiseLarkError(source_, key.location, "suffix attribute is specified more than once");
        }
        definition->suffix = std::move(suffix.text);
        definition->suffix_location = suffix.location;
      } else {
        RaiseLarkError(
            source_,
            key.location,
            "rule attribute '" + key.text + "' is not supported by XGrammar Lark"
        );
      }
      if (!Match(TokenType::kComma)) {
        break;
      }
    }
    Consume(TokenType::kRBracket, "expected ']' after rule attributes");
  }

  Node ParseChoice() {
    Location location = Peek().location;
    std::vector<Node> alternatives;
    alternatives.push_back(ParseSequence());
    while (MatchAlternativeSeparator()) {
      alternatives.push_back(ParseSequence());
    }
    if (alternatives.size() == 1) {
      return std::move(alternatives[0]);
    }
    Node result;
    result.kind = Node::Kind::kChoice;
    result.location = location;
    result.children = std::move(alternatives);
    return result;
  }

  bool MatchAlternativeSeparator() {
    if (Match(TokenType::kPipe)) {
      return true;
    }
    size_t saved_position = position_;
    ConsumeNewlines();
    if (Match(TokenType::kPipe)) {
      return true;
    }
    if (Peek().type == TokenType::kRParen || Peek().type == TokenType::kRBracket ||
        Peek().type == TokenType::kRBrace) {
      return false;
    }
    position_ = saved_position;
    return false;
  }

  Node ParseSequence() {
    Location location = Peek().location;
    std::vector<Node> elements;
    while (!IsSequenceEnd(Peek().type)) {
      elements.push_back(ParseExpr());
    }
    if (Match(TokenType::kArrow)) {
      Consume(TokenType::kName, "expected alias name after '->'");
    }
    if (Peek().type == TokenType::kAnd) {
      RaiseLarkError(source_, Peek().location, "terminal intersection '&' is not supported");
    }
    if (Peek().type == TokenType::kIf) {
      RaiseLarkError(source_, Peek().location, "parametric %if conditions are not supported");
    }
    Node result;
    result.kind = Node::Kind::kSequence;
    result.location = location;
    result.children = std::move(elements);
    return result;
  }

  static bool IsSequenceEnd(TokenType type) {
    return type == TokenType::kNewline || type == TokenType::kPipe || type == TokenType::kRParen ||
           type == TokenType::kRBracket || type == TokenType::kRBrace || type == TokenType::kEnd ||
           type == TokenType::kArrow || type == TokenType::kAnd || type == TokenType::kIf;
  }

  int32_t ParseInteger() {
    Token token = Consume(TokenType::kNumber, "expected integer");
    try {
      size_t parsed = 0;
      long long value = std::stoll(token.text, &parsed);
      if (parsed != token.text.size() || value < 0 || value > std::numeric_limits<int32_t>::max()) {
        RaiseLarkError(source_, token.location, "invalid non-negative repetition count");
      }
      return static_cast<int32_t>(value);
    } catch (const std::exception&) {
      RaiseLarkError(source_, token.location, "invalid repetition count");
    }
  }

  Node ParseExpr() {
    Location location = Peek().location;
    bool negated = Match(TokenType::kTilde);
    Node atom = ParseAtom();
    if (negated) {
      Node not_node;
      not_node.kind = Node::Kind::kNot;
      not_node.location = location;
      not_node.children.push_back(std::move(atom));
      atom = std::move(not_node);
    }

    int32_t min_repeat = -1;
    int32_t max_repeat = -1;
    if (Match(TokenType::kQuestion)) {
      min_repeat = 0;
      max_repeat = 1;
    } else if (Match(TokenType::kStar)) {
      min_repeat = 0;
      max_repeat = -1;
    } else if (Match(TokenType::kPlus)) {
      min_repeat = 1;
      max_repeat = -1;
    } else if (Match(TokenType::kTilde)) {
      min_repeat = ParseInteger();
      max_repeat = min_repeat;
      if (Match(TokenType::kDotDot)) {
        max_repeat = ParseInteger();
      }
    } else if (Match(TokenType::kLBrace)) {
      min_repeat = Peek().type == TokenType::kComma ? 0 : ParseInteger();
      if (Match(TokenType::kComma)) {
        max_repeat = Peek().type == TokenType::kRBrace ? -1 : ParseInteger();
      } else {
        max_repeat = min_repeat;
      }
      Consume(TokenType::kRBrace, "expected '}' after repetition range");
    }

    if (min_repeat == -1) {
      return atom;
    }
    if (max_repeat != -1 && max_repeat < min_repeat) {
      RaiseLarkError(source_, location, "repetition end must be greater than or equal to start");
    }
    Node repeat;
    repeat.kind = Node::Kind::kRepeat;
    repeat.location = location;
    repeat.min_repeat = min_repeat;
    repeat.max_repeat = max_repeat;
    repeat.children.push_back(std::move(atom));
    return repeat;
  }

  Node ParseAtom() {
    Token token = Peek();
    if (Match(TokenType::kLParen)) {
      Node result = ParseChoice();
      Consume(TokenType::kRParen, "expected ')' after group");
      return result;
    }
    if (Match(TokenType::kLBracket)) {
      Node inner = ParseChoice();
      Consume(TokenType::kRBracket, "expected ']' after optional group");
      Node result;
      result.kind = Node::Kind::kRepeat;
      result.location = token.location;
      result.min_repeat = 0;
      result.max_repeat = 1;
      result.children.push_back(std::move(inner));
      return result;
    }
    if (Match(TokenType::kString)) {
      Node result = ParseStringNode(token);
      if (Match(TokenType::kDotDot)) {
        Token end = Consume(TokenType::kString, "expected string after '..'");
        Node end_node = ParseStringNode(end);
        if (!result.flags.empty()) {
          RaiseLarkError(source_, token.location, "flags are not allowed on character ranges");
        }
        if (!end_node.flags.empty()) {
          RaiseLarkError(source_, end.location, "flags are not allowed on character ranges");
        }
        Node range;
        range.kind = Node::Kind::kRange;
        range.location = token.location;
        range.text = result.text;
        range.text2 = end_node.text;
        return range;
      }
      return result;
    }
    if (Match(TokenType::kRegex)) {
      Node result;
      result.kind = Node::Kind::kRegex;
      result.location = token.location;
      result.text = token.text;
      result.flags = token.flags;
      return result;
    }
    if (Match(TokenType::kName)) {
      Node result;
      result.kind = Node::Kind::kName;
      result.location = token.location;
      result.text = NormalizeRuleName(token.text);
      if (Peek().type == TokenType::kDoubleColon) {
        RaiseLarkError(source_, Peek().location, "parametric grammar is not supported");
      }
      if (Peek().type == TokenType::kLBrace && Peek(1).type != TokenType::kComma &&
          Peek(1).type != TokenType::kNumber) {
        RaiseLarkError(source_, Peek().location, "Lark templates are not supported");
      }
      return result;
    }
    if (Match(TokenType::kJson)) {
      Node result;
      result.kind = Node::Kind::kJson;
      result.location = token.location;
      result.text = token.text;
      return result;
    }
    if (Match(TokenType::kRegexExt)) {
      Node result;
      result.kind = Node::Kind::kRegexExt;
      result.location = token.location;
      result.text = token.text;
      return result;
    }
    if (Match(TokenType::kSpecialToken)) {
      Node result;
      result.kind = Node::Kind::kSpecialToken;
      result.location = token.location;
      result.text = token.text;
      return result;
    }
    if (Match(TokenType::kGrammarRef)) {
      Node result;
      result.kind = Node::Kind::kGrammarRef;
      result.location = token.location;
      result.text = token.text;
      return result;
    }
    if (Match(TokenType::kLark)) {
      Consume(TokenType::kLBrace, "expected '{' after %lark");
      Node result;
      result.kind = Node::Kind::kNestedLark;
      result.location = token.location;
      result.nested = std::make_shared<Document>(ParseDocument(true));
      Consume(TokenType::kRBrace, "expected '}' after nested Lark grammar");
      return result;
    }
    if (token.type == TokenType::kUnsupportedDirective) {
      RaiseLarkError(source_, token.location, "directive " + token.text + " is not supported");
    }
    RaiseLarkError(source_, token.location, "expected grammar expression");
  }

  Node ParseStringNode(const Token& token) {
    std::string json_string = token.text;
    std::string flags;
    if (!json_string.empty() && json_string.back() == 'i') {
      flags = "i";
      json_string.pop_back();
    }
    picojson::value value;
    std::string error = picojson::parse(value, json_string);
    if (!error.empty() || !value.is<std::string>()) {
      RaiseLarkError(source_, token.location, "invalid string literal: " + error);
    }
    Node result;
    result.kind = Node::Kind::kString;
    result.location = token.location;
    result.text = value.get<std::string>();
    result.flags = std::move(flags);
    return result;
  }

  const std::string& source_;
  std::vector<Token> tokens_;
  size_t position_ = 0;
};

const std::unordered_map<std::string, std::string>& CommonRegexes() {
  static const std::unordered_map<std::string, std::string> regexes = {
      {"common.DIGIT", "[0-9]"},
      {"common.HEXDIGIT", "[a-fA-F0-9]"},
      {"common.INT", "[0-9]+"},
      {"common.SIGNED_INT", "(\\+|-)?[0-9]+"},
      {"common.DECIMAL", "([0-9]+\\.[0-9]*)|(\\.[0-9]+)"},
      {"common._EXP", "[eE](\\+|-)?[0-9]+"},
      {"common.FLOAT", "([0-9]+\\.[0-9]*|\\.[0-9]+)([eE](\\+|-)?[0-9]+)?|[0-9]+[eE](\\+|-)?[0-9]+"},
      {"common.SIGNED_FLOAT",
       "(\\+|-)?(([0-9]+\\.[0-9]*|\\.[0-9]+)([eE](\\+|-)?[0-9]+)?|[0-9]+[eE](\\+|-)?[0-9]+)"},
      {"common.NUMBER",
       "([0-9]+)|([0-9]+\\.[0-9]*|\\.[0-9]+)([eE](\\+|-)?[0-9]+)?|[0-9]+[eE](\\+|-)?[0-9]+"},
      {"common.SIGNED_NUMBER",
       "(\\+|-)?(([0-9]+)|([0-9]+\\.[0-9]*|\\.[0-9]+)([eE](\\+|-)?[0-9]+)?|[0-9]+[eE](\\+|-)?[0-9]+"
       ")"},
      {"common.ESCAPED_STRING", "\\\"([^\\\"\\\\]|\\\\.)*\\\""},
      {"common.LCASE_LETTER", "[a-z]"},
      {"common.UCASE_LETTER", "[A-Z]"},
      {"common.LETTER", "[A-Za-z]"},
      {"common.WORD", "[A-Za-z]+"},
      {"common.CNAME", "[_A-Za-z][_A-Za-z0-9]*"},
      {"common.WS_INLINE", "[ \\t]+"},
      {"common.WS", "[ \\t\\f\\r\\n]+"},
      {"common.CR", "\\r"},
      {"common.LF", "\\n"},
      {"common.NEWLINE", "(\\r?\\n)+"},
      {"common.SH_COMMENT", "#[^\\n]*"},
      {"common.CPP_COMMENT", "//[^\\n]*"},
      {"common.C_COMMENT", "\\/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*\\/"},
      {"common.SQL_COMMENT", "--[^\\n]*"},
  };
  return regexes;
}

std::string Trim(std::string value) {
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::string RewriteRegexDots(const std::string& pattern, bool dot_matches_newline) {
  if (dot_matches_newline) {
    return pattern;
  }
  std::string result;
  result.reserve(pattern.size());
  bool escaped = false;
  bool in_character_class = false;
  for (char c : pattern) {
    if (escaped) {
      result.push_back(c);
      escaped = false;
      continue;
    }
    if (c == '\\') {
      result.push_back(c);
      escaped = true;
    } else if (c == '[') {
      result.push_back(c);
      in_character_class = true;
    } else if (c == ']' && in_character_class) {
      result.push_back(c);
      in_character_class = false;
    } else if (c == '.' && !in_character_class) {
      result += "[^\\n]";
    } else {
      result.push_back(c);
    }
  }
  return result;
}

std::optional<std::string> ParseFixedRegexLiteral(const std::string& pattern) {
  std::string result;
  for (size_t i = 0; i < pattern.size();) {
    char c = pattern[i++];
    if (c != '\\') {
      if (std::string(".^$*+?()[]{}|").find(c) != std::string::npos) {
        return std::nullopt;
      }
      result.push_back(c);
      continue;
    }
    if (i == pattern.size()) {
      return std::nullopt;
    }
    char escaped = pattern[i++];
    switch (escaped) {
      case 'n':
        result.push_back('\n');
        break;
      case 'r':
        result.push_back('\r');
        break;
      case 't':
        result.push_back('\t');
        break;
      case 'f':
        result.push_back('\f');
        break;
      case 'v':
        result.push_back('\v');
        break;
      case '0':
        result.push_back('\0');
        break;
      case '^':
      case '$':
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
      case '\\':
      case '/':
      case '-':
        result.push_back(escaped);
        break;
      case 'x':
      case 'u': {
        bool braced = escaped == 'u' && i < pattern.size() && pattern[i] == '{';
        TCodepoint codepoint = 0;
        if (braced) {
          ++i;
          int digit_count = 0;
          while (i < pattern.size() && HexCharToInt(pattern[i]) != -1 && digit_count < 6) {
            codepoint = codepoint * 16 + HexCharToInt(pattern[i++]);
            ++digit_count;
          }
          if (digit_count == 0 || i >= pattern.size() || pattern[i++] != '}') {
            return std::nullopt;
          }
        } else {
          int digit_count = escaped == 'x' ? 2 : 4;
          if (i + static_cast<size_t>(digit_count) > pattern.size()) {
            return std::nullopt;
          }
          for (int digit = 0; digit < digit_count; ++digit) {
            int value = HexCharToInt(pattern[i++]);
            if (value == -1) {
              return std::nullopt;
            }
            codepoint = codepoint * 16 + value;
          }
        }
        if (codepoint > 0x10FFFF || (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
          return std::nullopt;
        }
        result += CharToUTF8(codepoint);
        break;
      }
      default:
        return std::nullopt;
    }
  }
  return result;
}

struct NamedGrammarRegistry {
  std::unordered_map<std::string, std::variant<Grammar, std::string>> inputs;
  std::unordered_map<std::string, Grammar> compiled;
  std::vector<std::string> active;
};

class LarkCompiler {
 public:
  LarkCompiler(
      const std::string& source,
      Document document,
      const std::optional<TokenizerInfo>& tokenizer_info,
      NamedGrammarRegistry& named_grammars
  )
      : source_(source),
        document_(std::move(document)),
        tokenizer_info_(tokenizer_info),
        named_grammars_(named_grammars) {}

  Grammar Compile() {
    ExpandImports();
    ParseOptions();
    IndexDefinitions();
    ValidateTerminalCycles();

    for (const auto& definition : document_.definitions) {
      rule_ids_[definition.name] = builder_.AddEmptyRule(definition.name);
    }

    for (const auto& definition : document_.definitions) {
      if (definition.is_terminal) {
        builder_.UpdateRuleBody(
            rule_ids_.at(definition.name), CompileNode(definition.body, definition.name, true)
        );
      }
    }

    CompileIgnore();

    const Definition& start_definition = *definition_by_name_.at("start");
    std::optional<int32_t> dynamic_start_body = CompileDynamicStart(start_definition);

    for (const auto& definition : document_.definitions) {
      if (definition.is_terminal) {
        continue;
      }
      if (dynamic_unused_rules_.count(definition.name)) {
        if (definition.max_tokens.has_value()) {
          RaiseLarkError(
              source_,
              definition.max_tokens_location,
              "max_tokens is not supported on rules consumed by dynamic dispatch"
          );
        }
        if (definition.capture_name.has_value()) {
          RaiseLarkError(
              source_,
              definition.capture_location,
              "capture is not supported on rules consumed by dynamic dispatch"
          );
        }
        builder_.UpdateRuleBody(rule_ids_.at(definition.name), builder_.AddEmptyStr());
        continue;
      }
      int32_t body_expr_id;
      if (definition.name == "start") {
        if (dynamic_start_body.has_value()) {
          if (definition.max_tokens.has_value()) {
            RaiseLarkError(
                source_,
                definition.max_tokens_location,
                "max_tokens is not supported on a dynamic dispatch start rule"
            );
          }
          body_expr_id = dynamic_start_body.value();
        } else if (definition.max_tokens.has_value()) {
          body_expr_id = CompileMaxTokensRule(definition);
        } else if (HasLazySemantics(definition)) {
          body_expr_id = CompileLazyRule(definition);
        } else {
          body_expr_id = CompileNode(definition.body, definition.name, false);
        }
        if (allow_initial_skip_ && skip_rule_id_ != -1) {
          body_expr_id = builder_.AddSequence({builder_.AddRuleRef(skip_rule_id_), body_expr_id});
        }
      } else if (definition.max_tokens.has_value()) {
        body_expr_id = CompileMaxTokensRule(definition);
      } else if (HasLazySemantics(definition)) {
        body_expr_id = CompileLazyRule(definition);
      } else {
        body_expr_id = CompileNode(definition.body, definition.name, false);
      }
      builder_.UpdateRuleBody(rule_ids_.at(definition.name), body_expr_id);
      if (definition.capture_name.has_value()) {
        builder_.UpdateCaptureName(rule_ids_.at(definition.name), definition.capture_name.value());
      }
    }

    auto start_it = rule_ids_.find("start");
    if (start_it == rule_ids_.end()) {
      RaiseLarkError(source_, {1, 1}, "no start rule found");
    }
    return DeadCodeEliminator::Apply(GrammarNormalizer().Apply(builder_.Get(start_it->second)));
  }

 private:
  struct SpecialTokenSet {
    bool excluded = false;
    std::vector<int32_t> token_ids;
  };

  struct Trigger {
    enum class Level { kString, kToken } level;
    std::string string;
    std::vector<int32_t> token_ids;
    Location location;
  };

  struct DynamicAlternative {
    Trigger trigger;
    Node remainder;
  };

  static bool HasLazySemantics(const Definition& definition) {
    return definition.lazy || definition.suffix.has_value();
  }

  void ExpandImports() {
    for (const auto& import : document_.imports) {
      auto it = CommonRegexes().find(import.path);
      if (it == CommonRegexes().end()) {
        RaiseLarkError(source_, import.location, "unknown common import '" + import.path + "'");
      }
      Node regex;
      regex.kind = Node::Kind::kRegex;
      regex.location = import.location;
      regex.text = it->second;
      Definition definition;
      definition.name = import.local_name;
      definition.is_terminal = true;
      definition.body = std::move(regex);
      definition.location = import.location;
      document_.definitions.push_back(std::move(definition));
    }
  }

  void ParseOptions() {
    for (const auto& [value, location] : document_.options) {
      if (!value.is<picojson::object>()) {
        RaiseLarkError(source_, location, "%grammar_options value must be an object");
      }
      for (const auto& [key, option] : value.get<picojson::object>()) {
        if (key == "allow_initial_skip") {
          if (!option.is<bool>()) {
            RaiseLarkError(source_, location, "allow_initial_skip must be a boolean");
          }
          allow_initial_skip_ = allow_initial_skip_ || option.get<bool>();
        } else if (key == "no_forcing" || key == "allow_invalid_utf8") {
          if (!option.is<bool>()) {
            RaiseLarkError(source_, location, key + " must be a boolean");
          }
          if (option.get<bool>()) {
            RaiseLarkError(
                source_, location, "%grammar_options option '" + key + "' is not supported"
            );
          }
        } else {
          RaiseLarkError(source_, location, "unknown %grammar_options option '" + key + "'");
        }
      }
    }
  }

  void IndexDefinitions() {
    for (auto& definition : document_.definitions) {
      if (definition_by_name_.count(definition.name)) {
        RaiseLarkError(
            source_, definition.location, "duplicate rule or terminal '" + definition.name + "'"
        );
      }
      definition_by_name_[definition.name] = &definition;
    }
    if (!definition_by_name_.count("start")) {
      RaiseLarkError(source_, {1, 1}, "no start rule found");
    }
    if (definition_by_name_.at("start")->is_terminal) {
      RaiseLarkError(source_, definition_by_name_.at("start")->location, "start must be a rule");
    }
  }

  void CollectReferencedNames(const Node& node, std::vector<std::string>* names) const {
    if (node.kind == Node::Kind::kName) {
      names->push_back(node.text);
    }
    for (const Node& child : node.children) {
      CollectReferencedNames(child, names);
    }
  }

  void ValidateTerminalCycles() {
    std::unordered_map<std::string, int> states;
    for (const auto& definition : document_.definitions) {
      if (definition.is_terminal && states[definition.name] == 0) {
        VisitTerminal(definition, &states);
      }
    }
  }

  void VisitTerminal(const Definition& definition, std::unordered_map<std::string, int>* states) {
    (*states)[definition.name] = 1;
    std::vector<std::string> names;
    CollectReferencedNames(definition.body, &names);
    for (const std::string& name : names) {
      auto it = definition_by_name_.find(name);
      if (it == definition_by_name_.end()) {
        RaiseLarkError(source_, definition.location, "unknown name '" + name + "'");
      }
      if (!it->second->is_terminal) {
        RaiseLarkError(
            source_,
            definition.location,
            "terminal '" + definition.name + "' cannot reference rule '" + name + "'"
        );
      }
      if ((*states)[name] == 1) {
        RaiseLarkError(
            source_, definition.location, "circular reference in terminal '" + name + "'"
        );
      }
      if ((*states)[name] == 0) {
        VisitTerminal(*it->second, states);
      }
    }
    (*states)[definition.name] = 2;
  }

  void CompileIgnore() {
    if (document_.ignores.empty()) {
      return;
    }
    std::vector<int32_t> ignore_choices;
    for (const Node& ignore : document_.ignores) {
      ignore_choices.push_back(CompileNode(ignore, "lark_ignore", true));
    }
    int32_t ignore_body =
        ignore_choices.size() == 1 ? ignore_choices[0] : builder_.AddChoices(ignore_choices);
    int32_t ignore_item_rule = builder_.AddRuleWithHint("lark_ignore_item", ignore_body);
    int32_t ignore_repeat = builder_.AddRepeat(ignore_item_rule, 0, -1);
    skip_rule_id_ = builder_.AddRuleWithHint("lark_ignore", ignore_repeat);
  }

  int32_t CompileStringLiteral(const Node& node) {
    if (node.flags.empty()) {
      return node.text.empty() ? builder_.AddEmptyStr() : builder_.AddByteString(node.text);
    }
    if (node.flags != "i") {
      RaiseLarkError(
          source_, node.location, "unsupported string literal flags '" + node.flags + "'"
      );
    }
    std::vector<TCodepoint> codepoints = ParseUTF8(node.text.c_str());
    if (!node.text.empty() &&
        (codepoints.empty() || codepoints[0] == CharHandlingError::kInvalidUTF8)) {
      RaiseLarkError(source_, node.location, "case-insensitive string is not valid UTF-8");
    }
    std::vector<int32_t> elements;
    elements.reserve(codepoints.size());
    for (TCodepoint codepoint : codepoints) {
      if (codepoint > 0x7F) {
        RaiseLarkError(
            source_,
            node.location,
            "case-insensitive string literals currently support ASCII characters only"
        );
      }
      if ((codepoint >= 'a' && codepoint <= 'z') || (codepoint >= 'A' && codepoint <= 'Z')) {
        TCodepoint lowercase =
            static_cast<TCodepoint>(std::tolower(static_cast<unsigned char>(codepoint)));
        TCodepoint uppercase =
            static_cast<TCodepoint>(std::toupper(static_cast<unsigned char>(codepoint)));
        elements.push_back(
            builder_.AddCharacterClass({{lowercase, lowercase}, {uppercase, uppercase}})
        );
      } else {
        elements.push_back(builder_.AddByteString(CharToUTF8(codepoint)));
      }
    }
    if (elements.empty()) {
      return builder_.AddEmptyStr();
    }
    return elements.size() == 1 ? elements[0] : builder_.AddSequence(elements);
  }

  std::string PrepareRegexPattern(const Node& node) const {
    if (node.flags.empty()) {
      return RewriteRegexDots(node.text, false);
    }
    if (node.flags == "s") {
      return RewriteRegexDots(node.text, true);
    }
    if (node.flags.find('l') != std::string::npos) {
      RaiseLarkError(source_, node.location, "regular-expression flag 'l' is not supported");
    }
    RaiseLarkError(
        source_, node.location, "only the regular-expression flag 's' is currently supported"
    );
  }

  const Grammar& ResolveNamedGrammar(const std::string& name, const Location& location) {
    auto input_it = named_grammars_.inputs.find(name);
    if (input_it == named_grammars_.inputs.end()) {
      RaiseLarkError(source_, location, "unknown named grammar '@" + name + "'");
    }
    if (std::holds_alternative<Grammar>(input_it->second)) {
      return std::get<Grammar>(input_it->second);
    }
    auto compiled_it = named_grammars_.compiled.find(name);
    if (compiled_it != named_grammars_.compiled.end()) {
      return compiled_it->second;
    }

    auto active_it = std::find(named_grammars_.active.begin(), named_grammars_.active.end(), name);
    if (active_it != named_grammars_.active.end()) {
      std::ostringstream cycle;
      for (auto it = active_it; it != named_grammars_.active.end(); ++it) {
        if (it != active_it) {
          cycle << " -> ";
        }
        cycle << "@" << *it;
      }
      cycle << " -> @" << name;
      RaiseLarkError(source_, location, "circular named grammar reference: " + cycle.str());
    }

    named_grammars_.active.push_back(name);
    try {
      const std::string& named_source = std::get<std::string>(input_it->second);
      auto tokens = LarkLexer(named_source).Tokenize();
      auto document = LarkParser(named_source, std::move(tokens)).Parse();
      Grammar compiled =
          LarkCompiler(named_source, std::move(document), tokenizer_info_, named_grammars_)
              .Compile();
      auto compiled_it = named_grammars_.compiled.emplace(name, std::move(compiled)).first;
      named_grammars_.active.pop_back();
      return compiled_it->second;
    } catch (const std::exception& error) {
      named_grammars_.active.pop_back();
      RaiseLarkError(
          source_,
          location,
          "failed to compile named grammar '@" + name + "': " + std::string(error.what())
      );
    }
  }

  int32_t CompileNode(const Node& node, const std::string& rule_hint, bool terminal_mode) {
    switch (node.kind) {
      case Node::Kind::kSequence: {
        if (node.children.empty()) {
          return builder_.AddEmptyStr();
        }
        std::vector<int32_t> elements;
        elements.reserve(node.children.size());
        for (const Node& child : node.children) {
          elements.push_back(CompileNode(child, rule_hint, terminal_mode));
        }
        return elements.size() == 1 ? elements[0] : builder_.AddSequence(elements);
      }
      case Node::Kind::kChoice: {
        std::vector<int32_t> choices;
        choices.reserve(node.children.size());
        for (const Node& child : node.children) {
          choices.push_back(CompileNode(child, rule_hint, terminal_mode));
        }
        return choices.size() == 1 ? choices[0] : builder_.AddChoices(choices);
      }
      case Node::Kind::kRepeat: {
        int32_t child = CompileNode(node.children[0], rule_hint + "_repeat", terminal_mode);
        return builder_.AddRepeatFromExpr(
            rule_hint + "_repeat", child, node.min_repeat, node.max_repeat
        );
      }
      case Node::Kind::kName: {
        auto definition_it = definition_by_name_.find(node.text);
        if (definition_it == definition_by_name_.end()) {
          RaiseLarkError(source_, node.location, "unknown name '" + node.text + "'");
        }
        if (terminal_mode && !definition_it->second->is_terminal) {
          RaiseLarkError(
              source_, node.location, "terminal cannot reference rule '" + node.text + "'"
          );
        }
        int32_t result = builder_.AddRuleRef(rule_ids_.at(node.text));
        // Lazy rules are compiled like terminals (lexemes), so they also take a trailing skip.
        bool skip_after =
            definition_it->second->is_terminal || HasLazySemantics(*definition_it->second);
        return !terminal_mode && skip_after ? AppendSkip(result) : result;
      }
      case Node::Kind::kString: {
        int32_t result = CompileStringLiteral(node);
        return !terminal_mode && !node.text.empty() ? AppendSkip(result) : result;
      }
      case Node::Kind::kRange: {
        auto begin = ParseUTF8(node.text.c_str());
        auto end = ParseUTF8(node.text2.c_str());
        if (begin.size() != 1 || end.size() != 1 || begin[0] == CharHandlingError::kInvalidUTF8 ||
            end[0] == CharHandlingError::kInvalidUTF8) {
          RaiseLarkError(source_, node.location, "character range endpoints must be one character");
        }
        if (begin[0] > end[0]) {
          RaiseLarkError(source_, node.location, "character range start must not exceed end");
        }
        int32_t result = builder_.AddCharacterClass({{begin[0], end[0]}});
        return terminal_mode ? result : AppendSkip(result);
      }
      case Node::Kind::kRegex: {
        std::string pattern = PrepareRegexPattern(node);
        try {
          int32_t root = SubGrammarAdder::Apply(&builder_, Grammar::FromRegex(pattern));
          int32_t result = builder_.AddRuleRef(root);
          return terminal_mode ? result : AppendSkip(result);
        } catch (const std::exception& error) {
          RaiseLarkError(
              source_,
              node.location,
              "failed to compile regular expression: " + std::string(error.what())
          );
        }
      }
      case Node::Kind::kJson: {
        if (terminal_mode) {
          RaiseLarkError(source_, node.location, "%json cannot be used in terminals");
        }
        try {
          int32_t root = SubGrammarAdder::Apply(&builder_, Grammar::FromJSONSchema(node.text));
          int32_t result = builder_.AddRuleRef(root);
          return terminal_mode ? result : AppendSkip(result);
        } catch (const std::exception& error) {
          RaiseLarkError(
              source_,
              node.location,
              "failed to compile inline JSON schema: " + std::string(error.what())
          );
        }
      }
      case Node::Kind::kNestedLark: {
        if (terminal_mode) {
          RaiseLarkError(source_, node.location, "nested %lark cannot be used in terminals");
        }
        try {
          LarkCompiler compiler(source_, *node.nested, tokenizer_info_, named_grammars_);
          int32_t root = SubGrammarAdder::Apply(&builder_, compiler.Compile());
          int32_t result = builder_.AddRuleRef(root);
          return terminal_mode ? result : AppendSkip(result);
        } catch (const std::exception& error) {
          RaiseLarkError(
              source_,
              node.location,
              "failed to compile nested Lark grammar: " + std::string(error.what())
          );
        }
      }
      case Node::Kind::kSpecialToken: {
        if (terminal_mode) {
          RaiseLarkError(source_, node.location, "special tokens cannot be used in terminals");
        }
        SpecialTokenSet token_set = ResolveSpecialToken(node.text, node.location);
        int32_t result = token_set.excluded ? builder_.AddExcludeTokenSet(token_set.token_ids)
                                            : builder_.AddTokenSet(token_set.token_ids);
        return AppendSkip(result);
      }
      case Node::Kind::kRegexExt:
        RaiseLarkError(source_, node.location, "structured %regex is not supported");
      case Node::Kind::kGrammarRef: {
        if (terminal_mode) {
          RaiseLarkError(source_, node.location, "named grammars cannot be used in terminals");
        }
        std::string name = node.text.substr(1);
        auto root_it = named_grammar_roots_.find(name);
        if (root_it == named_grammar_roots_.end()) {
          int32_t root =
              SubGrammarAdder::Apply(&builder_, ResolveNamedGrammar(name, node.location));
          root_it = named_grammar_roots_.emplace(name, root).first;
        }
        return AppendSkip(builder_.AddRuleRef(root_it->second));
      }
      case Node::Kind::kNot:
        RaiseLarkError(
            source_, node.location, "regular-expression complement '~' is not supported"
        );
    }
    RaiseLarkError(source_, node.location, "unsupported grammar node");
  }

  int32_t AppendSkip(int32_t expression) {
    if (skip_rule_id_ == -1) {
      return expression;
    }
    return builder_.AddSequence({expression, builder_.AddRuleRef(skip_rule_id_)});
  }

  /*!
   * \brief Compile a rule with the max_tokens attribute. The body compiles normally and the
   * budget is recorded on the rule; the matcher then bounds each occurrence, forcing it to
   * end at the earliest possible position once the budget is exhausted. Bodies that can end
   * at any position (such as arbitrary text) therefore never exceed the budget.
   */
  int32_t CompileMaxTokensRule(const Definition& definition) {
    if (definition.lazy || definition.suffix.has_value()) {
      RaiseLarkError(
          source_,
          definition.max_tokens_location,
          "max_tokens cannot be combined with lazy or suffix"
      );
    }
    if (!IsAnyText(definition.body)) {
      XGRAMMAR_LOG(WARNING) << "max_tokens on rule '" << definition.name
                            << "' is best-effort: the budget may be exceeded when the rule "
                               "cannot end at the position where it runs out.";
    }
    builder_.UpdateMaxTokens(rule_ids_.at(definition.name), definition.max_tokens.value());
    return CompileNode(definition.body, definition.name, false);
  }

  SpecialTokenSet ResolveSpecialToken(const std::string& token, const Location& location) const {
    if (token.size() >= 4 && token.substr(0, 2) == "<[" && token.substr(token.size() - 2) == "]>") {
      std::string contents = token.substr(2, token.size() - 4);
      SpecialTokenSet result;
      if (!contents.empty() && contents[0] == '^') {
        result.excluded = true;
        contents.erase(contents.begin());
      }
      if (contents == "*") {
        if (result.excluded) {
          RaiseLarkError(source_, location, "negated wildcard special token is not supported");
        }
        if (!tokenizer_info_.has_value()) {
          RaiseLarkError(source_, location, "wildcard special token requires tokenizer_info");
        }
        result.token_ids.reserve(tokenizer_info_->GetVocabSize());
        for (int32_t token_id = 0; token_id < tokenizer_info_->GetVocabSize(); ++token_id) {
          result.token_ids.push_back(token_id);
        }
        return result;
      }
      if (contents.find('*') != std::string::npos) {
        RaiseLarkError(source_, location, "wildcard cannot be mixed with token ranges");
      }
      size_t offset = 0;
      while (offset <= contents.size()) {
        size_t comma = contents.find(',', offset);
        std::string range = Trim(contents.substr(offset, comma - offset));
        if (!range.empty()) {
          size_t dash = range.find('-');
          if (dash != std::string::npos && range.find('-', dash + 1) != std::string::npos) {
            RaiseLarkError(
                source_, location, "invalid numeric special-token range '" + range + "'"
            );
          }
          int64_t first;
          int64_t last;
          try {
            auto parse_token_id = [](const std::string& value) {
              std::string trimmed = Trim(value);
              size_t parsed = 0;
              int64_t result = std::stoll(trimmed, &parsed);
              if (parsed != trimmed.size()) {
                throw std::invalid_argument("trailing characters");
              }
              return result;
            };
            first = parse_token_id(range.substr(0, dash));
            last = dash == std::string::npos ? first : parse_token_id(range.substr(dash + 1));
          } catch (const std::exception&) {
            RaiseLarkError(
                source_, location, "invalid numeric special-token range '" + range + "'"
            );
          }
          if (first < 0 || last < first || last > std::numeric_limits<int32_t>::max()) {
            RaiseLarkError(
                source_, location, "invalid numeric special-token range '" + range + "'"
            );
          }
          if (last - first > 1'000'000) {
            RaiseLarkError(source_, location, "special-token range is too large");
          }
          for (int64_t token_id = first; token_id <= last; ++token_id) {
            result.token_ids.push_back(static_cast<int32_t>(token_id));
          }
        }
        if (comma == std::string::npos) {
          break;
        }
        offset = comma + 1;
      }
      if (result.token_ids.empty()) {
        RaiseLarkError(source_, location, "empty numeric special-token range");
      }
      std::sort(result.token_ids.begin(), result.token_ids.end());
      result.token_ids.erase(
          std::unique(result.token_ids.begin(), result.token_ids.end()), result.token_ids.end()
      );
      return result;
    }

    if (!tokenizer_info_.has_value()) {
      RaiseLarkError(
          source_, location, "named special token " + token + " requires tokenizer_info"
      );
    }
    SpecialTokenSet result;
    const auto& decoded_vocab = tokenizer_info_->GetDecodedVocab();
    for (int32_t token_id = 0; token_id < static_cast<int32_t>(decoded_vocab.size()); ++token_id) {
      if (decoded_vocab[token_id] == token) {
        result.token_ids.push_back(token_id);
      }
    }
    if (result.token_ids.empty()) {
      RaiseLarkError(source_, location, "unknown special token " + token);
    }
    return result;
  }

  static const Node* UnwrapSingle(const Node* node) {
    while (node->kind == Node::Kind::kSequence && node->children.size() == 1) {
      node = &node->children[0];
    }
    return node;
  }

  bool IsAnyText(const Node& node, std::unordered_set<std::string>* visiting = nullptr) const {
    if (node.kind == Node::Kind::kRegex) {
      std::string pattern;
      for (char c : node.text) {
        if (c != ' ' && c != '\t' && c != '\r') {
          pattern.push_back(c);
        }
      }
      if (node.flags == "s") {
        return pattern == ".*";
      }
      if (!node.flags.empty()) {
        return false;
      }
      return pattern == "(.|\\n)*" || pattern == "(\\n|.)*" || pattern == "(?s:.*)" ||
             pattern == "(?:.|\\n)*" || pattern == "(?:\\n|.)*" || pattern == "[\\s\\S]*";
    }
    if (node.kind == Node::Kind::kSequence && node.children.size() == 1) {
      return IsAnyText(node.children[0], visiting);
    }
    if (node.kind == Node::Kind::kName) {
      std::unordered_set<std::string> local_visiting;
      if (visiting == nullptr) {
        visiting = &local_visiting;
      }
      if (visiting->count(node.text)) {
        return false;
      }
      auto it = definition_by_name_.find(node.text);
      if (it == definition_by_name_.end()) {
        return false;
      }
      visiting->insert(node.text);
      bool result = IsAnyText(it->second->body, visiting);
      visiting->erase(node.text);
      return result;
    }
    return false;
  }

  std::optional<Trigger> ExtractLazyRegexTrigger(const Node& node) const {
    if (node.kind != Node::Kind::kRegex) {
      return std::nullopt;
    }
    std::vector<std::string> prefixes;
    if (node.flags.empty()) {
      prefixes = {"(.|\\n)*", "(\\n|.)*", "(?:.|\\n)*", "(?:\\n|.)*", "[\\s\\S]*", "(?s:.*)"};
    } else if (node.flags == "s") {
      prefixes = {".*"};
    } else {
      return std::nullopt;
    }
    for (const std::string& prefix : prefixes) {
      if (node.text.size() <= prefix.size() || node.text.compare(0, prefix.size(), prefix) != 0) {
        continue;
      }
      auto trigger = ParseFixedRegexLiteral(node.text.substr(prefix.size()));
      if (trigger.has_value() && !trigger->empty()) {
        return Trigger{Trigger::Level::kString, std::move(trigger.value()), {}, node.location};
      }
    }
    return std::nullopt;
  }

  std::optional<Trigger> ExtractLazyTrigger(const Definition& definition) const {
    if (definition.suffix.has_value()) {
      if (!IsAnyText(definition.body)) {
        return std::nullopt;
      }
      return Trigger{
          Trigger::Level::kString, definition.suffix.value(), {}, definition.suffix_location
      };
    }
    if (!definition.lazy) {
      return std::nullopt;
    }
    const Node* body = UnwrapSingle(&definition.body);
    if (body->kind == Node::Kind::kRegex) {
      auto regex_trigger = ExtractLazyRegexTrigger(*body);
      if (regex_trigger.has_value()) {
        return regex_trigger;
      }
    }
    if (definition.body.kind != Node::Kind::kSequence || definition.body.children.size() != 2 ||
        !IsAnyText(definition.body.children[0])) {
      return std::nullopt;
    }
    const Node& trigger = definition.body.children[1];
    if (trigger.kind == Node::Kind::kString && !trigger.text.empty() && trigger.flags.empty()) {
      return Trigger{Trigger::Level::kString, trigger.text, {}, trigger.location};
    }
    if (trigger.kind == Node::Kind::kSpecialToken) {
      SpecialTokenSet token_set = ResolveSpecialToken(trigger.text, trigger.location);
      if (token_set.excluded) {
        RaiseLarkError(source_, trigger.location, "lazy special-token trigger cannot be negated");
      }
      return Trigger{Trigger::Level::kToken, "", token_set.token_ids, trigger.location};
    }
    return std::nullopt;
  }

  int32_t CompileLazyRule(const Definition& definition) {
    if (definition.suffix.has_value()) {
      RaiseLarkError(
          source_,
          definition.location,
          "suffix is only supported on an ANY_TEXT head used by dynamic dispatch"
      );
    }
    const Node* body = UnwrapSingle(&definition.body);
    if (body->kind == Node::Kind::kRegex && ExtractLazyRegexTrigger(*body).has_value()) {
      RaiseLarkError(
          source_,
          definition.location,
          "lazy regex suffix is only supported on a head used by dynamic dispatch"
      );
    }
    auto trigger = ExtractLazyTrigger(definition);
    if (!trigger.has_value()) {
      // General committed-shortest lazy rule: compiled like a terminal (no skip insertion);
      // the terminal-like requirement is validated after grammar optimization.
      builder_.UpdateLazy(rule_ids_.at(definition.name), true);
      return CompileNode(definition.body, definition.name, true);
    }
    int32_t empty_rule = builder_.AddRuleWithHint("lark_lazy_end", builder_.AddEmptyStr());
    int32_t result;
    if (trigger->level == Trigger::Level::kString) {
      result = builder_.AddTagDispatch({{{trigger->string, empty_rule}}, false, {}});
    } else {
      Grammar::Impl::TokenTagDispatch dispatch;
      for (int32_t token_id : trigger->token_ids) {
        dispatch.trigger_rule_pairs.push_back({token_id, empty_rule});
      }
      dispatch.loop_after_dispatch = false;
      result = builder_.AddTokenTagDispatch(dispatch);
    }
    return AppendSkip(result);
  }

  static std::vector<Node> FlattenSequence(const Node& node) {
    if (node.kind == Node::Kind::kSequence) {
      return node.children;
    }
    return {node};
  }

  std::optional<int32_t> CompileDynamicStart(const Definition& start) {
    std::unordered_set<std::string> unused_rules;
    std::vector<Node> start_elements = FlattenSequence(start.body);
    if (start_elements.size() != 2) {
      return std::nullopt;
    }
    const Node* loop = UnwrapSingle(&start_elements[0]);
    if (loop->kind != Node::Kind::kRepeat || loop->min_repeat != 0 || loop->max_repeat != -1) {
      return std::nullopt;
    }
    const Node* loop_body = UnwrapSingle(&loop->children[0]);
    std::vector<std::string> tool_names;
    if (loop_body->kind == Node::Kind::kChoice) {
      for (const Node& alternative : loop_body->children) {
        const Node* name = UnwrapSingle(&alternative);
        if (name->kind != Node::Kind::kName) {
          return std::nullopt;
        }
        tool_names.push_back(name->text);
      }
    } else if (loop_body->kind == Node::Kind::kName) {
      tool_names.push_back(loop_body->text);
    } else {
      return std::nullopt;
    }

    const Node* tail_name = UnwrapSingle(&start_elements[1]);
    if (tail_name->kind != Node::Kind::kName) {
      return std::nullopt;
    }
    auto tail_it = definition_by_name_.find(tail_name->text);
    if (tail_it == definition_by_name_.end() || !IsAnyText(tail_it->second->body)) {
      return std::nullopt;
    }
    unused_rules.insert(tail_name->text);

    std::vector<DynamicAlternative> alternatives;
    for (const std::string& tool_name : tool_names) {
      auto tool_it = definition_by_name_.find(tool_name);
      if (tool_it == definition_by_name_.end() || tool_it->second->is_terminal) {
        return std::nullopt;
      }
      unused_rules.insert(tool_name);
      std::vector<Node> tool_elements = FlattenSequence(tool_it->second->body);
      if (tool_elements.empty()) {
        return std::nullopt;
      }

      std::optional<Trigger> trigger;
      size_t remainder_begin = 0;
      const Node* first = UnwrapSingle(&tool_elements[0]);
      if (first->kind == Node::Kind::kName) {
        auto head_it = definition_by_name_.find(first->text);
        if (head_it != definition_by_name_.end()) {
          trigger = ExtractLazyTrigger(*head_it->second);
          if (trigger.has_value()) {
            unused_rules.insert(first->text);
            remainder_begin = 1;
          }
        }
      }
      if (!trigger.has_value() && tool_elements.size() >= 2 && IsAnyText(tool_elements[0])) {
        const Node* token_trigger = UnwrapSingle(&tool_elements[1]);
        if (token_trigger->kind == Node::Kind::kSpecialToken) {
          SpecialTokenSet token_set =
              ResolveSpecialToken(token_trigger->text, token_trigger->location);
          if (token_set.excluded) {
            RaiseLarkError(
                source_, token_trigger->location, "dynamic special-token trigger cannot be negated"
            );
          }
          trigger =
              Trigger{Trigger::Level::kToken, "", token_set.token_ids, token_trigger->location};
          remainder_begin = 2;
        }
      }
      if (!trigger.has_value()) {
        return std::nullopt;
      }

      Node remainder;
      remainder.kind = Node::Kind::kSequence;
      remainder.location = tool_it->second->location;
      remainder.children.assign(
          tool_elements.begin() + static_cast<std::ptrdiff_t>(remainder_begin), tool_elements.end()
      );
      alternatives.push_back({std::move(trigger.value()), std::move(remainder)});
    }

    if (alternatives.empty()) {
      return std::nullopt;
    }
    Trigger::Level level = alternatives[0].trigger.level;
    for (const auto& alternative : alternatives) {
      if (alternative.trigger.level != level) {
        RaiseLarkError(
            source_,
            start.location,
            "a dynamic Lark start rule cannot mix string and token triggers"
        );
      }
    }

    if (level == Trigger::Level::kString) {
      std::unordered_map<std::string, std::vector<const DynamicAlternative*>> grouped;
      std::vector<std::string> trigger_order;
      for (const auto& alternative : alternatives) {
        if (!grouped.count(alternative.trigger.string)) {
          trigger_order.push_back(alternative.trigger.string);
        }
        grouped[alternative.trigger.string].push_back(&alternative);
      }
      Grammar::Impl::TagDispatch dispatch;
      dispatch.loop_after_dispatch = true;
      for (const std::string& trigger : trigger_order) {
        std::vector<int32_t> remainder_choices;
        for (const DynamicAlternative* alternative : grouped.at(trigger)) {
          remainder_choices.push_back(
              CompileNode(alternative->remainder, "lark_dynamic_body", false)
          );
        }
        int32_t body = remainder_choices.size() == 1 ? remainder_choices[0]
                                                     : builder_.AddChoices(remainder_choices);
        int32_t body_rule = builder_.AddRuleWithHint("lark_dynamic_body", body);
        dispatch.tag_rule_pairs.push_back({trigger, body_rule});
      }
      dynamic_unused_rules_ = std::move(unused_rules);
      return builder_.AddTagDispatch(dispatch);
    }

    std::unordered_map<int32_t, std::vector<const DynamicAlternative*>> grouped;
    std::vector<int32_t> token_order;
    for (const auto& alternative : alternatives) {
      for (int32_t token_id : alternative.trigger.token_ids) {
        if (!grouped.count(token_id)) {
          token_order.push_back(token_id);
        }
        grouped[token_id].push_back(&alternative);
      }
    }
    Grammar::Impl::TokenTagDispatch dispatch;
    dispatch.loop_after_dispatch = true;
    for (int32_t token_id : token_order) {
      std::vector<int32_t> remainder_choices;
      for (const DynamicAlternative* alternative : grouped.at(token_id)) {
        remainder_choices.push_back(
            CompileNode(alternative->remainder, "lark_dynamic_token_body", false)
        );
      }
      int32_t body = remainder_choices.size() == 1 ? remainder_choices[0]
                                                   : builder_.AddChoices(remainder_choices);
      int32_t body_rule = builder_.AddRuleWithHint("lark_dynamic_token_body", body);
      dispatch.trigger_rule_pairs.push_back({token_id, body_rule});
    }
    dynamic_unused_rules_ = std::move(unused_rules);
    return builder_.AddTokenTagDispatch(dispatch);
  }

  const std::string& source_;
  Document document_;
  const std::optional<TokenizerInfo>& tokenizer_info_;
  NamedGrammarRegistry& named_grammars_;
  GrammarBuilder builder_;
  std::unordered_map<std::string, Definition*> definition_by_name_;
  std::unordered_map<std::string, int32_t> rule_ids_;
  std::unordered_map<std::string, int32_t> named_grammar_roots_;
  int32_t skip_rule_id_ = -1;
  bool allow_initial_skip_ = false;
  std::unordered_set<std::string> dynamic_unused_rules_;
};

}  // namespace

Grammar LarkToGrammar(
    const std::string& lark_string,
    const std::optional<TokenizerInfo>& tokenizer_info,
    const std::vector<NamedGrammar>& named_grammars
) {
  NamedGrammarRegistry named_grammar_registry;
  for (const auto& [name, grammar_or_source] : named_grammars) {
    if (name.empty()) {
      throw XGrammarError("Named grammar names must not be empty");
    }
    if (!std::all_of(name.begin(), name.end(), [](unsigned char character) {
          return std::isalnum(character) || character == '_' || character == '-';
        })) {
      throw XGrammarError(
          "Invalid named grammar name '" + name +
          "': names may contain only letters, digits, underscores, and hyphens"
      );
    }
    if (!named_grammar_registry.inputs.emplace(name, grammar_or_source).second) {
      throw XGrammarError("Duplicate named grammar '" + name + "'");
    }
  }
  auto tokens = LarkLexer(lark_string).Tokenize();
  auto document = LarkParser(lark_string, std::move(tokens)).Parse();
  return LarkCompiler(lark_string, std::move(document), tokenizer_info, named_grammar_registry)
      .Compile();
}

}  // namespace xgrammar
