/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_parser.cc
 */

#include "grammar_parser.h"

#include <picojson.h>

#include "grammar_builder.h"
#include "grammar_data_structure.h"
#include "support/encoding.h"
#include "support/logging.h"

namespace xgrammar {

class EBNFLexer::Impl {
 public:
  using Token = EBNFLexer::Token;
  using TokenType = EBNFLexer::TokenType;

  explicit Impl(const std::string& input);
  std::vector<Token> Tokenize();

 private:
  const char* cur_ = nullptr;
  int cur_line_ = 1;
  int cur_column_ = 1;
  std::string input_;

  constexpr static int64_t kMaxIntegerInGrammar = 1e15;

  // Helper functions
  Token NextToken();
  Token ParseIdentifierOrBooleanToken();
  Token ParseStringToken();
  Token ParseCharClassToken();
  Token ParseIntegerToken();
  [[noreturn]] void ReportParseError(const std::string& msg);
  char Peek(int delta = 0) const;
  void Consume(int cnt = 1);
  void ConsumeSpace();
  std::string ParseIdentifier();
  static bool IsNameChar(char c, bool is_first = false);
};

// Constructor
EBNFLexer::Impl::Impl(const std::string& input) : input_(input) { cur_ = input_.c_str(); }

// Look at the next character
inline char EBNFLexer::Impl::Peek(int delta) const { return *(cur_ + delta); }

// Consume characters and update position information
inline void EBNFLexer::Impl::Consume(int cnt) {
  for (int i = 0; i < cnt; ++i) {
    // Newline\n \r \r\n
    if (*cur_ == '\n' || (*cur_ == '\r' && *(cur_ + 1) != '\n')) {
      ++cur_line_;
      cur_column_ = 1;
    } else {
      ++cur_column_;
    }
    ++cur_;
  }
}

// Skip whitespace and comments
void EBNFLexer::Impl::ConsumeSpace() {
  while (Peek() &&
         (Peek() == ' ' || Peek() == '\t' || Peek() == '#' || Peek() == '\n' || Peek() == '\r')) {
    Consume();
    if (Peek(-1) == '#') {
      while (Peek() && Peek() != '\n' && Peek() != '\r') {
        Consume();
      }
      if (!Peek()) {
        return;
      }
      Consume();
      if (Peek(-1) == '\r' && Peek() == '\n') {
        Consume();
      }
    }
  }
}

// Report parsing error
void EBNFLexer::Impl::ReportParseError(const std::string& msg) {
  XGRAMMAR_LOG(FATAL) << "EBNF lexer error at line " + std::to_string(cur_line_) + ", column " +
                             std::to_string(cur_column_) + ": " + msg;
}

// Check if a character can be part of an identifier
bool EBNFLexer::Impl::IsNameChar(char c, bool is_first) {
  return c == '_' || c == '-' || c == '.' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (!is_first && c >= '0' && c <= '9');
}

// Parse identifier
std::string EBNFLexer::Impl::ParseIdentifier() {
  const char* start = cur_;
  bool first_char = true;
  while (*cur_ && IsNameChar(*cur_, first_char)) {
    Consume();
    first_char = false;
  }
  if (start == cur_) {
    ReportParseError("Expect identifier");
  }
  return std::string(start, cur_ - start);
}

// Parse identifier or boolean value
EBNFLexer::Token EBNFLexer::Impl::ParseIdentifierOrBooleanToken() {
  int start_line = cur_line_;
  int start_column = cur_column_;

  std::string identifier = ParseIdentifier();

  // Check if it's a boolean value
  if (identifier == "true" || identifier == "false") {
    return {
        TokenType::Boolean, identifier, identifier == "true" ? "1" : "0", start_line, start_column
    };
  }

  // Otherwise it's an identifier
  return {TokenType::Identifier, identifier, identifier, start_line, start_column};
}

// Parse string literal
EBNFLexer::Token EBNFLexer::Impl::ParseStringToken() {
  int start_line = cur_line_;
  int start_column = cur_column_;
  const char* start_pos = cur_;

  Consume();  // Skip opening quote

  std::vector<int32_t> codepoints;
  while (*cur_ && *cur_ != '"' && *cur_ != '\n' && *cur_ != '\r') {
    auto [codepoint, len] = ParseNextUTF8OrEscaped(cur_);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      ReportParseError("Invalid UTF8 sequence");
    }
    if (codepoint == CharHandlingError::kInvalidEscape) {
      ReportParseError("Invalid escape sequence");
    }
    Consume(len);
    codepoints.push_back(codepoint);
  }

  if (*cur_ != '"') {
    ReportParseError("Expect \" in string literal");
  }
  Consume();  // Skip closing quote

  // Extract original lexeme
  std::string lexeme(start_pos, cur_ - start_pos);

  // Convert codepoints to UTF-8 string value
  std::string value;
  for (auto codepoint : codepoints) {
    value += PrintAsUTF8(codepoint);
  }

  return {TokenType::StringLiteral, lexeme, value, start_line, start_column};
}

// Parse character class.
EBNFLexer::Token EBNFLexer::Impl::ParseCharClassToken() {
  int start_line = cur_line_;
  int start_column = cur_column_;
  const char* start_pos = cur_;

  Consume();  // Skip '['

  if (*cur_ == '^') {
    Consume();
  }

  // Parse character class content
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {{'-', '-'}, {']', ']'}};

  while (*cur_ && *cur_ != ']') {
    if (*cur_ == '\r' || *cur_ == '\n') {
      ReportParseError("Character class should not contain newline");
    }

    auto [codepoint, len] = ParseNextUTF8OrEscaped(cur_, CUSTOM_ESCAPE_MAP);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      ReportParseError("Invalid UTF8 sequence");
    }
    if (codepoint == CharHandlingError::kInvalidEscape) {
      ReportParseError("Invalid escape sequence");
    }

    Consume(len);
  }

  if (!*cur_) {
    ReportParseError("Unterminated character class");
  }

  Consume();  // Skip ']'

  // Extract original lexeme
  std::string lexeme(start_pos, cur_ - start_pos);
  return {TokenType::CharClass, lexeme, lexeme, start_line, start_column};
}

// Parse integer
EBNFLexer::Token EBNFLexer::Impl::ParseIntegerToken() {
  int start_line = cur_line_;
  int start_column = cur_column_;
  const char* start_pos = cur_;

  int64_t num = 0;
  while (*cur_ && isdigit(*cur_)) {
    num = num * 10 + (*cur_ - '0');
    Consume();
    if (num > kMaxIntegerInGrammar) {
      ReportParseError(
          "Integer is too large: parsed " + std::to_string(num) + ", max allowed is " +
          std::to_string(kMaxIntegerInGrammar)
      );
    }
  }

  std::string lexeme(start_pos, cur_ - start_pos);
  std::string value = std::to_string(num);

  return {TokenType::IntegerLiteral, lexeme, value, start_line, start_column};
}

// Get the next token
EBNFLexer::Token EBNFLexer::Impl::NextToken() {
  ConsumeSpace();  // Skip whitespace and comments

  if (!*cur_) {
    return {TokenType::EndOfFile, "", "", cur_line_, cur_column_};
  }

  int start_line = cur_line_;
  int start_column = cur_column_;

  // Determine token type based on current character
  switch (*cur_) {
    case '(':
      if (Peek(1) == '=') {
        Consume(2);
        return {TokenType::LookaheadLParen, "(=", "", start_line, start_column};
      } else {
        Consume();
        return {TokenType::LParen, "(", "", start_line, start_column};
      }
    case ')':
      Consume();
      return {TokenType::RParen, ")", "", start_line, start_column};
    case '{':
      Consume();
      return {TokenType::LBrace, "{", "", start_line, start_column};
    case '}':
      Consume();
      return {TokenType::RBrace, "}", "", start_line, start_column};
    case '|':
      Consume();
      return {TokenType::Pipe, "|", "", start_line, start_column};
    case ',':
      Consume();
      return {TokenType::Comma, ",", "", start_line, start_column};
    case '*':
      Consume();
      return {TokenType::Star, "*", "", start_line, start_column};
    case '+':
      Consume();
      return {TokenType::Plus, "+", "", start_line, start_column};
    case '?':
      Consume();
      return {TokenType::Question, "?", "", start_line, start_column};
    case '=':
      Consume();
      return {TokenType::Equal, "=", "", start_line, start_column};
    case ':':
      if (Peek(1) == ':' && Peek(2) == '=') {
        Consume(3);
        return {TokenType::Assign, "::=", "", start_line, start_column};
      }
      ReportParseError("Unexpected character: ':'");
      break;
    case '"':
      return ParseStringToken();
    case '[':
      return ParseCharClassToken();
    default:
      if (IsNameChar(*cur_, true)) {
        return ParseIdentifierOrBooleanToken();
      } else if (isdigit(*cur_)) {
        return ParseIntegerToken();
      }

      // Unrecognized character, report error
      ReportParseError("Unexpected character: " + std::string(1, *cur_));
  }

  // Should not reach here
  return {TokenType::EndOfFile, "", "", cur_line_, cur_column_};
}

// Tokenize the entire input and return a vector of tokens
std::vector<EBNFLexer::Token> EBNFLexer::Impl::Tokenize() {
  std::vector<Token> tokens;

  // Reset position to the beginning
  cur_ = input_.c_str();
  cur_line_ = 1;
  cur_column_ = 1;

  // Collect all tokens
  while (true) {
    Token token = NextToken();
    tokens.push_back(token);

    // Stop when we reach the end of file
    if (token.type == TokenType::EndOfFile) {
      break;
    }
  }

  return tokens;
}

EBNFLexer::EBNFLexer(const std::string& input) : pimpl_(std::make_shared<Impl>(input)) {}

std::vector<EBNFLexer::Token> EBNFLexer::Tokenize() { return pimpl_->Tokenize(); }

class EBNFParser {
 public:
  /*! \brief The logic of parsing the grammar string. */
  Grammar Parse(const std::string& ebnf_string, const std::string& root_rule_name);

 private:
  using Rule = Grammar::Impl::Rule;
  using RuleExprType = Grammar::Impl::RuleExprType;

  // Parsing different parts of the grammar
  std::string ParseIdentifier(bool allow_empty = false);
  int32_t ParseCharacterClass();
  int32_t ParseString();
  int32_t ParseRuleRef();
  int32_t ParseElement();
  int64_t ParseInteger();
  std::pair<int64_t, int64_t> ParseRepetitionRange();
  int32_t ParseElementWithQuantifier();
  int32_t ParseLookaheadAssertion();
  int32_t ParseSequence();
  int32_t ParseChoices();
  std::pair<int32_t, int32_t> ParseTagDispatchElement();
  int32_t ParseTagDispatchOrChoices();
  Rule ParseRule();

  // Helper functions

  // Helper for ParseElementWithQuantifier
  int32_t HandleStarQuantifier(int32_t rule_expr_id);
  int32_t HandlePlusQuantifier(int32_t rule_expr_id);
  int32_t HandleQuestionQuantifier(int32_t rule_expr_id);
  int32_t HandleRepetitionRange(int32_t rule_expr_id, int64_t lower, int64_t upper);

  // When parsing, we first find the names of all rules, and build the mapping from name to rule id.
  void BuildRuleNameToId();
  // Consumes several spaces (newline, space, tab, comment, etc.)
  void ConsumeSpace(bool allow_newline = true);
  // Check the validity of a name
  static bool IsNameChar(TCodepoint c, bool first_char = false);
  // Reset the parser to the beginning of the string.
  void ResetStringIterator(const char* cur);

  // Consume a specified number of characters, and maintain the line and column number.
  void Consume(int cnt = 1) {
    for (int i = 0; i < cnt; ++i) {
      // \n \r \r\n
      if (Peek() == '\n' || (Peek() == '\r' && Peek(1) != '\n')) {
        ++cur_line_;
        cur_column_ = 1;
      } else {
        ++cur_column_;
      }
      ++cur_;
    }
  }

  // Peek the next character.
  char Peek(int delta = 0) const { return *(cur_ + delta); }

  // Report a parsing error with the given message and the line and column number.
  [[noreturn]] void ReportParseError(const std::string& msg) {
    XGRAMMAR_LOG(FATAL) << "EBNF parse error at line " + std::to_string(cur_line_) + ", column " +
                               std::to_string(cur_column_) + ": " + msg;
  }

  // The grammar builder
  GrammarBuilder builder_;
  // A pointer to the current parse position in the string
  const char* cur_ = nullptr;
  // The current line and column number
  int cur_line_ = 1;
  int cur_column_ = 1;
  // The current rule name. Help to generate a name for a new rule.
  std::string cur_rule_name_;
  // Whether the current element is in parentheses.
  // A sequence expression cannot contain newline, unless it is in parentheses.
  bool in_parentheses_ = false;

  // The name of the root rule
  std::string root_rule_name_;

  inline static constexpr int64_t MAX_INTEGER_IN_GRAMMAR = 1e10;
};

void EBNFParser::ConsumeSpace(bool allow_newline) {
  while (Peek() && (Peek() == ' ' || Peek() == '\t' || Peek() == '#' ||
                    (allow_newline && (Peek() == '\n' || Peek() == '\r')))) {
    Consume();
    if (Peek(-1) == '#') {
      auto start = cur_column_ - 1;
      while (Peek() && Peek() != '\n' && Peek() != '\r') {
        Consume();
      }
      if (!Peek() || start != 1 /* Reserve \n for inline comment */) {
        return;
      }
      Consume();
      if (Peek(-1) == '\r' && Peek() == '\n') {
        Consume();
      }
    }
  }
}

bool EBNFParser::IsNameChar(TCodepoint c, bool first_char) {
  return c == '_' || c == '-' || c == '.' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (!first_char && c >= '0' && c <= '9');
}

// name should be a char string (not a utf8 string)
std::string EBNFParser::ParseIdentifier(bool allow_empty) {
  auto start = cur_;
  bool first_char = true;
  while (Peek() && IsNameChar(Peek(), first_char)) {
    Consume();
    first_char = false;
  }
  if (start == cur_ && !allow_empty) {
    ReportParseError("Expect identifier");
  }
  return std::string(start, cur_);
}

// Character class:
// 1. Examples: [a-z] [ab] [a-zA-Z0-9] [^a-z] [测] [\u0123]
// 2. The "-" character is treated as a literal character if it is the last or the first (after
// the "^"", if present) character within the brackets. E.g. [a-] and [-a] means "a" or "-"
// 3. "-" and "]" should be escaped when used as a literal character:
// [\-] means -
// [\]] means ]
// Character class should not contain newlines.
int32_t EBNFParser::ParseCharacterClass() {
  static constexpr TCodepoint kUnknownUpperBound = -4;
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {{'-', '-'}, {']', ']'}};

  std::vector<GrammarBuilder::CharacterClassElement> elements;

  bool is_negated = false;
  if (Peek() == '^') {
    is_negated = true;
    Consume();
  }

  bool past_is_hyphen = false;
  bool past_is_single_char = false;
  while (Peek() && Peek() != ']') {
    if (Peek() == '\r' || Peek() == '\n') {
      ReportParseError("Character class should not contain newline");
    } else if (Peek() == '-' && Peek(1) != ']' && !past_is_hyphen && past_is_single_char) {
      Consume();
      past_is_hyphen = true;
      past_is_single_char = false;
      continue;
    }

    auto [codepoint, len] = ParseNextUTF8OrEscaped(cur_, CUSTOM_ESCAPE_MAP);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      ReportParseError("Invalid UTF8 sequence");
    }
    if (codepoint == CharHandlingError::kInvalidEscape) {
      ReportParseError("Invalid escape sequence");
    }
    Consume(len);
    if (past_is_hyphen) {
      XGRAMMAR_ICHECK(!elements.empty());
      if (elements.back().lower > codepoint) {
        ReportParseError("Invalid character class: lower bound is larger than upper bound");
      }
      elements.back().upper = codepoint;
      past_is_hyphen = false;
      XGRAMMAR_ICHECK(past_is_single_char == false);
    } else {
      elements.push_back({codepoint, kUnknownUpperBound});
      past_is_single_char = true;
    }
  }

  for (auto& element : elements) {
    if (element.upper == kUnknownUpperBound) {
      element.upper = element.lower;
    }
  }

  return builder_.AddCharacterClass(elements, is_negated);
}

// parse a c style string with utf8 support
int32_t EBNFParser::ParseString() {
  if (Peek() != '\"') {
    ReportParseError("Expect \" in string literal");
  }
  Consume();
  std::vector<int32_t> codepoints;
  while (Peek() && Peek() != '\"' && Peek() != '\n' && Peek() != '\r') {
    auto [codepoint, len] = ParseNextUTF8OrEscaped(cur_);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      ReportParseError("Invalid utf8 sequence");
    }
    if (codepoint == CharHandlingError::kInvalidEscape) {
      ReportParseError("Invalid escape sequence");
    }
    Consume(len);
    codepoints.push_back(codepoint);
  }
  if (Peek() != '\"') {
    ReportParseError("Expect \" in string literal");
  }
  Consume();

  if (codepoints.empty()) {
    return builder_.AddEmptyStr();
  }

  // convert codepoints to string
  std::string str;
  for (auto codepoint : codepoints) {
    str += PrintAsUTF8(codepoint);
  }
  // convert str to int32_t vector
  std::vector<int32_t> bytes;
  for (auto c : str) {
    bytes.push_back(static_cast<int32_t>(static_cast<uint8_t>(c)));
  }
  return builder_.AddByteString(bytes);
}

int32_t EBNFParser::ParseRuleRef() {
  std::string name = ParseIdentifier();
  auto rule_id = builder_.GetRuleId(name);
  if (rule_id == -1) {
    ReportParseError("Rule \"" + name + "\" is not defined");
  }
  return builder_.AddRuleRef(rule_id);
}

int32_t EBNFParser::ParseElement() {
  switch (Peek()) {
    case '(': {
      Consume();
      ConsumeSpace();
      if (Peek() == ')') {
        // Special case: ( )
        Consume();
        return builder_.AddEmptyStr();
      }
      auto prev_in_parentheses = in_parentheses_;
      in_parentheses_ = true;
      auto rule_expr_id = ParseChoices();
      ConsumeSpace();
      if (Peek() != ')') {
        ReportParseError("Expect )");
      }
      Consume();
      in_parentheses_ = prev_in_parentheses;
      return rule_expr_id;
    }
    case '[': {
      Consume();
      auto rule_expr_id = ParseCharacterClass();
      if (Peek() != ']') {
        ReportParseError("Expect ]");
      }
      Consume();
      return rule_expr_id;
    }
    case '\"': {
      return ParseString();
    }
    default: {
      if (IsNameChar(Peek(), true)) {
        return ParseRuleRef();
      }
      ReportParseError(
          "Expect element, but got character: " + std::string(1, static_cast<char>(Peek()))
      );
    }
  }
}

int32_t EBNFParser::HandleStarQuantifier(int32_t rule_expr_id) {
  Grammar::Impl::RuleExpr rule_expr = builder_.GetRuleExpr(rule_expr_id);
  if (rule_expr.type == GrammarBuilder::RuleExprType::kCharacterClass) {
    // We have special handling for character class star, e.g. [a-z]*
    rule_expr.type = GrammarBuilder::RuleExprType::kCharacterClassStar;
    // Copy rule expr because the grammar may change during insertion, and rule_expr is in the
    // grammar, so it may become invalid
    std::vector<int32_t> rule_expr_data(rule_expr.begin(), rule_expr.end());
    return builder_.AddRuleExpr({rule_expr.type, rule_expr_data.data(), rule_expr.data_len});
  } else {
    // For other star quantifiers, we transform it into a rule:
    // a*  -->  rule ::= a rule | ""
    auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
    auto new_rule_id = builder_.AddEmptyRule(new_rule_name);
    auto ref_to_new_rule = builder_.AddRuleRef(new_rule_id);
    auto new_rule_expr_id = builder_.AddChoices(
        {builder_.AddEmptyStr(), builder_.AddSequence({rule_expr_id, ref_to_new_rule})}
    );
    builder_.UpdateRuleBody(new_rule_id, new_rule_expr_id);

    // Return the reference to the new rule
    return builder_.AddRuleRef(new_rule_id);
  }
}

int32_t EBNFParser::HandlePlusQuantifier(int32_t rule_expr_id) {
  // a+  -->  rule ::= a rule | a
  auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
  auto new_rule_id = builder_.AddEmptyRule(new_rule_name);
  auto ref_to_new_rule = builder_.AddRuleRef(new_rule_id);
  auto new_rule_expr_id =
      builder_.AddChoices({builder_.AddSequence({rule_expr_id, ref_to_new_rule}), rule_expr_id});
  builder_.UpdateRuleBody(new_rule_id, new_rule_expr_id);

  // Return the reference to the new rule
  return builder_.AddRuleRef(new_rule_id);
}

int32_t EBNFParser::HandleQuestionQuantifier(int32_t rule_expr_id) {
  // a?  -->  rule ::= a | empty
  auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
  auto new_rule_expr_id = builder_.AddChoices({builder_.AddEmptyStr(), rule_expr_id});
  auto new_rule_id = builder_.AddRule({new_rule_name, new_rule_expr_id});
  return builder_.AddRuleRef(new_rule_id);
}

int64_t EBNFParser::ParseInteger() {
  if (!isdigit(Peek())) {
    ReportParseError("Expect integer");
  }
  int64_t num = 0;
  while (Peek() && isdigit(Peek())) {
    num = num * 10 + (Peek() - '0');
    Consume();
    if (num > MAX_INTEGER_IN_GRAMMAR) {
      ReportParseError(
          "Integer is too large: parsed " + std::to_string(num) + ", max allowed is " +
          std::to_string(MAX_INTEGER_IN_GRAMMAR)
      );
    }
  }
  return num;
}

// {x}: Match exactly x occurrences
// {x,}: Match at least x occurrences
// {x,y}: Match at least x occurrences, at most y occurrences
std::pair<int64_t, int64_t> EBNFParser::ParseRepetitionRange() {
  Consume();
  ConsumeSpace();
  int64_t lower = ParseInteger();
  ConsumeSpace();
  if (Peek() == ',') {
    Consume();
    ConsumeSpace();
    if (Peek() == '}') {
      Consume();
      return {lower, -1};
    }
    int64_t upper = ParseInteger();
    if (upper < lower) {
      ReportParseError(
          "Lower bound is larger than upper bound: " + std::to_string(lower) + " > " +
          std::to_string(upper)
      );
    }
    Consume();
    return {lower, upper};
  } else if (Peek() == '}') {
    Consume();
    return {lower, lower};
  }
  ReportParseError("Expect ',' or '}' in repetition range");
}

int32_t EBNFParser::HandleRepetitionRange(int32_t rule_expr_id, int64_t lower, int64_t upper) {
  // Construct expr expr ... expr (l times)
  std::vector<int32_t> elements;
  for (int64_t i = 0; i < lower; ++i) {
    elements.push_back(rule_expr_id);
  }

  // Case 1: {l}:
  // expr expr ... expr (l times)
  if (upper == lower) {
    return builder_.AddSequence(elements);
  }

  // Case 2: {l,}:
  // expr expr ... expr (l times) rest
  // rest ::= "" | expr rest
  if (upper == -1) {
    auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
    auto new_rule_id = builder_.AddEmptyRule(new_rule_name);
    auto ref_to_new_rule = builder_.AddRuleRef(new_rule_id);
    auto new_rule_expr_id = builder_.AddChoices(
        {builder_.AddEmptyStr(), builder_.AddSequence({rule_expr_id, ref_to_new_rule})}
    );
    builder_.UpdateRuleBody(new_rule_id, new_rule_expr_id);
    elements.push_back(builder_.AddRuleRef(new_rule_id));
    return builder_.AddSequence(elements);
  }

  // Case 3: {l, r} (r - l >= 1)
  // expr expr ... expr (l times) rest1
  // rest1 ::= "" | expr rest2
  // rest2 ::= "" | expr rest3
  // ...
  // rest(r - l) ::= "" | expr
  std::vector<int32_t> rest_rule_ids;

  for (int64_t i = 0; i < upper - lower; ++i) {
    auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
    rest_rule_ids.push_back(builder_.AddEmptyRule(new_rule_name));
  }
  for (int64_t i = 0; i < upper - lower - 1; ++i) {
    auto ref_to_next_rule = builder_.AddRuleRef(rest_rule_ids[i + 1]);
    auto new_rule_expr_id = builder_.AddChoices(
        {builder_.AddEmptyStr(), builder_.AddSequence({rule_expr_id, ref_to_next_rule})}
    );
    builder_.UpdateRuleBody(rest_rule_ids[i], new_rule_expr_id);
  }
  auto last_rule_expr_id = builder_.AddChoices({builder_.AddEmptyStr(), rule_expr_id});
  builder_.UpdateRuleBody(rest_rule_ids.back(), last_rule_expr_id);

  elements.push_back(builder_.AddRuleRef(rest_rule_ids[0]));
  return builder_.AddSequence(elements);
}

int32_t EBNFParser::ParseElementWithQuantifier() {
  int32_t rule_expr_id = ParseElement();
  ConsumeSpace(in_parentheses_);
  if (Peek() != '*' && Peek() != '+' && Peek() != '?' && Peek() != '{') {
    return rule_expr_id;
  }

  // Handle repetition range
  if (Peek() == '{') {
    auto [lower, upper] = ParseRepetitionRange();
    return HandleRepetitionRange(rule_expr_id, lower, upper);
  }

  // Handle quantifiers
  Consume();

  // We will transform a*, a+, a? into a rule, and return the reference to this rule
  switch (Peek(-1)) {
    case '*':
      // We assume that the star quantifier should be the body of some rule now
      return HandleStarQuantifier(rule_expr_id);
    case '+':
      return HandlePlusQuantifier(rule_expr_id);
    case '?':
      return HandleQuestionQuantifier(rule_expr_id);
    default:
      XGRAMMAR_LOG(FATAL) << "Unreachable";
  }
}

int32_t EBNFParser::ParseSequence() {
  std::vector<int32_t> elements;
  do {
    elements.push_back(ParseElementWithQuantifier());
    ConsumeSpace(in_parentheses_);
  } while (Peek() && Peek() != '|' && Peek() != ')' && Peek() != '\n' && Peek() != '\r' &&
           (Peek() != '(' || Peek(1) != '='));
  return builder_.AddSequence(elements);
}

int32_t EBNFParser::ParseChoices() {
  std::vector<int32_t> choices;

  choices.push_back(ParseSequence());
  ConsumeSpace();
  while (Peek() == '|') {
    Consume();
    ConsumeSpace();
    choices.push_back(ParseSequence());
    ConsumeSpace();
  }
  return builder_.AddChoices(choices);
}

// class MacroIR {
//  public:
//   struct Node;
//   using NodePtr = std::unique_ptr<Node>;
//   struct StringNode {
//     std::string str;
//   };
//   struct IntegerNode {
//     int64_t value;
//   };
//   struct BooleanNode {
//     bool value;
//   };
//   struct TupleNode {
//     std::vector<NodePtr> elements;
//   };
//   struct ArrayNode {
//     std::vector<NodePtr> elements;
//   };
//   using Node = std::variant<StringNode, IntegerNode, BooleanNode, TupleNode, ArrayNode>;

//   struct Arguments {
//     std::vector<NodePtr> arguments;
//     std::unordered_map<std::string, NodePtr> named_arguments;
//   };
// };

// std::pair<std::string, MacroIR::NodePtr> EBNFParser::ParseMacroArgument() {
//   if (Peek() == '[') {
//     ParseMacroArgumentArray();
//   }
// }

// MacroIR::Arguments EBNFParser::ParseMacro() {
//   ConsumeSpace();
//   if (Peek() != '(') {
//     ReportParseError("Expect ( in macro");
//   }
//   Consume();
//   ConsumeSpace();
//   MacroIR::Arguments args;
//   if (Peek() == ')') {
//     Consume();
//     return args;
//   }

//   while (true) {
//     auto& [name, value] = ParseMacroArgument();
//     if (name.empty()) {
//       args.arguments.push_back(value);
//     } else {
//       args.named_arguments[name] = value;
//     }
//     ConsumeSpace();
//     if (Peek() == ',') {
//       Consume();
//       ConsumeSpace();
//     } else if (Peek() == ')') {
//       Consume();
//       break;
//     } else {
//       ReportParseError("Expect , or ) in macro");
//     }
//   }
//   return args;
// }

std::pair<int32_t, int32_t> EBNFParser::ParseTagDispatchElement() {
  if (Peek() != '(') {
    ReportParseError("Expect ( in tag dispatch element");
  }
  Consume();
  ConsumeSpace();

  // Parse tag (a string literal)
  auto tag_id = ParseString();
  if (builder_.GetRuleExpr(tag_id).type == RuleExprType::kEmptyStr) {
    ReportParseError("Tag cannot be empty");
  }

  ConsumeSpace();
  if (Peek() != ',') {
    ReportParseError("Expect , in tag dispatch element");
  }
  Consume();
  ConsumeSpace();

  // Parse rule name (should refer to a rule in the grammar)
  auto rule_name = ParseIdentifier(false);

  // The rule cannot be the root rule and should be defined in the grammar
  if (rule_name == root_rule_name_) {
    ReportParseError("The root rule \"" + rule_name + "\" cannot be used as a tag");
  }
  auto rule_id = builder_.GetRuleId(rule_name);
  if (rule_id == -1) {
    ReportParseError("Rule \"" + rule_name + "\" is not defined");
  }

  ConsumeSpace();
  if (Peek() != ')') {
    ReportParseError("Expect ) in tag dispatch element");
  }
  Consume();

  return {tag_id, rule_id};
}

// TagDispatch:
// root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ...)
int32_t EBNFParser::ParseTagDispatchOrChoices() {
  auto prev_cursor = std::make_tuple(cur_, cur_line_, cur_column_, in_parentheses_);
  auto first_identifier = ParseIdentifier(true);
  if (first_identifier.empty() || first_identifier != "TagDispatch") {
    std::tie(cur_, cur_line_, cur_column_, in_parentheses_) = prev_cursor;
    return ParseChoices();
  }

  // TODO(yixin): Make tagdispatch general
  if (cur_rule_name_ != root_rule_name_) {
    ReportParseError("TagDispatch should only be used in the root rule");
  }

  ConsumeSpace();
  if (Peek() != '(') {
    ReportParseError("Expect ( after TagDispatch");
  }
  Consume();
  ConsumeSpace();
  std::vector<std::pair<int32_t, int32_t>> tag_dispatch_list;
  while (true) {
    auto tag_dispatch = ParseTagDispatchElement();
    tag_dispatch_list.push_back(tag_dispatch);
    ConsumeSpace();
    if (Peek() == ',') {
      Consume();
      ConsumeSpace();
    } else if (Peek() == ')') {
      Consume();
      break;
    } else {
      ReportParseError("Expect , or ) in macro function TagDispatch");
    }
  }
  return builder_.AddTagDispatch(tag_dispatch_list);
}

int32_t EBNFParser::ParseLookaheadAssertion() {
  if (Peek() != '(' || Peek(1) != '=') {
    return -1;
  }
  Consume(2);
  auto prev_in_parentheses = in_parentheses_;
  in_parentheses_ = true;
  ConsumeSpace(in_parentheses_);
  auto result = ParseSequence();
  ConsumeSpace(in_parentheses_);
  if (Peek() != ')') {
    ReportParseError("Expect )");
  }
  Consume();
  in_parentheses_ = prev_in_parentheses;
  return result;
}

EBNFParser::Rule EBNFParser::ParseRule() {
  std::string name = ParseIdentifier();
  cur_rule_name_ = name;
  ConsumeSpace();
  if (Peek() != ':' || Peek(1) != ':' || Peek(2) != '=') {
    ReportParseError("Expect ::=");
  }
  Consume(3);
  ConsumeSpace();
  auto body_id = ParseTagDispatchOrChoices();
  ConsumeSpace();
  auto lookahead_id = ParseLookaheadAssertion();
  return {name, body_id, lookahead_id};
}

void EBNFParser::BuildRuleNameToId() {
  ConsumeSpace();
  while (Peek()) {
    auto name = ParseIdentifier(true);
    ConsumeSpace(false);
    if (Peek() == ':' && Peek(1) == ':' && Peek(2) == '=') {
      if (name.empty()) {
        ReportParseError("Expect rule name");
      }
      Consume(3);
      if (builder_.GetRuleId(name) != -1) {
        ReportParseError("Rule \"" + name + "\" is defined multiple times");
      }
      builder_.AddEmptyRule(name);
    }
    while (Peek() && Peek() != '\n' && Peek() != '\r') {
      Consume();
    }
    ConsumeSpace();
  }
}

void EBNFParser::ResetStringIterator(const char* cur) {
  cur_ = cur;
  cur_line_ = 1;
  cur_column_ = 1;
  cur_rule_name_ = "";
  in_parentheses_ = false;
}

Grammar EBNFParser::Parse(const std::string& ebnf_string, const std::string& root_rule_name) {
  root_rule_name_ = root_rule_name;
  ResetStringIterator(ebnf_string.c_str());
  BuildRuleNameToId();

  ResetStringIterator(ebnf_string.c_str());
  ConsumeSpace();
  while (Peek()) {
    // Throw error when there are multiple lookahead assertions
    if (Peek() == '(' && Peek(1) == '=') {
      ReportParseError("Unexpected lookahead assertion");
    }
    auto new_rule = ParseRule();
    builder_.UpdateRuleBody(new_rule.name, new_rule.body_expr_id);
    // Update the lookahead assertion
    builder_.AddLookaheadAssertion(new_rule.name, new_rule.lookahead_assertion_id);

    ConsumeSpace();
  }

  // Check that the root rule is defined
  if (builder_.GetRuleId(root_rule_name) == -1) {
    ReportParseError("The root rule with name \"" + root_rule_name + "\" is not found.");
  }

  return builder_.Get(root_rule_name);
}

Grammar ParseEBNF(const std::string& ebnf_string, const std::string& root_rule_name) {
  EBNFParser parser;
  return parser.Parse(ebnf_string, root_rule_name);
}

}  // namespace xgrammar
