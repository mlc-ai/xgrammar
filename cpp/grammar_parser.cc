/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_parser.cc
 */

#include "grammar_parser.h"

#include <picojson.h>

#include <variant>

#include "grammar_builder.h"
#include "grammar_data_structure.h"
#include "support/encoding.h"
#include "support/logging.h"

namespace xgrammar {

class EBNFLexer::Impl {
 public:
  using Token = EBNFLexer::Token;
  using TokenType = EBNFLexer::TokenType;

  std::vector<Token> Tokenize(const std::string& input);

 private:
  std::string input_;
  const char* cur_ = nullptr;
  int cur_line_ = 1;
  int cur_column_ = 1;

  constexpr static int64_t kMaxIntegerInGrammar = 1e15;

  // Helper functions
  Token NextToken();
  Token ParseIdentifierOrBooleanToken();
  Token ParseStringToken();
  Token ParseCharClassToken();
  Token ParseIntegerToken();
  [[noreturn]] void ReportLexerError(const std::string& msg, int line = -1, int column = -1);
  char Peek(int delta = 0) const;
  void Consume(int cnt = 1);
  void ConsumeSpace();
  std::string ParseIdentifier();
  void ConvertIdentifierToRuleName(std::vector<Token>* tokens);
  static bool IsNameChar(char c, bool is_first = false);
};

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
void EBNFLexer::Impl::ReportLexerError(const std::string& msg, int line, int column) {
  int line_to_print = line == -1 ? cur_line_ : line;
  int column_to_print = column == -1 ? cur_column_ : column;
  XGRAMMAR_LOG(FATAL) << "EBNF lexer error at line " + std::to_string(line_to_print) + ", column " +
                             std::to_string(column_to_print) + ": " + msg;
  XGRAMMAR_UNREACHABLE();
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
    ReportLexerError("Expect identifier");
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
        TokenType::Boolean,
        identifier,
        identifier == "true" ? true : false,
        start_line,
        start_column
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
  while (Peek() && Peek() != '"' && Peek() != '\n' && Peek() != '\r') {
    auto [codepoint, len] = ParseNextUTF8OrEscaped(cur_);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      ReportLexerError("Invalid UTF8 sequence");
    }
    if (codepoint == CharHandlingError::kInvalidEscape) {
      ReportLexerError("Invalid escape sequence");
    }
    Consume(len);
    codepoints.push_back(codepoint);
  }

  if (Peek() != '"') {
    ReportLexerError("Expect \" in string literal");
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

  std::vector<TCodepoint> codepoints;

  if (Peek() == '^') {
    codepoints.push_back(static_cast<TCodepoint>(static_cast<uint8_t>('^')));
    Consume();
  }

  // Parse character class content
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {{'-', '-'}, {']', ']'}};

  while (Peek() && Peek() != ']') {
    if (Peek() == '\r' || Peek() == '\n') {
      ReportLexerError("Character class should not contain newline");
    }

    auto [codepoint, len] = ParseNextUTF8OrEscaped(cur_, CUSTOM_ESCAPE_MAP);
    if (codepoint == CharHandlingError::kInvalidUTF8) {
      ReportLexerError("Invalid UTF8 sequence");
    }
    if (codepoint == CharHandlingError::kInvalidEscape) {
      ReportLexerError("Invalid escape sequence");
    }

    Consume(len);
    codepoints.push_back(codepoint);
  }

  if (!Peek()) {
    ReportLexerError("Unterminated character class");
  }

  Consume();  // Skip ']'

  // Extract original lexeme
  std::string lexeme(start_pos, cur_ - start_pos);
  return {TokenType::CharClass, lexeme, codepoints, start_line, start_column};
}

// Parse integer
EBNFLexer::Token EBNFLexer::Impl::ParseIntegerToken() {
  int start_line = cur_line_;
  int start_column = cur_column_;
  const char* start_pos = cur_;
  bool is_negative = false;

  if (Peek() == '-') {
    is_negative = true;
    Consume();
  } else if (Peek() == '+') {
    Consume();
  }

  int64_t num = 0;
  while (Peek() && isdigit(Peek())) {
    num = num * 10 + (Peek() - '0');
    Consume();
    if (num > kMaxIntegerInGrammar) {
      ReportLexerError(
          "Integer is too large: parsed " + std::to_string(num) + ", max allowed is " +
          std::to_string(kMaxIntegerInGrammar)
      );
    }
  }

  std::string lexeme(start_pos, cur_ - start_pos);
  return {TokenType::IntegerLiteral, lexeme, is_negative ? -num : num, start_line, start_column};
}

// Get the next token
EBNFLexer::Token EBNFLexer::Impl::NextToken() {
  ConsumeSpace();  // Skip whitespace and comments

  if (!Peek()) {
    return {TokenType::EndOfFile, "", "", cur_line_, cur_column_};
  }

  int start_line = cur_line_;
  int start_column = cur_column_;

  // Determine token type based on current character
  switch (Peek()) {
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
      ReportLexerError("Unexpected character: ':'");
      break;
    case '"':
      return ParseStringToken();
    case '[':
      return ParseCharClassToken();
    default:
      if (IsNameChar(*cur_, true)) {
        return ParseIdentifierOrBooleanToken();
      } else if (isdigit(*cur_) || *cur_ == '-' || *cur_ == '+') {
        return ParseIntegerToken();
      }

      // Unrecognized character, report error
      ReportLexerError("Unexpected character: " + std::string(1, *cur_));
  }

  // Should not reach here
  return {TokenType::EndOfFile, "", "", cur_line_, cur_column_};
}

void EBNFLexer::Impl::ConvertIdentifierToRuleName(std::vector<Token>* tokens) {
  for (int i = 0; i < static_cast<int>(tokens->size()); ++i) {
    if (tokens->at(i).type == TokenType::Assign) {
      if (i == 0) {
        ReportLexerError(
            "Assign should not be the first token", tokens->at(i).line, tokens->at(i).column
        );
      }
      if (tokens->at(i - 1).type != TokenType::Identifier) {
        ReportLexerError(
            "Assign should be preceded by an identifier",
            tokens->at(i - 1).line,
            tokens->at(i - 1).column
        );
      }
      if (i >= 2 && tokens->at(i - 2).line == tokens->at(i - 1).line) {
        ReportLexerError(
            "The rule name should be at the beginning of the line",
            tokens->at(i - 1).line,
            tokens->at(i - 1).column
        );
      }
      tokens->at(i - 1).type = TokenType::RuleName;
    }
  }
}

// Tokenize the entire input and return a vector of tokens
std::vector<EBNFLexer::Token> EBNFLexer::Impl::Tokenize(const std::string& input) {
  // Reset position to the beginning
  input_ = input;
  cur_ = input_.c_str();
  cur_line_ = 1;
  cur_column_ = 1;

  // Collect all tokens
  std::vector<Token> tokens;

  while (true) {
    Token token = NextToken();
    tokens.push_back(token);

    // Stop when we reach the end of file
    if (token.type == TokenType::EndOfFile) {
      break;
    }
  }

  ConvertIdentifierToRuleName(&tokens);

  return tokens;
}

EBNFLexer::EBNFLexer() : pimpl_(std::make_shared<Impl>()) {}

std::vector<EBNFLexer::Token> EBNFLexer::Tokenize(const std::string& input) {
  return pimpl_->Tokenize(input);
}

class EBNFParser {
 public:
  /*! \brief The logic of parsing the grammar string. */
  Grammar Parse(const std::vector<EBNFLexer::Token>& tokens, const std::string& root_rule_name);

 private:
  using Rule = Grammar::Impl::Rule;
  using RuleExprType = Grammar::Impl::RuleExprType;
  using Token = EBNFLexer::Token;
  using TokenType = EBNFLexer::TokenType;

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
  Rule ParseRule();

  // Parser for macro
  class MacroIR {
   public:
    struct StringNode;
    struct IntegerNode;
    struct BooleanNode;
    struct IdentifierNode;
    struct TupleNode;

    using Node = std::variant<StringNode, IntegerNode, BooleanNode, IdentifierNode, TupleNode>;
    using NodePtr = std::unique_ptr<Node>;

    struct StringNode {
      std::string value;
    };
    struct IntegerNode {
      int64_t value;
    };
    struct BooleanNode {
      bool value;
    };
    struct IdentifierNode {
      std::string name;
    };
    struct TupleNode {
      std::vector<NodePtr> elements;
    };

    struct Arguments {
      std::vector<NodePtr> arguments;
      std::unordered_map<std::string, NodePtr> named_arguments;
    };
  };
  MacroIR::Arguments ParseMacroArguments();
  MacroIR::NodePtr ParseMacroValue();

  int32_t ParseTagDispatch();

  // Helper functions

  // Helper for ParseElementWithQuantifier
  int32_t HandleStarQuantifier(int32_t rule_expr_id);
  int32_t HandlePlusQuantifier(int32_t rule_expr_id);
  int32_t HandleQuestionQuantifier(int32_t rule_expr_id);
  int32_t HandleRepetitionRange(int32_t rule_expr_id, int64_t lower, int64_t upper);

  // When parsing, we first find the names of all rules, and build the mapping from name to rule id.
  void InitRuleNames();

  // Consume a token and advance to the next
  void Consume(int cnt = 1);

  // Peek at the current token with optional offset
  const Token& Peek(int delta = 0) const;

  // Consume token if it matches expected type, otherwise report error
  void PeekAndConsume(TokenType type, const std::string& message);

  // Report a parsing error with the given message
  [[noreturn]] void ReportParseError(const std::string& msg, int delta_element = 0);

  // The grammar builder
  GrammarBuilder builder_;

  // The current token pointer
  const Token* current_token_ = nullptr;

  // Tokens from lexer
  std::vector<Token> tokens_;

  // The current rule name. Help to generate a name for a new rule.
  std::string cur_rule_name_;

  // The name of the root rule
  std::string root_rule_name_;

  static const std::unordered_map<std::string, std::function<int32_t(EBNFParser*)>> kMacroFunctions;
};

const std::unordered_map<std::string, std::function<int32_t(EBNFParser*)>>
    EBNFParser::kMacroFunctions = {
        {"TagDispatch", [](EBNFParser* parser) { return parser->ParseTagDispatch(); }},
};

const EBNFParser::Token& EBNFParser::Peek(int delta) const { return *(current_token_ + delta); }

void EBNFParser::Consume(int cnt) { current_token_ += cnt; }

void EBNFParser::PeekAndConsume(TokenType type, const std::string& message) {
  if (Peek().type != type) {
    ReportParseError(message);
  }
  Consume();
}

void EBNFParser::ReportParseError(const std::string& msg, int delta_element) {
  XGRAMMAR_DCHECK(current_token_ < tokens_.data() + tokens_.size());
  int line_to_print = Peek(delta_element).line;
  int column_to_print = Peek(delta_element).column;
  XGRAMMAR_LOG(FATAL) << "EBNF parser error at line " + std::to_string(line_to_print) +
                             ", column " + std::to_string(column_to_print) + ": " + msg;
  XGRAMMAR_UNREACHABLE();
}

std::string EBNFParser::ParseIdentifier(bool allow_empty) {
  if (Peek().type != TokenType::Identifier) {
    if (allow_empty) {
      return "";
    }
    ReportParseError("Expect identifier");
  }
  std::string identifier = Peek().lexeme;
  Consume();
  return identifier;
}

int32_t EBNFParser::ParseCharacterClass() {
  if (Peek().type != TokenType::CharClass) {
    ReportParseError("Expect character class");
  }

  std::vector<TCodepoint> codepoints = std::any_cast<std::vector<TCodepoint>>(Peek().value);
  Consume();

  std::vector<GrammarBuilder::CharacterClassElement> elements;

  // Check if the character class is negated (first codepoint is '^')
  bool is_negated = false;
  int start_idx = 0;
  if (!codepoints.empty() && codepoints[0] == static_cast<TCodepoint>(static_cast<uint8_t>('^'))) {
    is_negated = true;
    start_idx = 1;
  }

  // Process the codepoints to build character class elements
  for (int i = start_idx; i < static_cast<int>(codepoints.size()); i++) {
    // Check for range expression (a-z)
    if (i + 2 < static_cast<int>(codepoints.size()) &&
        codepoints[i + 1] == static_cast<TCodepoint>(static_cast<uint8_t>('-'))) {
      TCodepoint lower = codepoints[i];
      TCodepoint upper = codepoints[i + 2];

      if (lower > upper) {
        ReportParseError("Invalid character class: lower bound is larger than upper bound", -1);
      }

      elements.push_back({lower, upper});
      i += 2;  // Skip the hyphen and upper bound
    } else {
      // Single character
      elements.push_back({codepoints[i], codepoints[i]});
    }
  }

  return builder_.AddCharacterClass(elements, is_negated);
}

int32_t EBNFParser::ParseString() {
  if (Peek().type != TokenType::StringLiteral) {
    ReportParseError("Expect string literal");
  }

  std::string str_value = std::any_cast<std::string>(Peek().value);
  Consume();

  if (str_value.empty()) {
    return builder_.AddEmptyStr();
  }

  // Convert string to bytes
  std::vector<int32_t> bytes;
  for (auto c : str_value) {
    bytes.push_back(static_cast<int32_t>(static_cast<uint8_t>(c)));
  }
  return builder_.AddByteString(bytes);
}

int32_t EBNFParser::ParseRuleRef() {
  std::string name = ParseIdentifier();
  auto rule_id = builder_.GetRuleId(name);
  if (rule_id == -1) {
    ReportParseError("Rule \"" + name + "\" is not defined", -1);
  }
  return builder_.AddRuleRef(rule_id);
}

int32_t EBNFParser::ParseElement() {
  if (Peek().type == TokenType::LParen) {
    Consume();
    if (Peek().type == TokenType::RParen) {
      // Special case: ( )
      Consume();
      return builder_.AddEmptyStr();
    }
    auto rule_expr_id = ParseChoices();
    PeekAndConsume(TokenType::RParen, "Expect )");
    return rule_expr_id;
  } else if (Peek().type == TokenType::CharClass) {
    return ParseCharacterClass();
  } else if (Peek().type == TokenType::StringLiteral) {
    return ParseString();
  } else if (Peek().type == TokenType::Identifier) {
    if (kMacroFunctions.count(Peek().lexeme)) {
      return kMacroFunctions.at(Peek().lexeme)(this);
    } else {
      return ParseRuleRef();
    }
  } else {
    ReportParseError("Expect element, but got " + Peek().lexeme);
  }
}

int64_t EBNFParser::ParseInteger() {
  if (Peek().type != TokenType::IntegerLiteral) {
    ReportParseError("Expect integer, but got " + Peek().lexeme);
  }
  int64_t num = std::any_cast<int64_t>(Peek().value);
  Consume();
  return num;
}

std::pair<int64_t, int64_t> EBNFParser::ParseRepetitionRange() {
  PeekAndConsume(TokenType::LBrace, "Expect {");

  int64_t lower = ParseInteger();

  if (lower < 0) {
    ReportParseError("Lower bound cannot be negative", -1);
  }

  if (Peek().type == TokenType::Comma) {
    Consume();
    if (Peek().type == TokenType::RBrace) {
      Consume();
      return {lower, -1};
    }
    int64_t upper = ParseInteger();
    if (upper < lower) {
      ReportParseError(
          "Lower bound is larger than upper bound: " + std::to_string(lower) + " > " +
              std::to_string(upper),
          -1
      );
    }
    PeekAndConsume(TokenType::RBrace, "Expect }");
    return {lower, upper};
  } else if (Peek().type == TokenType::RBrace) {
    Consume();
    return {lower, lower};
  }

  ReportParseError("Expect ',' or '}' in repetition range");
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

  if (Peek().type == TokenType::Star) {
    Consume();
    return HandleStarQuantifier(rule_expr_id);
  } else if (Peek().type == TokenType::Plus) {
    Consume();
    return HandlePlusQuantifier(rule_expr_id);
  } else if (Peek().type == TokenType::Question) {
    Consume();
    return HandleQuestionQuantifier(rule_expr_id);
  } else if (Peek().type == TokenType::LBrace) {
    auto [lower, upper] = ParseRepetitionRange();
    return HandleRepetitionRange(rule_expr_id, lower, upper);
  }

  return rule_expr_id;
}

int32_t EBNFParser::ParseSequence() {
  std::vector<int32_t> elements;

  do {
    elements.push_back(ParseElementWithQuantifier());
  } while (Peek().type != TokenType::Pipe && Peek().type != TokenType::RParen &&
           Peek().type != TokenType::LookaheadLParen && Peek().type != TokenType::RuleName &&
           Peek().type != TokenType::EndOfFile);

  return builder_.AddSequence(elements);
}

int32_t EBNFParser::ParseChoices() {
  std::vector<int32_t> choices;

  choices.push_back(ParseSequence());

  while (Peek().type == TokenType::Pipe) {
    Consume();
    choices.push_back(ParseSequence());
  }

  return builder_.AddChoices(choices);
}

// Parse macro arguments and return a MacroIR::Arguments structure
EBNFParser::MacroIR::Arguments EBNFParser::ParseMacroArguments() {
  MacroIR::Arguments args;

  PeekAndConsume(TokenType::LParen, "Expect ( after macro function name");

  // Parse arguments
  if (Peek().type != TokenType::RParen) {
    while (true) {
      // Check if it's a named argument (identifier = value)
      if (Peek().type == TokenType::Identifier && Peek(1).type == TokenType::Equal) {
        std::string name = Peek().lexeme;
        Consume();  // Consume identifier
        Consume();  // Consume =

        // Parse the value
        args.named_arguments[name] = ParseMacroValue();
      } else {
        // Regular positional argument
        args.arguments.push_back(ParseMacroValue());
      }

      // Check for comma or end of arguments
      if (Peek().type == TokenType::Comma) {
        Consume();
      } else if (Peek().type == TokenType::RParen) {
        break;
      } else {
        ReportParseError("Expect , or ) in macro arguments");
      }
    }
  }

  PeekAndConsume(TokenType::RParen, "Expect ) after macro arguments");
  return args;
}

// Parse a single macro value (string, integer, boolean, or tuple)
EBNFParser::MacroIR::NodePtr EBNFParser::ParseMacroValue() {
  if (Peek().type == TokenType::StringLiteral) {
    // String value
    std::string value = std::any_cast<std::string>(Peek().value);
    Consume();
    return std::make_unique<MacroIR::Node>(MacroIR::StringNode{value});
  } else if (Peek().type == TokenType::IntegerLiteral) {
    // Integer value
    int64_t value = std::any_cast<int64_t>(Peek().value);
    Consume();
    return std::make_unique<MacroIR::Node>(MacroIR::IntegerNode{value});
  } else if (Peek().type == TokenType::Boolean) {
    // Boolean value
    bool value = std::any_cast<bool>(Peek().value);
    Consume();
    return std::make_unique<MacroIR::Node>(MacroIR::BooleanNode{value});
  } else if (Peek().type == TokenType::Identifier) {
    // Identifier value
    std::string name = Peek().lexeme;
    Consume();
    return std::make_unique<MacroIR::Node>(MacroIR::IdentifierNode{name});
  } else if (Peek().type == TokenType::LParen) {
    // Tuple value
    Consume();  // Consume (

    MacroIR::TupleNode tuple;

    // Parse tuple elements
    if (Peek().type != TokenType::RParen) {
      while (true) {
        tuple.elements.push_back(ParseMacroValue());

        if (Peek().type == TokenType::Comma) {
          Consume();
        } else if (Peek().type == TokenType::RParen) {
          break;
        } else {
          ReportParseError("Expect , or ) in tuple");
        }
      }
    }

    Consume();  // Consume )
    return std::make_unique<MacroIR::Node>(std::move(tuple));
  } else {
    ReportParseError("Expect string, integer, boolean, or tuple in macro argument");
  }
}

int32_t EBNFParser::ParseTagDispatch() {
  Consume();  // Consume TagDispatch operator
  auto start = current_token_;
  auto args = ParseMacroArguments();
  auto delta_element = start - current_token_;  // Used to report parse errors
  // Process the arguments for TagDispatch
  std::vector<std::pair<int32_t, int32_t>> tag_rule_pairs;

  // Process each argument in the form of ("tag", rule_name)
  for (const auto& arg : args.arguments) {
    auto tuple_node = std::get_if<MacroIR::TupleNode>(arg.get());
    if (tuple_node == nullptr) {
      ReportParseError("Each tag dispatch element must be a tuple", delta_element);
    }

    if (tuple_node->elements.size() != 2) {
      ReportParseError("Each tag dispatch element must be a pair (tag, rule)", delta_element);
    }

    // First element should be a string (tag)
    auto tag_str_node = std::get_if<MacroIR::StringNode>(tuple_node->elements[0].get());
    if (tag_str_node == nullptr || tag_str_node->value.empty()) {
      ReportParseError("Tag must be a non-empty string literal", delta_element);
    }
    auto tag_id = builder_.AddByteString(tag_str_node->value);

    // Second element should be an identifier (rule name)
    auto rule_name_node = std::get_if<MacroIR::IdentifierNode>(tuple_node->elements[1].get());
    if (rule_name_node == nullptr) {
      ReportParseError("Rule reference must be an identifier", delta_element);
    }

    auto rule_id = builder_.GetRuleId(rule_name_node->name);
    if (rule_id == -1) {
      ReportParseError("Rule \"" + rule_name_node->name + "\" is not defined", delta_element);
    }

    tag_rule_pairs.push_back({tag_id, rule_id});
  }

  return builder_.AddTagDispatch(tag_rule_pairs);
}

// std::pair<int32_t, int32_t> EBNFParser::ParseTagDispatchElement() {
//   PeekAndConsume(TokenType::LParen, "Expect ( in tag dispatch element");

//   // Parse tag (a string literal)
//   if (Peek().type != TokenType::StringLiteral) {
//     ReportParseError("Expect string literal for tag");
//   }
//   auto tag_id = ParseString();
//   if (builder_.GetRuleExpr(tag_id).type == RuleExprType::kEmptyStr) {
//     ReportParseError("Tag cannot be empty");
//   }

//   PeekAndConsume(TokenType::Comma, "Expect , in tag dispatch element");

//   // Parse rule name (should refer to a rule in the grammar)
//   std::string rule_name = ParseIdentifier(false);

//   // The rule cannot be the root rule and should be defined in the grammar
//   if (rule_name == root_rule_name_) {
//     ReportParseError("The root rule \"" + rule_name + "\" cannot be used as a tag");
//   }
//   auto rule_id = builder_.GetRuleId(rule_name);
//   if (rule_id == -1) {
//     ReportParseError("Rule \"" + rule_name + "\" is not defined");
//   }

//   PeekAndConsume(TokenType::RParen, "Expect ) in tag dispatch element");

//   return {tag_id, rule_id};
// }

// int32_t EBNFParser::ParseTagDispatchOrChoices() {
//   const Token* saved_token = current_token_;

//   if (Peek().type == TokenType::Identifier && Peek().lexeme == "TagDispatch") {
//     Consume();

//     // TODO(yixin): Make tagdispatch general
//     if (cur_rule_name_ != root_rule_name_) {
//       ReportParseError("TagDispatch should only be used in the root rule");
//     }

//     PeekAndConsume(TokenType::LParen, "Expect ( after TagDispatch");

//     std::vector<std::pair<int32_t, int32_t>> tag_dispatch_list;
//     while (true) {
//       auto tag_dispatch = ParseTagDispatchElement();
//       tag_dispatch_list.push_back(tag_dispatch);

//       if (Peek().type == TokenType::Comma) {
//         Consume();
//       } else if (Peek().type == TokenType::RParen) {
//         Consume();
//         break;
//       } else {
//         ReportParseError("Expect , or ) in macro function TagDispatch");
//       }
//     }

//     return builder_.AddTagDispatch(tag_dispatch_list);
//   }

//   // Reset token position if not TagDispatch
//   current_token_ = saved_token;
//   return ParseChoices();
// }

int32_t EBNFParser::ParseLookaheadAssertion() {
  PeekAndConsume(TokenType::LookaheadLParen, "Expect (= in lookahead assertion");
  auto result = ParseSequence();
  PeekAndConsume(TokenType::RParen, "Expect )");
  return result;
}

EBNFParser::Rule EBNFParser::ParseRule() {
  if (Peek().type != TokenType::RuleName) {
    ReportParseError("Expect rule name");
  }
  cur_rule_name_ = Peek().lexeme;
  Consume();

  PeekAndConsume(TokenType::Assign, "Expect ::=");

  auto body_id = ParseChoices();

  int32_t lookahead_id = -1;
  if (Peek().type == TokenType::LookaheadLParen) {
    lookahead_id = ParseLookaheadAssertion();
  }

  return {cur_rule_name_, body_id, lookahead_id};
}

void EBNFParser::InitRuleNames() {
  int delta_element = 0;
  for (auto& token : tokens_) {
    if (token.type == TokenType::RuleName) {
      auto name = std::any_cast<std::string>(token.value);
      if (builder_.GetRuleId(name) != -1) {
        ReportParseError("Rule \"" + name + "\" is defined multiple times", delta_element);
      }
      builder_.AddEmptyRule(name);
    }
    ++delta_element;
  }
  if (builder_.GetRuleId(root_rule_name_) == -1) {
    ReportParseError("The root rule with name \"" + root_rule_name_ + "\" is not found", 0);
  }
}

Grammar EBNFParser::Parse(
    const std::vector<EBNFLexer::Token>& tokens, const std::string& root_rule_name
) {
  tokens_ = tokens;
  current_token_ = tokens_.data();
  root_rule_name_ = root_rule_name;

  // First collect rule names
  InitRuleNames();

  // Then parse all the rules
  while (Peek().type != TokenType::EndOfFile) {
    auto new_rule = ParseRule();
    builder_.UpdateRuleBody(new_rule.name, new_rule.body_expr_id);
    builder_.UpdateLookaheadAssertion(new_rule.name, new_rule.lookahead_assertion_id);
  }

  return builder_.Get(root_rule_name);
}

Grammar ParseEBNF(const std::string& ebnf_string, const std::string& root_rule_name) {
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(ebnf_string);
  EBNFParser parser;
  return parser.Parse(std::move(tokens), root_rule_name);
}

}  // namespace xgrammar
