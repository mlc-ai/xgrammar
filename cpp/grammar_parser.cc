/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_parser.cc
 */

#include "grammar_parser.h"

#include <picojson.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "grammar_builder.h"
#include "grammar_impl.h"
#include "support/encoding.h"
#include "support/logging.h"
#include "xgrammar/grammar.h"

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

  /*!
   * \brief Consume a character sequence and return the next token. Return a token if it's a
   * single token, or a vector of tokens if it's a sequence of tokens.
   *
   * \return std::variant<Token, std::vector<Token>>
   */
  std::variant<Token, std::vector<Token>> NextToken();
  Token ParseIdentifierOrBooleanToken();
  Token ParseStringToken();
  std::vector<Token> ParseCharClassToken();
  Token ParseIntegerToken();
  [[noreturn]] void ReportLexerError(const std::string& msg, int line = -1, int column = -1);
  char Peek(int delta = 0) const;
  void Consume(int cnt = 1);
  void ConsumeSpace();
  std::string ParseIdentifierToken();
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
std::string EBNFLexer::Impl::ParseIdentifierToken() {
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

  std::string identifier = ParseIdentifierToken();

  // Check if it's a boolean value
  if (identifier == "true" || identifier == "false") {
    return {
        TokenType::BooleanLiteral,
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
    value += CharToUTF8(codepoint);
  }

  return {TokenType::StringLiteral, lexeme, value, start_line, start_column};
}

// Parse character class.
std::vector<EBNFLexer::Token> EBNFLexer::Impl::ParseCharClassToken() {
  std::vector<Token> tokens;

  tokens.push_back({TokenType::LBracket, "[", "", cur_line_, cur_column_});
  Consume();  // Skip '['

  if (Peek() == '^') {
    tokens.push_back({TokenType::Caret, "^", "", cur_line_, cur_column_});
    Consume();
  }

  static const std::unordered_map<char, TCodepoint> kRegexEscapeChars = {
      // clang-format off
      {'^', '^'}, {'$', '$'}, {'\\', '\\'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}, {'-', '-'}  // clang-format on
  };

  static const std::unordered_set<char> kRegexSpecialEscapes = {'d', 'D', 's', 'S', 'w', 'W'};

  while (Peek() && Peek() != ']') {
    if (Peek() == '\r' || Peek() == '\n') {
      ReportLexerError("Character class should not contain newline");
    } else if (Peek() == '-') {
      // Handle dash; this dash could be a range expression or a normal dash.
      // It will further be handled in EBNFParser::ParseCharClass.
      tokens.push_back({TokenType::Dash, "-", "", cur_line_, cur_column_});
      Consume();
    } else if (Peek() == '\\' && kRegexSpecialEscapes.count(Peek(1))) {
      // Handle escaped characters with special function
      tokens.push_back(
          {TokenType::EscapeInCharClass,
           std::string(cur_, cur_ + 2),
           std::string(cur_ + 1, cur_ + 2),
           cur_line_,
           cur_column_}
      );
      Consume(2);
    } else {
      // Handle normal characters
      auto [codepoint, len] = ParseNextUTF8OrEscaped(cur_, kRegexEscapeChars);
      if (codepoint == CharHandlingError::kInvalidUTF8) {
        ReportLexerError("Invalid UTF8 sequence");
      }

      if (codepoint == CharHandlingError::kInvalidEscape) {
        ReportLexerError("Invalid escape sequence" + std::string(cur_, cur_ + 2));
      }

      tokens.push_back(
          {TokenType::CharInCharClass,
           std::string(cur_, cur_ + len),
           codepoint,
           cur_line_,
           cur_column_}
      );
      Consume(len);
    }
  }

  if (!Peek()) {
    ReportLexerError("Unterminated character class");
  }

  tokens.push_back({TokenType::RBracket, "]", "", cur_line_, cur_column_});
  Consume();  // Skip ']'

  return tokens;
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
std::variant<EBNFLexer::Token, std::vector<EBNFLexer::Token>> EBNFLexer::Impl::NextToken() {
  ConsumeSpace();  // Skip whitespace and comments

  auto start_line = cur_line_;
  auto start_column = cur_column_;

  if (!Peek()) {
    return EBNFLexer::Token{TokenType::EndOfFile, "", "", start_line, start_column};
  }

  // Determine token type based on current character
  switch (Peek()) {
    case '(':
      if (Peek(1) == '=') {
        Consume(2);
        return EBNFLexer::Token{TokenType::LookaheadLParen, "(=", "", start_line, start_column};
      } else {
        Consume();
        return EBNFLexer::Token{TokenType::LParen, "(", "", start_line, start_column};
      }
    case ')':
      Consume();
      return EBNFLexer::Token{TokenType::RParen, ")", "", start_line, start_column};
    case '{':
      Consume();
      return EBNFLexer::Token{TokenType::LBrace, "{", "", start_line, start_column};
    case '}':
      Consume();
      return EBNFLexer::Token{TokenType::RBrace, "}", "", start_line, start_column};
    case '|':
      Consume();
      return EBNFLexer::Token{TokenType::Pipe, "|", "", start_line, start_column};
    case ',':
      Consume();
      return EBNFLexer::Token{TokenType::Comma, ",", "", start_line, start_column};
    case '*':
      Consume();
      return EBNFLexer::Token{TokenType::Star, "*", "", start_line, start_column};
    case '+':
      Consume();
      return EBNFLexer::Token{TokenType::Plus, "+", "", start_line, start_column};
    case '?':
      Consume();
      return EBNFLexer::Token{TokenType::Question, "?", "", start_line, start_column};
    case '=':
      Consume();
      return EBNFLexer::Token{TokenType::Equal, "=", "", start_line, start_column};
    case ':':
      if (Peek(1) == ':' && Peek(2) == '=') {
        Consume(3);
        return EBNFLexer::Token{TokenType::Assign, "::=", "", start_line, start_column};
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
  XGRAMMAR_UNREACHABLE();
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
    auto token = NextToken();

    if (auto* token_value = std::get_if<Token>(&token)) {
      tokens.push_back(*token_value);
      // Stop when we reach the end of file
      if (token_value->type == TokenType::EndOfFile) {
        break;
      }
    } else {
      auto vec = std::get_if<std::vector<Token>>(&token);
      XGRAMMAR_DCHECK(vec != nullptr);
      tokens.insert(tokens.end(), vec->begin(), vec->end());
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
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  using Token = EBNFLexer::Token;
  using TokenType = EBNFLexer::TokenType;

  // Parsing different parts of the grammar
  std::string ParseIdentifier();
  int32_t ParseCharClass();
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
  int32_t HandleStarQuantifier(int32_t grammar_expr_id);
  int32_t HandlePlusQuantifier(int32_t grammar_expr_id);
  int32_t HandleQuestionQuantifier(int32_t grammar_expr_id);
  int32_t HandleRepetitionRange(int32_t grammar_expr_id, int64_t lower, int64_t upper);
  int32_t LegacyHandleRepetitionRange(int32_t grammar_expr_id, int64_t lower, int64_t upper);

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
  XGRAMMAR_DCHECK(current_token_ + delta_element < tokens_.data() + tokens_.size());
  int line_to_print = Peek(delta_element).line;
  int column_to_print = Peek(delta_element).column;
  XGRAMMAR_LOG(FATAL) << "EBNF parser error at line " + std::to_string(line_to_print) +
                             ", column " + std::to_string(column_to_print) + ": " + msg;
  XGRAMMAR_UNREACHABLE();
}

std::string EBNFParser::ParseIdentifier() {
  if (Peek().type != TokenType::Identifier) {
    ReportParseError("Expect identifier");
  }
  std::string identifier = std::any_cast<std::string>(Peek().value);
  Consume();
  return identifier;
}

int32_t EBNFParser::ParseCharClass() {
  PeekAndConsume(TokenType::LBracket, "Expect [ in character class");

  std::vector<GrammarBuilder::CharacterClassElement> elements;
  bool is_negated = false;

  if (Peek().type == TokenType::Caret) {
    is_negated = true;
    Consume();
  }

  while (Peek().type != TokenType::RBracket && Peek().type != TokenType::EndOfFile) {
    if (Peek().type == TokenType::EscapeInCharClass) {
      ReportParseError("Character class escape is not supported yet in EBNF");
    }

    TCodepoint codepoint;
    if (Peek().type == TokenType::CharInCharClass) {
      codepoint = std::any_cast<TCodepoint>(Peek().value);
    } else if (Peek().type == TokenType::Dash) {
      codepoint = static_cast<TCodepoint>(static_cast<uint8_t>('-'));
    } else {
      ReportParseError("Unexpected character in character class: " + Peek().lexeme);
    }
    Consume();

    if (Peek().type == TokenType::Dash &&
        (Peek(1).type == TokenType::CharInCharClass || Peek(1).type == TokenType::Dash)) {
      // Range expression
      TCodepoint codepoint2;
      if (Peek(1).type == TokenType::CharInCharClass) {
        codepoint2 = std::any_cast<TCodepoint>(Peek(1).value);
      } else {
        XGRAMMAR_DCHECK(Peek(1).type == TokenType::Dash);
        codepoint2 = static_cast<TCodepoint>(static_cast<uint8_t>('-'));
      }

      if (codepoint > codepoint2) {
        ReportParseError("Invalid character class: lower bound is larger than upper bound", -1);
      }
      elements.push_back({codepoint, codepoint2});
      Consume(2);
    } else {
      // Single character
      elements.push_back({codepoint, codepoint});
    }
  }

  PeekAndConsume(TokenType::RBracket, "Expect ] in character class");

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

  return builder_.AddByteString(str_value);
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
    auto grammar_expr_id = ParseChoices();
    PeekAndConsume(TokenType::RParen, "Expect )");
    return grammar_expr_id;
  } else if (Peek().type == TokenType::LBracket) {
    return ParseCharClass();
  } else if (Peek().type == TokenType::StringLiteral) {
    return ParseString();
  } else if (Peek().type == TokenType::Identifier) {
    auto id = std::any_cast<std::string>(Peek().value);
    if (kMacroFunctions.count(id)) {
      return kMacroFunctions.at(id)(this);
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

int32_t EBNFParser::HandleStarQuantifier(int32_t grammar_expr_id) {
  Grammar::Impl::GrammarExpr grammar_expr = builder_.GetGrammarExpr(grammar_expr_id);
  if (grammar_expr.type == GrammarBuilder::GrammarExprType::kCharacterClass) {
    // We have special handling for character class star, e.g. [a-z]*
    grammar_expr.type = GrammarBuilder::GrammarExprType::kCharacterClassStar;
    // Copy grammar expr because the grammar may change during insertion, and grammar_expr is in the
    // grammar, so it may become invalid
    std::vector<int32_t> grammar_expr_data(grammar_expr.begin(), grammar_expr.end());
    return builder_.AddGrammarExpr(
        {grammar_expr.type, grammar_expr_data.data(), grammar_expr.data_len}
    );
  } else {
    // For other star quantifiers, we transform it into a rule:
    // a*  -->  rule ::= a rule | ""
    auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
    auto new_rule_id = builder_.AddEmptyRule(new_rule_name);
    auto ref_to_new_rule = builder_.AddRuleRef(new_rule_id);
    auto new_grammar_expr_id = builder_.AddChoices(
        {builder_.AddEmptyStr(), builder_.AddSequence({grammar_expr_id, ref_to_new_rule})}
    );
    builder_.UpdateRuleBody(new_rule_id, new_grammar_expr_id);

    // Return the reference to the new rule
    return builder_.AddRuleRef(new_rule_id);
  }
}

int32_t EBNFParser::HandlePlusQuantifier(int32_t grammar_expr_id) {
  // a+  -->  rule ::= a rule | a
  auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
  auto new_rule_id = builder_.AddEmptyRule(new_rule_name);
  auto ref_to_new_rule = builder_.AddRuleRef(new_rule_id);
  auto new_grammar_expr_id = builder_.AddChoices(
      {builder_.AddSequence({grammar_expr_id, ref_to_new_rule}), grammar_expr_id}
  );
  builder_.UpdateRuleBody(new_rule_id, new_grammar_expr_id);

  // Return the reference to the new rule
  return builder_.AddRuleRef(new_rule_id);
}

int32_t EBNFParser::HandleQuestionQuantifier(int32_t grammar_expr_id) {
  // a?  -->  rule ::= a | empty
  auto new_rule_name = builder_.GetNewRuleName(cur_rule_name_);
  auto new_grammar_expr_id = builder_.AddChoices({builder_.AddEmptyStr(), grammar_expr_id});
  auto new_rule_id = builder_.AddRule({new_rule_name, new_grammar_expr_id});
  return builder_.AddRuleRef(new_rule_id);
}

int32_t EBNFParser::LegacyHandleRepetitionRange(
    int32_t grammar_expr_id, int64_t lower, int64_t upper
) {
  // Construct expr expr ... expr (l times)

  std::vector<int32_t> elements;
  for (int64_t i = 0; i < lower; ++i) {
    elements.push_back(grammar_expr_id);
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
    auto new_grammar_expr_id = builder_.AddChoices(
        {builder_.AddEmptyStr(), builder_.AddSequence({grammar_expr_id, ref_to_new_rule})}
    );
    builder_.UpdateRuleBody(new_rule_id, new_grammar_expr_id);
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
    auto new_grammar_expr_id = builder_.AddChoices(
        {builder_.AddEmptyStr(), builder_.AddSequence({grammar_expr_id, ref_to_next_rule})}
    );
    builder_.UpdateRuleBody(rest_rule_ids[i], new_grammar_expr_id);
  }
  auto last_grammar_expr_id = builder_.AddChoices({builder_.AddEmptyStr(), grammar_expr_id});
  builder_.UpdateRuleBody(rest_rule_ids.back(), last_grammar_expr_id);

  elements.push_back(builder_.AddRuleRef(rest_rule_ids[0]));
  return builder_.AddSequence(elements);
}

int32_t EBNFParser::HandleRepetitionRange(
    const int32_t grammar_expr_id, int64_t lower, int64_t upper
) {
  static const int64_t kUnzipThreshold = 128;
  XGRAMMAR_DCHECK(lower >= 0);
  XGRAMMAR_DCHECK(upper == -1 || upper >= lower);
  // Case 1.1 small upper (<=threshold), unzip the repetition.
  // Case 1.2 unbounded upper, and lower is also small (<=threshold), unzip the lower part.
  if ((upper != -1 && upper <= kUnzipThreshold) || (upper == -1 && lower <= kUnzipThreshold)) {
    return LegacyHandleRepetitionRange(grammar_expr_id, lower, upper);
  }

  // Case 2. upper is unbounded, and lower is large (>threshold).
  // Or upper is bounded, but upper > threshold.

  // Case 2.1.1. lower is smaller than threshold, and upper is large. Transform {lower, upper} into:
  // {threshold, upper} | {lower, threshold}
  std::vector<int32_t> choices;
  if (lower < kUnzipThreshold) {
    choices.push_back(LegacyHandleRepetitionRange(grammar_expr_id, lower, kUnzipThreshold - 1));
    lower = kUnzipThreshold;
  }

  std::optional<int32_t> infinite_repetition_id = std::nullopt;
  std::vector<int32_t> repeated_sequence;
  // Now, we transform {lower, upper} into {max{threshold, lower}, upper}.
  // Case 2.2 upper is unbounded. We will transform it into {lower} {0, inf}.
  if (upper == -1) {
    const auto& rule_expr = builder_.GetGrammarExpr(grammar_expr_id);
    if (rule_expr.type == GrammarBuilder::GrammarExprType::kCharacterClass) {
      std::vector<GrammarBuilder::CharacterClassElement> character_ranges;
      bool is_negative = rule_expr[0];
      for (int i = 1; i < static_cast<int>(rule_expr.size()); i += 2) {
        character_ranges.push_back({rule_expr[i], rule_expr[i + 1]});
      }
      infinite_repetition_id = builder_.AddCharacterClassStar(character_ranges, is_negative);
    } else {
      const auto& unbounded_rule_id =
          builder_.AddEmptyRule(builder_.GetNewRuleName(cur_rule_name_ + "_repeat_inf"));
      int recursion_sequence =
          builder_.AddSequence({grammar_expr_id, builder_.AddRuleRef(unbounded_rule_id)});
      int recursion_choice = builder_.AddChoices({builder_.AddEmptyStr(), recursion_sequence});
      builder_.UpdateRuleBody(unbounded_rule_id, recursion_choice);
      infinite_repetition_id = builder_.AddRuleRef(unbounded_rule_id);
    }
    upper = lower;
  }

  // Handle the {lower, upper} part, where threshold <= lower <= upper.
  const auto repeat_name = cur_rule_name_ + "_repeat_1";
  XGRAMMAR_DCHECK(lower >= kUnzipThreshold && upper >= lower);

  // If we have infinite repetition part, add it to the sequence.
  if (infinite_repetition_id.has_value()) {
    repeated_sequence.push_back(infinite_repetition_id.value());
  }

  // The repetition body.
  if (upper != kUnzipThreshold) {
    XGRAMMAR_DCHECK(upper > kUnzipThreshold);
    auto new_grammar_expr_id = builder_.AddChoices({builder_.AddSequence({grammar_expr_id})});
    auto new_rule_id = builder_.AddRuleWithHint(repeat_name, new_grammar_expr_id);
    auto new_repeated_ref_rule_expr = builder_.AddChoices({builder_.AddSequence(
        {builder_.AddRepeat(new_rule_id, lower - kUnzipThreshold, upper - kUnzipThreshold)}
    )});
    auto new_repeated_rule_id =
        builder_.AddRuleWithHint(repeat_name + "_inner", new_repeated_ref_rule_expr);
    repeated_sequence.push_back(builder_.AddRuleRef(new_repeated_rule_id));
    std::vector<int32_t> repetition_lookahead(kUnzipThreshold, grammar_expr_id);
    builder_.UpdateLookaheadAssertion(new_rule_id, builder_.AddSequence(repetition_lookahead));
  }

  // Add the last threshold grammar_expr_id to the sequence.
  for (int i = 0; i < kUnzipThreshold; ++i) {
    repeated_sequence.push_back(grammar_expr_id);
  }

  // Add the sequence to choices.
  choices.push_back(builder_.AddSequence(repeated_sequence));
  return builder_.AddChoices(choices);
}

int32_t EBNFParser::ParseElementWithQuantifier() {
  int32_t grammar_expr_id = ParseElement();

  if (Peek().type == TokenType::Star) {
    Consume();
    return HandleStarQuantifier(grammar_expr_id);
  } else if (Peek().type == TokenType::Plus) {
    Consume();
    return HandlePlusQuantifier(grammar_expr_id);
  } else if (Peek().type == TokenType::Question) {
    Consume();
    return HandleQuestionQuantifier(grammar_expr_id);
  } else if (Peek().type == TokenType::LBrace) {
    auto [lower, upper] = ParseRepetitionRange();
    return HandleRepetitionRange(grammar_expr_id, lower, upper);
  }

  return grammar_expr_id;
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
        std::string name = std::any_cast<std::string>(Peek().value);
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
  } else if (Peek().type == TokenType::BooleanLiteral) {
    // Boolean value
    bool value = std::any_cast<bool>(Peek().value);
    Consume();
    return std::make_unique<MacroIR::Node>(MacroIR::BooleanNode{value});
  } else if (Peek().type == TokenType::Identifier) {
    // Identifier value
    std::string name = std::any_cast<std::string>(Peek().value);
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

  Grammar::Impl::TagDispatch tag_dispatch;

  // Position parameters: ("tag", rule_name)
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
    // Second element should be an identifier (rule name)
    auto rule_name_node = std::get_if<MacroIR::IdentifierNode>(tuple_node->elements[1].get());
    if (rule_name_node == nullptr) {
      ReportParseError("Rule reference must be an identifier", delta_element);
    }

    auto rule_id = builder_.GetRuleId(rule_name_node->name);
    if (rule_id == -1) {
      ReportParseError("Rule \"" + rule_name_node->name + "\" is not defined", delta_element);
    }

    tag_dispatch.tag_rule_pairs.push_back({tag_str_node->value, rule_id});
  }

  // stop_eos
  tag_dispatch.stop_eos = true;
  if (auto it = args.named_arguments.find("stop_eos"); it != args.named_arguments.end()) {
    auto bool_node = std::get_if<MacroIR::BooleanNode>(it->second.get());
    if (bool_node == nullptr) {
      ReportParseError("stop_eos must be a boolean literal", delta_element);
    }
    tag_dispatch.stop_eos = bool_node->value;
  }

  // stop_str
  if (auto it = args.named_arguments.find("stop_str"); it != args.named_arguments.end()) {
    auto tuple_node = std::get_if<MacroIR::TupleNode>(it->second.get());
    if (tuple_node == nullptr) {
      ReportParseError("Stop strings must be a tuple", delta_element);
    }

    for (const auto& element : tuple_node->elements) {
      auto stop_str_node = std::get_if<MacroIR::StringNode>(element.get());
      if (stop_str_node == nullptr || stop_str_node->value.empty()) {
        ReportParseError("Stop string must be a non-empty string literal", delta_element);
      }
      tag_dispatch.stop_str.push_back(stop_str_node->value);
    }
  }

  // loop_after_dispatch
  tag_dispatch.loop_after_dispatch = true;
  if (auto it = args.named_arguments.find("loop_after_dispatch");
      it != args.named_arguments.end()) {
    auto bool_node = std::get_if<MacroIR::BooleanNode>(it->second.get());
    if (bool_node == nullptr) {
      ReportParseError("loop_after_dispatch must be a boolean literal", delta_element);
    }
    tag_dispatch.loop_after_dispatch = bool_node->value;
  }

  // Well formed check
  if (!tag_dispatch.stop_eos && tag_dispatch.stop_str.empty()) {
    ReportParseError(
        "The TagDispatch must have stop_eos=true or stop_str is not empty", delta_element
    );
  }

  return builder_.AddTagDispatch(tag_dispatch);
}

int32_t EBNFParser::ParseLookaheadAssertion() {
  PeekAndConsume(TokenType::LookaheadLParen, "Expect (= in lookahead assertion");
  auto result = ParseChoices();
  PeekAndConsume(TokenType::RParen, "Expect )");
  return result;
}

EBNFParser::Rule EBNFParser::ParseRule() {
  if (Peek().type != TokenType::RuleName) {
    ReportParseError("Expect rule name");
  }
  cur_rule_name_ = std::any_cast<std::string>(Peek().value);
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
