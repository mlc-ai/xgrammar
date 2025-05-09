/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_parser.h
 * \brief The header for the parser of BNF/EBNF grammar into BNF AST.
 */

#ifndef XGRAMMAR_GRAMMAR_PARSER_H_
#define XGRAMMAR_GRAMMAR_PARSER_H_

#include <xgrammar/xgrammar.h>

namespace xgrammar {

class EBNFLexer {
 public:
  // Token types
  enum class TokenType {
    Identifier,      // e.g.: root, rule1, TagDispatch
    StringLiteral,   // e.g.: "tag1", "hello"
    IntegerLiteral,  // 123
    CharClass,       // Character class, e.g.: [a-z], [0-9], [^a-z]
    LParen,          // (
    RParen,          // )
    LBrace,          // {
    RBrace,          // }
    Pipe,            // |
    Comma,           // ,
    EndOfFile,       // End of file

    // Boolean type
    Boolean,  // true, false

    // Symbols and quantifiers
    Assign,    // ::=
    Equal,     // =
    Star,      // *
    Plus,      // +
    Question,  // ?

    // Special structures
    LookaheadLParen,  // (=
  };

  // Token structure
  struct Token {
    TokenType type;
    std::string lexeme;  // original text
    std::string value;   // processed value
    int line;
    int column;
  };

  explicit EBNFLexer(const std::string& input);
  std::vector<Token> Tokenize();

  XGRAMMAR_DEFINE_PIMPL_METHODS(EBNFLexer);
};

/*!
 * \brief This class parses a BNF/EBNF grammar string into an BNF abstract syntax tree (AST).
 * \details This function accepts the EBNF notation defined in the W3C XML Specification
 * (https://www.w3.org/TR/xml/#sec-notation), which is a popular standard, with the following
 * changes:
 * - Using # as comment mark instead of C-style comments
 * - Accept C-style unicode escape sequence \u01AB, \U000001AB, \xAB instead of #x0123
 * - Rule A-B (match A and not match B) is not supported yet
 *
 * See tests/python/serve/json.ebnf for an example.
 * \param ebnf_string The grammar string.
 * \param root_rule_name The name of the root rule. Default is "root".
 * \return The parsed grammar.
 */
Grammar ParseEBNF(const std::string& ebnf_string, const std::string& root_rule_name = "root");

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_PARSER_H_
