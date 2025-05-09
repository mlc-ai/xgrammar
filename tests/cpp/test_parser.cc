#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include "grammar_parser.h"
#include "test_utils.h"

using namespace xgrammar;

// Note: the inputs to the lexer tests may not be valid EBNF
TEST(XGrammarLexerTest, BasicTokenization) {
  // Test basic token types
  std::string input =
      "rule1 ::= \"string\" | [a-z] | 123 | (expr) | {1,3} | * | + | ? | true | false";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 28);  // 27 tokens + EOF

  // Check token types
  EXPECT_EQ(tokens[0].type, EBNFLexer::TokenType::RuleName);
  EXPECT_EQ(tokens[0].lexeme, "rule1");

  EXPECT_EQ(tokens[1].type, EBNFLexer::TokenType::Assign);
  EXPECT_EQ(tokens[1].lexeme, "::=");

  EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::StringLiteral);
  EXPECT_EQ(tokens[2].lexeme, "\"string\"");
  EXPECT_EQ(tokens[2].value, "string");

  EXPECT_EQ(tokens[3].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[4].type, EBNFLexer::TokenType::CharClass);
  EXPECT_EQ(tokens[4].lexeme, "[a-z]");

  EXPECT_EQ(tokens[5].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[6].type, EBNFLexer::TokenType::IntegerLiteral);
  EXPECT_EQ(tokens[6].lexeme, "123");
  EXPECT_EQ(tokens[6].value, "123");

  EXPECT_EQ(tokens[7].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[8].type, EBNFLexer::TokenType::LParen);
  EXPECT_EQ(tokens[9].type, EBNFLexer::TokenType::Identifier);
  EXPECT_EQ(tokens[10].type, EBNFLexer::TokenType::RParen);

  EXPECT_EQ(tokens[11].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[12].type, EBNFLexer::TokenType::LBrace);
  EXPECT_EQ(tokens[13].type, EBNFLexer::TokenType::IntegerLiteral);
  EXPECT_EQ(tokens[14].type, EBNFLexer::TokenType::Comma);
  EXPECT_EQ(tokens[15].type, EBNFLexer::TokenType::IntegerLiteral);
  EXPECT_EQ(tokens[16].type, EBNFLexer::TokenType::RBrace);

  EXPECT_EQ(tokens[17].type, EBNFLexer::TokenType::Pipe);
  EXPECT_EQ(tokens[18].type, EBNFLexer::TokenType::Star);
  EXPECT_EQ(tokens[19].type, EBNFLexer::TokenType::Pipe);
  EXPECT_EQ(tokens[20].type, EBNFLexer::TokenType::Plus);
  EXPECT_EQ(tokens[21].type, EBNFLexer::TokenType::Pipe);
  EXPECT_EQ(tokens[22].type, EBNFLexer::TokenType::Question);
  EXPECT_EQ(tokens[23].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[24].type, EBNFLexer::TokenType::Boolean);
  EXPECT_EQ(tokens[24].lexeme, "true");
  EXPECT_EQ(tokens[24].value, "1");

  EXPECT_EQ(tokens[25].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[26].type, EBNFLexer::TokenType::Boolean);
  EXPECT_EQ(tokens[26].lexeme, "false");
  EXPECT_EQ(tokens[26].value, "0");

  EXPECT_EQ(tokens[27].type, EBNFLexer::TokenType::EndOfFile);
}

TEST(XGrammarLexerTest, CommentsAndWhitespace) {
  std::string input = "rule1 ::= expr1 # This is a comment\n  | expr2 # Another comment";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 6);  // 5 tokens + EOF
  EXPECT_EQ(tokens[0].type, EBNFLexer::TokenType::RuleName);
  EXPECT_EQ(tokens[0].lexeme, "rule1");
  EXPECT_EQ(tokens[1].type, EBNFLexer::TokenType::Assign);
  EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::Identifier);
  EXPECT_EQ(tokens[2].lexeme, "expr1");
  EXPECT_EQ(tokens[3].type, EBNFLexer::TokenType::Pipe);
  EXPECT_EQ(tokens[4].type, EBNFLexer::TokenType::Identifier);
  EXPECT_EQ(tokens[4].lexeme, "expr2");
}

TEST(XGrammarLexerTest, StringLiterals) {
  // Test string literals with escape sequences
  std::string input = "rule ::= \"normal string\" | \"escaped \\\"quotes\\\"\" | \"\\n\\r\\t\\\\\"";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 8);  // 7 tokens + EOF
  EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::StringLiteral);
  EXPECT_EQ(tokens[2].value, "normal string");

  EXPECT_EQ(tokens[4].type, EBNFLexer::TokenType::StringLiteral);
  EXPECT_EQ(tokens[4].value, "escaped \"quotes\"");

  EXPECT_EQ(tokens[6].type, EBNFLexer::TokenType::StringLiteral);
  EXPECT_EQ(tokens[6].value, "\n\r\t\\");
}

TEST(XGrammarLexerTest, CharacterClasses) {
  std::string input =
      "rule ::= [a-z] | [0-9] | [^a-z] | [\\-\\]\\\\] | [\\u0041-\\u005A] | [测试] | [\\t\\r\\n] | "
      "[\\b\\f]";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 18);  // 17 tokens + EOF
  EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::CharClass);
  EXPECT_EQ(tokens[2].lexeme, "[a-z]");

  EXPECT_EQ(tokens[4].type, EBNFLexer::TokenType::CharClass);
  EXPECT_EQ(tokens[4].lexeme, "[0-9]");

  EXPECT_EQ(tokens[6].type, EBNFLexer::TokenType::CharClass);
  EXPECT_EQ(tokens[6].lexeme, "[^a-z]");

  EXPECT_EQ(tokens[8].type, EBNFLexer::TokenType::CharClass);
  EXPECT_EQ(tokens[8].lexeme, "[\\-\\]\\\\]");

  // Unicode escape sequences (A-Z in Unicode code points)
  EXPECT_EQ(tokens[10].type, EBNFLexer::TokenType::CharClass);
  EXPECT_EQ(tokens[10].lexeme, "[\\u0041-\\u005A]");

  // UTF-8 characters directly in the character class
  EXPECT_EQ(tokens[12].type, EBNFLexer::TokenType::CharClass);
  EXPECT_EQ(tokens[12].lexeme, "[测试]");

  // Common escape sequences: tab, carriage return, newline
  EXPECT_EQ(tokens[14].type, EBNFLexer::TokenType::CharClass);
  EXPECT_EQ(tokens[14].lexeme, "[\\t\\r\\n]");

  // Additional escape sequences: backspace, form feed
  EXPECT_EQ(tokens[16].type, EBNFLexer::TokenType::CharClass);
  EXPECT_EQ(tokens[16].lexeme, "[\\b\\f]");
}

TEST(XGrammarLexerTest, BooleanValues) {
  std::string input = "rule ::= true | false";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 6);  // 5 tokens + EOF
  EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::Boolean);
  EXPECT_EQ(tokens[2].lexeme, "true");
  EXPECT_EQ(tokens[2].value, "1");

  EXPECT_EQ(tokens[4].type, EBNFLexer::TokenType::Boolean);
  EXPECT_EQ(tokens[4].lexeme, "false");
  EXPECT_EQ(tokens[4].value, "0");
}

TEST(XGrammarLexerTest, LookaheadAssertion) {
  std::string input = "rule ::= \"a\" (= lookahead)";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 7);  // 6 tokens + EOF
  EXPECT_EQ(tokens[3].type, EBNFLexer::TokenType::LookaheadLParen);
  EXPECT_EQ(tokens[3].lexeme, "(=");
  EXPECT_EQ(tokens[5].type, EBNFLexer::TokenType::RParen);
}

TEST(XGrammarLexerTest, LineAndColumnTracking) {
  std::string input = "rule1 ::= expr1\nrule2 ::= expr2";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 7);  // 6 tokens + EOF

  // First line tokens
  EXPECT_EQ(tokens[0].line, 1);
  EXPECT_EQ(tokens[0].column, 1);
  EXPECT_EQ(tokens[1].line, 1);
  EXPECT_EQ(tokens[2].line, 1);

  // Second line tokens
  EXPECT_EQ(tokens[3].line, 2);
  EXPECT_EQ(tokens[3].column, 1);
  EXPECT_EQ(tokens[4].line, 2);
  EXPECT_EQ(tokens[5].line, 2);
}

TEST(XGrammarLexerTest, ComplexGrammar) {
  std::string input =
      "# JSON Grammar\n"
      "root ::= value\n"
      "value ::= object | array | string | number | \"true\" | \"false\" | \"null\"\n"
      "object ::= \"{\" (member (\",\" member)*)? \"}\"\n"
      "member ::= string \":\" value\n"
      "array ::= \"[\" (value (\",\" value)*)? \"]\"\n"
      "string ::= \"\\\"\" char* \"\\\"\"\n"
      "char ::= [^\"\\\\] | \"\\\\\\\"\"\n"
      "number ::= int frac? exp?\n"
      "int ::= \"-\"? ([1-9] [0-9]* | \"0\")\n"
      "frac ::= \".\" [0-9]+\n"
      "exp ::= [eE] [+\\-]? [0-9]+";

  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  // Just verify we have a reasonable number of tokens and no crashes
  EXPECT_GT(tokens.size(), 50);
  EXPECT_EQ(tokens.back().type, EBNFLexer::TokenType::EndOfFile);
}

TEST(XGrammarLexerTest, EdgeCases) {
  // Empty input
  {
    std::string input = "";
    EBNFLexer lexer;
    auto tokens = lexer.Tokenize(input);
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].type, EBNFLexer::TokenType::EndOfFile);
  }

  // Only whitespace and comments
  {
    std::string input = "  \t\n # Comment\n  # Another comment";
    EBNFLexer lexer;
    auto tokens = lexer.Tokenize(input);
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].type, EBNFLexer::TokenType::EndOfFile);
  }

  // Various newline formats
  {
    std::string input = "rule1 ::= expr1\nrule2 ::= expr2\r\nrule3 ::= expr3\rrule4 ::= expr4";
    EBNFLexer lexer;
    auto tokens = lexer.Tokenize(input);
    ASSERT_EQ(tokens.size(), 13);  // 12 tokens + EOF
  }

  // Integer boundary
  {
    std::string input = "rule ::= 999999999999999";  // 15 digits (max allowed)
    EBNFLexer lexer;
    auto tokens = lexer.Tokenize(input);
    ASSERT_EQ(tokens.size(), 4);  // 3 tokens + EOF
    EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::IntegerLiteral);
    EXPECT_EQ(tokens[2].lexeme, "999999999999999");
  }

  // Special identifiers
  {
    std::string input = "rule-name ::= _special.identifier-123";
    EBNFLexer lexer;
    auto tokens = lexer.Tokenize(input);
    ASSERT_EQ(tokens.size(), 4);  // 3 tokens + EOF
    EXPECT_EQ(tokens[0].type, EBNFLexer::TokenType::RuleName);
    EXPECT_EQ(tokens[0].lexeme, "rule-name");
    EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::Identifier);
    EXPECT_EQ(tokens[2].lexeme, "_special.identifier-123");
  }
}

TEST(XGrammarLexerTest, QuantifierTokens) {
  std::string input = "rule ::= expr? | expr* | expr+ | expr{1} | expr{1,} | expr{1,5}";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  // Verify question mark, star, plus, and brace tokens
  EXPECT_EQ(tokens[3].type, EBNFLexer::TokenType::Question);
  EXPECT_EQ(tokens[6].type, EBNFLexer::TokenType::Star);
  EXPECT_EQ(tokens[9].type, EBNFLexer::TokenType::Plus);
  EXPECT_EQ(tokens[12].type, EBNFLexer::TokenType::LBrace);
  EXPECT_EQ(tokens[13].type, EBNFLexer::TokenType::IntegerLiteral);
  EXPECT_EQ(tokens[14].type, EBNFLexer::TokenType::RBrace);
  EXPECT_EQ(tokens[17].type, EBNFLexer::TokenType::LBrace);
  EXPECT_EQ(tokens[18].type, EBNFLexer::TokenType::IntegerLiteral);
  EXPECT_EQ(tokens[19].type, EBNFLexer::TokenType::Comma);
  EXPECT_EQ(tokens[20].type, EBNFLexer::TokenType::RBrace);
}

// Test for UTF-8 handling in string literals
TEST(XGrammarLexerTest, UTF8Handling) {
  std::string input = "rule ::= \"UTF-8: \\u00A9 \\u2603 \\U0001F600\"";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 4);  // 3 tokens + EOF
  EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::StringLiteral);
  // The value should contain the actual UTF-8 characters
  EXPECT_EQ(tokens[2].value, "UTF-8: © ☃ 😀");
}

TEST(XGrammarLexerTest, LexerErrorCases) {
  // Test for unterminated string
  {
    std::string input = "rule ::= \"unterminated string";
    XGRAMMAR_EXPECT_THROW(
        EBNFLexer().Tokenize(input), std::exception, "Expect \" in string literal"
    );
  }

  // Test for unterminated character class
  {
    std::string input = "rule ::= [a-z";
    XGRAMMAR_EXPECT_THROW(
        EBNFLexer().Tokenize(input), std::exception, "Unterminated character class"
    );
  }

  // Test for invalid UTF-8 sequence in string
  {
    std::string input = "rule ::= \"\xC2\x20\"";  // Invalid UTF-8 sequence
    XGRAMMAR_EXPECT_THROW(EBNFLexer().Tokenize(input), std::exception, "Invalid UTF8 sequence");
  }

  // Test for invalid escape sequence in string
  {
    std::string input = "rule ::= \"\\z\"";  // Invalid escape sequence
    XGRAMMAR_EXPECT_THROW(EBNFLexer().Tokenize(input), std::exception, "Invalid escape sequence");
  }

  // Test for newline in character class
  {
    std::string input = "rule ::= [a-z\n]";
    XGRAMMAR_EXPECT_THROW(
        EBNFLexer().Tokenize(input), std::exception, "Character class should not contain newline"
    );
  }

  // Test for invalid UTF-8 sequence in character class
  {
    std::string input = "rule ::= [\xC2\x20]";  // Invalid UTF-8 sequence
    XGRAMMAR_EXPECT_THROW(EBNFLexer().Tokenize(input), std::exception, "Invalid UTF8 sequence");
  }

  // Test for invalid escape sequence in character class
  {
    std::string input = "rule ::= [\\z]";  // Invalid escape sequence
    XGRAMMAR_EXPECT_THROW(EBNFLexer().Tokenize(input), std::exception, "Invalid escape sequence");
  }

  // Test for integer too large
  {
    std::string input = "rule ::= expr{1000000000000000000}";  // Integer > 1e15
    XGRAMMAR_EXPECT_THROW(EBNFLexer().Tokenize(input), std::exception, "Integer is too large");
  }

  // Test for unexpected character
  {
    std::string input = "rule ::= @";
    XGRAMMAR_EXPECT_THROW(EBNFLexer().Tokenize(input), std::exception, "Unexpected character");
  }

  // Test for unexpected colon
  {
    std::string input = "rule : expr";
    XGRAMMAR_EXPECT_THROW(EBNFLexer().Tokenize(input), std::exception, "Unexpected character: ':'");
  }

  // Test for assign preceded by non-identifier
  {
    std::string input = "\"string\" ::= expr";
    XGRAMMAR_EXPECT_THROW(
        EBNFLexer().Tokenize(input), std::exception, "Assign should be preceded by an identifier"
    );
  }

  // Test for assign as first token
  {
    std::string input = "::= expr";
    XGRAMMAR_EXPECT_THROW(
        EBNFLexer().Tokenize(input), std::exception, "Assign should not be the first token"
    );
  }

  // Test for rule name not at beginning of line
  {
    std::string input = "token token ::= expr";
    XGRAMMAR_EXPECT_THROW(
        EBNFLexer().Tokenize(input),
        std::exception,
        "The rule name should be at the beginning of the line"
    );
  }
}
