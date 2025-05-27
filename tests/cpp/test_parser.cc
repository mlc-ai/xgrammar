#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include "grammar_parser.h"
#include "support/encoding.h"
#include "test_utils.h"

using namespace xgrammar;

// Note: the inputs to the lexer tests may not be valid EBNF
TEST(XGrammarLexerTest, BasicTokenization) {
  // Test basic token types
  std::string input =
      "rule1 ::= \"string\" | [a-z] | 123 | (expr) | {1,3} | * | + | ? | true | false";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 32);  // 27 tokens + EOF

  // Check token types
  EXPECT_EQ(tokens[0].type, EBNFLexer::TokenType::RuleName);
  EXPECT_EQ(tokens[0].lexeme, "rule1");

  EXPECT_EQ(tokens[1].type, EBNFLexer::TokenType::Assign);
  EXPECT_EQ(tokens[1].lexeme, "::=");

  EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::StringLiteral);
  EXPECT_EQ(tokens[2].lexeme, "\"string\"");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[2].value, std::string, "string");

  EXPECT_EQ(tokens[3].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[4].type, EBNFLexer::TokenType::LBracket);
  EXPECT_EQ(tokens[5].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[5].lexeme, "a");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[5].value, TCodepoint, 'a');
  EXPECT_EQ(tokens[6].type, EBNFLexer::TokenType::Dash);
  EXPECT_EQ(tokens[7].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[7].lexeme, "z");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[7].value, TCodepoint, 'z');
  EXPECT_EQ(tokens[8].type, EBNFLexer::TokenType::RBracket);

  EXPECT_EQ(tokens[9].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[10].type, EBNFLexer::TokenType::IntegerLiteral);
  EXPECT_EQ(tokens[10].lexeme, "123");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[10].value, int64_t, 123);

  EXPECT_EQ(tokens[11].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[12].type, EBNFLexer::TokenType::LParen);
  EXPECT_EQ(tokens[13].type, EBNFLexer::TokenType::Identifier);
  EXPECT_EQ(tokens[14].type, EBNFLexer::TokenType::RParen);

  EXPECT_EQ(tokens[15].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[16].type, EBNFLexer::TokenType::LBrace);
  EXPECT_EQ(tokens[17].type, EBNFLexer::TokenType::IntegerLiteral);
  EXPECT_EQ(tokens[18].type, EBNFLexer::TokenType::Comma);
  EXPECT_EQ(tokens[19].type, EBNFLexer::TokenType::IntegerLiteral);
  EXPECT_EQ(tokens[20].type, EBNFLexer::TokenType::RBrace);

  EXPECT_EQ(tokens[21].type, EBNFLexer::TokenType::Pipe);
  EXPECT_EQ(tokens[22].type, EBNFLexer::TokenType::Star);
  EXPECT_EQ(tokens[23].type, EBNFLexer::TokenType::Pipe);
  EXPECT_EQ(tokens[24].type, EBNFLexer::TokenType::Plus);
  EXPECT_EQ(tokens[25].type, EBNFLexer::TokenType::Pipe);
  EXPECT_EQ(tokens[26].type, EBNFLexer::TokenType::Question);
  EXPECT_EQ(tokens[27].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[28].type, EBNFLexer::TokenType::BooleanLiteral);
  EXPECT_EQ(tokens[28].lexeme, "true");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[28].value, bool, true);

  EXPECT_EQ(tokens[29].type, EBNFLexer::TokenType::Pipe);

  EXPECT_EQ(tokens[30].type, EBNFLexer::TokenType::BooleanLiteral);
  EXPECT_EQ(tokens[30].lexeme, "false");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[30].value, bool, false);

  EXPECT_EQ(tokens[31].type, EBNFLexer::TokenType::EndOfFile);
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
  XGRAMMAR_EXPECT_ANY_EQ(tokens[2].value, std::string, "normal string");

  EXPECT_EQ(tokens[4].type, EBNFLexer::TokenType::StringLiteral);
  XGRAMMAR_EXPECT_ANY_EQ(tokens[4].value, std::string, "escaped \"quotes\"");

  EXPECT_EQ(tokens[6].type, EBNFLexer::TokenType::StringLiteral);
  XGRAMMAR_EXPECT_ANY_EQ(tokens[6].value, std::string, "\n\r\t\\");
}

TEST(XGrammarLexerTest, CharacterClasses) {
  std::string input =
      "rule ::= [a-z] | [0-9] | [^a-z] | [\\-\\]\\\\] | [\\u0041-\\u005A] | [测试] | [\\t\\r\\n] | "
      "[\\b\\f]";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 49);  // 45 tokens + EOF

  // [a-z]
  EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::LBracket);
  EXPECT_EQ(tokens[3].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[3].lexeme, "a");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[3].value, TCodepoint, 'a');
  EXPECT_EQ(tokens[4].type, EBNFLexer::TokenType::Dash);
  EXPECT_EQ(tokens[5].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[5].lexeme, "z");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[5].value, TCodepoint, 'z');
  EXPECT_EQ(tokens[6].type, EBNFLexer::TokenType::RBracket);

  EXPECT_EQ(tokens[7].type, EBNFLexer::TokenType::Pipe);

  // [0-9]
  EXPECT_EQ(tokens[8].type, EBNFLexer::TokenType::LBracket);
  EXPECT_EQ(tokens[9].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[9].lexeme, "0");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[9].value, TCodepoint, '0');
  EXPECT_EQ(tokens[10].type, EBNFLexer::TokenType::Dash);
  EXPECT_EQ(tokens[11].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[11].lexeme, "9");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[11].value, TCodepoint, '9');
  EXPECT_EQ(tokens[12].type, EBNFLexer::TokenType::RBracket);

  EXPECT_EQ(tokens[13].type, EBNFLexer::TokenType::Pipe);

  // [^a-z]
  EXPECT_EQ(tokens[14].type, EBNFLexer::TokenType::LBracket);
  EXPECT_EQ(tokens[15].type, EBNFLexer::TokenType::Caret);
  EXPECT_EQ(tokens[16].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[16].lexeme, "a");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[16].value, TCodepoint, 'a');
  EXPECT_EQ(tokens[17].type, EBNFLexer::TokenType::Dash);
  EXPECT_EQ(tokens[18].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[18].lexeme, "z");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[18].value, TCodepoint, 'z');
  EXPECT_EQ(tokens[19].type, EBNFLexer::TokenType::RBracket);

  EXPECT_EQ(tokens[20].type, EBNFLexer::TokenType::Pipe);

  // [\-\]\\]
  EXPECT_EQ(tokens[21].type, EBNFLexer::TokenType::LBracket);
  EXPECT_EQ(tokens[22].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[22].lexeme, "\\-");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[22].value, TCodepoint, '-');
  EXPECT_EQ(tokens[23].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[23].lexeme, "\\]");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[23].value, TCodepoint, ']');
  EXPECT_EQ(tokens[24].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[24].lexeme, "\\\\");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[24].value, TCodepoint, '\\');
  EXPECT_EQ(tokens[25].type, EBNFLexer::TokenType::RBracket);

  EXPECT_EQ(tokens[26].type, EBNFLexer::TokenType::Pipe);

  // [\u0041-\u005A]
  EXPECT_EQ(tokens[27].type, EBNFLexer::TokenType::LBracket);
  EXPECT_EQ(tokens[28].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[28].lexeme, "\\u0041");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[28].value, TCodepoint, 0x41);  // 'A'
  EXPECT_EQ(tokens[29].type, EBNFLexer::TokenType::Dash);
  EXPECT_EQ(tokens[30].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[30].lexeme, "\\u005A");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[30].value, TCodepoint, 0x5A);  // 'Z'
  EXPECT_EQ(tokens[31].type, EBNFLexer::TokenType::RBracket);

  EXPECT_EQ(tokens[32].type, EBNFLexer::TokenType::Pipe);

  // [测试]
  EXPECT_EQ(tokens[33].type, EBNFLexer::TokenType::LBracket);
  EXPECT_EQ(tokens[34].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[34].lexeme, "测");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[34].value, TCodepoint, 0x6D4B);
  EXPECT_EQ(tokens[35].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[35].lexeme, "试");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[35].value, TCodepoint, 0x8BD5);
  EXPECT_EQ(tokens[36].type, EBNFLexer::TokenType::RBracket);

  EXPECT_EQ(tokens[37].type, EBNFLexer::TokenType::Pipe);

  // [\t\r\n]
  EXPECT_EQ(tokens[38].type, EBNFLexer::TokenType::LBracket);
  EXPECT_EQ(tokens[39].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[39].lexeme, "\\t");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[39].value, TCodepoint, '\t');
  EXPECT_EQ(tokens[40].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[40].lexeme, "\\r");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[40].value, TCodepoint, '\r');
  EXPECT_EQ(tokens[41].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[41].lexeme, "\\n");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[41].value, TCodepoint, '\n');
  EXPECT_EQ(tokens[42].type, EBNFLexer::TokenType::RBracket);

  EXPECT_EQ(tokens[43].type, EBNFLexer::TokenType::Pipe);

  // [\b\f]
  EXPECT_EQ(tokens[44].type, EBNFLexer::TokenType::LBracket);
  EXPECT_EQ(tokens[45].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[45].lexeme, "\\b");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[45].value, TCodepoint, '\b');
  EXPECT_EQ(tokens[46].type, EBNFLexer::TokenType::CharInCharClass);
  EXPECT_EQ(tokens[46].lexeme, "\\f");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[46].value, TCodepoint, '\f');
  EXPECT_EQ(tokens[47].type, EBNFLexer::TokenType::RBracket);
}

TEST(XGrammarLexerTest, BooleanValues) {
  std::string input = "rule ::= true | false";
  EBNFLexer lexer;
  auto tokens = lexer.Tokenize(input);

  ASSERT_EQ(tokens.size(), 6);  // 5 tokens + EOF
  EXPECT_EQ(tokens[2].type, EBNFLexer::TokenType::BooleanLiteral);
  EXPECT_EQ(tokens[2].lexeme, "true");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[2].value, bool, true);

  EXPECT_EQ(tokens[4].type, EBNFLexer::TokenType::BooleanLiteral);
  EXPECT_EQ(tokens[4].lexeme, "false");
  XGRAMMAR_EXPECT_ANY_EQ(tokens[4].value, bool, false);
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
  EXPECT_EQ(tokens[1].column, 7);
  EXPECT_EQ(tokens[2].line, 1);
  EXPECT_EQ(tokens[2].column, 11);

  // Second line tokens
  EXPECT_EQ(tokens[3].line, 2);
  EXPECT_EQ(tokens[3].column, 1);
  EXPECT_EQ(tokens[4].line, 2);
  EXPECT_EQ(tokens[4].column, 7);
  EXPECT_EQ(tokens[5].line, 2);
  EXPECT_EQ(tokens[5].column, 11);
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
  XGRAMMAR_EXPECT_ANY_EQ(tokens[2].value, std::string, "UTF-8: © ☃ 😀");
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

  // Test for unterminated character class with escaped bracket
  {
    std::string input = "rule ::= [a-z\\-\\\\\\]";
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
