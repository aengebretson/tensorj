#include "gtest/gtest.h"
#include "lexer/lexer.hpp" // Adjust path as needed
#include "lexer/token.hpp"   // For TokenType and Token

using namespace JInterpreter;

TEST(LexerTest, TokenizeSimpleInteger) {
    Lexer lexer("123");
    std::vector<Token> tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 2); // Number + EOF
    EXPECT_EQ(tokens[0].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[0].lexeme, "123");
    ASSERT_TRUE(std::holds_alternative<long long>(tokens[0].literal_value));
    EXPECT_EQ(std::get<long long>(tokens[0].literal_value), 123);
    EXPECT_EQ(tokens[1].type, TokenType::END_OF_FILE);
}

TEST(LexerTest, TokenizeNegativeIntegerJStyle) {
    Lexer lexer("_5");
    std::vector<Token> tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 2); // Number + EOF
    EXPECT_EQ(tokens[0].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[0].lexeme, "_5");
    ASSERT_TRUE(std::holds_alternative<long long>(tokens[0].literal_value));
    EXPECT_EQ(std::get<long long>(tokens[0].literal_value), -5);
}


TEST(LexerTest, TokenizeSimpleString) {
    Lexer lexer("'hello world'");
    std::vector<Token> tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 2); // String + EOF
    EXPECT_EQ(tokens[0].type, TokenType::NOUN_STRING);
    EXPECT_EQ(tokens[0].lexeme, "'hello world'");
    ASSERT_TRUE(std::holds_alternative<std::string>(tokens[0].literal_value));
    EXPECT_EQ(std::get<std::string>(tokens[0].literal_value), "hello world");
}

TEST(LexerTest, TokenizeStringWithEscapedQuote) {
    Lexer lexer("'it''s nice'");
    std::vector<Token> tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TokenType::NOUN_STRING);
    EXPECT_EQ(tokens[0].lexeme, "'it''s nice'");
    ASSERT_TRUE(std::holds_alternative<std::string>(tokens[0].literal_value));
    EXPECT_EQ(std::get<std::string>(tokens[0].literal_value), "it's nice");
}


TEST(LexerTest, TokenizeBasicVerb) {
    Lexer lexer("+");
    std::vector<Token> tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0].type, TokenType::VERB);
    EXPECT_EQ(tokens[0].lexeme, "+");
}

TEST(LexerTest, TokenizeAssignment) {
    Lexer lexer("name =. 1");
    std::vector<Token> tokens = lexer.tokenize();
    // Expected: NAME, ASSIGN_LOCAL, NOUN_INTEGER, EOF
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[0].type, TokenType::NAME);
    EXPECT_EQ(tokens[0].lexeme, "name");
    EXPECT_EQ(tokens[1].type, TokenType::ASSIGN_LOCAL);
    EXPECT_EQ(tokens[1].lexeme, "=.");
    EXPECT_EQ(tokens[2].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[2].lexeme, "1");
}

TEST(LexerTest, TokenizeNBComment) {
    Lexer lexer("NB. this is a comment\n123");
    std::vector<Token> tokens = lexer.tokenize();
    // Expected: NOUN_INTEGER(123), NEWLINE, EOF (if comments are skipped and newline kept)
    // Or COMMENT, NEWLINE, NOUN_INTEGER, EOF if comments are tokenized
    // Current lexer skips comments but tokenizes newlines:
    ASSERT_EQ(tokens.size(), 3);
    EXPECT_EQ(tokens[0].type, TokenType::NEWLINE);
    EXPECT_EQ(tokens[1].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[1].lexeme, "123");
    EXPECT_EQ(tokens[2].type, TokenType::END_OF_FILE);
}

TEST(LexerTest, TokenizeMultipleTokens) {
    Lexer lexer("i =. _1 + 2 NB. sum\n'string'");
    std::vector<Token> tokens = lexer.tokenize();
    // i, =., _1, +, 2, \n, 'string', EOF
    ASSERT_EQ(tokens.size(), 8);
    EXPECT_EQ(tokens[0].type, TokenType::NAME);
    EXPECT_EQ(tokens[0].lexeme, "i");
    EXPECT_EQ(tokens[1].type, TokenType::ASSIGN_LOCAL);
    EXPECT_EQ(tokens[2].type, TokenType::NOUN_INTEGER); // _1
    EXPECT_EQ(tokens[3].type, TokenType::VERB); // +
    EXPECT_EQ(tokens[4].type, TokenType::NOUN_INTEGER); // 2
    EXPECT_EQ(tokens[5].type, TokenType::NEWLINE);
    EXPECT_EQ(tokens[6].type, TokenType::NOUN_STRING);
    EXPECT_EQ(tokens[7].type, TokenType::END_OF_FILE);
}

// Test tokenizing space-separated numbers (vector literals)
TEST(LexerTest, TokenizeSpaceSeparatedNumbers) {
    Lexer lexer("1 2 3");
    std::vector<Token> tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4); // Three numbers + EOF
    
    EXPECT_EQ(tokens[0].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[0].lexeme, "1");
    ASSERT_TRUE(std::holds_alternative<long long>(tokens[0].literal_value));
    EXPECT_EQ(std::get<long long>(tokens[0].literal_value), 1);
    
    EXPECT_EQ(tokens[1].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[1].lexeme, "2");
    ASSERT_TRUE(std::holds_alternative<long long>(tokens[1].literal_value));
    EXPECT_EQ(std::get<long long>(tokens[1].literal_value), 2);
    
    EXPECT_EQ(tokens[2].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[2].lexeme, "3");
    ASSERT_TRUE(std::holds_alternative<long long>(tokens[2].literal_value));
    EXPECT_EQ(std::get<long long>(tokens[2].literal_value), 3);
    
    EXPECT_EQ(tokens[3].type, TokenType::END_OF_FILE);
}

TEST(LexerTest, TokenizeMixedIntegerFloat) {
    Lexer lexer("1 2.5 3");
    std::vector<Token> tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 4); // Three numbers + EOF
    
    EXPECT_EQ(tokens[0].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[0].lexeme, "1");
    
    EXPECT_EQ(tokens[1].type, TokenType::NOUN_FLOAT);
    EXPECT_EQ(tokens[1].lexeme, "2.5");
    ASSERT_TRUE(std::holds_alternative<double>(tokens[1].literal_value));
    EXPECT_DOUBLE_EQ(std::get<double>(tokens[1].literal_value), 2.5);
    
    EXPECT_EQ(tokens[2].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[2].lexeme, "3");
}

TEST(LexerTest, TokenizeVectorAdditionExpression) {
    Lexer lexer("1 2 3 + 4 5 6");
    std::vector<Token> tokens = lexer.tokenize();
    ASSERT_EQ(tokens.size(), 8); // Six numbers + one verb + EOF
    
    // First vector: 1 2 3
    EXPECT_EQ(tokens[0].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[0].lexeme, "1");
    EXPECT_EQ(tokens[1].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[1].lexeme, "2");
    EXPECT_EQ(tokens[2].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[2].lexeme, "3");
    
    // Verb: +
    EXPECT_EQ(tokens[3].type, TokenType::VERB);
    EXPECT_EQ(tokens[3].lexeme, "+");
    
    // Second vector: 4 5 6
    EXPECT_EQ(tokens[4].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[4].lexeme, "4");
    EXPECT_EQ(tokens[5].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[5].lexeme, "5");
    EXPECT_EQ(tokens[6].type, TokenType::NOUN_INTEGER);
    EXPECT_EQ(tokens[6].lexeme, "6");
    
    EXPECT_EQ(tokens[7].type, TokenType::END_OF_FILE);
}

// Add more tests for all token types, edge cases, errors etc.
