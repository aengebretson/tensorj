#include "gtest/gtest.h"
#include "lexer/lexer.hpp"
#include "parser/parser.hpp" // Adjust path
#include "ast/ast_nodes.hpp" // For checking AST structure

using namespace JInterpreter;

TEST(ParserTest, ParseSingleInteger) {
    Lexer lexer("123");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::NOUN_LITERAL);
    auto* noun_node = dynamic_cast<NounLiteralNode*>(root.get());
    ASSERT_NE(noun_node, nullptr);
    ASSERT_TRUE(std::holds_alternative<long long>(noun_node->value));
    EXPECT_EQ(std::get<long long>(noun_node->value), 123);
}

TEST(ParserTest, ParseSimpleMonadicApplication) {
    Lexer lexer("# 'test'"); // Count 'test'
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::MONADIC_APPLICATION);
    auto* app_node = dynamic_cast<MonadicApplicationNode*>(root.get());
    ASSERT_NE(app_node, nullptr);

    ASSERT_NE(app_node->verb, nullptr);
    EXPECT_EQ(app_node->verb->type, AstNodeType::VERB);
    auto* verb_node = dynamic_cast<VerbNode*>(app_node->verb.get());
    ASSERT_NE(verb_node, nullptr);
    EXPECT_EQ(verb_node->identifier, "#");

    ASSERT_NE(app_node->argument, nullptr);
    EXPECT_EQ(app_node->argument->type, AstNodeType::NOUN_LITERAL);
    auto* noun_arg_node = dynamic_cast<NounLiteralNode*>(app_node->argument.get());
    ASSERT_NE(noun_arg_node, nullptr);
    ASSERT_TRUE(std::holds_alternative<std::string>(noun_arg_node->value));
    EXPECT_EQ(std::get<std::string>(noun_arg_node->value), "test");
}


TEST(ParserTest, ParseSimpleDyadicApplication) {
    // Note: The current simple parser is left-associative and doesn't handle J's right-to-left.
    // This test will pass for `(1+2)` but `1+2*3` would be `(1+2)*3`.
    // This test reflects the current STUB parser's behavior.
    Lexer lexer("1 + 2");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    // root->print(std::cout); // For debugging test failures

    EXPECT_EQ(root->type, AstNodeType::DYADIC_APPLICATION);
    auto* app_node = dynamic_cast<DyadicApplicationNode*>(root.get());
    ASSERT_NE(app_node, nullptr);

    ASSERT_NE(app_node->left_argument, nullptr);
    EXPECT_EQ(app_node->left_argument->type, AstNodeType::NOUN_LITERAL);

    ASSERT_NE(app_node->verb, nullptr);
    EXPECT_EQ(app_node->verb->type, AstNodeType::VERB);

    ASSERT_NE(app_node->right_argument, nullptr);
    EXPECT_EQ(app_node->right_argument->type, AstNodeType::NOUN_LITERAL);
}

TEST(ParserTest, ParseParenthesizedExpression) {
    Lexer lexer("(5)");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    // root->print(std::cout);
    // Depending on if ParenExpressionNode is used or elided:
    EXPECT_EQ(root->type, AstNodeType::NOUN_LITERAL); // Current parser elides the paren node
    auto* noun_node = dynamic_cast<NounLiteralNode*>(root.get());
    ASSERT_NE(noun_node, nullptr);
    EXPECT_EQ(std::get<long long>(noun_node->value), 5LL);
}


// Add many more tests for different J constructs as you implement them.
// Especially for right-to-left evaluation, trains, adverbs, conjunctions.
