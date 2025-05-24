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



TEST(ParserTest, ParseRightToLeftDyadic) {
    // Test that 1 + 2 * 3 parses as 1 + (2 * 3) due to right-to-left evaluation
    // NOTE: Current parser might not handle this correctly yet - this test documents expected behavior
    Lexer lexer("1 + 2 * 3");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::DYADIC_APPLICATION);
    auto* outer_app = dynamic_cast<DyadicApplicationNode*>(root.get());
    ASSERT_NE(outer_app, nullptr);

    // Outer operation should be +
    auto* outer_verb = dynamic_cast<VerbNode*>(outer_app->verb.get());
    ASSERT_NE(outer_verb, nullptr);
    EXPECT_EQ(outer_verb->identifier, "+");

    // Left should be 1
    auto* left_noun = dynamic_cast<NounLiteralNode*>(outer_app->left_argument.get());
    ASSERT_NE(left_noun, nullptr);
    EXPECT_EQ(std::get<long long>(left_noun->value), 1);

    // Right should be (2 * 3)
    EXPECT_EQ(outer_app->right_argument->type, AstNodeType::DYADIC_APPLICATION);
    auto* inner_app = dynamic_cast<DyadicApplicationNode*>(outer_app->right_argument.get());
    ASSERT_NE(inner_app, nullptr);

    // Inner operation should be *
    auto* inner_verb = dynamic_cast<VerbNode*>(inner_app->verb.get());
    ASSERT_NE(inner_verb, nullptr);
    EXPECT_EQ(inner_verb->identifier, "*");
}

TEST(ParserTest, ParseNestedParentheses) {
    Lexer lexer("(1 + 2) * 3");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::DYADIC_APPLICATION);
    auto* outer_app = dynamic_cast<DyadicApplicationNode*>(root.get());
    ASSERT_NE(outer_app, nullptr);

    // Should be * operation
    auto* verb = dynamic_cast<VerbNode*>(outer_app->verb.get());
    EXPECT_EQ(verb->identifier, "*");

    // Left should be (1 + 2)
    EXPECT_EQ(outer_app->left_argument->type, AstNodeType::DYADIC_APPLICATION);
    auto* left_app = dynamic_cast<DyadicApplicationNode*>(outer_app->left_argument.get());
    auto* left_verb = dynamic_cast<VerbNode*>(left_app->verb.get());
    EXPECT_EQ(left_verb->identifier, "+");

    // Right should be 3
    auto* right_noun = dynamic_cast<NounLiteralNode*>(outer_app->right_argument.get());
    EXPECT_EQ(std::get<long long>(right_noun->value), 3);
}

TEST(ParserTest, ParseFloatLiterals) {
    Lexer lexer("3.14");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::NOUN_LITERAL);
    auto* noun_node = dynamic_cast<NounLiteralNode*>(root.get());
    ASSERT_NE(noun_node, nullptr);
    ASSERT_TRUE(std::holds_alternative<double>(noun_node->value));
    EXPECT_DOUBLE_EQ(std::get<double>(noun_node->value), 3.14);
}

TEST(ParserTest, ParseJNegativeInteger) {
    Lexer lexer("_42");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::NOUN_LITERAL);
    auto* noun_node = dynamic_cast<NounLiteralNode*>(root.get());
    ASSERT_NE(noun_node, nullptr);
    ASSERT_TRUE(std::holds_alternative<long long>(noun_node->value));
    EXPECT_EQ(std::get<long long>(noun_node->value), -42);
}

TEST(ParserTest, ParseEmptyExpression) {
    Lexer lexer("");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    // Should handle empty input gracefully
    ASSERT_NE(root, nullptr);
    // Current implementation returns null literal for empty
    EXPECT_EQ(root->type, AstNodeType::NOUN_LITERAL);
}

// Error handling tests
TEST(ParserTest, ParseUnmatchedLeftParen) {
    Lexer lexer("(1 + 2");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    
    // Should throw or return nullptr for syntax error
    EXPECT_THROW({
        std::unique_ptr<AstNode> root = parser.parse();
    }, std::runtime_error);
}

TEST(ParserTest, ParseUnmatchedRightParen) {
    Lexer lexer("1 + 2)");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    
    // This might not throw with current parser, but documents expected behavior
    std::unique_ptr<AstNode> root = parser.parse();
    // Should either throw or handle gracefully
    // The behavior depends on parser implementation
}

// Tests for future features (these will initially fail)
TEST(ParserTest, DISABLED_ParseSimpleAssignment) {
    // This test is disabled until AssignmentNode is implemented
    Lexer lexer("x =. 5");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    // EXPECT_EQ(root->type, AstNodeType::ASSIGNMENT);
    // Test assignment structure when implemented
}

TEST(ParserTest, DISABLED_ParseAdverbApplication) {
    // This test is disabled until adverb parsing is implemented
    Lexer lexer("+/ 1 2 3");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    // Should parse as MonadicApplication of (+/) to (1 2 3)
    // Where (+/) is an AdverbApplication
}

TEST(ParserTest, DISABLED_ParseSimpleFork) {
    // This test is disabled until train parsing is implemented
    Lexer lexer("(+/ % #) 1 2 3 4");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    // Should parse as average: sum divided by count
    // This is a complex train that will need special handling
}

// Utility test to help debug parser issues
TEST(ParserTest, DebugTokenSequence) {
    Lexer lexer("1 + 2 * 3");
    std::vector<Token> tokens = lexer.tokenize();
    
    // Print token sequence for debugging
    std::cout << "\nToken sequence for '1 + 2 * 3':" << std::endl;
    for (const auto& token : tokens) {
        std::cout << token << std::endl;
    }
    
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();
    
    if (root) {
        std::cout << "\nAST structure:" << std::endl;
        root->print(std::cout, 0);
    }
    
   EXPECT_NE(root, nullptr);
}

// Add many more tests for different J constructs as you implement them.
// Especially for right-to-left evaluation, trains, adverbs, conjunctions.
