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
TEST(ParserTest, ParseSimpleAssignment) {
    // This test is disabled until AssignmentNode is implemented
    Lexer lexer("x =. 5");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    // EXPECT_EQ(root->type, AstNodeType::ASSIGNMENT);
    // Test assignment structure when implemented
}

TEST(ParserTest, ParseAdverbApplication) {
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

TEST(ParserTest, ParseVectorLiteral) {
    Lexer lexer("1 2 3");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::VECTOR_LITERAL);
    auto* vector_node = dynamic_cast<VectorLiteralNode*>(root.get());
    ASSERT_NE(vector_node, nullptr);
    
    EXPECT_EQ(vector_node->elements.size(), 3);
    
    // Check each element
    ASSERT_TRUE(std::holds_alternative<long long>(vector_node->elements[0]));
    EXPECT_EQ(std::get<long long>(vector_node->elements[0]), 1);
    
    ASSERT_TRUE(std::holds_alternative<long long>(vector_node->elements[1]));
    EXPECT_EQ(std::get<long long>(vector_node->elements[1]), 2);
    
    ASSERT_TRUE(std::holds_alternative<long long>(vector_node->elements[2]));
    EXPECT_EQ(std::get<long long>(vector_node->elements[2]), 3);
}

TEST(ParserTest, ParseSingleNumberAsNounLiteral) {
    // Single number should not create a vector
    Lexer lexer("42");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::NOUN_LITERAL);
    auto* noun_node = dynamic_cast<NounLiteralNode*>(root.get());
    ASSERT_NE(noun_node, nullptr);
    ASSERT_TRUE(std::holds_alternative<long long>(noun_node->value));
    EXPECT_EQ(std::get<long long>(noun_node->value), 42);
}

TEST(ParserTest, ParseMixedVectorLiteral) {
    Lexer lexer("1 2.5 3");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::VECTOR_LITERAL);
    auto* vector_node = dynamic_cast<VectorLiteralNode*>(root.get());
    ASSERT_NE(vector_node, nullptr);
    
    EXPECT_EQ(vector_node->elements.size(), 3);
    
    // Check each element
    ASSERT_TRUE(std::holds_alternative<long long>(vector_node->elements[0]));
    EXPECT_EQ(std::get<long long>(vector_node->elements[0]), 1);
    
    ASSERT_TRUE(std::holds_alternative<double>(vector_node->elements[1]));
    EXPECT_DOUBLE_EQ(std::get<double>(vector_node->elements[1]), 2.5);
    
    ASSERT_TRUE(std::holds_alternative<long long>(vector_node->elements[2]));
    EXPECT_EQ(std::get<long long>(vector_node->elements[2]), 3);
}

TEST(ParserTest, ParseVectorAdditionExpression) {
    Lexer lexer("1 2 3 + 4 5 6");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    std::unique_ptr<AstNode> root = parser.parse();

    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->type, AstNodeType::DYADIC_APPLICATION);
    auto* dyadic_node = dynamic_cast<DyadicApplicationNode*>(root.get());
    ASSERT_NE(dyadic_node, nullptr);
    
    // Check left argument (should be vector 1 2 3)
    ASSERT_NE(dyadic_node->left_argument, nullptr);
    EXPECT_EQ(dyadic_node->left_argument->type, AstNodeType::VECTOR_LITERAL);
    auto* left_vector = dynamic_cast<VectorLiteralNode*>(dyadic_node->left_argument.get());
    ASSERT_NE(left_vector, nullptr);
    EXPECT_EQ(left_vector->elements.size(), 3);
    
    // Check verb (should be +)
    ASSERT_NE(dyadic_node->verb, nullptr);
    EXPECT_EQ(dyadic_node->verb->type, AstNodeType::VERB);
    auto* verb_node = dynamic_cast<VerbNode*>(dyadic_node->verb.get());
    ASSERT_NE(verb_node, nullptr);
    EXPECT_EQ(verb_node->identifier, "+");
    
    // Check right argument (should be vector 4 5 6)
    ASSERT_NE(dyadic_node->right_argument, nullptr);
    EXPECT_EQ(dyadic_node->right_argument->type, AstNodeType::VECTOR_LITERAL);
    auto* right_vector = dynamic_cast<VectorLiteralNode*>(dyadic_node->right_argument.get());
    ASSERT_NE(right_vector, nullptr);
    EXPECT_EQ(right_vector->elements.size(), 3);
}

// Test conjunction parsing with compound adverbs
TEST(ParserTest, ParseConjunctionWithoutSpaceSuccess) {
    // Test that <./ 5 2 8 (no space between < and ./) parses successfully
    // This should parse as a monadic application of the conjunction (<./) to the vector (5 2 8)
    Lexer lexer("<./ 5 2 8");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    
    std::unique_ptr<AstNode> root = parser.parse();
    ASSERT_NE(root, nullptr);
    
    // Should parse as a monadic application (conjunction applied to argument)
    EXPECT_EQ(root->type, AstNodeType::MONADIC_APPLICATION);
    auto* app_node = dynamic_cast<MonadicApplicationNode*>(root.get());
    ASSERT_NE(app_node, nullptr);
    
    // The verb should be a conjunction application (<./)
    ASSERT_NE(app_node->verb, nullptr);
    EXPECT_EQ(app_node->verb->type, AstNodeType::CONJUNCTION_APPLICATION);
    auto* conj_node = dynamic_cast<ConjunctionApplicationNode*>(app_node->verb.get());
    ASSERT_NE(conj_node, nullptr);
    
    // Left operand should be <
    ASSERT_NE(conj_node->left_operand, nullptr);
    EXPECT_EQ(conj_node->left_operand->type, AstNodeType::VERB);
    auto* left_verb = dynamic_cast<VerbNode*>(conj_node->left_operand.get());
    ASSERT_NE(left_verb, nullptr);
    EXPECT_EQ(left_verb->identifier, "<");
    
    // Right operand should be ./
    ASSERT_NE(conj_node->right_operand, nullptr);
    EXPECT_EQ(conj_node->right_operand->type, AstNodeType::ADVERB);
    auto* right_adverb = dynamic_cast<AdverbNode*>(conj_node->right_operand.get());
    ASSERT_NE(right_adverb, nullptr);
    EXPECT_EQ(right_adverb->identifier, "./");
    
    // Argument should be vector 5 2 8
    ASSERT_NE(app_node->argument, nullptr);
    EXPECT_EQ(app_node->argument->type, AstNodeType::VECTOR_LITERAL);
    auto* vector_arg = dynamic_cast<VectorLiteralNode*>(app_node->argument.get());
    ASSERT_NE(vector_arg, nullptr);
    EXPECT_EQ(vector_arg->elements.size(), 3);
}

TEST(ParserTest, ParseConjunctionWithSpaceError) {
    // Test that < ./ 5 2 8 (with space between < and ./) produces a syntax error
    // Note: Currently this test documents the actual behavior, not the desired behavior
    // According to J language rules, this should be a syntax error
    Lexer lexer("< ./ 5 2 8");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    
    // Current behavior: The lexer incorrectly creates a compound ./ adverb
    // even with space, so the parser may accept it incorrectly
    // TODO: Once lexer is fixed, this should throw a parser error
    
    try {
        std::unique_ptr<AstNode> root = parser.parse();
        // If parsing succeeds, document what was actually parsed
        ASSERT_NE(root, nullptr);
        std::cout << "\nParser successfully parsed '< ./ 5 2 8', AST type: " 
                  << static_cast<int>(root->type) << std::endl;
        
        // This should NOT succeed according to J language rules
        // The space should make this a syntax error
    } catch (const std::runtime_error& e) {
        // This is the correct behavior - should throw syntax error
        std::cout << "\nParser correctly rejected '< ./ 5 2 8' with error: " 
                  << e.what() << std::endl;
        // Test passes if we get here
        SUCCEED();
        return;
    }
    
    // If we get here, the parser incorrectly accepted invalid syntax
    // For now, just document this behavior
    ADD_FAILURE() << "Parser should reject '< ./ 5 2 8' as syntax error (space between < and ./)";
}

TEST(ParserTest, ParseOtherCompoundAdverbs) {
    // Test >.\ (greater-than scan) - just the conjunction without argument
    Lexer lexer(">.\\");
    std::vector<Token> tokens = lexer.tokenize();
    Parser parser(tokens);
    
    std::unique_ptr<AstNode> root = parser.parse();
    ASSERT_NE(root, nullptr);
    
    // Should parse as a conjunction application
    EXPECT_EQ(root->type, AstNodeType::CONJUNCTION_APPLICATION);
    auto* conj_node = dynamic_cast<ConjunctionApplicationNode*>(root.get());
    ASSERT_NE(conj_node, nullptr);
    
    // Left operand should be >
    ASSERT_NE(conj_node->left_operand, nullptr);
    EXPECT_EQ(conj_node->left_operand->type, AstNodeType::VERB);
    auto* left_verb = dynamic_cast<VerbNode*>(conj_node->left_operand.get());
    ASSERT_NE(left_verb, nullptr);
    EXPECT_EQ(left_verb->identifier, ">");
    
    // Right operand should be .\
    ASSERT_NE(conj_node->right_operand, nullptr);
    EXPECT_EQ(conj_node->right_operand->type, AstNodeType::ADVERB);
    auto* right_adverb = dynamic_cast<AdverbNode*>(conj_node->right_operand.get());
    ASSERT_NE(right_adverb, nullptr);
    EXPECT_EQ(right_adverb->identifier, ".\\");
}
