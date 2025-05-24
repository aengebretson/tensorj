#include "gtest/gtest.h"
#include "lexer/lexer.hpp"

using namespace JInterpreter;

TEST(DebugTest, TokenizeAdverbApplication) {
    Lexer lexer("+/ i. 5");
    std::vector<Token> tokens = lexer.tokenize();
    
    std::cout << "Tokenizing: +/ i. 5" << std::endl;
    for (const auto& token : tokens) {
        std::cout << token << std::endl;
    }
    
    // Just to have an assertion
    EXPECT_GT(tokens.size(), 0);
}
