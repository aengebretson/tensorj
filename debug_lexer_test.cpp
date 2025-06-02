#include "src/lexer/lexer.hpp"
#include <iostream>

int main() {
    std::cout << "Testing lexer tokenization of compound adverbs and comma:" << std::endl;
    
    // Test 1: <./
    std::string test1 = "<./ 5 2 8";
    JInterpreter::Lexer lexer1(test1);
    auto tokens1 = lexer1.tokenize();
    
    std::cout << "\nTokens for '<./ 5 2 8':" << std::endl;
    for (const auto& token : tokens1) {
        std::cout << token << std::endl;
    }
    
    // Test 2: comma
    std::string test2 = "1 2 , 3 4 5";
    JInterpreter::Lexer lexer2(test2);
    auto tokens2 = lexer2.tokenize();
    
    std::cout << "\nTokens for '1 2 , 3 4 5':" << std::endl;
    for (const auto& token : tokens2) {
        std::cout << token << std::endl;
    }
    
    return 0;
}
