#include "src/lexer/lexer.hpp"
#include <iostream>

using namespace JInterpreter;

int main() {
    std::string input = "+/ i. 5";
    std::cout << "Tokenizing: " << input << std::endl;
    
    Lexer lexer(input);
    std::vector<Token> tokens = lexer.tokenize();
    
    for (const auto& token : tokens) {
        std::cout << token << std::endl;
    }
    
    return 0;
}
