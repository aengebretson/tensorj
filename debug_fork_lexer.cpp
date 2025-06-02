#include "src/lexer/lexer.hpp"
#include <iostream>

using namespace JInterpreter;

int main() {
    std::string input = "(+/ % #) 1 2 3 4";
    std::cout << "Tokenizing: " << input << std::endl;
    
    Lexer lexer(input);
    std::vector<Token> tokens = lexer.tokenize();
    
    std::cout << "Tokens:" << std::endl;
    for (const auto& token : tokens) {
        std::cout << "  " << token << std::endl;
    }
    
    return 0;
}
