#include "src/lexer/lexer.hpp"
#include <iostream>

using namespace JInterpreter;

int main() {
    std::string input = "< ./ 5 2 8";
    std::cout << "Tokenizing: '" << input << "'" << std::endl;
    
    Lexer lexer(input);
    std::vector<Token> tokens = lexer.tokenize();
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "Token " << i << ": " << tokens[i] << std::endl;
    }
    
    return 0;
}
