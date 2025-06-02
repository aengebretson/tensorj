#include "src/lexer/lexer.hpp"
#include <iostream>

using namespace JInterpreter;

int main() {
    Lexer lexer("< ./");
    auto tokens = lexer.tokenize();
    
    std::cout << "Tokens for '< ./':" << std::endl;
    for (const auto& token : tokens) {
        std::cout << "  Type: " << static_cast<int>(token.type) << ", Lexeme: '" << token.lexeme << "'" << std::endl;
    }
    
    return 0;
}
