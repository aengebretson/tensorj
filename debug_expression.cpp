#include "src/lexer/lexer.hpp"
#include <iostream>

using namespace JInterpreter;

int main() {
    Lexer lexer("A +.* B");
    auto tokens = lexer.tokenize();
    
    std::cout << "Token count: " << tokens.size() << std::endl;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "Token " << i << ": '" << tokens[i].lexeme << "' (type: " << static_cast<int>(tokens[i].type) << ")" << std::endl;
    }
    
    return 0;
}
