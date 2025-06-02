#include "src/lexer/lexer.hpp"
#include <iostream>

using namespace JInterpreter;

int main() {
    Lexer lexer("(+/ % #)");
    std::vector<Token> tokens = lexer.tokenize();
    
    std::cout << "Total tokens: " << tokens.size() << std::endl;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << i << ": " << tokens[i] << std::endl;
    }
    
    return 0;
}
