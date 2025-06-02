#include "src/lexer/lexer.hpp"
#include "src/parser/parser.hpp"
#include <iostream>

using namespace JInterpreter;

void debug_tokenization(const std::string& input) {
    std::cout << "=== Debugging: \"" << input << "\" ===" << std::endl;
    
    Lexer lexer(input);
    auto tokens = lexer.tokenize();
    
    std::cout << "Tokens:" << std::endl;
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << i << ": " << tokens[i].lexeme << " (" << static_cast<int>(tokens[i].type) << ")" << std::endl;
    }
    
    std::cout << "\nParsing attempt:" << std::endl;
    try {
        Parser parser(tokens);
        auto ast = parser.parse();
        if (ast) {
            std::cout << "Parse successful!" << std::endl;
        } else {
            std::cout << "Parse returned null" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Parse error: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Test the failing expressions from the test
    debug_tokenization("< ./ 5 2 8");
    debug_tokenization("> ./  5 2 8");
    debug_tokenization("(+/ % #) 5 2 8");
    
    // Test what should work in J
    debug_tokenization("<./ 5 2 8");  // No space
    debug_tokenization(">./ 5 2 8");  // No space
    
    return 0;
}
