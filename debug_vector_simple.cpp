#include <iostream>
#include "src/lexer/lexer.hpp"
#include "src/parser/parser.hpp"
#include "src/ast/ast_nodes.hpp"

using namespace JInterpreter;

int main() {
    std::string input = "1 2 3 + 4 5 6";
    std::cout << "Parsing: " << input << std::endl;
    
    Lexer lexer(input);
    auto tokens = lexer.tokenize();
    
    std::cout << "Tokens: ";
    for (const auto& token : tokens) {
        std::cout << "[" << static_cast<int>(token.type) << ":" << token.lexeme << "] ";
    }
    std::cout << std::endl;
    
    Parser parser(tokens);
    auto ast = parser.parse();
    
    if (ast) {
        std::cout << "AST parsed successfully:" << std::endl;
        ast->print(std::cout, 0);
    } else {
        std::cout << "Failed to parse AST" << std::endl;
    }
    
    return 0;
}
