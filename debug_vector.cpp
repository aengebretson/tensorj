#include <iostream>
#include "src/lexer/lexer.hpp"
#include "src/parser/parser.hpp"
#include "src/interpreter/interpreter.hpp"

using namespace JInterpreter;

int main() {
    try {
        std::string input = "1 2 3";
        std::cout << "Parsing: " << input << std::endl;
        
        // Tokenize
        Lexer lexer(input);
        auto tokens = lexer.tokenize();
        std::cout << "Tokens:" << std::endl;
        for (const auto& token : tokens) {
            std::cout << "  " << token << std::endl;
        }
        
        // Parse
        Parser parser(std::move(tokens));
        auto ast = parser.parse();
        if (!ast) {
            std::cout << "Failed to parse" << std::endl;
            return 1;
        }
        
        std::cout << "AST:" << std::endl;
        ast->print(std::cout, 0);
        
        // Evaluate
        Interpreter interpreter;
        auto result = interpreter.evaluate(ast.get());
        
        if (std::holds_alternative<std::shared_ptr<JTensor>>(result)) {
            auto tensor = std::get<std::shared_ptr<JTensor>>(result);
            if (tensor) {
                std::cout << "Result tensor rank: " << tensor->rank() << std::endl;
                if (tensor->rank() > 0) {
                    std::cout << "Shape: [";
                    for (int i = 0; i < tensor->rank(); ++i) {
                        if (i > 0) std::cout << ", ";
                        std::cout << tensor->shape()[i];
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                std::cout << "Null tensor result" << std::endl;
            }
        } else {
            std::cout << "Result is not a tensor" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    
    return 0;
}
