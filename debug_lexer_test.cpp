#include "src/lexer/lexer.hpp"
#include <iostream>

int main() {
    std::cout << "Testing lexer tokenization of compound adverbs and comma:" << std::endl;
    
    // Test cases
    std::vector<std::string> test_cases = {
        "<./ 5 2 8",
        ">./  5 2 8",
        "(+/ % #) 5 2 8",
        "1 2 , 3 4 5"
    };
    
    for (const auto& test : test_cases) {
        std::cout << "Tokenizing: '" << test << "'" << std::endl;
        JInterpreter::Lexer lexer(test);
        auto tokens = lexer.tokenize();
        
        for (const auto& token : tokens) {
            std::cout << "  " << token << std::endl;
        }
        std::cout << std::endl;
    }
    
    return 0;
}
