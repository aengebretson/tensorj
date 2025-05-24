#include <iostream>
#include <string>
#include <vector>
#include "j_interpreter_lib.hpp" // Umbrella header

void print_tokens(const std::vector<JInterpreter::Token>& tokens) {
    for (const auto& token : tokens) {
        std::cout << token << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "J Interpreter (C++ Prototype)" << std::endl;
    std::cout << "Current Date (Compile Time): " << __DATE__ << " " << __TIME__ << std::endl;
    // Note: The user's request for current date during execution needs runtime handling, not compile time.
    // For simplicity, this example doesn't include runtime date fetching for the prompt.

    JInterpreter::Interpreter j_interpreter_instance; // Create an instance

    std::string line;
    while (true) {
        std::cout << "   "; // J's typical prompt
        if (!std::getline(std::cin, line)) {
            break; // EOF or error
        }
        if (line == "quit" || line == "exit") {
            break;
        }
        if (line.empty()) {
            continue;
        }

        try {
            JInterpreter::Lexer lexer(line);
            std::vector<JInterpreter::Token> tokens = lexer.tokenize();
            std::cout << "Tokens:" << std::endl;
            print_tokens(tokens);

            if (tokens.empty() || (tokens.size() == 1 && tokens[0].type == JInterpreter::TokenType::END_OF_FILE)) {
                continue;
            }
            
            JInterpreter::Parser parser(tokens);
            std::unique_ptr<JInterpreter::AstNode> ast_root = parser.parse();

            if (ast_root) {
                std::cout << "AST:" << std::endl;
                ast_root->print(std::cout, 0);

                std::cout << "Evaluation Result:" << std::endl;
                JInterpreter::JValue result = j_interpreter_instance.evaluate(ast_root.get());
                // Print JValue (needs a proper print function for the variant)
                 std::visit([](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::nullptr_t>) {
                        std::cout << "(null)";
                    } else if constexpr (std::is_same_v<T, std::string>) {
                        std::cout << "'" << arg << "'";
                    } else if constexpr (std::is_same_v<T, std::shared_ptr<JInterpreter::JTensor>>) {
                        if (arg) {
                            arg->print(std::cout);
                        } else {
                            std::cout << "(null tensor)";
                        }
                    } else {
                        std::cout << arg;
                    }
                }, result);
                std::cout << std::endl;

            } else {
                std::cout << "Parser returned null AST (likely parse error)." << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
