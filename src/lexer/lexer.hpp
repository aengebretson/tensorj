#ifndef J_INTERPRETER_LEXER_HPP
#define J_INTERPRETER_LEXER_HPP

#include <string>
#include <vector>
#include "token.hpp"

namespace JInterpreter {

class Lexer {
public:
    explicit Lexer(std::string source);

    std::vector<Token> tokenize();

private:
    std::string m_source;
    size_t m_current_pos = 0;
    int m_current_line = 1;
    int m_line_start_pos = 0; // To calculate column

    char peek(size_t offset = 0) const;
    char advance();
    bool is_at_end() const;
    SourceLocation current_location() const;

    Token make_token(TokenType type, const std::string& lexeme = "");
    Token make_token_with_literal(TokenType type, const std::string& lexeme,
                                  const std::variant<std::monostate, long long, double, std::string>& literal);


    // Token recognition methods
    Token scan_token();
    Token number();
    Token string_literal();
    Token identifier_or_keyword(); // Handles names and control words
    Token comment();
    Token handle_verb_with_dot(char c); // Handles verb variations with . and :
    // void skip_whitespace_and_comments(); // Or tokenize them if significant

    // Helper for multi-char ops like =. =:
    bool match(char expected);
};

} // namespace JInterpreter

#endif // J_INTERPRETER_LEXER_HPP
