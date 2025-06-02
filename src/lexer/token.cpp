#include "token.hpp"

namespace JInterpreter {

std::ostream& operator<<(std::ostream& os, TokenType type) {
    switch (type) {
        case TokenType::NOUN_INTEGER: os << "NOUN_INTEGER"; break;
        case TokenType::NOUN_FLOAT: os << "NOUN_FLOAT"; break;
        case TokenType::NOUN_STRING: os << "NOUN_STRING"; break;
        case TokenType::VERB: os << "VERB"; break;
        case TokenType::ADVERB: os << "ADVERB"; break;
        case TokenType::CONJUNCTION: os << "CONJUNCTION"; break;
        case TokenType::NAME: os << "NAME"; break;
        case TokenType::ASSIGN_LOCAL: os << "ASSIGN_LOCAL"; break;
        case TokenType::ASSIGN_GLOBAL: os << "ASSIGN_GLOBAL"; break;
        case TokenType::LEFT_PAREN: os << "LEFT_PAREN"; break;
        case TokenType::RIGHT_PAREN: os << "RIGHT_PAREN"; break;
        case TokenType::COMMA: os << "COMMA"; break;
        case TokenType::APOSTROPHE: os << "APOSTROPHE"; break;
        case TokenType::COLON: os << "COLON"; break;
        case TokenType::IF: os << "IF"; break;
        // ... add all other token types
        case TokenType::COMMENT: os << "COMMENT"; break;
        case TokenType::NEWLINE: os << "NEWLINE"; break;
        case TokenType::WHITESPACE: os << "WHITESPACE"; break;
        case TokenType::END_OF_FILE: os << "END_OF_FILE"; break;
        case TokenType::UNKNOWN: os << "UNKNOWN"; break;
        default: os << "UNHANDLED_TOKEN_TYPE"; break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Token& token) {
    os << "Token(" << token.type << ", '" << token.lexeme << "', " << token.location;
    std::visit([&os](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            if constexpr (std::is_same_v<T, std::string>) {
                 os << ", Lit:'" << arg << "'";
            } else {
                 os << ", Lit:" << arg;
            }
        }
    }, token.literal_value);
    os << ")";
    return os;
}

} // namespace JInterpreter
