#ifndef J_INTERPRETER_TOKEN_HPP
#define J_INTERPRETER_TOKEN_HPP

#include <string>
#include <variant>
#include <ostream>
#include "common/source_location.hpp"

namespace JInterpreter {

enum class TokenType {
    // Literals
    NOUN_INTEGER,
    NOUN_FLOAT,
    NOUN_STRING,
    // NOUN_BOX_START, // Might be '<' and parser differentiates

    // Primitives (can be further categorized if needed)
    VERB,       // e.g. +, -, *, %, #, <, >
    ADVERB,     // e.g. /, \, /:, ~., }.
    CONJUNCTION,// e.g. ^:, ., :, !.

    // Names/Identifiers
    NAME,

    // Assignment
    ASSIGN_LOCAL, // =.
    ASSIGN_GLOBAL,// =:

    // Punctuation / Operators not covered above
    LEFT_PAREN,   // (
    RIGHT_PAREN,  // )
    COMMA,        // , (for array separation)
    APOSTROPHE,   // ' (for string delimiting, might be part of NOUN_STRING processing)
    COLON,        // : (for explicit definitions, conjunctions)

    // Control Words
    IF, DO, ELSE, ELSEIF, END, SELECT, CASE, TRY, CATCH, WHILE, FOR_FRAMENAME,
    // (Add all relevant J control words)

    // Special
    COMMENT,      // NB.
    NEWLINE,
    WHITESPACE,   // Usually skipped but can be tokenized
    END_OF_FILE,
    UNKNOWN
};

// Overload for printing TokenType
std::ostream& operator<<(std::ostream& os, TokenType type);


struct Token {
    TokenType type;
    std::string lexeme; // The actual text of the token
    // For literals, the actual value can be stored here or processed later
    // For simplicity, lexeme is often enough for the parser, value conversion later.
    std::variant<std::monostate, long long, double, std::string> literal_value;
    SourceLocation location;

    Token(TokenType t, std::string lex, SourceLocation loc,
          std::variant<std::monostate, long long, double, std::string> val = std::monostate{})
        : type(t), lexeme(std::move(lex)), location(loc), literal_value(std::move(val)) {}

    friend std::ostream& operator<<(std::ostream& os, const Token& token);
};

} // namespace JInterpreter

#endif // J_INTERPRETER_TOKEN_HPP
