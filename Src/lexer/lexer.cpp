#include "lexer.hpp"
#include <stdexcept> // For error reporting (can be refined)
#include <cctype>    // For isalpha, isdigit
#include <iostream>  // For debug prints (temporary)
#include <unordered_map>

namespace JInterpreter {

// Basic map for J keywords (can be expanded)
// J is case-sensitive for names, but control words are specific.
const std::unordered_map<std::string, TokenType> keywords = {
    {"NB.", TokenType::COMMENT}, // Special case, usually handled before identifier
    {"if.", TokenType::IF},
    {"do.", TokenType::DO},
    {"else.", TokenType::ELSE},
    {"elseif.", TokenType::ELSEIF},
    {"end.", TokenType::END},
    {"select.", TokenType::SELECT},
    {"case.", TokenType::CASE},
    {"while.", TokenType::WHILE},
    {"for.", TokenType::FOR_FRAMENAME}, // for_x.
    // Add other J keywords
};

// J primitives (a very small subset for stub)
// In J, these are often single characters, but context (noun/verb/etc.) is key.
// The lexer might tokenize generically, parser assigns role, or lexer uses simple rules.
// For now, we'll tokenize based on common patterns.
const std::unordered_map<std::string, TokenType> primitives = {
    {"+", TokenType::VERB}, {"-", TokenType::VERB}, {"*", TokenType::VERB},
    {"%", TokenType::VERB}, {"#", TokenType::VERB}, {"<", TokenType::VERB}, // < can be BOX or VERB
    {">", TokenType::VERB},
    {"/", TokenType::ADVERB}, {"\\", TokenType::ADVERB},
    {"^:", TokenType::CONJUNCTION}, {".", TokenType::VERB}, // . can be part of numbers or verb/conj
    {":", TokenType::COLON}, // Can be part of conjunction or explicit def
    {"(LS)", TokenType::CONJUNCTION}, // <: Left Shoe (example, this isn't how <: is typically tokenized)
};


Lexer::Lexer(std::string source) : m_source(std::move(source)) {}

char Lexer::peek(size_t offset) const {
    if (m_current_pos + offset >= m_source.length()) {
        return '\0';
    }
    return m_source[m_current_pos + offset];
}

char Lexer::advance() {
    if (!is_at_end()) {
        m_current_pos++;
    }
    return m_source[m_current_pos - 1];
}

bool Lexer::is_at_end() const {
    return m_current_pos >= m_source.length();
}

SourceLocation Lexer::current_location() const {
    // Column is 1-based
    return SourceLocation(m_current_line, static_cast<int>(m_current_pos - m_line_start_pos) +1);
}

Token Lexer::make_token(TokenType type, const std::string& lexeme_override) {
    std::string lexeme_val = lexeme_override;
    if (lexeme_val.empty() && !is_at_end()) { // Default to single char if no override
        lexeme_val = m_source.substr(m_current_pos -1, 1); // Assuming advance was called
    } else if (lexeme_val.empty() && is_at_end() && type == TokenType::END_OF_FILE) {
        lexeme_val = "EOF";
    }
    // Adjust location slightly if making token after advancing for multi-char
    int col = static_cast<int>(m_current_pos - m_line_start_pos);
    if (!lexeme_val.empty()) col = static_cast<int>(m_current_pos - m_line_start_pos - lexeme_val.length()) +1;


    return Token(type, lexeme_val, SourceLocation(m_current_line, col));
}

Token Lexer::make_token_with_literal(TokenType type, const std::string& lexeme,
                                     const std::variant<std::monostate, long long, double, std::string>& literal) {
    int col = static_cast<int>(m_current_pos - m_line_start_pos - lexeme.length()) +1;
    return Token(type, lexeme, SourceLocation(m_current_line, col), literal);
}


bool Lexer::match(char expected) {
    if (is_at_end()) return false;
    if (peek() != expected) return false;
    m_current_pos++;
    return true;
}


Token Lexer::number() {
    size_t start_pos = m_current_pos -1; // Already consumed the first digit
    bool is_float = false;
    bool is_negative_literal = false; // J's `_` for negative, e.g. `_5`

    if (m_source[start_pos] == '_') { // J negative literal
        is_negative_literal = true;
        start_pos = m_current_pos; // Start actual number after _
        // advance(); // Consume the char after _ if it's a digit
    }


    while (isdigit(peek())) {
        advance();
    }

    if (peek() == '.' && isdigit(peek(1))) {
        is_float = true;
        advance(); // Consume '.'
        while (isdigit(peek())) {
            advance();
        }
    }
    // Could add J's 'e' for scientific notation or 'j' for complex, 'x' for extended, 'r' for rational

    std::string lexeme = m_source.substr(start_pos, m_current_pos - start_pos);
    if (is_negative_literal) lexeme.insert(0, "_");


    if (is_float) {
        try {
            // J uses _ for negative, std::stod doesn't directly support it.
            // We'd need to replace `_` with `-` before parsing for standard libraries.
            std::string val_str = lexeme;
            if (val_str.front() == '_') val_str[0] = '-';
            return make_token_with_literal(TokenType::NOUN_FLOAT, lexeme, std::stod(val_str));
        } catch (const std::out_of_range& oor) {
            // Handle error: number too large/small
            return make_token(TokenType::UNKNOWN, lexeme);
        } catch (const std::invalid_argument& ia) {
            return make_token(TokenType::UNKNOWN, lexeme);
        }
    } else {
         try {
            std::string val_str = lexeme;
            if (val_str.front() == '_') val_str[0] = '-';
            return make_token_with_literal(TokenType::NOUN_INTEGER, lexeme, std::stoll(val_str));
        } catch (const std::out_of_range& oor) {
            return make_token(TokenType::UNKNOWN, lexeme);
        } catch (const std::invalid_argument& ia) {
            return make_token(TokenType::UNKNOWN, lexeme);
        }
    }
}

Token Lexer::string_literal() {
    size_t start_pos = m_current_pos; // After the opening '
    std::string value;
    while (peek() != '\'' && !is_at_end()) {
        if (peek() == '\'' && peek(1) == '\'') { // Escaped apostrophe ''
            value += '\'';
            advance(); // consume first '
        } else {
            value += peek();
        }
        advance();
    }

    if (is_at_end()) {
        // Unterminated string
        return make_token(TokenType::UNKNOWN, m_source.substr(start_pos -1, m_current_pos - (start_pos-1)));
    }

    advance(); // Consume the closing '
    std::string lexeme = m_source.substr(start_pos -1, m_current_pos - (start_pos -1));
    return make_token_with_literal(TokenType::NOUN_STRING, lexeme, value);
}

Token Lexer::identifier_or_keyword() {
    size_t start_pos = m_current_pos - 1; // Already consumed first char
    // J names can end with . or : for type indication, or contain _
    // Simple identifier: letter followed by letters, digits, or underscore.
    while (isalnum(peek()) || peek() == '_' ) { // J names can contain _
        advance();
    }
    // J specific: names can end with . or :
    if (peek() == '.' || peek() == ':') {
         if (isalnum(peek(-1))) { // Ensure it's part of the name, not a separate operator
            advance();
         }
    }


    std::string lexeme = m_source.substr(start_pos, m_current_pos - start_pos);
    
    // Check for keywords like if. do. etc.
    // Note: J keywords often end with a dot.
    auto it_kw = keywords.find(lexeme);
    if (it_kw != keywords.end()) {
        return make_token(it_kw->second, lexeme);
    }
    
    // Check for user-defined names that might shadow primitives (handled by parser context)
    // For now, if not a keyword, it's a NAME
    return make_token(TokenType::NAME, lexeme);
}


Token Lexer::comment() { // Assumes "NB." has been identified
    size_t start_pos = m_current_pos - 3; // Start of "NB."
    while (peek() != '\n' && !is_at_end()) {
        advance();
    }
    std::string lexeme = m_source.substr(start_pos, m_current_pos - start_pos);
    return make_token(TokenType::COMMENT, lexeme);
}


Token Lexer::scan_token() {
    char c = advance();

    // Whitespace
    if (isspace(c) && c != '\n') {
        // Could optionally produce WHITESPACE tokens, or skip.
        // For J, usually skip.
        while (isspace(peek()) && peek() != '\n') advance();
        return scan_token(); // Re-scan after skipping
    }

    switch (c) {
        case '\n':
            m_current_line++;
            m_line_start_pos = m_current_pos;
            return make_token(TokenType::NEWLINE, "\\n");
        case '(': return make_token(TokenType::LEFT_PAREN, "(");
        case ')': return make_token(TokenType::RIGHT_PAREN, ")");
        case '\'': return string_literal();
        
        case '_': // Could be start of negative number, or part of name
            if (isdigit(peek())) return number(); // Start of negative number like _5
            // else, could be part of an identifier if first char (less common in J)
            // or treated as a verb if standalone. For now, let identifier handle it.
            break; 

        case '.': // Can be part of number, assign, verb, conjunction
            if (match(' ')) { /* ignore for now, context dependent */ }
            if (isdigit(peek())) { // Part of a float like .5, J might not use this form often
                 // Put back the '.' to be consumed by number() if it expects it
                 m_current_pos--;
                 return number();
            }
            // Check for =. (handled by '=' case) or other dot-ending ops
            // If standalone, it's a VERB or CONJUNCTION based on context (parser differentiates)
            // For now, a simple rule:
            if (match('=')) return make_token(TokenType::ASSIGN_LOCAL, "=."); // J is specific =. not .=
            // Check for specific J primitives ending in dot (e.g., some adverbs/conjunctions)
            // If just '.', it's often a verb or conjunction.
            return make_token(TokenType::VERB, "."); // Placeholder, parser resolves

        case ':': // Can be assign, verb, conjunction
            if (match('=')) return make_token(TokenType::ASSIGN_GLOBAL, "=:");// J is specific =: not :=
            // Could be part of ^: or other primitives
            // If standalone, often a CONJUNCTION
            return make_token(TokenType::COLON, ":"); // Placeholder, parser resolves

        case '=': // =, =., =:
            if (match('.')) return make_token(TokenType::ASSIGN_LOCAL, "=.");
            if (match(':')) return make_token(TokenType::ASSIGN_GLOBAL, "=:");
            return make_token(TokenType::VERB, "="); // Equality verb

        // Single char verbs/adverbs/conjunctions (simplified)
        case '+': case '-': case '*': case '%': case '#':
        case '<': case '>': case '$': case '~': case '|':
        // ... many more J primitives
            // Check for multi-char primitives starting with this char
            // E.g. <: >: %: etc.
            if (c == '<' && match(':')) return make_token(TokenType::VERB, "<:");
            if (c == '>' && match(':')) return make_token(TokenType::VERB, ">:");
            // ... other multi-char primitives
            return make_token(TokenType::VERB, std::string(1, c)); // Default to VERB, parser differentiates role

        case '/': case '\\': // Adverbs
             // Check for /: \:
            if (match(':')) return make_token(TokenType::ADVERB, std::string(1,c) + ":");
            return make_token(TokenType::ADVERB, std::string(1,c));

        case '^': // Conjunction
            if (match(':')) return make_token(TokenType::CONJUNCTION, "^:");
            return make_token(TokenType::CONJUNCTION, "^"); // If ^ itself is a primitive

        // ... Add more cases for all J special characters and primitives
    }

    if (isalpha(c) || c == '_') { // Identifiers can start with letter or _ (though _ more for negatives)
        // Check for NB. comment first if not handled by a direct switch
        if (c == 'N' && peek(0) == 'B' && peek(1) == '.') {
            advance(); advance(); // Consume B.
            return comment();
        }
        return identifier_or_keyword();
    }
    if (isdigit(c)) {
        return number();
    }


    return make_token(TokenType::UNKNOWN, std::string(1,c));
}


std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    m_current_pos = 0;
    m_current_line = 1;
    m_line_start_pos = 0;

    while (!is_at_end()) {
        // Store start before scan_token might skip whitespace
        size_t start_current_pos = m_current_pos;
        int start_line = m_current_line;
        int start_line_pos = m_line_start_pos;

        Token token = scan_token();

        // If scan_token skipped whitespace and returned by recursive call,
        // the current_pos might have advanced significantly.
        // We only add non-whitespace tokens here if scan_token doesn't already skip.
        // The current stub for scan_token *does* skip initial whitespace then re-calls.
        if (token.type != TokenType::WHITESPACE && token.type != TokenType::COMMENT) { // Optionally filter comments
             tokens.push_back(token);
        }
        if (token.type == TokenType::UNKNOWN) {
             // Potentially log error or throw
             std::cerr << "Unknown token: " << token.lexeme << " at " << token.location << std::endl;
        }
        if (token.type == TokenType::END_OF_FILE) break; // Should not happen here if !is_at_end()

        // Safety break for misbehaving scan_token in dev
        if (m_current_pos == start_current_pos && !is_at_end() && token.type != TokenType::END_OF_FILE) {
            if (!is_at_end()) advance(); // Force progress
            std::cerr << "Lexer stuck! Force advancing." << std::endl;
            tokens.push_back(make_token(TokenType::UNKNOWN, m_source.substr(start_current_pos,1)));
            if (m_current_pos >= m_source.length()) break;
        }
    }

    tokens.push_back(Token(TokenType::END_OF_FILE, "EOF", current_location()));
    return tokens;
}

} // namespace JInterpreter
