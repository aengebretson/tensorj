#ifndef J_INTERPRETER_PARSER_HPP
#define J_INTERPRETER_PARSER_HPP

#include <vector>
#include <memory> // For std::unique_ptr for AST nodes
#include "lexer/token.hpp"
#include "ast/ast_nodes.hpp"

namespace JInterpreter {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    std::unique_ptr<AstNode> parse(); // Returns the root of the AST (e.g., a StatementList)

private:
    std::vector<Token> m_tokens;
    size_t m_current_token_idx = 0;

    const Token& peek(size_t offset = 0) const;
    const Token& previous() const;
    const Token& advance();
    bool is_at_end() const;
    bool check(TokenType type) const; // Check current token type without consuming
    bool match(const std::vector<TokenType>& types); // Consume if current token matches one of types

    SourceLocation current_loc() const; // Location of current token

    // Error handling (can be more sophisticated)
    void error(const Token& token, const std::string& message);
    // void synchronize(); // For error recovery

    // Parsing methods for different grammar rules (recursive descent / Pratt)
    // These will be built out incrementally
    std::unique_ptr<AstNode> parse_statement();
    std::unique_ptr<AstNode> parse_expression(int min_precedence); 
    std::unique_ptr<AstNode> parse_expression(); // Main entry for expression parsing
    
    // Pratt parser style methods:
    // NUD (Null Denotation): For tokens that start an expression (literals, prefixes)
    std::unique_ptr<AstNode> nud(const Token& token);
    // LED (Left Denotation): For tokens that are infix or suffix
    std::unique_ptr<AstNode> led(const Token& token, std::unique_ptr<AstNode> left_node);
    
    // Precedence levels for operators/verbs (J specific)
    int get_token_precedence(const Token& token) const;


    // Simpler recursive descent stubs (can be replaced/augmented by Pratt)
    std::unique_ptr<AstNode> parse_primary();        // Literals, names, parenthesized expr
    std::unique_ptr<AstNode> parse_dyadic_expression(); // Right-to-left dyadic expressions
    std::unique_ptr<AstNode> parse_assignment(std::unique_ptr<AstNode> name_node); // If an expr starts with a name that is then assigned

    // J specific parsing
    std::unique_ptr<AstNode> parse_verb_application(std::unique_ptr<AstNode> left_operand); // Handles monadic/dyadic/trains
    std::unique_ptr<AstNode> parse_train(std::unique_ptr<AstNode> first_verb_expr);

    // Helper functions for monadic application detection
    bool is_verb_like(const AstNode* node) const;
    bool can_be_argument(const Token& token) const;

    // Utility to consume a token
    const Token& consume(TokenType type, const std::string& error_message);
};

} // namespace JInterpreter

#endif // J_INTERPRETER_PARSER_HPP
