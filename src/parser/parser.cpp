#include "parser.hpp"
#include <stdexcept> // For errors
#include <iostream>  // For debug/error messages
#include <variant>   // For std::holds_alternative and std::get

namespace JInterpreter {

Parser::Parser(std::vector<Token> tokens) : m_tokens(std::move(tokens)) {}

const Token& Parser::peek(size_t offset) const {
    if (m_current_token_idx + offset >= m_tokens.size()) {
        return m_tokens.back(); // Should be EOF
    }
    return m_tokens[m_current_token_idx + offset];
}

const Token& Parser::previous() const {
    if (m_current_token_idx == 0) {
        // This case should ideally not happen if previous() is called logically
        static Token start_sentinel(TokenType::UNKNOWN, "", {0,0}, std::monostate{});
        return start_sentinel; // Or throw
    }
    return m_tokens[m_current_token_idx - 1];
}


const Token& Parser::advance() {
    if (!is_at_end()) {
        m_current_token_idx++;
    }
    return previous();
}

bool Parser::is_at_end() const {
    return peek().type == TokenType::END_OF_FILE;
}

bool Parser::check(TokenType type) const {
    if (is_at_end()) return false;
    return peek().type == type;
}

bool Parser::match(const std::vector<TokenType>& types) {
    for (TokenType type : types) {
        if (check(type)) {
            advance();
            return true;
        }
    }
    return false;
}

SourceLocation Parser::current_loc() const {
    return peek().location;
}

void Parser::error(const Token& token, const std::string& message) {
    std::cerr << "Parse Error at " << token.location << " (token: '" << token.lexeme << "' " << token.type << "): " << message << std::endl;
    // For now, we'll throw to stop. Real parsers might try to synchronize.
    throw std::runtime_error("Parser error: " + message);
}

const Token& Parser::consume(TokenType type, const std::string& error_message) {
    if (check(type)) return advance();
    error(peek(), error_message);
    // Should not be reached due to throw in error()
    return peek(); // To satisfy compiler if error doesn't throw
}


// --- Pratt Parser Inspired Stubs ---
// J's grammar is not strictly infix, so a pure Pratt might need adaptation.
// The "right-to-left" nature is key.
// We might parse operands, then look left for verbs, or parse right-to-left.

// Simplified precedence for demonstration. J's is more nuanced.
// Higher numbers bind tighter.
// Nouns (operands) are not operators, but they terminate a chain to their left.
// Verbs have precedence. Adverbs/Conjunctions modify verbs and effectively have higher precedence w.r.t their verb.
int Parser::get_token_precedence(const Token& token) const {
    switch (token.type) {
        // For J, assignment might be lower precedence than most verbs
        case TokenType::ASSIGN_LOCAL:
        case TokenType::ASSIGN_GLOBAL:
            return 10;

        // Dyadic verbs (example)
        case TokenType::VERB: // General VERB token, specific verbs (+, -, *, %) might have diff precedence
            // For J, most verbs have similar "right-associative" behavior
            // Let's give a baseline for dyadic application
            if (token.lexeme == "+" || token.lexeme == "-") return 20;
            if (token.lexeme == "*" || token.lexeme == "%") return 30;
            // Monadic verbs are handled by NUD if they are prefix
            return 5; // Default low for unhandled verbs as ops

        // Adverbs and Conjunctions (when applied, they form a new verb-like thing)
        // These effectively have high precedence with the verb they modify.
        // This is handled by how they are parsed (e.g. verb_phrase -> verb adverb)
        // rather than as separate infix operators in a Pratt loop.

        default:
            return 0; // Not an operator in this context
    }
}


std::unique_ptr<AstNode> Parser::nud(const Token& token) {
    switch (token.type) {
        case TokenType::NOUN_INTEGER:
        case TokenType::NOUN_FLOAT:
        case TokenType::NOUN_STRING:
            // The literal value from token should be used
            if (std::holds_alternative<long long>(token.literal_value)) {
                return std::make_unique<NounLiteralNode>(std::get<long long>(token.literal_value), token.location);
            } else if (std::holds_alternative<double>(token.literal_value)) {
                return std::make_unique<NounLiteralNode>(std::get<double>(token.literal_value), token.location);
            } else if (std::holds_alternative<std::string>(token.literal_value)) {
                return std::make_unique<NounLiteralNode>(std::get<std::string>(token.literal_value), token.location);
            }
            error(token, "Unsupported literal type in NUD.");
            return nullptr; // Should not reach

        case TokenType::NAME:
            return std::make_unique<NameNode>(token.lexeme, token.location);

        case TokenType::LEFT_PAREN: {
            auto expr = parse_expression(); // Or parse_expression_with_precedence(0) for Pratt
            consume(TokenType::RIGHT_PAREN, "Expected ')' after parenthesized expression.");
            return expr; // Could wrap in ParenExpressionNode if distinction is needed
        }

        case TokenType::VERB: // Monadic prefix verbs, e.g., `- 5` (negate), `# table` (count)
            {
                // This is a monadic application if no left operand was present.
                // The Pratt `parse_expression_with_precedence` loop would call NUD for this.
                // The right operand is parsed with a precedence that allows the monadic verb to bind.
                // Example: `- x * y` should be `(-x) * y` if `-` is high-precedence monadic.
                // In J: `- y` (negate y). `V y`.
                // For J's right-to-left, this is more direct: parse verb, then parse its right operand.
                auto verb_node = std::make_unique<VerbNode>(token.lexeme, token.location);
                auto right_operand = parse_expression(); // This needs refinement for precedence.
                                                         // For J, it's often parse_expression_until_lower_precedence_or_end_of_scope
                return std::make_unique<MonadicApplicationNode>(std::move(verb_node), std::move(right_operand), token.location);
            }


        // TODO: Add cases for monadic adverbs/conjunctions if they can start expressions
        // e.g. a defined adverb used monadically if J syntax allows.

        default:
            error(token, "Expected an expression (literal, name, (, or prefix operator).");
            return nullptr; // Should not reach
    }
}

std::unique_ptr<AstNode> Parser::led(const Token& token, std::unique_ptr<AstNode> left_node) {
    // This is for infix-like operators. J's dyadic verbs are like this.
    // `x V y`. `left_node` is `x`, `token` is `V`.
    switch (token.type) {
        case TokenType::VERB: {
            auto verb_node = std::make_unique<VerbNode>(token.lexeme, token.location);
            // Parse the right-hand operand with precedence of the current verb
            // In J, this is generally "the rest of the expression to the right"
            // or up to the next verb of lower "conceptual" precedence (though J is mostly right-to-left).
            auto right_operand = parse_expression(); // Needs refinement for precedence.
                                                     // Or parse_expression_with_precedence(get_token_precedence(token)) for Pratt.
            return std::make_unique<DyadicApplicationNode>(std::move(left_node), std::move(verb_node), std::move(right_operand), token.location);
        }

        case TokenType::ASSIGN_LOCAL:
        case TokenType::ASSIGN_GLOBAL: {
            if (left_node->type != AstNodeType::NAME_IDENTIFIER) {
                error(token, "Left-hand side of assignment must be a name.");
            }
            auto target_name = std::unique_ptr<NameNode>(static_cast<NameNode*>(left_node.release()));
            auto value_expr = parse_expression(); // Or parse_expression_with_precedence(get_token_precedence(token)-1) to make it right-associative
            // TODO: Create AssignmentNode
            error(token, "AssignmentNode not yet implemented in LED.");
            return nullptr;
        }

        // Adverbs and Conjunctions in J are typically not "infix" in the Pratt sense.
        // They modify a verb. `verb adverb` or `verb1 conjunction verb2`.
        // This is usually handled by parsing a "verb phrase" rather than a simple LED.
        // E.g., if `left_node` is a verb, and `token` is an adverb, LED could form AdverbApplication.

        default:
            error(token, "Unexpected token in LED (expected infix operator or verb).");
            return nullptr; // Should not reach
    }
}


// Main parsing function using right-to-left recursive descent for J
std::unique_ptr<AstNode> Parser::parse_expression() {
    if (is_at_end()) {
        return std::make_unique<NounLiteralNode>(nullptr, current_loc()); // Empty expression
    }

    return parse_dyadic_expression();
}

// Parse dyadic expressions with right-to-left associativity
std::unique_ptr<AstNode> Parser::parse_dyadic_expression() {
    std::unique_ptr<AstNode> left = parse_primary();
    
    // If parse_primary failed to parse anything (e.g., unexpected token), return null
    if (!left) {
        return nullptr;
    }

    // Check if there's a verb following the left operand
    if (!is_at_end() && peek().type == TokenType::VERB) {
        Token verb_token = advance(); // Consume verb
        auto verb_node = std::make_unique<VerbNode>(verb_token.lexeme, verb_token.location);
        
        // For right-to-left associativity, recursively parse the entire right expression
        // This will handle chains like "1 + 2 * 3" as "1 + (2 * 3)"
        std::unique_ptr<AstNode> right = parse_dyadic_expression();
        
        // If the right side failed to parse, we still have a valid left + verb, so this is an error
        if (!right) {
            error(verb_token, "Expected expression after verb.");
            return nullptr;
        }
        
        return std::make_unique<DyadicApplicationNode>(std::move(left), std::move(verb_node), std::move(right), verb_token.location);
    } else if (!is_at_end() && (peek().type == TokenType::ASSIGN_LOCAL || peek().type == TokenType::ASSIGN_GLOBAL)) {
        if (left->type != AstNodeType::NAME_IDENTIFIER) {
            error(peek(), "Left-hand side of assignment must be a name.");
        }
        Token assign_token = advance();
        // Create AssignmentNode...
        std::cout << "Assignment parsing not fully implemented yet." << std::endl;
        auto value_expr = parse_dyadic_expression(); // Recursive call
        // For now, just return the value to avoid more errors
        return value_expr;
    }
    
    return left;
}


std::unique_ptr<AstNode> Parser::parse_primary() {
    if (match({TokenType::NOUN_INTEGER, TokenType::NOUN_FLOAT, TokenType::NOUN_STRING})) {
        const Token& t = previous();
         if (std::holds_alternative<long long>(t.literal_value)) {
            return std::make_unique<NounLiteralNode>(std::get<long long>(t.literal_value), t.location);
        } else if (std::holds_alternative<double>(t.literal_value)) {
            return std::make_unique<NounLiteralNode>(std::get<double>(t.literal_value), t.location);
        } else if (std::holds_alternative<std::string>(t.literal_value)) {
            return std::make_unique<NounLiteralNode>(std::get<std::string>(t.literal_value), t.location);
        }
        error(t, "Unsupported literal type in primary.");
        return nullptr;
    }
    if (match({TokenType::NAME})) {
        return std::make_unique<NameNode>(previous().lexeme, previous().location);
    }
    if (match({TokenType::LEFT_PAREN})) {
        SourceLocation paren_loc = previous().location;
        auto expr = parse_expression();
        consume(TokenType::RIGHT_PAREN, "Expected ')' after expression.");
        // Optionally wrap in a ParenExpressionNode if distinction is needed
        // return std::make_unique<ParenExpressionNode>(std::move(expr), paren_loc);
        return expr;
    }

    // Handle monadic prefix verbs if parse_expression doesn't cover them via NUD
    if (peek().type == TokenType::VERB) { // E.g. `# table` or `- value`
        Token verb_token = advance();
        auto verb_ast_node = std::make_unique<VerbNode>(verb_token.lexeme, verb_token.location);
        auto operand_ast_node = parse_primary(); // Recursive call for the operand
        return std::make_unique<MonadicApplicationNode>(std::move(verb_ast_node), std::move(operand_ast_node), verb_token.location);
    }

    // Handle unexpected tokens - for certain tokens like RIGHT_PAREN, return null to stop parsing gracefully
    if (peek().type == TokenType::RIGHT_PAREN) {
        // Don't consume the token, just return null to indicate we can't parse a primary here
        // This allows the caller to handle the unexpected token appropriately
        return nullptr;
    }

    // For other unexpected tokens, we should throw an error
    error(peek(), "Expected primary expression (literal, name, or '(').");
    return nullptr; // Should not reach
}

// Placeholder for a statement
std::unique_ptr<AstNode> Parser::parse_statement() {
    // For J, a line is usually a single expression.
    // Could also be an assignment.
    // If we see NAME followed by ASSIGN_LOCAL/GLOBAL, it's an assignment.
    if (peek(0).type == TokenType::NAME && 
        (peek(1).type == TokenType::ASSIGN_LOCAL || peek(1).type == TokenType::ASSIGN_GLOBAL)) {
        
        Token name_token = advance();
        Token assign_token = advance();

        auto name_node = std::make_unique<NameNode>(name_token.lexeme, name_token.location);
        auto value_expr = parse_expression();
        
        // TODO: Create and return an AssignmentNode
        // return std::make_unique<AssignmentNode>(std::move(name_node), std::move(value_expr), assign_token.type == TokenType::ASSIGN_GLOBAL, assign_token.location);
        std::cout << "Assignment parsing not fully implemented yet." << std::endl;
        return value_expr; // Temporary
    }

    auto expr = parse_expression();
    
    // If parse_expression failed (returned nullptr), we should handle the error
    if (!expr) {
        // For now, just return a null literal to indicate a failed parse
        // In a real parser, this might trigger error recovery
        return std::make_unique<NounLiteralNode>(nullptr, current_loc());
    }
    
    return expr;
}


std::unique_ptr<AstNode> Parser::parse() {
    std::vector<std::unique_ptr<AstNode>> statements;
    while (!is_at_end()) {
        if (peek().type == TokenType::NEWLINE) { // Skip empty lines or treat as statement separators
            advance();
            continue;
        }
        
        auto stmt = parse_statement();
        if (stmt) {
            statements.push_back(std::move(stmt));
        }
        
        // After a statement, if there are still tokens and they're not newline or EOF,
        // we have a syntax error (like an unmatched right paren)
        if (!is_at_end() && peek().type != TokenType::NEWLINE) {
            // Check for common error cases
            if (peek().type == TokenType::RIGHT_PAREN) {
                // Unmatched right parenthesis - consume it and continue
                advance(); // Consume the problematic token
                // Return what we've parsed so far
                break;
            }
            // For other unexpected tokens, we could add more error handling here
            // For now, just break to avoid infinite loops
            break;
        }
    }

    if (statements.empty()) {
         return std::make_unique<NounLiteralNode>(nullptr, SourceLocation{1,1}); // Or a specific "EmptyProgramNode"
    }
    // For now, return the first statement's AST. A full program would be a list.
    // TODO: return a StatementListNode or similar
    return std::move(statements[0]);
}


} // namespace JInterpreter
