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
// int Parser::get_token_precedence(const Token& token) const {
//     switch (token.type) {
//         // For J, assignment might be lower precedence than most verbs
//         case TokenType::ASSIGN_LOCAL:
//         case TokenType::ASSIGN_GLOBAL:
//             return 10;

//         // Dyadic verbs (example)
//         case TokenType::VERB: // General VERB token, specific verbs (+, -, *, %) might have diff precedence
//             // For J, most verbs have similar "right-associative" behavior
//             // Let's give a baseline for dyadic application
//             if (token.lexeme == "+" || token.lexeme == "-") return 20;
//             if (token.lexeme == "*" || token.lexeme == "%") return 30;
//             // Monadic verbs are handled by NUD if they are prefix
//             return 5; // Default low for unhandled verbs as ops

//         // Adverbs and Conjunctions (when applied, they form a new verb-like thing)
//         // These effectively have high precedence with the verb they modify.
//         // This is handled by how they are parsed (e.g. verb_phrase -> verb adverb)
//         // rather than as separate infix operators in a Pratt loop.

//         default:
//             return 0; // Not an operator in this context
//     }
// }


std::unique_ptr<AstNode> Parser::nud(const Token& token) {
    switch (token.type) {
        case TokenType::NOUN_INTEGER:
        case TokenType::NOUN_FLOAT: {
            // Check for space-separated numbers to form vectors (e.g., "1 2 3")
            std::vector<NounValue> vector_elements;
            SourceLocation start_location = token.location;
            
            // Add the current token to vector elements
            if (std::holds_alternative<long long>(token.literal_value)) {
                vector_elements.push_back(std::get<long long>(token.literal_value));
            } else if (std::holds_alternative<double>(token.literal_value)) {
                vector_elements.push_back(std::get<double>(token.literal_value));
            } else {
                error(token, "Unsupported numeric literal type.");
                return nullptr;
            }
            
            // Collect consecutive numeric tokens (without advancing past them in the main parse loop)
            while (check(TokenType::NOUN_INTEGER) || check(TokenType::NOUN_FLOAT)) {
                Token num_token = advance();
                if (std::holds_alternative<long long>(num_token.literal_value)) {
                    vector_elements.push_back(std::get<long long>(num_token.literal_value));
                } else if (std::holds_alternative<double>(num_token.literal_value)) {
                    vector_elements.push_back(std::get<double>(num_token.literal_value));
                } else {
                    error(num_token, "Unsupported numeric literal type.");
                    return nullptr;
                }
            }
            
            // If we collected multiple numbers, create a vector
            if (vector_elements.size() > 1) {
                return std::make_unique<VectorLiteralNode>(std::move(vector_elements), start_location);
            } 
            // If only one number, create a single noun literal
            else {
                return std::make_unique<NounLiteralNode>(std::move(vector_elements[0]), start_location);
            }
        }
        
        case TokenType::NOUN_STRING:
            // Handle string literals separately (they don't form vectors)
            if (std::holds_alternative<std::string>(token.literal_value)) {
                return std::make_unique<NounLiteralNode>(std::get<std::string>(token.literal_value), token.location);
            }
            error(token, "Unsupported literal type in NUD.");
            return nullptr; // Should not reach

        case TokenType::NAME:
            return std::make_unique<NameNode>(token.lexeme, token.location);

        case TokenType::LEFT_PAREN: {
            SourceLocation paren_loc = token.location;
            
            // Look ahead to detect if this is a train pattern
            // Save current position to restore if not a train
            size_t saved_position = m_current_token_idx;
            bool is_train_pattern = false;
            int verb_count = 0;
            
            // Scan ahead to detect train pattern (multiple verbs/adverbs)
            while (!is_at_end() && peek().type != TokenType::RIGHT_PAREN && peek().type != TokenType::END_OF_FILE) {
                if (peek().type == TokenType::VERB) {
                    verb_count++;
                    advance();
                    // Skip optional adverb after verb
                    if (!is_at_end() && peek().type == TokenType::ADVERB) {
                        advance();
                    }
                } else if (peek().type == TokenType::ADVERB) {
                    verb_count++;
                    advance();
                } else {
                    // Non-verb/adverb token found, not a train
                    break;
                }
            }
            
            // Check if we found a train pattern (2+ verbs/adverbs followed by closing paren)
            is_train_pattern = (verb_count >= 2 && !is_at_end() && peek().type == TokenType::RIGHT_PAREN);
            
            // Restore position
            m_current_token_idx = saved_position;
            
            if (is_train_pattern) {
                // Parse as train - start with first verb/adverb and use parse_train method
                std::unique_ptr<AstNode> first_verb_expr = nullptr;
                
                // Parse the first verb/adverb combination
                if (match({TokenType::VERB})) {
                    Token verb_token = previous();
                    auto verb_node = std::make_unique<VerbNode>(verb_token.lexeme, verb_token.location);
                    
                    // Check if this verb is followed by an adverb
                    if (match({TokenType::ADVERB})) {
                        Token adverb_token = previous();
                        auto adverb_node = std::make_unique<AdverbNode>(adverb_token.lexeme, adverb_token.location);
                        
                        // Create adverb application
                        first_verb_expr = std::make_unique<AdverbApplicationNode>(
                            std::move(verb_node), std::move(adverb_node), verb_token.location);
                    } else {
                        first_verb_expr = std::move(verb_node);
                    }
                } else if (match({TokenType::ADVERB})) {
                    Token adverb_token = previous();
                    first_verb_expr = std::make_unique<AdverbNode>(adverb_token.lexeme, adverb_token.location);
                }
                
                if (!first_verb_expr) {
                    error(previous(), "Expected verb or adverb in train expression.");
                    return nullptr;
                }
                
                // Use parse_train method to collect the rest
                auto train_node = parse_train(std::move(first_verb_expr));
                consume(TokenType::RIGHT_PAREN, "Expected ')' after train expression.");
                return train_node;
            } else {
                // Parse as regular parenthesized expression
                auto expr = parse_expression();
                consume(TokenType::RIGHT_PAREN, "Expected ')' after expression.");
                return expr;
            }
        }

        case TokenType::VERB: // Monadic prefix verbs, e.g., `- 5` (negate), `# table` (count)
            {
                // This is a monadic application if no left operand was present.
                auto verb_node = std::make_unique<VerbNode>(token.lexeme, token.location);
                
                // Check if there's an adverb following the verb
                if (check(TokenType::ADVERB)) {
                    Token adverb_token = advance();
                    auto adverb_node = std::make_unique<AdverbNode>(adverb_token.lexeme, adverb_token.location);
                    
                    // Create adverb application node
                    auto adverb_app = std::make_unique<AdverbApplicationNode>(
                        std::move(verb_node), std::move(adverb_node), token.location);
                    
                    // Parse the right operand with a recursive call to NUD
                    auto right_operand = parse_primary();
                    if (!right_operand) {
                        error(adverb_token, "Expected expression after adverb application.");
                        return nullptr;
                    }
                    
                    return std::make_unique<MonadicApplicationNode>(
                        std::move(adverb_app), std::move(right_operand), token.location);
                }
                // Check if there's a conjunction following the verb
                else if (check(TokenType::CONJUNCTION)) {
                    Token conj_token = advance();
                    auto conj_node = std::make_unique<ConjunctionNode>(conj_token.lexeme, conj_token.location);
                    
                    // Create conjunction application node
                    auto conj_app = std::make_unique<ConjunctionApplicationNode>(
                        std::move(verb_node), std::move(conj_node), token.location);
                    
                    // Parse the right operand with a recursive call to NUD
                    auto right_operand = parse_primary();
                    if (!right_operand) {
                        error(conj_token, "Expected expression after conjunction application.");
                        return nullptr;
                    }
                    
                    return std::make_unique<MonadicApplicationNode>(
                        std::move(conj_app), std::move(right_operand), token.location);
                }
                
                // Regular monadic verb case
                auto right_operand = parse_expression(); // This needs refinement for precedence.
                return std::make_unique<MonadicApplicationNode>(
                    std::move(verb_node), std::move(right_operand), token.location);
            }

        default:
            error(token, "Expected an expression (literal, name, (, or prefix operator).");
            return nullptr; // Should not reach
    }
}


std::unique_ptr<AstNode> Parser::parse_expression(int min_precedence) {
    // Get the current token and advance
    Token token = advance();
    
    // Handle prefix/primary expressions (NUD - Null Denotation)
    std::unique_ptr<AstNode> left = nud(token);
    
    // Main Pratt parsing loop
    while (!is_at_end() && get_token_precedence(peek()) > min_precedence) {
        Token op_token = advance();
        
        // Handle infix expressions (LED - Left Denotation)
        left = led(op_token, std::move(left));
    }
    
    // After the main loop, check for monadic applications
    // In J, if we have a verb (or train) followed by a potential argument, create monadic application
    if (left && is_verb_like(left.get()) && !is_at_end() && can_be_argument(peek())) {
        auto argument = parse_expression(100); // High precedence to parse the argument
        if (argument) {
            return std::make_unique<MonadicApplicationNode>(std::move(left), std::move(argument), left->location);
        }
    }
    
    return left;
}

// Overload for initial call
std::unique_ptr<AstNode> Parser::parse_expression() {
    return parse_expression(0); // Start with minimum precedence
}

std::unique_ptr<AstNode> Parser::led(const Token& token, std::unique_ptr<AstNode> left_node) {
    switch (token.type) {
        case TokenType::VERB: 
        case TokenType::COMMA: {  // Handle commas just like verbs for concatenation
            auto verb_node = std::make_unique<VerbNode>(token.lexeme, token.location);
            
            // KEY: For right-associativity, use (precedence - 1) instead of precedence
            // This makes operators of the same precedence associate to the right
            int current_precedence = get_token_precedence(token);
            auto right_operand = parse_expression(current_precedence - 1); // RIGHT-ASSOCIATIVE!
            
            return std::make_unique<DyadicApplicationNode>(
                std::move(left_node), 
                std::move(verb_node), 
                std::move(right_operand), 
                token.location
            );
        }
        
        case TokenType::ASSIGN_LOCAL:
        case TokenType::ASSIGN_GLOBAL: {
            if (left_node->type != AstNodeType::NAME_IDENTIFIER) {
                error(token, "Left-hand side of assignment must be a name.");
            }
            auto target_name = std::unique_ptr<NameNode>(static_cast<NameNode*>(left_node.release()));
            
            // Assignment is typically right-associative: a = b = c means a = (b = c)
            int current_precedence = get_token_precedence(token);
            auto value_expr = parse_expression(current_precedence - 1); // RIGHT-ASSOCIATIVE!
            
            // TODO: Create and return AssignmentNode when implemented
            // return std::make_unique<AssignmentNode>(std::move(target_name), std::move(value_expr), 
            //                                        token.type == TokenType::ASSIGN_GLOBAL, token.location);
            
            // Temporary placeholder
            std::cout << "Assignment parsing not fully implemented yet." << std::endl;
            return value_expr;
        }
        
        default:
            error(token, "Unexpected token in LED (expected infix operator or verb).");
            return nullptr;
    }
}

int Parser::get_token_precedence(const Token& token) const {
    switch (token.type) {
        // Assignment has lowest precedence
        case TokenType::ASSIGN_LOCAL:
        case TokenType::ASSIGN_GLOBAL:
            return 10;

        // Comma has lower precedence than other verbs
        case TokenType::COMMA:
            return 15; // Set lower than normal verbs but higher than assignment

        // J verbs - in J, most verbs have the same precedence and are right-associative
        case TokenType::VERB:
            // You might want different precedence for different verbs later
            if (token.lexeme == "+" || token.lexeme == "-") return 20;
            if (token.lexeme == "*" || token.lexeme == "%") return 20; // Same as +/- for right-associativity
            if (token.lexeme == "^") return 30; // Power might be higher
            return 20; // Default verb precedence

        // Adverbs and conjunctions would have higher precedence when they bind to verbs
        // But they're typically handled differently in J parsing
        
        default:
            return 0; // Not an operator
    }
}

// Old
// Main parsing function using right-to-left recursive descent for J
// std::unique_ptr<AstNode> Parser::parse_expression() {
//     if (is_at_end()) {
//         return std::make_unique<NounLiteralNode>(nullptr, current_loc()); // Empty expression
//     }

//     return parse_dyadic_expression();
// }

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
        
        // Check if there's an adverb following the verb (e.g., +/)
        if (!is_at_end() && peek().type == TokenType::ADVERB) {
            Token adverb_token = advance(); // Consume adverb
            auto adverb_node = std::make_unique<AdverbNode>(adverb_token.lexeme, adverb_token.location);
            
            // Create adverb application node (verb + adverb)
            auto adverb_app = std::make_unique<AdverbApplicationNode>(std::move(verb_node), std::move(adverb_node), verb_token.location);
            
            // Now parse the right operand for the monadic application
            std::unique_ptr<AstNode> right = parse_dyadic_expression();
            if (!right) {
                error(adverb_token, "Expected expression after adverb application.");
                return nullptr;
            }
            
            // Return monadic application of the adverb application to the right operand
            return std::make_unique<MonadicApplicationNode>(std::move(adverb_app), std::move(right), verb_token.location);
        } 
        // Check if there's a conjunction following the verb (e.g., +.*)
        else if (!is_at_end() && peek().type == TokenType::CONJUNCTION) {
            Token conj_token = advance(); // Consume conjunction
            auto conj_node = std::make_unique<ConjunctionNode>(conj_token.lexeme, conj_token.location);
            
            // Create conjunction application node (verb + conjunction)
            auto conj_app = std::make_unique<ConjunctionApplicationNode>(std::move(verb_node), std::move(conj_node), verb_token.location);
            
            // Now parse the right operand for the monadic application
            std::unique_ptr<AstNode> right = parse_dyadic_expression();
            if (!right) {
                error(conj_token, "Expected expression after conjunction application.");
                return nullptr;
            }
            
            // Return monadic application of the conjunction application to the right operand
            return std::make_unique<MonadicApplicationNode>(std::move(conj_app), std::move(right), verb_token.location);
        } else {
            // Regular dyadic verb application
            // For right-to-left associativity, recursively parse the entire right expression
            // This will handle chains like "1 + 2 * 3" as "1 + (2 * 3)"
            std::unique_ptr<AstNode> right = parse_dyadic_expression();
            
            // If the right side failed to parse, we still have a valid left + verb, so this is an error
            if (!right) {
                error(verb_token, "Expected expression after verb.");
                return nullptr;
            }
            
            return std::make_unique<DyadicApplicationNode>(std::move(left), std::move(verb_node), std::move(right), verb_token.location);
        }
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
    // Check for space-separated numbers to form vectors (e.g., "1 2 3")
    if (check(TokenType::NOUN_INTEGER) || check(TokenType::NOUN_FLOAT)) {
        std::vector<NounValue> vector_elements;
        SourceLocation start_location = peek().location;
        
        // Collect consecutive numeric tokens
        while (check(TokenType::NOUN_INTEGER) || check(TokenType::NOUN_FLOAT)) {
            Token num_token = advance();
            if (std::holds_alternative<long long>(num_token.literal_value)) {
                vector_elements.push_back(std::get<long long>(num_token.literal_value));
            } else if (std::holds_alternative<double>(num_token.literal_value)) {
                vector_elements.push_back(std::get<double>(num_token.literal_value));
            } else {
                error(num_token, "Unsupported numeric literal type.");
                return nullptr;
            }
        }
        
        // If we collected multiple numbers, create a vector
        if (vector_elements.size() > 1) {
            return std::make_unique<VectorLiteralNode>(std::move(vector_elements), start_location);
        } 
        // If only one number, create a single noun literal
        else if (vector_elements.size() == 1) {
            return std::make_unique<NounLiteralNode>(std::move(vector_elements[0]), start_location);
        }
    }
    
    // Handle string literals separately (they don't form vectors)
    if (match({TokenType::NOUN_STRING})) {
        const Token& t = previous();
        if (std::holds_alternative<std::string>(t.literal_value)) {
            return std::make_unique<NounLiteralNode>(std::get<std::string>(t.literal_value), t.location);
        }
        error(t, "Unsupported string literal type.");
        return nullptr;
    }
    if (match({TokenType::NAME})) {
        return std::make_unique<NameNode>(previous().lexeme, previous().location);
    }

    // Handle monadic prefix verbs if parse_expression doesn't cover them via NUD
    if (match({TokenType::VERB})) { // E.g. `# table` or `- value` or `+/ array`
        Token verb_token = previous();
        auto verb_ast_node = std::make_unique<VerbNode>(verb_token.lexeme, verb_token.location);
        
        // Check if there's an adverb following the verb (e.g., +/)
        if (match({TokenType::ADVERB})) {
            Token adverb_token = previous(); // Get the consumed adverb token
            auto adverb_node = std::make_unique<AdverbNode>(adverb_token.lexeme, adverb_token.location);
            
            // Create adverb application node (verb + adverb)
            auto adverb_app = std::make_unique<AdverbApplicationNode>(std::move(verb_ast_node), std::move(adverb_node), verb_token.location);
            
            // Parse the operand for the monadic application
            auto operand_ast_node = parse_primary(); // Recursive call for the operand
            if (!operand_ast_node) {
                error(adverb_token, "Expected operand after adverb application.");
                return nullptr;
            }
            
            return std::make_unique<MonadicApplicationNode>(std::move(adverb_app), std::move(operand_ast_node), verb_token.location);
        }
        // Check if there's a conjunction following the verb (e.g., < ./ or +.*)
        else if (check(TokenType::CONJUNCTION)) {
            Token conj_token = advance(); // Consume the conjunction
            auto conj_node = std::make_unique<ConjunctionNode>(conj_token.lexeme, conj_token.location);
            
            // Parse the right operand of the conjunction
            std::unique_ptr<AstNode> right_verb_node = nullptr;
            
            // The right operand could be a verb, an adverb application, or a complex expression
            if (check(TokenType::VERB)) {
                Token right_verb_token = advance();
                right_verb_node = std::make_unique<VerbNode>(right_verb_token.lexeme, right_verb_token.location);
                
                // Check if the right verb is followed by an adverb (e.g., ./)
                if (check(TokenType::ADVERB)) {
                    Token right_adverb_token = advance();
                    auto right_adverb_node = std::make_unique<AdverbNode>(right_adverb_token.lexeme, right_adverb_token.location);
                    
                    // Create adverb application for the right side
                    right_verb_node = std::make_unique<AdverbApplicationNode>(
                        std::move(right_verb_node), std::move(right_adverb_node), right_verb_token.location);
                }
            } else if (check(TokenType::ADVERB)) {
                // Handle case where right operand starts with adverb (need to parse preceding verb)
                error(conj_token, "Expected verb before adverb in conjunction.");
                return nullptr;
            } else {
                // Try to parse a more complex expression as right operand
                right_verb_node = parse_primary();
                if (!right_verb_node) {
                    error(conj_token, "Expected right operand for conjunction.");
                    return nullptr;
                }
            }
            
            // Create conjunction application with both left and right operands using the new constructor
            auto conj_app = std::make_unique<ConjunctionApplicationNode>(
                std::move(verb_ast_node), std::move(conj_node), std::move(right_verb_node), verb_token.location);
            
            // Parse the argument that the conjunction will be applied to
            auto operand_ast_node = parse_primary(); // Recursive call for the operand
            if (!operand_ast_node) {
                error(conj_token, "Expected operand after conjunction application.");
                return nullptr;
            }
            
            return std::make_unique<MonadicApplicationNode>(std::move(conj_app), std::move(operand_ast_node), verb_token.location);
        } else {
            // Regular monadic verb application
            auto operand_ast_node = parse_primary(); // Recursive call for the operand
            if (!operand_ast_node) {
                error(verb_token, "Expected operand after verb.");
                return nullptr;
            }
            return std::make_unique<MonadicApplicationNode>(std::move(verb_ast_node), std::move(operand_ast_node), verb_token.location);
        }
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


// J specific parsing methods
std::unique_ptr<AstNode> Parser::parse_train(std::unique_ptr<AstNode> first_verb_expr) {
    std::vector<std::unique_ptr<AstNode>> train_verbs;
    train_verbs.push_back(std::move(first_verb_expr));
    
    SourceLocation train_location = train_verbs[0]->location;
    
    // Continue collecting verbs/adverbs in the train
    while (!is_at_end() && (check(TokenType::VERB) || check(TokenType::ADVERB))) {
        if (check(TokenType::VERB)) {
            Token verb_token = advance();
            auto verb_node = std::make_unique<VerbNode>(verb_token.lexeme, verb_token.location);
            
            // Check if this verb is followed by an adverb
            if (check(TokenType::ADVERB)) {
                Token adverb_token = advance();
                auto adverb_node = std::make_unique<AdverbNode>(adverb_token.lexeme, adverb_token.location);
                
                // Create adverb application and add to train
                auto adverb_app = std::make_unique<AdverbApplicationNode>(
                    std::move(verb_node), std::move(adverb_node), verb_token.location);
                train_verbs.push_back(std::move(adverb_app));
            } else {
                train_verbs.push_back(std::move(verb_node));
            }
        } else if (check(TokenType::ADVERB)) {
            // Standalone adverbs in trains are less common but possible
            Token adverb_token = advance();
            auto adverb_node = std::make_unique<AdverbNode>(adverb_token.lexeme, adverb_token.location);
            train_verbs.push_back(std::move(adverb_node));
        }
    }
    
    // A train needs at least 2 verbs to be meaningful
    if (train_verbs.size() < 2) {
        // Not actually a train, return the single verb
        return std::move(train_verbs[0]);
    }
    
    return std::make_unique<TrainExpressionNode>(std::move(train_verbs), train_location);
}

bool Parser::is_verb_like(const AstNode* node) const {
    if (!node) return false;
    
    // Check for node types that can act like verbs (i.e., can be applied)
    switch (node->type) {
        case AstNodeType::VERB:
        case AstNodeType::ADVERB:
        case AstNodeType::CONJUNCTION:
        case AstNodeType::ADVERB_APPLICATION:
        case AstNodeType::CONJUNCTION_APPLICATION:
        case AstNodeType::TRAIN_EXPRESSION:
            return true;
        default:
            return false;
    }
}

bool Parser::can_be_argument(const Token& token) const {
    // Check for token types that can start a valid argument expression
    switch (token.type) {
        case TokenType::NOUN_INTEGER:
        case TokenType::NOUN_FLOAT:
        case TokenType::NOUN_STRING:
        case TokenType::NAME:
        case TokenType::LEFT_PAREN:
        case TokenType::VERB:      // Can be start of another verb application
        case TokenType::ADVERB:    // Can be start of adverb application
        case TokenType::CONJUNCTION:
            return true;
        default:
            return false;
    }
}

} // namespace JInterpreter
