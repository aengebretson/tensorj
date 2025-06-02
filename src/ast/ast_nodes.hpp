#ifndef J_INTERPRETER_AST_NODES_HPP
#define J_INTERPRETER_AST_NODES_HPP

#include <string>
#include <vector>
#include <memory> // For std::unique_ptr
#include <variant> // For NounLiteralNode value

#include "common/source_location.hpp" // Use common source location

namespace JInterpreter {

// Forward declarations if needed
// struct ExpressionNode;

enum class AstNodeType {
    // Literals
    NOUN_LITERAL,
    VECTOR_LITERAL,         // For space-separated vectors like "1 2 3"
    // Names
    NAME_IDENTIFIER,
    // Verbs, Adverbs, Conjunctions (can be primitive or derived)
    VERB,
    ADVERB,
    CONJUNCTION,
    // Applications
    MONADIC_APPLICATION,
    DYADIC_APPLICATION,
    ADVERB_APPLICATION,     // e.g., +/
    CONJUNCTION_APPLICATION,// e.g., verb ^: noun
    // Structural
    PARENTHESIZED_EXPRESSION,
    TRAIN_EXPRESSION,       // For hooks, forks if represented explicitly
    ASSIGNMENT,
    EXPLICIT_DEFINITION,
    // Control Flow
    IF_EXPRESSION,
    // Top Level
    STATEMENT_LIST,
    EMPTY // Represents nothing, e.g., an empty line or comment result
};

struct AstNode {
    AstNodeType type;
    SourceLocation location;

    AstNode(AstNodeType t, SourceLocation loc) : type(t), location(loc) {}
    virtual ~AstNode() = default; // Important for proper cleanup with unique_ptr

    // For RTTI or visitors, you might add a virtual print or accept method
    virtual void print(std::ostream& os, int indent = 0) const = 0;
};

// --- Example Noun Literal Node ---
using NounValue = std::variant<
    long long,                // Integer
    double,                   // Float
    std::string,              // String
    // std::complex<double>,  // If supporting complex numbers
    // std::vector<std::shared_ptr<AstNode>> // For boxed arrays (simplified)
    std::nullptr_t            // For an uninitialized or error state if needed
>;

struct NounLiteralNode : public AstNode {
    NounValue value;

    NounLiteralNode(NounValue val, SourceLocation loc)
        : AstNode(AstNodeType::NOUN_LITERAL, loc), value(std::move(val)) {}

    void print(std::ostream& os, int indent = 0) const override;
};

// --- Vector Literal Node for space-separated sequences like "1 2 3" ---
struct VectorLiteralNode : public AstNode {
    std::vector<NounValue> elements;

    VectorLiteralNode(std::vector<NounValue> elems, SourceLocation loc)
        : AstNode(AstNodeType::VECTOR_LITERAL, loc), elements(std::move(elems)) {}

    void print(std::ostream& os, int indent = 0) const override;
};

// --- Example Name Identifier Node ---
struct NameNode : public AstNode {
    std::string name;

    NameNode(std::string n, SourceLocation loc)
        : AstNode(AstNodeType::NAME_IDENTIFIER, loc), name(std::move(n)) {}

    void print(std::ostream& os, int indent = 0) const override;
};


// --- Example Monadic Application Node ---
struct MonadicApplicationNode : public AstNode {
    std::unique_ptr<AstNode> verb;
    std::unique_ptr<AstNode> argument;

    MonadicApplicationNode(std::unique_ptr<AstNode> v, std::unique_ptr<AstNode> arg, SourceLocation loc)
        : AstNode(AstNodeType::MONADIC_APPLICATION, loc), verb(std::move(v)), argument(std::move(arg)) {}

    void print(std::ostream& os, int indent = 0) const override;
};

// --- Example Dyadic Application Node ---
struct DyadicApplicationNode : public AstNode {
    std::unique_ptr<AstNode> left_argument;
    std::unique_ptr<AstNode> verb;
    std::unique_ptr<AstNode> right_argument;

    DyadicApplicationNode(std::unique_ptr<AstNode> l_arg, std::unique_ptr<AstNode> v, std::unique_ptr<AstNode> r_arg, SourceLocation loc)
        : AstNode(AstNodeType::DYADIC_APPLICATION, loc), left_argument(std::move(l_arg)), verb(std::move(v)), right_argument(std::move(r_arg)) {}
    
    void print(std::ostream& os, int indent = 0) const override;
};


// --- Example Verb Node (could be primitive or a NameNode pointing to a definition) ---
struct VerbNode : public AstNode {
    std::string identifier; // e.g., "+", "#", or "myUserDefinedVerb"

    VerbNode(std::string id, SourceLocation loc)
        : AstNode(AstNodeType::VERB, loc), identifier(std::move(id)) {}

    void print(std::ostream& os, int indent = 0) const override;
};

// --- Adverb Node ---
struct AdverbNode : public AstNode {
    std::string identifier; // e.g., "/", "\", "~"

    AdverbNode(std::string id, SourceLocation loc)
        : AstNode(AstNodeType::ADVERB, loc), identifier(std::move(id)) {}

    void print(std::ostream& os, int indent = 0) const override;
};

// --- Conjunction Node ---
struct ConjunctionNode : public AstNode {
    std::string identifier; // e.g., ".", "*", ".*"

    ConjunctionNode(std::string id, SourceLocation loc)
        : AstNode(AstNodeType::CONJUNCTION, loc), identifier(std::move(id)) {}

    void print(std::ostream& os, int indent = 0) const override;
};

// --- Adverb Application Node (e.g., +/) ---
struct AdverbApplicationNode : public AstNode {
    std::unique_ptr<AstNode> verb;    // The verb being modified (e.g., "+")
    std::unique_ptr<AstNode> adverb;  // The adverb (e.g., "/")

    AdverbApplicationNode(std::unique_ptr<AstNode> v, std::unique_ptr<AstNode> adv, SourceLocation loc)
        : AstNode(AstNodeType::ADVERB_APPLICATION, loc), verb(std::move(v)), adverb(std::move(adv)) {}

    void print(std::ostream& os, int indent = 0) const override;
};

// --- Conjunction Application Node (e.g., +.*) ---
struct ConjunctionApplicationNode : public AstNode {
    std::unique_ptr<AstNode> left_operand;   // The left verb (e.g., "+")
    std::unique_ptr<AstNode> conjunction;    // The conjunction (e.g., ".*")
    std::unique_ptr<AstNode> right_operand;  // The right verb (e.g., "*") - optional for some conjunctions

    // Constructor with both operands (for full conjunctions like < ./)
    ConjunctionApplicationNode(std::unique_ptr<AstNode> left, std::unique_ptr<AstNode> conj, std::unique_ptr<AstNode> right, SourceLocation loc)
        : AstNode(AstNodeType::CONJUNCTION_APPLICATION, loc), left_operand(std::move(left)), conjunction(std::move(conj)), right_operand(std::move(right)) {}

    // Constructor for backwards compatibility (single operand conjunctions)
    ConjunctionApplicationNode(std::unique_ptr<AstNode> v, std::unique_ptr<AstNode> conj, SourceLocation loc)
        : AstNode(AstNodeType::CONJUNCTION_APPLICATION, loc), left_operand(std::move(v)), conjunction(std::move(conj)), right_operand(nullptr) {}

    void print(std::ostream& os, int indent = 0) const override;
};

// Add more node types as you design them (AdverbNode, ConjunctionNode, AssignmentNode, etc.)

} // namespace JInterpreter

#endif // J_INTERPRETER_AST_NODES_HPP
