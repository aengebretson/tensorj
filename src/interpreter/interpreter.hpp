#ifndef J_INTERPRETER_INTERPRETER_HPP
#define J_INTERPRETER_INTERPRETER_HPP

#include "ast/ast_nodes.hpp"
#include <memory>

// Forward declare TensorFlow types if you use them directly here
// Or use an adapter layer.
// namespace tensorflow { class Session; class Tensor; }


namespace JInterpreter {

// Represents the runtime value in J (often an array/tensor)
// This will evolve significantly, especially with TensorFlow.
// For now, a simple placeholder.
using JValue = NounValue; // Reusing NounValue from AST for extreme simplicity initially

class Interpreter /* : public AstVisitor (if using visitor pattern) */ {
public:
    Interpreter();
    ~Interpreter();

    JValue evaluate(AstNode* node); // Or pass std::unique_ptr<AstNode>

private:
    // Environment for storing variables
    // std::unordered_map<std::string, JValue> m_environment;

    // TensorFlow session, context, etc.
    // std::unique_ptr<tensorflow::Session> m_tf_session;

    // Visit methods if using Visitor pattern for AST traversal
    // JValue visitNounLiteralNode(NounLiteralNode* node);
    // JValue visitNameNode(NameNode* node);
    // JValue visitMonadicApplicationNode(MonadicApplicationNode* node);
    // ... etc.
};

} // namespace JInterpreter

#endif // J_INTERPRETER_INTERPRETER_HPP
