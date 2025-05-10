#include "interpreter.hpp"
#include <stdexcept>
#include <iostream>

// #include "tensorflow/core/public/session.h" // Example include
// #include "tensorflow/core/platform/env.h"

namespace JInterpreter {

Interpreter::Interpreter() {
    // Initialize TensorFlow session, etc.
    // Example:
    // tensorflow::SessionOptions options;
    // m_tf_session.reset(tensorflow::NewSession(options));
    // if (!m_tf_session) {
    //    throw std::runtime_error("Could not create TensorFlow session.");
    // }
    std::cout << "J Interpreter stub initialized." << std::endl;
}

Interpreter::~Interpreter() {
    // if (m_tf_session) m_tf_session->Close();
}

JValue Interpreter::evaluate(AstNode* node) {
    if (!node) {
        // throw std::runtime_error("Cannot evaluate null AST node.");
        std::cerr << "Warning: evaluate called with null AST node." << std::endl;
        return nullptr;
    }

    // Basic dispatch based on node type (replace with Visitor pattern for cleaner code)
    switch (node->type) {
        case AstNodeType::NOUN_LITERAL: {
            auto* noun_node = static_cast<NounLiteralNode*>(node);
            return noun_node->value; // Directly return the NounValue
        }
        case AstNodeType::NAME_IDENTIFIER:
            std::cerr << "Evaluation of NameNode not implemented yet." << std::endl;
            // TODO: Lookup in environment
            break;
        case AstNodeType::MONADIC_APPLICATION: {
            std::cerr << "Evaluation of MonadicApplicationNode not implemented yet." << std::endl;
            // auto* app_node = static_cast<MonadicApplicationNode*>(node);
            // JValue operand_val = evaluate(app_node->argument.get());
            // JValue verb_val = evaluate(app_node->verb.get()); // This is tricky, verb itself is not a value
            // TODO: Execute the verb with the operand
            break;
        }
         case AstNodeType::DYADIC_APPLICATION: {
            std::cerr << "Evaluation of DyadicApplicationNode not implemented yet." << std::endl;
            break;
        }
        // ... other cases
        default:
            std::cerr << "Evaluation for AST node type " << static_cast<int>(node->type) << " not implemented." << std::endl;
    }
    return nullptr; // Placeholder for unhandled or void expressions
}

} // namespace JInterpreter