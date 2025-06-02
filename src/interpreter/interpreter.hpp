#ifndef J_INTERPRETER_INTERPRETER_HPP
#define J_INTERPRETER_INTERPRETER_HPP

#include "ast/ast_nodes.hpp"
#include "tf_operations.hpp"
#include <memory>
#include <unordered_map>

namespace JInterpreter {

// JValue now uses the enhanced type system with TensorFlow support
// using JValue = ... (defined in tf_operations.hpp)

class Interpreter {
public:
    Interpreter();
    ~Interpreter();

    JValue evaluate(AstNode* node);

private:
    // Environment for storing variables
    std::unordered_map<std::string, JValue> m_environment;

    // TensorFlow session for operations
    std::unique_ptr<TFSession> m_tf_session;

    // Helper methods for different node types
    JValue evaluate_noun_literal(NounLiteralNode* node);
    JValue evaluate_vector_literal(VectorLiteralNode* node);
    JValue evaluate_name_identifier(NameNode* node);
    JValue evaluate_monadic_application(MonadicApplicationNode* node);
    JValue evaluate_dyadic_application(DyadicApplicationNode* node);
    JValue evaluate_adverb_application(AdverbApplicationNode* node);
    
    // Helper methods for J operations
    JValue execute_monadic_verb(const std::string& verb_name, const JValue& operand);
    JValue execute_dyadic_verb(const std::string& verb_name, const JValue& left, const JValue& right);
    JValue execute_adverb_application(AdverbApplicationNode* adverb_app, const JValue& operand);
    JValue execute_fold(const std::string& verb_name, const JValue& operand);
    
    // Utility methods
    std::shared_ptr<JTensor> to_tensor(const JValue& value);
    JValue from_tensor(std::shared_ptr<JTensor> tensor);
    bool is_tensor_value(const JValue& value);
    
    // J verb implementations
    JValue j_plus(const JValue& left, const JValue& right);  // +
    JValue j_minus(const JValue& left, const JValue& right); // -
    JValue j_times(const JValue& left, const JValue& right); // *
    JValue j_divide(const JValue& left, const JValue& right); // %
    JValue j_iota(const JValue& operand); // i.
    JValue j_shape(const JValue& operand); // $
    JValue j_reshape(const JValue& shape, const JValue& data); // $ (dyadic)
};

} // namespace JInterpreter

#endif // J_INTERPRETER_INTERPRETER_HPP
