#ifndef J_INTERPRETER_INTERPRETER_HPP
#define J_INTERPRETER_INTERPRETER_HPP

#include "ast/ast_nodes.hpp"
#include "tf_operations.hpp"
#include "tf_graph.hpp"
#include <memory>
#include <unordered_map>

namespace JInterpreter {

// JValue now uses the enhanced type system with TensorFlow support
// using JValue = ... (defined in tf_operations.hpp)

enum class ExecutionMode {
    EAGER,  // Default eager execution
    GRAPH   // Graph-based deferred execution
};

class Interpreter {
public:
    Interpreter();
    ~Interpreter();

    JValue evaluate(AstNode* node);
    
    // Graph mode control
    void set_execution_mode(ExecutionMode mode) { m_execution_mode = mode; }
    ExecutionMode get_execution_mode() const { return m_execution_mode; }
    
    // Public accessor for TFSession
    TFSession* getTFSession() const { return m_tf_session.get(); }

private:
    // Environment for storing variables
    std::unordered_map<std::string, JValue> m_environment;

    // TensorFlow session for operations
    std::unique_ptr<TFSession> m_tf_session;
    
    // Execution mode
    ExecutionMode m_execution_mode;
    
    // Graph builder for deferred execution
    std::unique_ptr<JGraphBuilder> m_graph_builder;

    // Helper methods for different node types
    JValue evaluate_noun_literal(NounLiteralNode* node);
    JValue evaluate_vector_literal(VectorLiteralNode* node);
    JValue evaluate_name_identifier(NameNode* node);
    JValue evaluate_monadic_application(MonadicApplicationNode* node);
    JValue evaluate_dyadic_application(DyadicApplicationNode* node);
    JValue evaluate_adverb_application(AdverbApplicationNode* node);
    JValue evaluate_conjunction_application(ConjunctionApplicationNode* node);
    JValue evaluate_assignment(AssignmentNode* node);
    JValue evaluate_train_expression(TrainExpressionNode* node, const JValue& argument);
    
    // Graph mode execution
    JValue evaluate_train_expression_graph(TrainExpressionNode* node, const JValue& argument);
    
    // Helper methods for J operations
    JValue execute_monadic_verb(const std::string& verb_name, const JValue& operand);
    JValue execute_dyadic_verb(const std::string& verb_name, const JValue& left, const JValue& right);
    JValue execute_adverb_application(AdverbApplicationNode* adverb_app, const JValue& operand);
    JValue execute_conjunction_application(ConjunctionApplicationNode* conj_app, const JValue& operand);
    JValue execute_fold(const std::string& verb_name, const JValue& operand);
    JValue execute_inner_product(const std::string& verb_name, const JValue& left, const JValue& right);
    
    // Graph-based helper methods for deferred execution
    std::shared_ptr<DeferredTensor> execute_monadic_verb_graph(const std::string& verb_name, std::shared_ptr<DeferredTensor> operand);
    std::shared_ptr<DeferredTensor> execute_dyadic_verb_graph(const std::string& verb_name, std::shared_ptr<DeferredTensor> left, std::shared_ptr<DeferredTensor> right);
    
    // Utility methods
    std::shared_ptr<JTensor> to_tensor(const JValue& value);
    JValue from_tensor(std::shared_ptr<JTensor> tensor);
    bool is_tensor_value(const JValue& value);
    
    // J verb implementations
    JValue j_plus(const JValue& left, const JValue& right);  // +
    JValue j_minus(const JValue& left, const JValue& right); // -
    JValue j_times(const JValue& left, const JValue& right); // *
    JValue j_divide(const JValue& left, const JValue& right); // %
    JValue j_power(const JValue& left, const JValue& right); // ^ (dyadic power)
    JValue j_negate(const JValue& operand); // - (monadic negation)
    JValue j_square(const JValue& operand); // *: (monadic square)
    JValue j_reciprocal(const JValue& operand); // % (monadic reciprocal)
    JValue j_iota(const JValue& operand); // i.
    JValue j_shape(const JValue& operand); // $
    JValue j_tally(const JValue& operand); // # (count/tally)
    JValue j_reshape(const JValue& shape, const JValue& data); // $ (dyadic)
    
    // Comparison operations
    JValue j_equal(const JValue& left, const JValue& right); // =
    JValue j_less_than(const JValue& left, const JValue& right); // <
    JValue j_greater_than(const JValue& left, const JValue& right); // >
    JValue j_less_equal(const JValue& left, const JValue& right); // <:
    JValue j_greater_equal(const JValue& left, const JValue& right); // >:
    
    // Array operations
    JValue j_concatenate(const JValue& left, const JValue& right); // ,
};

} // namespace JInterpreter

#endif // J_INTERPRETER_INTERPRETER_HPP
