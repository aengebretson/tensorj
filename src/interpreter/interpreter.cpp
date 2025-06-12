#include "interpreter.hpp"
#include <stdexcept>
#include <iostream>
#include <variant>
#include <type_traits>

namespace JInterpreter {

Interpreter::Interpreter() : m_execution_mode(ExecutionMode::EAGER) {
    m_tf_session = std::make_unique<TFSession>();
    if (!m_tf_session->is_initialized()) {
        std::cerr << "Warning: TensorFlow session initialization failed. Using fallback mode." << std::endl;
    }
    m_graph_builder = std::make_unique<JGraphBuilder>();
    std::cout << "J Interpreter initialized." << std::endl;
}

Interpreter::~Interpreter() = default;

JValue Interpreter::evaluate(AstNode* node) {
    if (!node) {
        std::cerr << "Warning: evaluate called with null AST node." << std::endl;
        return nullptr;
    }

    switch (node->type) {
        case AstNodeType::NOUN_LITERAL:
            return evaluate_noun_literal(static_cast<NounLiteralNode*>(node));
            
        case AstNodeType::VECTOR_LITERAL:
            return evaluate_vector_literal(static_cast<VectorLiteralNode*>(node));
            
        case AstNodeType::NAME_IDENTIFIER:
            return evaluate_name_identifier(static_cast<NameNode*>(node));
            
        case AstNodeType::MONADIC_APPLICATION:
            return evaluate_monadic_application(static_cast<MonadicApplicationNode*>(node));
            
        case AstNodeType::DYADIC_APPLICATION:
            return evaluate_dyadic_application(static_cast<DyadicApplicationNode*>(node));
            
        case AstNodeType::ADVERB_APPLICATION:
            return evaluate_adverb_application(static_cast<AdverbApplicationNode*>(node));
            
        case AstNodeType::CONJUNCTION_APPLICATION:
            return evaluate_conjunction_application(static_cast<ConjunctionApplicationNode*>(node));
            
        case AstNodeType::TRAIN_EXPRESSION:
            std::cerr << "Complex verb evaluation not implemented yet (train expressions require argument)." << std::endl;
            return nullptr;
            
        default:
            std::cerr << "Evaluation for AST node type " << static_cast<int>(node->type) << " not implemented." << std::endl;
            return nullptr;
    }
}

JValue Interpreter::evaluate_noun_literal(NounLiteralNode* node) {
    // Convert NounValue to JValue, creating tensors for numeric values
    return std::visit([this](auto&& value) -> JValue {
        using T = std::decay_t<decltype(value)>;
        
        if constexpr (std::is_same_v<T, long long>) {
            // Create a scalar tensor for integers
            return JTensor::scalar(value);
        } else if constexpr (std::is_same_v<T, double>) {
            // Create a scalar tensor for floats
            return JTensor::scalar(value);
        } else if constexpr (std::is_same_v<T, std::string>) {
            // Keep strings as strings for now
            return value;
        } else {
            return nullptr;
        }
    }, node->value);
}

JValue Interpreter::evaluate_vector_literal(VectorLiteralNode* node) {
    if (node->elements.empty()) {
        // Empty vector - create rank-1 tensor with size 0
        return JTensor::from_data(std::vector<double>{}, {0});
    }
    
    // Check if all elements are of the same type and collect them
    bool all_integers = true;
    bool all_floats = true;
    
    std::vector<long long> int_values;
    std::vector<double> float_values;
    int_values.reserve(node->elements.size());
    float_values.reserve(node->elements.size());
    
    for (const auto& element : node->elements) {
        std::visit([&](auto&& value) {
            using T = std::decay_t<decltype(value)>;
            
            if constexpr (std::is_same_v<T, long long>) {
                int_values.push_back(value);
                float_values.push_back(static_cast<double>(value));
                // Don't mark all_floats as false - allow mixed types to create float vector
            } else if constexpr (std::is_same_v<T, double>) {
                float_values.push_back(value);
                all_integers = false;
            } else {
                // Strings or other types - not supported in numeric vectors
                all_integers = false;
                all_floats = false;
            }
        }, element);
    }
    
    if (all_integers) {
        // All elements are integers - create integer vector
        return JTensor::from_data(int_values, {static_cast<long long>(int_values.size())});
    } else if (all_integers || all_floats) {
        // All elements are floats OR mixed numeric - create float vector
        return JTensor::from_data(float_values, {static_cast<long long>(float_values.size())});
    } else {
        std::cerr << "Vector literal contains non-numeric elements" << std::endl;
        return nullptr;
    }
}

JValue Interpreter::evaluate_name_identifier(NameNode* node) {
    // Look up variable in environment
    auto it = m_environment.find(node->name);
    if (it != m_environment.end()) {
        return it->second;
    }
    
    std::cerr << "Undefined variable: " << node->name << std::endl;
    return nullptr;
}

JValue Interpreter::evaluate_monadic_application(MonadicApplicationNode* node) {
    // First evaluate the argument
    JValue operand = evaluate(node->argument.get());
    if (std::holds_alternative<std::nullptr_t>(operand)) {
        return nullptr;
    }
    
    // Check if verb is a simple name (primitive verb)
    if (node->verb->type == AstNodeType::NAME_IDENTIFIER) {
        auto* verb_node = static_cast<NameNode*>(node->verb.get());
        return execute_monadic_verb(verb_node->name, operand);
    } else if (node->verb->type == AstNodeType::VERB) {
        auto* verb_node = static_cast<VerbNode*>(node->verb.get());
        return execute_monadic_verb(verb_node->identifier, operand);
    } else if (node->verb->type == AstNodeType::ADVERB_APPLICATION) {
        auto* adverb_app_node = static_cast<AdverbApplicationNode*>(node->verb.get());
        return execute_adverb_application(adverb_app_node, operand);
    } else if (node->verb->type == AstNodeType::CONJUNCTION_APPLICATION) {
        auto* conj_app_node = static_cast<ConjunctionApplicationNode*>(node->verb.get());
        return execute_conjunction_application(conj_app_node, operand);
    } else if (node->verb->type == AstNodeType::TRAIN_EXPRESSION) {
        auto* train_node = static_cast<TrainExpressionNode*>(node->verb.get());
        if (m_execution_mode == ExecutionMode::GRAPH) {
            return evaluate_train_expression_graph(train_node, operand);
        } else {
            return evaluate_train_expression(train_node, operand);
        }
    }
    
    std::cerr << "Complex verb evaluation not implemented yet." << std::endl;
    return nullptr;
}

JValue Interpreter::evaluate_dyadic_application(DyadicApplicationNode* node) {
    // Evaluate both arguments
    JValue left = evaluate(node->left_argument.get());
    JValue right = evaluate(node->right_argument.get());
    
    if (std::holds_alternative<std::nullptr_t>(left) || std::holds_alternative<std::nullptr_t>(right)) {
        return nullptr;
    }
    
    // Check if verb is a simple name (primitive verb)
    if (node->verb->type == AstNodeType::NAME_IDENTIFIER) {
        auto* verb_node = static_cast<NameNode*>(node->verb.get());
        return execute_dyadic_verb(verb_node->name, left, right);
    } else if (node->verb->type == AstNodeType::VERB) {
        auto* verb_node = static_cast<VerbNode*>(node->verb.get());
        return execute_dyadic_verb(verb_node->identifier, left, right);
    }
    
    std::cerr << "Complex verb evaluation not implemented yet." << std::endl;
    return nullptr;
}

JValue Interpreter::evaluate_adverb_application(AdverbApplicationNode* node) {
    // For now, handle only the "/" adverb (fold/reduce)
    if (node->adverb->type != AstNodeType::ADVERB) {
        std::cerr << "Expected adverb node in adverb application." << std::endl;
        return nullptr;
    }
    
    auto* adverb_node = static_cast<AdverbNode*>(node->adverb.get());
    if (adverb_node->identifier == "/") {
        // This is a fold/reduce operation
        // The adverb application itself doesn't have operands - it creates a derived verb
        // The actual operand will be provided by the MonadicApplicationNode that contains this
        
        // For fold operations, we need to return a representation that can be used later
        // For now, we'll return the AdverbApplicationNode itself wrapped as a JValue
        // This is a simplified approach - in a full implementation, we'd have a more sophisticated verb system
        
        std::cerr << "Adverb application evaluation not fully implemented yet." << std::endl;
        return nullptr;
    }
    
    std::cerr << "Unknown adverb: " << adverb_node->identifier << std::endl;
    return nullptr;
}

JValue Interpreter::execute_monadic_verb(const std::string& verb_name, const JValue& operand) {
    if (verb_name == "i.") {
        return j_iota(operand);
    } else if (verb_name == "$") {
        return j_shape(operand);
    } else if (verb_name == "#") {
        return j_tally(operand);
    } else if (verb_name == "-") {
        return j_negate(operand);
    } else if (verb_name == "*:") {
        return j_square(operand);
    } else if (verb_name == "%") {
        return j_reciprocal(operand);
    }
    
    std::cerr << "Unknown monadic verb: " << verb_name << std::endl;
    return nullptr;
}

JValue Interpreter::execute_dyadic_verb(const std::string& verb_name, const JValue& left, const JValue& right) {
    if (verb_name == "+") {
        return j_plus(left, right);
    } else if (verb_name == "-") {
        return j_minus(left, right);
    } else if (verb_name == "*") {
        return j_times(left, right);
    } else if (verb_name == "%") {
        return j_divide(left, right);
    } else if (verb_name == "^") {
        return j_power(left, right);
    } else if (verb_name == "$") {
        return j_reshape(left, right);
    } else if (verb_name == "=") {
        return j_equal(left, right);
    } else if (verb_name == "<") {
        return j_less_than(left, right);
    } else if (verb_name == ">") {
        return j_greater_than(left, right);
    } else if (verb_name == "<:") {
        return j_less_equal(left, right);
    } else if (verb_name == ">:") {
        return j_greater_equal(left, right);
    } else if (verb_name == ",") {
        return j_concatenate(left, right);
    }
    
    std::cerr << "Unknown dyadic verb: " << verb_name << std::endl;
    return nullptr;
}

JValue Interpreter::execute_adverb_application(AdverbApplicationNode* adverb_app, const JValue& operand) {
    // Get the base verb and adverb
    if (adverb_app->verb->type != AstNodeType::VERB || adverb_app->adverb->type != AstNodeType::ADVERB) {
        std::cerr << "Invalid adverb application structure." << std::endl;
        return nullptr;
    }
    
    auto* verb_node = static_cast<VerbNode*>(adverb_app->verb.get());
    auto* adverb_node = static_cast<AdverbNode*>(adverb_app->adverb.get());
    
    // Handle the "/" adverb (fold/reduce)
    if (adverb_node->identifier == "/") {
        return execute_fold(verb_node->identifier, operand);
    }
    // Handle the "./" adverb (also fold/reduce - compound adverb)
    else if (adverb_node->identifier == "./") {
        return execute_fold(verb_node->identifier, operand);
    }
    
    std::cerr << "Unknown adverb in application: " << adverb_node->identifier << std::endl;
    return nullptr;
}

JValue Interpreter::execute_fold(const std::string& verb_name, const JValue& operand) {
    // Convert operand to tensor
    auto tensor = to_tensor(operand);
    if (!tensor) {
        std::cerr << "Cannot convert operand to tensor for fold operation." << std::endl;
        return nullptr;
    }
    
    // Handle "+" verb for sum reduction
    if (verb_name == "+") {
        // Use TensorFlow's reduce_sum operation
        auto result = m_tf_session->reduce_sum(tensor);
        return from_tensor(result);
    }
    // Handle "*" verb for product reduction
    else if (verb_name == "*") {
        // Use TensorFlow's reduce_product operation
        auto result = m_tf_session->reduce_product(tensor);
        return from_tensor(result);
    }
    // Handle "<" verb for min reduction (in J, < means minimum when used with ./)
    else if (verb_name == "<") {
        auto result = m_tf_session->reduce_min(tensor);
        return from_tensor(result);
    }
    // Handle ">" verb for max reduction (in J, > means maximum when used with ./)
    else if (verb_name == ">") {
        auto result = m_tf_session->reduce_max(tensor);
        return from_tensor(result);
    }
    
    std::cerr << "Fold operation not implemented for verb: " << verb_name << std::endl;
    return nullptr;
}

std::shared_ptr<JTensor> Interpreter::to_tensor(const JValue& value) {
    return std::visit([](auto&& val) -> std::shared_ptr<JTensor> {
        using T = std::decay_t<decltype(val)>;
        
        if constexpr (std::is_same_v<T, std::shared_ptr<JTensor>>) {
            return val;
        } else if constexpr (std::is_same_v<T, long long>) {
            return JTensor::scalar(val);
        } else if constexpr (std::is_same_v<T, double>) {
            return JTensor::scalar(val);
        } else {
            return nullptr;
        }
    }, value);
}

JValue Interpreter::from_tensor(std::shared_ptr<JTensor> tensor) {
    if (!tensor) return nullptr;
    return tensor;
}

bool Interpreter::is_tensor_value(const JValue& value) {
    return std::holds_alternative<std::shared_ptr<JTensor>>(value);
}

// J verb implementations

JValue Interpreter::j_plus(const JValue& left, const JValue& right) {
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for addition" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->add(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_minus(const JValue& left, const JValue& right) {
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for subtraction" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->subtract(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_times(const JValue& left, const JValue& right) {
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for multiplication" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->multiply(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_divide(const JValue& left, const JValue& right) {
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for division" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->divide(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_iota(const JValue& operand) {
    // i. n creates vector 0 1 2 ... n-1
    auto tensor = to_tensor(operand);
    if (!tensor) {
        std::cerr << "Cannot convert operand to tensor for iota" << std::endl;
        return nullptr;
    }
    
    if (tensor->rank() != 0) {
        std::cerr << "Iota requires scalar argument" << std::endl;
        return nullptr;
    }
    
    long long n = tensor->get_scalar<long long>();
    auto result = m_tf_session->iota(n);
    return from_tensor(result);
}

JValue Interpreter::j_shape(const JValue& operand) {
    // $ y returns the shape of y
    auto tensor = to_tensor(operand);
    if (!tensor) {
        std::cerr << "Cannot convert operand to tensor for shape" << std::endl;
        return nullptr;
    }
    
    auto shape = tensor->shape();
    // Create a vector tensor with the shape dimensions
    // The result should always be a rank-1 tensor containing the shape dimensions
    std::vector<long long> result_shape = {static_cast<long long>(shape.size())};
    auto result = JTensor::from_data(shape, result_shape);
    return from_tensor(result);
}

JValue Interpreter::j_tally(const JValue& operand) {
    // # y returns the tally (count) of y - the number of items in the first dimension
    auto tensor = to_tensor(operand);
    if (!tensor) {
        std::cerr << "Cannot convert operand to tensor for tally" << std::endl;
        return nullptr;
    }
    
    long long count;
    if (tensor->rank() == 0) {
        // For scalars, tally is 1
        count = 1;
    } else {
        // For arrays, tally is the size of the first dimension
        count = tensor->shape()[0];
    }
    
    // Return as a scalar tensor
    auto result = JTensor::scalar(count);
    return from_tensor(result);
}

JValue Interpreter::j_reshape(const JValue& shape, const JValue& data) {
    // shape $ data reshapes data to shape
    auto shape_tensor = to_tensor(shape);
    auto data_tensor = to_tensor(data);
    
    if (!shape_tensor || !data_tensor) {
        std::cerr << "Cannot convert operands to tensors for reshape" << std::endl;
        return nullptr;
    }
    
    // Extract new shape from shape tensor
    std::vector<long long> new_shape;
    if (shape_tensor->rank() == 0) {
        // Scalar shape - convert to 1D vector
        new_shape.push_back(shape_tensor->get_scalar<long long>());
    } else {
        // Vector of dimensions
        auto shape_data = shape_tensor->shape();
        // For simplicity, assume shape tensor is 1D and contains the new shape
        // This is a simplified implementation
        new_shape = shape_tensor->shape();
    }
    
    auto result = m_tf_session->reshape(data_tensor, new_shape);
    return from_tensor(result);
}

// Comparison operations
JValue Interpreter::j_equal(const JValue& left, const JValue& right) {
    // = comparison: returns boolean tensor
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for equality comparison" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->equal(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_less_than(const JValue& left, const JValue& right) {
    // < comparison: returns boolean tensor
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for less than comparison" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->less_than(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_greater_than(const JValue& left, const JValue& right) {
    // > comparison: returns boolean tensor
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for greater than comparison" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->greater_than(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_less_equal(const JValue& left, const JValue& right) {
    // <: comparison: returns boolean tensor
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for less equal comparison" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->less_equal(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_greater_equal(const JValue& left, const JValue& right) {
    // >: comparison: returns boolean tensor
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for greater equal comparison" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->greater_equal(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_concatenate(const JValue& left, const JValue& right) {
    // , concatenation: joins arrays along first axis
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for concatenation" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->concatenate(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_power(const JValue& left, const JValue& right) {
    // ^ dyadic power: x ^ y = x to the power of y
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for power operation" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->power(left_tensor, right_tensor);
    return from_tensor(result);
}

JValue Interpreter::j_negate(const JValue& operand) {
    // - monadic negation: -x
    auto tensor = to_tensor(operand);
    
    if (!tensor) {
        std::cerr << "Cannot convert operand to tensor for negation" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->negate(tensor);
    return from_tensor(result);
}

JValue Interpreter::j_square(const JValue& operand) {
    // *: monadic square: x^2
    auto tensor = to_tensor(operand);
    
    if (!tensor) {
        std::cerr << "Cannot convert operand to tensor for square operation" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->square(tensor);
    return from_tensor(result);
}

JValue Interpreter::j_reciprocal(const JValue& operand) {
    // % monadic reciprocal: 1/x
    auto tensor = to_tensor(operand);
    
    if (!tensor) {
        std::cerr << "Cannot convert operand to tensor for reciprocal operation" << std::endl;
        return nullptr;
    }
    
    auto result = m_tf_session->reciprocal(tensor);
    return from_tensor(result);
}

JValue Interpreter::evaluate_conjunction_application(ConjunctionApplicationNode* node) {
    // For now, we handle conjunctions as operations on their arguments
    // In J, conjunctions take two arguments (typically verbs) and create a new verb
    
    if (!node) {
        std::cerr << "Conjunction application node is null" << std::endl;
        return nullptr;
    }
    
    // Handle the .* inner product conjunction specifically
    if (node->conjunction && node->conjunction->type == AstNodeType::CONJUNCTION) {
        ConjunctionNode* conj = static_cast<ConjunctionNode*>(node->conjunction.get());
        if (conj->identifier == ".*") {
            // Inner product: left_verb .* right_verb
            // For now, assume left verb is multiplication (*) and right is addition (+)
            if (node->left_operand) {
                JValue verb = evaluate(node->left_operand.get());
                return execute_inner_product("*", verb, verb);
            }
        }
    }
    
    std::cerr << "Conjunction application not fully implemented for: " << node->conjunction << std::endl;
    return nullptr;
}

JValue Interpreter::execute_conjunction_application(ConjunctionApplicationNode* conj_app, const JValue& operand) {
    // This handles applying a conjunction result to an operand
    // For example, if we have (+/ .* -/) applied to a matrix
    
    if (!conj_app || !conj_app->conjunction) {
        std::cerr << "Invalid conjunction application" << std::endl;
        return nullptr;
    }
    
    // Handle .* inner product specifically
    if (conj_app->conjunction->type == AstNodeType::CONJUNCTION) {
        ConjunctionNode* conj = static_cast<ConjunctionNode*>(conj_app->conjunction.get());
        if (conj->identifier == ".*") {
            // Apply inner product operation to the operand
            // For matrices, this typically means matrix multiplication
            auto tensor = to_tensor(operand);
            if (!tensor) {
                std::cerr << "Cannot convert operand to tensor for inner product" << std::endl;
                return nullptr;
            }
            
            // For now, treat as identity operation - more complex logic needed
            return operand;
        }
    }
    
    std::cerr << "Conjunction application execution not implemented for conjunction type" << std::endl;
    return nullptr;
}

JValue Interpreter::execute_inner_product(const std::string& verb_name, const JValue& left, const JValue& right) {
    // Inner product implementation: combines two arrays using specified operations
    // Format: left_verb .* right_verb applied to arrays
    // For matrices: performs matrix multiplication when verbs are * and +
    
    auto left_tensor = to_tensor(left);
    auto right_tensor = to_tensor(right);
    
    if (!left_tensor || !right_tensor) {
        std::cerr << "Cannot convert operands to tensors for inner product" << std::endl;
        return nullptr;
    }
    
    // Get shapes to determine operation type
    auto left_shape = left_tensor->shape();
    auto right_shape = right_tensor->shape();
    
    // For matrix multiplication (most common inner product case)
    if (verb_name == "*" && left_shape.size() >= 2 && right_shape.size() >= 2) {
        // Use TensorFlow matrix multiplication
        auto result = m_tf_session->matrix_multiply(left_tensor, right_tensor);
        return from_tensor(result);
    }
    
    // For vector inner product (dot product)
    if (verb_name == "*" && left_shape.size() == 1 && right_shape.size() == 1) {
        // Element-wise multiply then sum
        auto multiply_result = m_tf_session->multiply(left_tensor, right_tensor);
        auto sum_result = m_tf_session->reduce_sum(multiply_result, {0});
        return from_tensor(sum_result);
    }
    
    // General case: element-wise operation followed by reduction
    // This is a simplified implementation
    JValue mult_result = j_times(left, right);
    if (std::holds_alternative<std::nullptr_t>(mult_result)) {
        return nullptr;
    }
    
    // Sum over the last axis for inner product
    auto mult_tensor = to_tensor(mult_result);
    if (!mult_tensor) {
        return mult_result;
    }
    
    auto shape = mult_tensor->shape();
    if (shape.empty()) {
        return mult_result; // Already a scalar
    }
    
    // Sum over the last dimension
    auto result = m_tf_session->reduce_sum(mult_tensor, {static_cast<int>(shape.size() - 1)});
    return from_tensor(result);
}

JValue Interpreter::evaluate_train_expression(TrainExpressionNode* node, const JValue& argument) {
    // Handle train expressions (forks, hooks, etc.)
    // For now, implement 3-verb forks: (f g h) y = (f y) g (h y)
    
    if (node->verbs.size() == 3) {
        // This is a fork: (f g h) y = (f y) g (h y)
        
        // Apply first verb (f) to argument
        JValue left_result;
        if (node->verbs[0]->type == AstNodeType::NAME_IDENTIFIER) {
            auto* verb_node = static_cast<NameNode*>(node->verbs[0].get());
            left_result = execute_monadic_verb(verb_node->name, argument);
        } else if (node->verbs[0]->type == AstNodeType::VERB) {
            auto* verb_node = static_cast<VerbNode*>(node->verbs[0].get());
            left_result = execute_monadic_verb(verb_node->identifier, argument);
        } else if (node->verbs[0]->type == AstNodeType::ADVERB_APPLICATION) {
            auto* adverb_app_node = static_cast<AdverbApplicationNode*>(node->verbs[0].get());
            left_result = execute_adverb_application(adverb_app_node, argument);
        } else {
            std::cerr << "Unsupported verb type in train expression." << std::endl;
            return nullptr;
        }
        
        // Apply third verb (h) to argument  
        JValue right_result;
        if (node->verbs[2]->type == AstNodeType::NAME_IDENTIFIER) {
            auto* verb_node = static_cast<NameNode*>(node->verbs[2].get());
            right_result = execute_monadic_verb(verb_node->name, argument);
        } else if (node->verbs[2]->type == AstNodeType::VERB) {
            auto* verb_node = static_cast<VerbNode*>(node->verbs[2].get());
            right_result = execute_monadic_verb(verb_node->identifier, argument);
        } else if (node->verbs[2]->type == AstNodeType::ADVERB_APPLICATION) {
            auto* adverb_app_node = static_cast<AdverbApplicationNode*>(node->verbs[2].get());
            right_result = execute_adverb_application(adverb_app_node, argument);
        } else {
            std::cerr << "Unsupported verb type in train expression." << std::endl;
            return nullptr;
        }
        
        // Check if either result failed
        if (std::holds_alternative<std::nullptr_t>(left_result) || std::holds_alternative<std::nullptr_t>(right_result)) {
            return nullptr;
        }
        
        // Apply middle verb (g) dyadically: left_result g right_result
        if (node->verbs[1]->type == AstNodeType::NAME_IDENTIFIER) {
            auto* verb_node = static_cast<NameNode*>(node->verbs[1].get());
            return execute_dyadic_verb(verb_node->name, left_result, right_result);
        } else if (node->verbs[1]->type == AstNodeType::VERB) {
            auto* verb_node = static_cast<VerbNode*>(node->verbs[1].get());
            return execute_dyadic_verb(verb_node->identifier, left_result, right_result);
        } else {
            std::cerr << "Unsupported middle verb type in fork expression." << std::endl;
            return nullptr;
        }
    } else if (node->verbs.size() == 2) {
        // This is a hook: (g h) y = y g (h y) 
        std::cerr << "Hook train expressions not yet implemented." << std::endl;
        return nullptr;
    } else {
        std::cerr << "Train expressions with " << node->verbs.size() << " verbs not yet supported." << std::endl;
        return nullptr;
    }
}

JValue Interpreter::evaluate_train_expression_graph(TrainExpressionNode* node, const JValue& argument) {
    // Graph-based version of train expression evaluation using deferred execution
    
    if (node->verbs.size() == 3) {
        // This is a fork: (f g h) y = (f y) g (h y)
        
        // Convert argument to deferred tensor if it's a regular tensor
        std::shared_ptr<DeferredTensor> deferred_arg;
        if (std::holds_alternative<std::shared_ptr<JTensor>>(argument)) {
            auto tensor = std::get<std::shared_ptr<JTensor>>(argument);
            deferred_arg = DeferredTensor::from_tensor(m_graph_builder->get_graph(), tensor);
        } else {
            // For now, only support tensor arguments in graph mode
            std::cerr << "Graph mode only supports tensor arguments for train expressions." << std::endl;
            return nullptr;
        }
        
        // Apply first verb (f) to argument - create deferred operation
        std::shared_ptr<DeferredTensor> left_result;
        if (node->verbs[0]->type == AstNodeType::NAME_IDENTIFIER) {
            auto* verb_node = static_cast<NameNode*>(node->verbs[0].get());
            left_result = execute_monadic_verb_graph(verb_node->name, deferred_arg);
        } else if (node->verbs[0]->type == AstNodeType::VERB) {
            auto* verb_node = static_cast<VerbNode*>(node->verbs[0].get());
            left_result = execute_monadic_verb_graph(verb_node->identifier, deferred_arg);
        } else {
            std::cerr << "Unsupported verb type in graph train expression." << std::endl;
            return nullptr;
        }
        
        // Apply third verb (h) to argument - create deferred operation
        std::shared_ptr<DeferredTensor> right_result;
        if (node->verbs[2]->type == AstNodeType::NAME_IDENTIFIER) {
            auto* verb_node = static_cast<NameNode*>(node->verbs[2].get());
            right_result = execute_monadic_verb_graph(verb_node->name, deferred_arg);
        } else if (node->verbs[2]->type == AstNodeType::VERB) {
            auto* verb_node = static_cast<VerbNode*>(node->verbs[2].get());
            right_result = execute_monadic_verb_graph(verb_node->identifier, deferred_arg);
        } else {
            std::cerr << "Unsupported verb type in graph train expression." << std::endl;
            return nullptr;
        }
        
        // Check if either result failed
        if (!left_result || !right_result) {
            return nullptr;
        }
        
        // Apply middle verb (g) dyadically: left_result g right_result
        std::shared_ptr<DeferredTensor> final_result;
        if (node->verbs[1]->type == AstNodeType::NAME_IDENTIFIER) {
            auto* verb_node = static_cast<NameNode*>(node->verbs[1].get());
            final_result = execute_dyadic_verb_graph(verb_node->name, left_result, right_result);
        } else if (node->verbs[1]->type == AstNodeType::VERB) {
            auto* verb_node = static_cast<VerbNode*>(node->verbs[1].get());
            final_result = execute_dyadic_verb_graph(verb_node->identifier, left_result, right_result);
        } else {
            std::cerr << "Unsupported middle verb type in graph fork expression." << std::endl;
            return nullptr;
        }
        
        return final_result;
    } else if (node->verbs.size() == 2) {
        // This is a hook: (g h) y = y g (h y) 
        std::cerr << "Hook train expressions not yet implemented in graph mode." << std::endl;
        return nullptr;
    } else {
        std::cerr << "Graph train expressions with " << node->verbs.size() << " verbs not yet supported." << std::endl;
        return nullptr;
    }
}

std::shared_ptr<DeferredTensor> Interpreter::execute_monadic_verb_graph(const std::string& verb_name, std::shared_ptr<DeferredTensor> operand) {
    // Graph-based monadic verb execution - creates deferred operations
    return m_graph_builder->apply_monadic_verb(verb_name, operand);
}

std::shared_ptr<DeferredTensor> Interpreter::execute_dyadic_verb_graph(const std::string& verb_name, std::shared_ptr<DeferredTensor> left, std::shared_ptr<DeferredTensor> right) {
    // Graph-based dyadic verb execution - creates deferred operations
    return m_graph_builder->apply_dyadic_verb(verb_name, left, right);
}

} // namespace JInterpreter