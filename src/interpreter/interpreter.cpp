#include "interpreter.hpp"
#include <stdexcept>
#include <iostream>
#include <variant>
#include <type_traits>

namespace JInterpreter {

Interpreter::Interpreter() {
    m_tf_session = std::make_unique<TFSession>();
    if (!m_tf_session->is_initialized()) {
        std::cerr << "Warning: TensorFlow session initialization failed. Using fallback mode." << std::endl;
    }
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
                all_floats = false;  // Mixed types favor float conversion
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
    } else if (all_floats) {
        // All elements are floats or mixed numeric - create float vector
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
    } else if (verb_name == "$") {
        return j_reshape(left, right);
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

} // namespace JInterpreter