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
            
        case AstNodeType::NAME_IDENTIFIER:
            return evaluate_name_identifier(static_cast<NameNode*>(node));
            
        case AstNodeType::MONADIC_APPLICATION:
            return evaluate_monadic_application(static_cast<MonadicApplicationNode*>(node));
            
        case AstNodeType::DYADIC_APPLICATION:
            return evaluate_dyadic_application(static_cast<DyadicApplicationNode*>(node));
            
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
    auto result = JTensor::from_data(shape);
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