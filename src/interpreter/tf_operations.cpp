// src/interpreter/tf_operations_simple.cpp
#include "tf_operations.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace JInterpreter {

// ===== JTensor Implementation =====

JTensor::JTensor() : m_dtype(DataType::UNKNOWN) {}

JTensor::~JTensor() = default;

std::shared_ptr<JTensor> JTensor::scalar(double value) {
    auto tensor = std::make_shared<JTensor>();
    tensor->m_shape = {}; // Empty shape for scalar
    tensor->m_dtype = DataType::FLOAT64;
    tensor->m_float_data = {value};
    return tensor;
}

std::shared_ptr<JTensor> JTensor::scalar(long long value) {
    auto tensor = std::make_shared<JTensor>();
    tensor->m_shape = {}; // Empty shape for scalar
    tensor->m_dtype = DataType::INT64;
    tensor->m_int_data = {value};
    return tensor;
}

std::shared_ptr<JTensor> JTensor::from_data(const std::vector<double>& data, const std::vector<long long>& shape) {
    auto tensor = std::make_shared<JTensor>();
    tensor->m_dtype = DataType::FLOAT64;
    tensor->m_float_data = data;
    
    // If shape is provided, use it; if empty and data is single element, keep empty (scalar)
    if (shape.empty() && data.size() == 1) {
        tensor->m_shape = {}; // Scalar tensor
    } else if (shape.empty()) {
        tensor->m_shape = {static_cast<long long>(data.size())}; // Vector
    } else {
        tensor->m_shape = shape;
    }
    
    return tensor;
}

std::shared_ptr<JTensor> JTensor::from_data(const std::vector<long long>& data, const std::vector<long long>& shape) {
    auto tensor = std::make_shared<JTensor>();
    tensor->m_dtype = DataType::INT64;
    tensor->m_int_data = data;
    
    // If shape is provided, use it; if empty and data is single element, keep empty (scalar)
    if (shape.empty() && data.size() == 1) {
        tensor->m_shape = {}; // Scalar tensor
    } else if (shape.empty()) {
        tensor->m_shape = {static_cast<long long>(data.size())}; // Vector
    } else {
        tensor->m_shape = shape;
    }
    
    return tensor;
}

std::vector<long long> JTensor::shape() const {
    return m_shape;
}

size_t JTensor::rank() const {
    return m_shape.size();
}

size_t JTensor::size() const {
    if (m_shape.empty()) return 1; // Scalar
    return std::accumulate(m_shape.begin(), m_shape.end(), 1LL, std::multiplies<long long>());
}

template<>
double JTensor::get_scalar() const {
    if (!m_shape.empty()) {
        throw std::runtime_error("Tensor is not a scalar");
    }
    
    if (m_dtype == DataType::FLOAT64 && !m_float_data.empty()) {
        return m_float_data[0];
    } else if (m_dtype == DataType::INT64 && !m_int_data.empty()) {
        return static_cast<double>(m_int_data[0]);
    }
    
    return 0.0;
}

template<>
long long JTensor::get_scalar() const {
    if (!m_shape.empty()) {
        throw std::runtime_error("Tensor is not a scalar");
    }
    
    if (m_dtype == DataType::INT64 && !m_int_data.empty()) {
        return m_int_data[0];
    } else if (m_dtype == DataType::FLOAT64 && !m_float_data.empty()) {
        return static_cast<long long>(m_float_data[0]);
    }
    
    return 0;
}

template<>
std::vector<long long> JTensor::get_flat() const {
    if (m_dtype != DataType::INT64) {
        throw std::runtime_error("Tensor is not INT64 type");
    }
    return m_int_data;
}

template<>
std::vector<double> JTensor::get_flat() const {
    if (m_dtype != DataType::FLOAT64) {
        throw std::runtime_error("Tensor is not FLOAT64 type");
    }
    return m_float_data;
}

JTensor::DataType JTensor::dtype() const {
    return m_dtype;
}

std::string JTensor::dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::INT64:
            return "INT64";
        case DataType::FLOAT64:
            return "FLOAT64";
        case DataType::STRING:
            return "STRING";
        case DataType::UNKNOWN:
        default:
            return "UNKNOWN";
    }
}

void JTensor::print(std::ostream& os) const {
    os << "JTensor(shape=[";
    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (i > 0) os << ", ";
        os << m_shape[i];
    }
    os << "], dtype=" << (m_dtype == DataType::INT64 ? "INT64" : "FLOAT64");
    os << ", data=";
    
    if (m_shape.empty()) {
        // Scalar
        if (m_dtype == DataType::FLOAT64) {
            os << get_scalar<double>();
        } else {
            os << get_scalar<long long>();
        }
    } else if (size() <= 10) {
        // Small tensor - print all values
        os << "[";
        if (m_dtype == DataType::FLOAT64) {
            for (size_t i = 0; i < m_float_data.size(); ++i) {
                if (i > 0) os << ", ";
                os << m_float_data[i];
            }
        } else {
            for (size_t i = 0; i < m_int_data.size(); ++i) {
                if (i > 0) os << ", ";
                os << m_int_data[i];
            }
        }
        os << "]";
    } else {
        os << "<" << size() << " elements>";
    }
    
    os << ")";
}

// ===== TFSession Implementation =====

TFSession::TFSession() : m_initialized(true) {
    std::cout << "TFSession initialized (stub mode)" << std::endl;
}

TFSession::~TFSession() = default;

bool TFSession::is_initialized() const {
    return m_initialized;
}

std::shared_ptr<JTensor> TFSession::add(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // Support broadcasting for scalar + tensor and tensor + scalar
    bool a_is_scalar = (a->rank() == 0);
    bool b_is_scalar = (b->rank() == 0);
    
    // If neither is scalar, shapes must match exactly
    if (!a_is_scalar && !b_is_scalar && a->shape() != b->shape()) {
        std::cerr << "Shape mismatch in addition" << std::endl;
        return nullptr;
    }
    
    // Determine result shape and data type
    auto result_shape = a_is_scalar ? b->shape() : a->shape();
    bool result_is_float = (a->dtype() == JTensor::DataType::FLOAT64 || b->dtype() == JTensor::DataType::FLOAT64);
    
    if (result_is_float) {
        // Convert to double and perform addition
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<double> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = a_is_scalar ? 0 : i;
            size_t b_idx = b_is_scalar ? 0 : i;
            result_data[i] = a_data[a_idx] + b_data[b_idx];
        }
        
        return JTensor::from_data(result_data, result_shape);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = a_is_scalar ? 0 : i;
            size_t b_idx = b_is_scalar ? 0 : i;
            result_data[i] = a_data[a_idx] + b_data[b_idx];
        }
        
        return JTensor::from_data(result_data, result_shape);
    }
}

std::shared_ptr<JTensor> TFSession::subtract(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // Support broadcasting for scalar + tensor and tensor + scalar
    bool a_is_scalar = (a->rank() == 0);
    bool b_is_scalar = (b->rank() == 0);
    
    // If neither is scalar, shapes must match exactly
    if (!a_is_scalar && !b_is_scalar && a->shape() != b->shape()) {
        std::cerr << "Shape mismatch in subtraction" << std::endl;
        return nullptr;
    }
    
    // Determine result shape and data type
    auto result_shape = a_is_scalar ? b->shape() : a->shape();
    bool result_is_float = (a->dtype() == JTensor::DataType::FLOAT64 || b->dtype() == JTensor::DataType::FLOAT64);
    
    if (result_is_float) {
        // Convert to double and perform subtraction
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<double> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = a_is_scalar ? 0 : i;
            size_t b_idx = b_is_scalar ? 0 : i;
            result_data[i] = a_data[a_idx] - b_data[b_idx];
        }
        
        return JTensor::from_data(result_data, result_shape);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = a_is_scalar ? 0 : i;
            size_t b_idx = b_is_scalar ? 0 : i;
            result_data[i] = a_data[a_idx] - b_data[b_idx];
        }
        
        return JTensor::from_data(result_data, result_shape);
    }
}

std::shared_ptr<JTensor> TFSession::multiply(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // Support broadcasting for scalar + tensor and tensor + scalar
    bool a_is_scalar = (a->rank() == 0);
    bool b_is_scalar = (b->rank() == 0);
    
    // If neither is scalar, shapes must match exactly
    if (!a_is_scalar && !b_is_scalar && a->shape() != b->shape()) {
        std::cerr << "Shape mismatch in multiplication" << std::endl;
        return nullptr;
    }
    
    // Determine result shape and data type
    auto result_shape = a_is_scalar ? b->shape() : a->shape();
    bool result_is_float = (a->dtype() == JTensor::DataType::FLOAT64 || b->dtype() == JTensor::DataType::FLOAT64);
    
    if (result_is_float) {
        // Convert to double and perform multiplication
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<double> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = a_is_scalar ? 0 : i;
            size_t b_idx = b_is_scalar ? 0 : i;
            result_data[i] = a_data[a_idx] * b_data[b_idx];
        }
        
        return JTensor::from_data(result_data, result_shape);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = a_is_scalar ? 0 : i;
            size_t b_idx = b_is_scalar ? 0 : i;
            result_data[i] = a_data[a_idx] * b_data[b_idx];
        }
        
        return JTensor::from_data(result_data, result_shape);
    }
}

std::shared_ptr<JTensor> TFSession::divide(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) {
        return nullptr;
    }
    
    // Check if shapes are compatible for broadcasting
    if (a->shape() != b->shape() && a->size() != 1 && b->size() != 1) {
        std::cerr << "Shape mismatch in division" << std::endl;
        return nullptr;
    }
    
    // Always return double for division
    std::vector<double> a_data;
    if (a->dtype() == JTensor::DataType::INT64) {
        auto a_int_data = a->get_flat<long long>();
        a_data.assign(a_int_data.begin(), a_int_data.end());
    } else {
        a_data = a->get_flat<double>();
    }
    
    std::vector<double> b_data;
    if (b->dtype() == JTensor::DataType::INT64) {
        auto b_int_data = b->get_flat<long long>();
        b_data.assign(b_int_data.begin(), b_int_data.end());
    } else {
        b_data = b->get_flat<double>();
    }
    
    // Determine result shape and size
    auto result_shape = (a->size() >= b->size()) ? a->shape() : b->shape();
    size_t result_size = std::max(a_data.size(), b_data.size());
    std::vector<double> result_data(result_size);
    
    for (size_t i = 0; i < result_size; ++i) {
        // Handle broadcasting: use modulo for indexing
        size_t a_idx = (a_data.size() == 1) ? 0 : i;
        size_t b_idx = (b_data.size() == 1) ? 0 : i;
        
        if (b_data[b_idx] == 0.0) {
            std::cerr << "Division by zero" << std::endl;
            return nullptr;
        }
        result_data[i] = a_data[a_idx] / b_data[b_idx];
    }
    
    return JTensor::from_data(result_data, result_shape);
}

std::shared_ptr<JTensor> TFSession::power(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // Support broadcasting for scalar + tensor and tensor + scalar
    bool a_is_scalar = (a->rank() == 0);
    bool b_is_scalar = (b->rank() == 0);
    
    // If neither is scalar, shapes must match exactly
    if (!a_is_scalar && !b_is_scalar && a->shape() != b->shape()) {
        std::cerr << "Shape mismatch in power operation" << std::endl;
        return nullptr;
    }
    
    // Determine result shape
    auto result_shape = a_is_scalar ? b->shape() : a->shape();
    
    // Always convert to double for power operation (due to potential fractional results)
    std::vector<double> a_data, b_data;
    
    if (a->dtype() == JTensor::DataType::INT64) {
        auto a_int_data = a->get_flat<long long>();
        a_data.assign(a_int_data.begin(), a_int_data.end());
    } else {
        a_data = a->get_flat<double>();
    }
    
    if (b->dtype() == JTensor::DataType::INT64) {
        auto b_int_data = b->get_flat<long long>();
        b_data.assign(b_int_data.begin(), b_int_data.end());
    } else {
        b_data = b->get_flat<double>();
    }
    
    size_t result_size = std::max(a_data.size(), b_data.size());
    std::vector<double> result_data(result_size);
    
    for (size_t i = 0; i < result_size; ++i) {
        size_t a_idx = a_is_scalar ? 0 : i;
        size_t b_idx = b_is_scalar ? 0 : i;
        result_data[i] = std::pow(a_data[a_idx], b_data[b_idx]);
    }
    
    return JTensor::from_data(result_data, result_shape);
}

std::shared_ptr<JTensor> TFSession::negate(const std::shared_ptr<JTensor>& tensor) {
    if (!tensor) return nullptr;
    
    if (tensor->dtype() == JTensor::DataType::INT64) {
        auto data = tensor->get_flat<long long>();
        std::vector<long long> result_data(data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            result_data[i] = -data[i];
        }
        
        return JTensor::from_data(result_data, tensor->shape());
    } else {
        auto data = tensor->get_flat<double>();
        std::vector<double> result_data(data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            result_data[i] = -data[i];
        }
        
        return JTensor::from_data(result_data, tensor->shape());
    }
}

std::shared_ptr<JTensor> TFSession::square(const std::shared_ptr<JTensor>& tensor) {
    if (!tensor) return nullptr;
    
    if (tensor->dtype() == JTensor::DataType::INT64) {
        auto data = tensor->get_flat<long long>();
        std::vector<long long> result_data(data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            result_data[i] = data[i] * data[i];
        }
        
        return JTensor::from_data(result_data, tensor->shape());
    } else {
        auto data = tensor->get_flat<double>();
        std::vector<double> result_data(data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            result_data[i] = data[i] * data[i];
        }
        
        return JTensor::from_data(result_data, tensor->shape());
    }
}

std::shared_ptr<JTensor> TFSession::reciprocal(const std::shared_ptr<JTensor>& tensor) {
    if (!tensor) return nullptr;
    
    // Always convert to double for reciprocal operation (to handle fractional results)
    std::vector<double> data;
    
    if (tensor->dtype() == JTensor::DataType::INT64) {
        auto int_data = tensor->get_flat<long long>();
        data.assign(int_data.begin(), int_data.end());
    } else {
        data = tensor->get_flat<double>();
    }
    
    std::vector<double> result_data(data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] == 0.0) {
            std::cerr << "Division by zero in reciprocal operation" << std::endl;
            return nullptr;
        }
        result_data[i] = 1.0 / data[i];
    }
    
    return JTensor::from_data(result_data, tensor->shape());
}

std::shared_ptr<JTensor> TFSession::iota(long long n) {
    std::vector<long long> data;
    for (long long i = 0; i < n; ++i) {
        data.push_back(i);
    }
    return JTensor::from_data(data, {n});
}

std::shared_ptr<JTensor> TFSession::reduce_sum(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes) {
    if (!tensor) return nullptr;
    
    // Simple implementation: sum all elements
    if (tensor->dtype() == JTensor::DataType::INT64) {
        auto data = tensor->get_flat<long long>();
        long long sum = 0;
        for (auto val : data) {
            sum += val;
        }
        return JTensor::scalar(sum);
    } else {
        auto data = tensor->get_flat<double>();
        double sum = 0.0;
        for (auto val : data) {
            sum += val;
        }
        return JTensor::scalar(sum);
    }
}

JValue TFSession::reduce_sum(const JValue& operand) {
    if (!std::holds_alternative<std::shared_ptr<JTensor>>(operand)) {
        throw std::runtime_error("Operand for reduce_sum must be a JTensor");
    }
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(operand);
    return reduce_sum(tensor);
}

std::shared_ptr<JTensor> TFSession::reduce_product(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes) {
    if (!tensor) return nullptr;
    
    // Simple implementation: multiply all elements
    if (tensor->dtype() == JTensor::DataType::INT64) {
        auto data = tensor->get_flat<long long>();
        long long product = 1;
        for (auto val : data) {
            product *= val;
        }
        return JTensor::scalar(product);
    } else {
        auto data = tensor->get_flat<double>();
        double product = 1.0;
        for (auto val : data) {
            product *= val;
        }
        return JTensor::scalar(product);
    }
}

JValue TFSession::reduce_product(const JValue& operand) {
    if (!std::holds_alternative<std::shared_ptr<JTensor>>(operand)) {
        throw std::runtime_error("Operand for reduce_product must be a JTensor");
    }
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(operand);
    return reduce_product(tensor);
}

std::shared_ptr<JTensor> TFSession::reshape(const std::shared_ptr<JTensor>& tensor, const std::vector<long long>& new_shape) {
    if (!tensor) return nullptr;
    
    // Calculate sizes
    size_t old_size = tensor->size();
    size_t new_size = 1;
    for (auto dim : new_shape) {
        new_size *= dim;
    }
    
    // J language reshape: cycle/repeat data to fill new shape
    if (tensor->dtype() == JTensor::DataType::INT64) {
        auto old_data = tensor->get_flat<long long>();
        std::vector<long long> new_data;
        new_data.reserve(new_size);
        
        // Cycle through old data to fill new shape
        for (size_t i = 0; i < new_size; ++i) {
            new_data.push_back(old_data[i % old_size]);
        }
        
        return JTensor::from_data(new_data, new_shape);
    } else {
        auto old_data = tensor->get_flat<double>();
        std::vector<double> new_data;
        new_data.reserve(new_size);
        
        // Cycle through old data to fill new shape
        for (size_t i = 0; i < new_size; ++i) {
            new_data.push_back(old_data[i % old_size]);
        }
        
        return JTensor::from_data(new_data, new_shape);
    }
}

std::shared_ptr<JTensor> TFSession::transpose(const std::shared_ptr<JTensor>& tensor) {
    if (!tensor) return nullptr;
    
    // For now, just return a copy
    std::cout << "Transpose not fully implemented (returning copy)" << std::endl;
    
    if (tensor->dtype() == JTensor::DataType::INT64) {
        return JTensor::from_data(tensor->get_flat<long long>(), tensor->shape());
    } else {
        return JTensor::from_data(tensor->get_flat<double>(), tensor->shape());
    }
}

// Advanced reduction operations
std::shared_ptr<JTensor> TFSession::reduce_min(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes) {
    if (!tensor) return nullptr;
    
    if (tensor->dtype() == JTensor::DataType::INT64) {
        auto data = tensor->get_flat<long long>();
        if (data.empty()) return JTensor::scalar(0LL);
        
        long long min_val = data[0];
        for (auto val : data) {
            if (val < min_val) min_val = val;
        }
        return JTensor::scalar(min_val);
    } else {
        auto data = tensor->get_flat<double>();
        if (data.empty()) return JTensor::scalar(0.0);
        
        double min_val = data[0];
        for (auto val : data) {
            if (val < min_val) min_val = val;
        }
        return JTensor::scalar(min_val);
    }
}

JValue TFSession::reduce_min(const JValue& operand) {
    if (!std::holds_alternative<std::shared_ptr<JTensor>>(operand)) {
        throw std::runtime_error("Operand for reduce_min must be a JTensor");
    }
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(operand);
    return reduce_min(tensor);
}

std::shared_ptr<JTensor> TFSession::reduce_max(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes) {
    if (!tensor) return nullptr;
    
    if (tensor->dtype() == JTensor::DataType::INT64) {
        auto data = tensor->get_flat<long long>();
        if (data.empty()) return JTensor::scalar(0LL);
        
        long long max_val = data[0];
        for (auto val : data) {
            if (val > max_val) max_val = val;
        }
        return JTensor::scalar(max_val);
    } else {
        auto data = tensor->get_flat<double>();
        if (data.empty()) return JTensor::scalar(0.0);
        
        double max_val = data[0];
        for (auto val : data) {
            if (val > max_val) max_val = val;
        }
        return JTensor::scalar(max_val);
    }
}

JValue TFSession::reduce_max(const JValue& operand) {
    if (!std::holds_alternative<std::shared_ptr<JTensor>>(operand)) {
        throw std::runtime_error("Operand for reduce_max must be a JTensor");
    }
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(operand);
    return reduce_max(tensor);
}

std::shared_ptr<JTensor> TFSession::reduce_mean(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes) {
    if (!tensor) return nullptr;
    
    if (tensor->dtype() == JTensor::DataType::INT64) {
        auto data = tensor->get_flat<long long>();
        if (data.empty()) return JTensor::scalar(0.0);
        
        double sum = 0.0;
        for (auto val : data) {
            sum += static_cast<double>(val);
        }
        double mean = sum / static_cast<double>(data.size());
        return JTensor::scalar(mean);
    } else {
        auto data = tensor->get_flat<double>();
        if (data.empty()) return JTensor::scalar(0.0);
        
        double sum = 0.0;
        for (auto val : data) {
            sum += val;
        }
        double mean = sum / static_cast<double>(data.size());
        return JTensor::scalar(mean);
    }
}

JValue TFSession::reduce_mean(const JValue& operand) {
    if (!std::holds_alternative<std::shared_ptr<JTensor>>(operand)) {
        throw std::runtime_error("Operand for reduce_mean must be a JTensor");
    }
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(operand);
    return reduce_mean(tensor);
}

// Comparison operations
std::shared_ptr<JTensor> TFSession::equal(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // Handle mixed types - promote to float if needed
    bool has_float = (a->dtype() == JTensor::DataType::FLOAT64) || (b->dtype() == JTensor::DataType::FLOAT64);
    
    if (has_float) {
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] == b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] == b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    }
}

std::shared_ptr<JTensor> TFSession::less_than(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // Handle mixed types - promote to float if needed
    bool has_float = (a->dtype() == JTensor::DataType::FLOAT64) || (b->dtype() == JTensor::DataType::FLOAT64);
    
    if (has_float) {
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] < b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] < b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    }
}

std::shared_ptr<JTensor> TFSession::greater_than(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // Handle mixed types - promote to float if needed
    bool has_float = (a->dtype() == JTensor::DataType::FLOAT64) || (b->dtype() == JTensor::DataType::FLOAT64);
    
    if (has_float) {
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] > b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] > b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    }
}

std::shared_ptr<JTensor> TFSession::less_equal(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // Handle mixed types - promote to float if needed
    bool has_float = (a->dtype() == JTensor::DataType::FLOAT64) || (b->dtype() == JTensor::DataType::FLOAT64);
    
    if (has_float) {
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] <= b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] <= b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    }
}

std::shared_ptr<JTensor> TFSession::greater_equal(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // Handle mixed types - promote to float if needed
    bool has_float = (a->dtype() == JTensor::DataType::FLOAT64) || (b->dtype() == JTensor::DataType::FLOAT64);
    
    if (has_float) {
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] >= b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        size_t result_size = std::max(a_data.size(), b_data.size());
        std::vector<long long> result_data(result_size);
        
        for (size_t i = 0; i < result_size; ++i) {
            size_t a_idx = (a_data.size() == 1) ? 0 : i;
            size_t b_idx = (b_data.size() == 1) ? 0 : i;
            result_data[i] = (a_data[a_idx] >= b_data[b_idx]) ? 1 : 0;
        }
        
        auto result_shape = (a_data.size() > b_data.size()) ? a->shape() : b->shape();
        return JTensor::from_data(result_data, result_shape);
    }
}

// Array operations
std::shared_ptr<JTensor> TFSession::concatenate(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b, int axis) {
    if (!a || !b) return nullptr;
    
    // For now, implement simple concatenation along axis 0 (rows)
    if (axis != 0) {
        std::cerr << "Only axis=0 concatenation supported currently" << std::endl;
        return nullptr;
    }
    
    // Handle mixed types - promote to float if needed
    bool has_float = (a->dtype() == JTensor::DataType::FLOAT64) || (b->dtype() == JTensor::DataType::FLOAT64);
    
    if (has_float) {
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        // Concatenate the data
        std::vector<double> result_data;
        result_data.reserve(a_data.size() + b_data.size());
        result_data.insert(result_data.end(), a_data.begin(), a_data.end());
        result_data.insert(result_data.end(), b_data.begin(), b_data.end());
        
        // Calculate result shape
        std::vector<long long> result_shape;
        if (a->rank() == 0 && b->rank() == 0) {
            result_shape = {2}; // Two scalars become a 2-element vector
        } else if (a->rank() == 1 && b->rank() == 1) {
            result_shape = {static_cast<long long>(a_data.size() + b_data.size())};
        } else {
            result_shape = {static_cast<long long>(result_data.size())};
        }
        
        return JTensor::from_data(result_data, result_shape);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        std::vector<long long> result_data;
        result_data.reserve(a_data.size() + b_data.size());
        result_data.insert(result_data.end(), a_data.begin(), a_data.end());
        result_data.insert(result_data.end(), b_data.begin(), b_data.end());
        
        // Calculate result shape
        std::vector<long long> result_shape;
        if (a->rank() == 0 && b->rank() == 0) {
            result_shape = {2}; // Two scalars become a 2-element vector
        } else if (a->rank() == 1 && b->rank() == 1) {
            result_shape = {static_cast<long long>(a_data.size() + b_data.size())};
        } else {
            result_shape = {static_cast<long long>(result_data.size())};
        }
        
        return JTensor::from_data(result_data, result_shape);
    }
}

std::shared_ptr<JTensor> TFSession::matrix_multiply(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    // For now, implement simple dot product for 1D vectors
    if (a->rank() != 1 || b->rank() != 1) {
        std::cerr << "Matrix multiplication currently only supports 1D vectors (dot product)" << std::endl;
        return nullptr;
    }
    
    if (a->shape()[0] != b->shape()[0]) {
        std::cerr << "Vector dimensions must match for dot product" << std::endl;
        return nullptr;
    }
    
    // Handle mixed types - promote to float if needed
    bool has_float = (a->dtype() == JTensor::DataType::FLOAT64) || (b->dtype() == JTensor::DataType::FLOAT64);
    
    if (has_float) {
        std::vector<double> a_data, b_data;
        
        if (a->dtype() == JTensor::DataType::INT64) {
            auto a_int_data = a->get_flat<long long>();
            a_data.assign(a_int_data.begin(), a_int_data.end());
        } else {
            a_data = a->get_flat<double>();
        }
        
        if (b->dtype() == JTensor::DataType::INT64) {
            auto b_int_data = b->get_flat<long long>();
            b_data.assign(b_int_data.begin(), b_int_data.end());
        } else {
            b_data = b->get_flat<double>();
        }
        
        double result = 0.0;
        for (size_t i = 0; i < a_data.size(); ++i) {
            result += a_data[i] * b_data[i];
        }
        
        return JTensor::scalar(result);
    } else {
        // Both are integers
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        
        long long result = 0;
        for (size_t i = 0; i < a_data.size(); ++i) {
            result += a_data[i] * b_data[i];
        }
        
        return JTensor::scalar(result);
    }
}

} // namespace JInterpreter