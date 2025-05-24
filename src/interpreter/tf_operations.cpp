// src/interpreter/tf_operations_simple.cpp
#include "tf_operations.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>

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
    
    if (shape.empty()) {
        tensor->m_shape = {static_cast<long long>(data.size())};
    } else {
        tensor->m_shape = shape;
    }
    
    return tensor;
}

std::shared_ptr<JTensor> JTensor::from_data(const std::vector<long long>& data, const std::vector<long long>& shape) {
    auto tensor = std::make_shared<JTensor>();
    tensor->m_dtype = DataType::INT64;
    tensor->m_int_data = data;
    
    if (shape.empty()) {
        tensor->m_shape = {static_cast<long long>(data.size())};
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
    
    // Simple element-wise addition for matching shapes
    if (a->shape() != b->shape()) {
        std::cerr << "Shape mismatch in addition" << std::endl;
        return nullptr;
    }
    
    if (a->dtype() == JTensor::DataType::INT64) {
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        std::vector<long long> result_data(a_data.size());
        
        for (size_t i = 0; i < a_data.size(); ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
        
        return JTensor::from_data(result_data, a->shape());
    } else {
        auto a_data = a->get_flat<double>();
        auto b_data = b->get_flat<double>();
        std::vector<double> result_data(a_data.size());
        
        for (size_t i = 0; i < a_data.size(); ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
        
        return JTensor::from_data(result_data, a->shape());
    }
}

std::shared_ptr<JTensor> TFSession::subtract(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    if (a->shape() != b->shape()) {
        std::cerr << "Shape mismatch in subtraction" << std::endl;
        return nullptr;
    }
    
    if (a->dtype() == JTensor::DataType::INT64) {
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        std::vector<long long> result_data(a_data.size());
        
        for (size_t i = 0; i < a_data.size(); ++i) {
            result_data[i] = a_data[i] - b_data[i];
        }
        
        return JTensor::from_data(result_data, a->shape());
    } else {
        auto a_data = a->get_flat<double>();
        auto b_data = b->get_flat<double>();
        std::vector<double> result_data(a_data.size());
        
        for (size_t i = 0; i < a_data.size(); ++i) {
            result_data[i] = a_data[i] - b_data[i];
        }
        
        return JTensor::from_data(result_data, a->shape());
    }
}

std::shared_ptr<JTensor> TFSession::multiply(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    if (a->shape() != b->shape()) {
        std::cerr << "Shape mismatch in multiplication" << std::endl;
        return nullptr;
    }
    
    if (a->dtype() == JTensor::DataType::INT64) {
        auto a_data = a->get_flat<long long>();
        auto b_data = b->get_flat<long long>();
        std::vector<long long> result_data(a_data.size());
        
        for (size_t i = 0; i < a_data.size(); ++i) {
            result_data[i] = a_data[i] * b_data[i];
        }
        
        return JTensor::from_data(result_data, a->shape());
    } else {
        auto a_data = a->get_flat<double>();
        auto b_data = b->get_flat<double>();
        std::vector<double> result_data(a_data.size());
        
        for (size_t i = 0; i < a_data.size(); ++i) {
            result_data[i] = a_data[i] * b_data[i];
        }
        
        return JTensor::from_data(result_data, a->shape());
    }
}

std::shared_ptr<JTensor> TFSession::divide(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b) {
    if (!a || !b) return nullptr;
    
    if (a->shape() != b->shape()) {
        std::cerr << "Shape mismatch in division" << std::endl;
        return nullptr;
    }
    
    // Always return double for division
    auto a_data = (a->dtype() == JTensor::DataType::INT64) 
        ? std::vector<double>(a->get_flat<long long>().begin(), a->get_flat<long long>().end())
        : a->get_flat<double>();
    
    auto b_data = (b->dtype() == JTensor::DataType::INT64) 
        ? std::vector<double>(b->get_flat<long long>().begin(), b->get_flat<long long>().end())
        : b->get_flat<double>();
    
    std::vector<double> result_data(a_data.size());
    
    for (size_t i = 0; i < a_data.size(); ++i) {
        if (b_data[i] == 0.0) {
            std::cerr << "Division by zero" << std::endl;
            return nullptr;
        }
        result_data[i] = a_data[i] / b_data[i];
    }
    
    return JTensor::from_data(result_data, a->shape());
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

std::shared_ptr<JTensor> TFSession::reshape(const std::shared_ptr<JTensor>& tensor, const std::vector<long long>& new_shape) {
    if (!tensor) return nullptr;
    
    // Verify size compatibility
    size_t old_size = tensor->size();
    size_t new_size = 1;
    for (auto dim : new_shape) {
        new_size *= dim;
    }
    
    if (old_size != new_size) {
        std::cerr << "Reshape: size mismatch" << std::endl;
        return nullptr;
    }
    
    // Create new tensor with same data but different shape
    if (tensor->dtype() == JTensor::DataType::INT64) {
        return JTensor::from_data(tensor->get_flat<long long>(), new_shape);
    } else {
        return JTensor::from_data(tensor->get_flat<double>(), new_shape);
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

} // namespace JInterpreter