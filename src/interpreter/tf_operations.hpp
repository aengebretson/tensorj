#ifndef J_INTERPRETER_TF_OPERATIONS_HPP
#define J_INTERPRETER_TF_OPERATIONS_HPP

#include <memory>
#include <vector>
#include <variant>
#include <string>
#include <iostream>
#include <cstring>

// Only include TensorFlow headers if we actually have the C++ API
// For C API, we'll use a different approach
#ifdef TENSORFLOW_ENABLED
  #if __has_include("tensorflow/core/public/session.h")
    #define HAS_TF_CC_API 1
    #include "tensorflow/core/public/session.h"
    #include "tensorflow/core/framework/tensor.h"
  #elif __has_include("tensorflow/c/c_api.h")
    #define HAS_TF_C_API 1
    #include "tensorflow/c/c_api.h"
  #endif
#endif

namespace JInterpreter {

// Forward declarations
class JTensor;
using JValue = std::variant<
    long long,
    double, 
    std::string,
    std::shared_ptr<JTensor>,
    std::nullptr_t
>;

// Wrapper class for tensors - works with or without TensorFlow
class JTensor {
public:
    JTensor();
    ~JTensor();
    
    // Factory methods
    static std::shared_ptr<JTensor> zeros(const std::vector<long long>& shape);
    static std::shared_ptr<JTensor> from_data(const std::vector<double>& data, const std::vector<long long>& shape = {});
    static std::shared_ptr<JTensor> from_data(const std::vector<long long>& data, const std::vector<long long>& shape = {});
    static std::shared_ptr<JTensor> scalar(double value);
    static std::shared_ptr<JTensor> scalar(long long value);
    static std::shared_ptr<JTensor> copy(const JTensor& other);
    
    // Basic operations
    std::vector<long long> shape() const;
    size_t rank() const;
    size_t size() const;
    
    // Element access
    template<typename T>
    T get_scalar() const;
    
    template<typename T>
    std::vector<T> get_flat() const;
    
    // Type information
    enum class DataType { INT64, FLOAT64, STRING, UNKNOWN };
    DataType dtype() const;
    
    // Print for debugging
    void print(std::ostream& os) const;
    
private:
    std::vector<long long> m_shape;
    DataType m_dtype;
    
    // Always use internal storage - TensorFlow is optional
    std::vector<double> m_float_data;
    std::vector<long long> m_int_data;
    std::vector<std::string> m_string_data;
    
#ifdef HAS_TF_CC_API
    tensorflow::Tensor m_tf_tensor;
    bool m_has_tf_tensor = false;
#elif defined(HAS_TF_C_API)
    TF_Tensor* m_c_tensor = nullptr;
#endif
    
    void init_from_data(const std::vector<double>& data, const std::vector<long long>& shape);
    void init_from_data(const std::vector<long long>& data, const std::vector<long long>& shape);
};

// TensorFlow session wrapper - provides same interface regardless of backend
class TFSession {
public:
    TFSession();
    ~TFSession();
    
    bool is_initialized() const;
    
    // Basic arithmetic operations
    std::shared_ptr<JTensor> add(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> subtract(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> multiply(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> divide(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    
    // Array operations
    std::shared_ptr<JTensor> reshape(const std::shared_ptr<JTensor>& tensor, const std::vector<long long>& new_shape);
    std::shared_ptr<JTensor> transpose(const std::shared_ptr<JTensor>& tensor);
    std::shared_ptr<JTensor> concatenate(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b, int axis = 0);
    std::shared_ptr<JTensor> matrix_multiply(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    
    // Reduction operations
    std::shared_ptr<JTensor> reduce_sum(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes = {});
    JValue reduce_sum(const JValue& operand);
    std::shared_ptr<JTensor> reduce_product(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes = {});
    JValue reduce_product(const JValue& operand);
    std::shared_ptr<JTensor> reduce_min(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes = {});
    JValue reduce_min(const JValue& operand);
    std::shared_ptr<JTensor> reduce_max(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes = {});
    JValue reduce_max(const JValue& operand);
    std::shared_ptr<JTensor> reduce_mean(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes = {});
    JValue reduce_mean(const JValue& operand);
    
    // Comparison operations
    std::shared_ptr<JTensor> equal(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> less_than(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> greater_than(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> less_equal(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> greater_equal(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    
    // Array generation
    std::shared_ptr<JTensor> iota(long long n);
    
private:
    bool m_initialized;
    
#ifdef HAS_TF_CC_API
    std::unique_ptr<tensorflow::Session> m_session;
    tensorflow::GraphDef m_graph_def;
#elif defined(HAS_TF_C_API)
    TF_Status* m_status = nullptr;
    TF_Session* m_session = nullptr;
    TF_Graph* m_graph = nullptr;
    TF_SessionOptions* m_session_options = nullptr;
#endif
};

// Template specializations declarations
template<> double JTensor::get_scalar() const;
template<> long long JTensor::get_scalar() const;
template<> std::vector<double> JTensor::get_flat() const;
template<> std::vector<long long> JTensor::get_flat() const;

} // namespace JInterpreter

#endif // J_INTERPRETER_TF_OPERATIONS_HPP