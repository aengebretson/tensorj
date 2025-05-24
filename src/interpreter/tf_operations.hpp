#ifndef J_INTERPRETER_TF_OPERATIONS_HPP
#define J_INTERPRETER_TF_OPERATIONS_HPP

#include <memory>
#include <vector>
#include <variant>
#include <string>

#ifdef TENSORFLOW_ENABLED
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"
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

// Wrapper class for TensorFlow tensors
class JTensor {
public:
    JTensor();
    
    // Create tensor with given shape, filled with zeros
    static std::shared_ptr<JTensor> zeros(const std::vector<long long>& shape);
    
    // Create tensor from data (shape will be inferred as 1D if not provided)
    static std::shared_ptr<JTensor> from_data(const std::vector<double>& data, const std::vector<long long>& shape = {});
    static std::shared_ptr<JTensor> from_data(const std::vector<long long>& data, const std::vector<long long>& shape = {});
    
    // Create scalar tensors
    static std::shared_ptr<JTensor> scalar(double value);
    static std::shared_ptr<JTensor> scalar(long long value);
    
    // Copy a tensor
    static std::shared_ptr<JTensor> copy(const JTensor& other);
    
#ifdef TENSORFLOW_ENABLED
    explicit JTensor(tensorflow::Tensor tf_tensor);
    const tensorflow::Tensor& get_tf_tensor() const { return m_tf_tensor; }
    tensorflow::Tensor& get_tf_tensor() { return m_tf_tensor; }
#endif
    
    ~JTensor();
    
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
    
#ifdef TENSORFLOW_ENABLED
    tensorflow::Tensor m_tf_tensor;
    bool m_has_tf_tensor;
#else
    // Stub storage for when TensorFlow is not available
    std::vector<double> m_float_data;
    std::vector<long long> m_int_data;
    std::vector<std::string> m_string_data;
#endif
    
    void init_from_data(const std::vector<double>& data, const std::vector<long long>& shape);
    void init_from_data(const std::vector<long long>& data, const std::vector<long long>& shape);
};

// TensorFlow session wrapper
class TFSession {
public:
    TFSession();
    ~TFSession();
    
    bool is_initialized() const;
    
    // Execute operations
    std::shared_ptr<JTensor> add(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> subtract(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> multiply(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    std::shared_ptr<JTensor> divide(const std::shared_ptr<JTensor>& a, const std::shared_ptr<JTensor>& b);
    
    // Array operations (J-specific)
    std::shared_ptr<JTensor> reshape(const std::shared_ptr<JTensor>& tensor, const std::vector<long long>& new_shape);
    std::shared_ptr<JTensor> transpose(const std::shared_ptr<JTensor>& tensor);
    std::shared_ptr<JTensor> reduce_sum(const std::shared_ptr<JTensor>& tensor, const std::vector<int>& axes = {});
    std::shared_ptr<JTensor> iota(long long n); // J's i. verb
    
private:
#ifdef TENSORFLOW_ENABLED
    std::unique_ptr<tensorflow::Session> m_session;
    tensorflow::GraphDef m_graph_def;
    bool m_initialized;
    
    // Helper methods for TensorFlow operations
    std::string create_binary_op(const std::string& op_name, 
                                const std::string& input1_name, 
                                const std::string& input2_name,
                                const std::string& output_name);
#else
    bool m_initialized;
    
    // Stub implementations
    std::shared_ptr<JTensor> stub_binary_op(const std::shared_ptr<JTensor>& a, 
                                          const std::shared_ptr<JTensor>& b,
                                          const std::string& op_name);
#endif
};

} // namespace JInterpreter

#endif // J_INTERPRETER_TF_OPERATIONS_HPP
