#ifndef TF_GRAPH_HPP
#define TF_GRAPH_HPP

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <any>
#include <functional>
#include "tf_operations.hpp"

// TensorFlow GraphDef includes for true graph execution
#if HAS_TF_CC_API
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#endif

namespace JInterpreter {

enum class GraphOpType {
    INPUT,
    CONSTANT,
    ADD, SUBTRACT, MULTIPLY, DIVIDE,
    REDUCE_SUM, REDUCE_PRODUCT, REDUCE_MIN, REDUCE_MAX, REDUCE_MEAN,
    MATRIX_MULTIPLY,
    RESHAPE, CONCATENATE,
    EQUAL, LESS_THAN, GREATER_THAN, LESS_EQUAL, GREATER_EQUAL,
    IOTA, SHAPE, TALLY
};

struct GraphNode {
    std::string id;
    GraphOpType op_type;
    std::vector<std::string> input_ids;
    std::vector<long long> shape;
    std::string dtype;
    
    // Operation-specific parameters
    std::unordered_map<std::string, std::any> parameters;
    
    GraphNode(const std::string& node_id, GraphOpType type, 
              const std::vector<std::string>& inputs = {},
              const std::vector<long long>& node_shape = {},
              const std::string& data_type = "float64")
        : id(node_id), op_type(type), input_ids(inputs), shape(node_shape), dtype(data_type) {}
};

class TFGraph {
private:
    std::vector<std::unique_ptr<GraphNode>> nodes_;
    std::unordered_map<std::string, size_t> node_map_;
    size_t next_node_counter_;
    
#if HAS_TF_CC_API
    // TensorFlow graph for true graph execution
    std::unique_ptr<tensorflow::Graph> tf_graph_;
    std::unordered_map<std::string, tensorflow::Node*> tf_nodes_;
#endif
    
public:
    TFGraph();
    ~TFGraph() = default;
    
    // Graph building methods
    std::string add_input(const std::vector<long long>& shape, const std::string& dtype = "float64");
    std::string add_constant(std::shared_ptr<JTensor> tensor);
    std::string add_operation(GraphOpType op_type, const std::vector<std::string>& inputs, 
                             const std::unordered_map<std::string, std::any>& params = {});
    
    // Graph execution
    std::unordered_map<std::string, std::shared_ptr<JTensor>> execute(
        std::shared_ptr<TFSession> tf_session,
        const std::unordered_map<std::string, std::shared_ptr<JTensor>>& inputs);
    
    // True TensorFlow graph execution
    std::unordered_map<std::string, std::shared_ptr<JTensor>> execute_with_graphdef(
        std::shared_ptr<TFSession> tf_session,
        const std::unordered_map<std::string, std::shared_ptr<JTensor>>& inputs);
    
    // Graph introspection
    const GraphNode* get_node(const std::string& node_id) const;
    std::vector<std::string> get_output_nodes() const;
    void print_graph() const;
    size_t node_count() const { return nodes_.size(); }
    
    // Graph optimization
    void optimize();
    
private:
    std::string generate_node_id();
    void topological_sort(std::vector<size_t>& sorted_indices) const;
    bool has_cycle() const;
    void execute_node(const GraphNode* node, 
                     const std::unordered_map<std::string, std::shared_ptr<JTensor>>& node_results,
                     std::shared_ptr<TFSession> tf_session,
                     std::unordered_map<std::string, std::shared_ptr<JTensor>>& results);
    
#if HAS_TF_CC_API
    // TensorFlow graph building methods
    void build_tensorflow_graph();
    tensorflow::Node* create_tf_node(const GraphNode* node);
    tensorflow::DataType get_tf_data_type(const std::string& dtype);
    tensorflow::TensorShape get_tf_tensor_shape(const std::vector<long long>& shape);
#endif
};

// Deferred computation class
class DeferredTensor {
private:
    std::shared_ptr<TFGraph> graph_;
    std::string node_id_;
    std::vector<long long> shape_;
    std::string dtype_;
    bool is_materialized_;
    std::shared_ptr<JTensor> materialized_value_;
    
public:
    DeferredTensor(std::shared_ptr<TFGraph> graph, const std::string& node_id, 
                   const std::vector<long long>& shape, const std::string& dtype = "float64")
        : graph_(graph), node_id_(node_id), shape_(shape), dtype_(dtype), is_materialized_(false) {}
    
    // Static factory methods
    static std::shared_ptr<DeferredTensor> from_tensor(std::shared_ptr<TFGraph> graph, 
                                                       std::shared_ptr<JTensor> tensor);
    static std::shared_ptr<DeferredTensor> input(std::shared_ptr<TFGraph> graph,
                                                  const std::vector<long long>& shape,
                                                  const std::string& dtype = "float64");
    
    // Force evaluation
    std::shared_ptr<JTensor> materialize(std::shared_ptr<TFSession> tf_session,
                                        const std::unordered_map<std::string, std::shared_ptr<JTensor>>& inputs = {});
    
    // Deferred operations
    std::shared_ptr<DeferredTensor> add(std::shared_ptr<DeferredTensor> other);
    std::shared_ptr<DeferredTensor> subtract(std::shared_ptr<DeferredTensor> other);
    std::shared_ptr<DeferredTensor> multiply(std::shared_ptr<DeferredTensor> other);
    std::shared_ptr<DeferredTensor> divide(std::shared_ptr<DeferredTensor> other);
    std::shared_ptr<DeferredTensor> reduce_sum();
    std::shared_ptr<DeferredTensor> reduce_min();
    std::shared_ptr<DeferredTensor> reduce_max();
    std::shared_ptr<DeferredTensor> tally();
    
    // Accessors
    const std::vector<long long>& shape() const { return shape_; }
    const std::string& node_id() const { return node_id_; }
    const std::string& dtype() const { return dtype_; }
    std::shared_ptr<TFGraph> graph() const { return graph_; }
};

// Graph builder for J expressions
class JGraphBuilder {
private:
    std::shared_ptr<TFGraph> graph_;
    
public:
    JGraphBuilder() : graph_(std::make_shared<TFGraph>()) {}
    
    // Convert JValue to deferred tensor
    std::shared_ptr<DeferredTensor> from_jvalue(const JValue& value);
    
    // Build fork expression: (f g h) y = (f y) g (h y)
    std::shared_ptr<DeferredTensor> build_fork(
        std::shared_ptr<DeferredTensor> arg,
        const std::string& left_verb,
        const std::string& middle_verb, 
        const std::string& right_verb);
    
    // Apply verb to deferred tensor
    std::shared_ptr<DeferredTensor> apply_monadic_verb(const std::string& verb, 
                                                       std::shared_ptr<DeferredTensor> operand);
    std::shared_ptr<DeferredTensor> apply_dyadic_verb(const std::string& verb,
                                                      std::shared_ptr<DeferredTensor> left,
                                                      std::shared_ptr<DeferredTensor> right);
    
    std::shared_ptr<TFGraph> get_graph() const { return graph_; }
};

} // namespace JInterpreter

#endif // TF_GRAPH_HPP
