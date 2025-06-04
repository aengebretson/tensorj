#include "tf_graph.hpp"
#include <iostream>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <variant>
#include <any>

#if HAS_TF_CC_API
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#endif

namespace JInterpreter {

// TFGraph implementation

TFGraph::TFGraph() : next_node_counter_(0) {
#if HAS_TF_CC_API
    // Initialize TensorFlow graph
    tf_graph_ = std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());
#endif
}

std::string TFGraph::generate_node_id() {
    return "node_" + std::to_string(next_node_counter_++);
}

std::string TFGraph::add_input(const std::vector<long long>& shape, const std::string& dtype) {
    std::string node_id = generate_node_id();
    auto node = std::make_unique<GraphNode>(node_id, GraphOpType::INPUT, std::vector<std::string>{}, shape, dtype);
    
    node_map_[node_id] = nodes_.size();
    nodes_.push_back(std::move(node));
    
    return node_id;
}

std::string TFGraph::add_constant(std::shared_ptr<JTensor> tensor) {
    std::string node_id = generate_node_id();
    auto node = std::make_unique<GraphNode>(node_id, GraphOpType::CONSTANT, std::vector<std::string>{}, 
                                          tensor->shape(), JTensor::dtype_to_string(tensor->dtype()));
    
    // Store the tensor data in parameters
    node->parameters["tensor_data"] = tensor;
    
    node_map_[node_id] = nodes_.size();
    nodes_.push_back(std::move(node));
    
    return node_id;
}

std::string TFGraph::add_operation(GraphOpType op_type, const std::vector<std::string>& inputs, 
                                  const std::unordered_map<std::string, std::any>& params) {
    std::string node_id = generate_node_id();
    
    // Determine output shape based on operation type and input shapes
    std::vector<long long> output_shape;
    std::string output_dtype = "float64";
    
    if (!inputs.empty()) {
        auto input_node = get_node(inputs[0]);
        if (input_node) {
            output_shape = input_node->shape;
            output_dtype = input_node->dtype;
        }
    }
    
    // Adjust shape based on operation type
    switch (op_type) {
        case GraphOpType::REDUCE_SUM:
        case GraphOpType::REDUCE_MIN:
        case GraphOpType::REDUCE_MAX:
        case GraphOpType::REDUCE_MEAN:
        case GraphOpType::TALLY:
            output_shape = {}; // Scalar output
            break;
        case GraphOpType::ADD:
        case GraphOpType::SUBTRACT:
        case GraphOpType::MULTIPLY:
        case GraphOpType::DIVIDE:
            // For binary ops, use broadcasting rules (simplified)
            if (inputs.size() >= 2) {
                auto input1_node = get_node(inputs[0]);
                auto input2_node = get_node(inputs[1]);
                if (input1_node && input2_node) {
                    // Use the larger shape (simplified broadcasting)
                    if (input1_node->shape.size() >= input2_node->shape.size()) {
                        output_shape = input1_node->shape;
                        output_dtype = input1_node->dtype;
                    } else {
                        output_shape = input2_node->shape;
                        output_dtype = input2_node->dtype;
                    }
                }
            }
            break;
        default:
            break;
    }
    
    auto node = std::make_unique<GraphNode>(node_id, op_type, inputs, output_shape, output_dtype);
    node->parameters = params;
    
    node_map_[node_id] = nodes_.size();
    nodes_.push_back(std::move(node));
    
    return node_id;
}

const GraphNode* TFGraph::get_node(const std::string& node_id) const {
    auto it = node_map_.find(node_id);
    if (it != node_map_.end()) {
        return nodes_[it->second].get();
    }
    return nullptr;
}

std::vector<std::string> TFGraph::get_output_nodes() const {
    std::unordered_set<std::string> input_nodes;
    
    // Collect all nodes that are inputs to other nodes
    for (const auto& node : nodes_) {
        for (const auto& input_id : node->input_ids) {
            input_nodes.insert(input_id);
        }
    }
    
    // Find nodes that are not inputs to any other node
    std::vector<std::string> output_nodes;
    for (const auto& node : nodes_) {
        if (input_nodes.find(node->id) == input_nodes.end()) {
            output_nodes.push_back(node->id);
        }
    }
    
    return output_nodes;
}

void TFGraph::topological_sort(std::vector<size_t>& sorted_indices) const {
    std::vector<int> in_degree(nodes_.size(), 0);
    std::unordered_map<std::string, size_t> id_to_index;
    
    // Build id to index mapping
    for (size_t i = 0; i < nodes_.size(); ++i) {
        id_to_index[nodes_[i]->id] = i;
    }
    
    // Calculate in-degrees
    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (const auto& input_id : nodes_[i]->input_ids) {
            auto it = id_to_index.find(input_id);
            if (it != id_to_index.end()) {
                in_degree[i]++;
            }
        }
    }
    
    // Kahn's algorithm
    std::queue<size_t> zero_in_degree;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (in_degree[i] == 0) {
            zero_in_degree.push(i);
        }
    }
    
    sorted_indices.clear();
    while (!zero_in_degree.empty()) {
        size_t current = zero_in_degree.front();
        zero_in_degree.pop();
        sorted_indices.push_back(current);
        
        // Update in-degrees of dependent nodes
        for (size_t i = 0; i < nodes_.size(); ++i) {
            for (const auto& input_id : nodes_[i]->input_ids) {
                if (input_id == nodes_[current]->id) {
                    in_degree[i]--;
                    if (in_degree[i] == 0) {
                        zero_in_degree.push(i);
                    }
                }
            }
        }
    }
}

void TFGraph::execute_node(const GraphNode* node, 
                          const std::unordered_map<std::string, std::shared_ptr<JTensor>>& node_results,
                          std::shared_ptr<TFSession> tf_session,
                          std::unordered_map<std::string, std::shared_ptr<JTensor>>& results) {
    
    switch (node->op_type) {
        case GraphOpType::INPUT:
            // Input nodes should already be in the results from the input map
            break;
            
        case GraphOpType::CONSTANT: {
            auto it = node->parameters.find("tensor_data");
            if (it != node->parameters.end()) {
                results[node->id] = std::any_cast<std::shared_ptr<JTensor>>(it->second);
            }
            break;
        }
        
        case GraphOpType::ADD: {
            if (node->input_ids.size() >= 2) {
                auto left = node_results.at(node->input_ids[0]);
                auto right = node_results.at(node->input_ids[1]);
                results[node->id] = tf_session->add(left, right);
            }
            break;
        }
        
        case GraphOpType::SUBTRACT: {
            if (node->input_ids.size() >= 2) {
                auto left = node_results.at(node->input_ids[0]);
                auto right = node_results.at(node->input_ids[1]);
                results[node->id] = tf_session->subtract(left, right);
            }
            break;
        }
        
        case GraphOpType::MULTIPLY: {
            if (node->input_ids.size() >= 2) {
                auto left = node_results.at(node->input_ids[0]);
                auto right = node_results.at(node->input_ids[1]);
                results[node->id] = tf_session->multiply(left, right);
            }
            break;
        }
        
        case GraphOpType::DIVIDE: {
            if (node->input_ids.size() >= 2) {
                auto left = node_results.at(node->input_ids[0]);
                auto right = node_results.at(node->input_ids[1]);
                results[node->id] = tf_session->divide(left, right);
            }
            break;
        }
        
        case GraphOpType::REDUCE_SUM: {
            if (!node->input_ids.empty()) {
                auto input = node_results.at(node->input_ids[0]);
                results[node->id] = tf_session->reduce_sum(input);
            }
            break;
        }
        
        case GraphOpType::REDUCE_MIN: {
            if (!node->input_ids.empty()) {
                auto input = node_results.at(node->input_ids[0]);
                results[node->id] = tf_session->reduce_min(input);
            }
            break;
        }
        
        case GraphOpType::REDUCE_MAX: {
            if (!node->input_ids.empty()) {
                auto input = node_results.at(node->input_ids[0]);
                results[node->id] = tf_session->reduce_max(input);
            }
            break;
        }
        
        case GraphOpType::REDUCE_MEAN: {
            if (!node->input_ids.empty()) {
                auto input = node_results.at(node->input_ids[0]);
                results[node->id] = tf_session->reduce_mean(input);
            }
            break;
        }
        
        case GraphOpType::TALLY: {
            if (!node->input_ids.empty()) {
                auto input = node_results.at(node->input_ids[0]);
                long long count;
                if (input->rank() == 0) {
                    count = 1;
                } else {
                    count = input->shape()[0];
                }
                results[node->id] = JTensor::scalar(count);
            }
            break;
        }
        
        default:
            std::cerr << "Execution not implemented for graph operation type: " 
                     << static_cast<int>(node->op_type) << std::endl;
            break;
    }
}

std::unordered_map<std::string, std::shared_ptr<JTensor>> TFGraph::execute(
    std::shared_ptr<TFSession> tf_session,
    const std::unordered_map<std::string, std::shared_ptr<JTensor>>& inputs) {
    
#if HAS_TF_CC_API
    // Use true TensorFlow graph execution when available
    return execute_with_graphdef(tf_session, inputs);
#else
    // Fallback to eager execution for compatibility
    std::unordered_map<std::string, std::shared_ptr<JTensor>> results = inputs;
    
    // Topologically sort the nodes
    std::vector<size_t> sorted_indices;
    topological_sort(sorted_indices);
    
    // Execute nodes in topological order
    for (size_t index : sorted_indices) {
        const GraphNode* node = nodes_[index].get();
        execute_node(node, results, tf_session, results);
    }
    
    return results;
#endif
}

std::unordered_map<std::string, std::shared_ptr<JTensor>> TFGraph::execute_with_graphdef(
    std::shared_ptr<TFSession> tf_session,
    const std::unordered_map<std::string, std::shared_ptr<JTensor>>& inputs) {
    
    std::unordered_map<std::string, std::shared_ptr<JTensor>> results;
    
#if HAS_TF_CC_API
    if (!tf_session->is_initialized()) {
        std::cerr << "TensorFlow session not initialized" << std::endl;
        return results;
    }
    
    // Build the TensorFlow graph
    build_tensorflow_graph();
    
    // Convert graph to GraphDef
    tensorflow::GraphDef graph_def;
    tf_graph_->ToGraphDef(&graph_def);
    
    // Create a new session and run the graph
    tensorflow::SessionOptions session_options;
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
    
    tensorflow::Status status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << "Failed to create TensorFlow session from GraphDef: " << status.ToString() << std::endl;
        return results;
    }
    
    // Prepare input tensors
    std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
    for (const auto& input_pair : inputs) {
        const std::string& input_name = input_pair.first;
        const auto& input_tensor = input_pair.second;
        
        // Convert JTensor to TensorFlow tensor
        tensorflow::Tensor tf_tensor;
        if (input_tensor->dtype() == JTensor::DataType::FLOAT64) {
            auto shape = get_tf_tensor_shape(input_tensor->shape());
            tf_tensor = tensorflow::Tensor(tensorflow::DT_DOUBLE, shape);
            auto flat = tf_tensor.flat<double>();
            auto data = input_tensor->get_flat<double>();
            for (size_t i = 0; i < data.size(); ++i) {
                flat(i) = data[i];
            }
        } else if (input_tensor->dtype() == JTensor::DataType::INT64) {
            auto shape = get_tf_tensor_shape(input_tensor->shape());
            tf_tensor = tensorflow::Tensor(tensorflow::DT_INT64, shape);
            auto flat = tf_tensor.flat<long long>();
            auto data = input_tensor->get_flat<long long>();
            for (size_t i = 0; i < data.size(); ++i) {
                flat(i) = data[i];
            }
        }
        
        feed_dict.emplace_back(input_name, tf_tensor);
    }
    
    // Determine output nodes (nodes with no outgoing edges)
    std::vector<std::string> output_names = get_output_nodes();
    
    // Run the session
    std::vector<tensorflow::Tensor> output_tensors;
    status = session->Run(feed_dict, output_names, {}, &output_tensors);
    
    if (!status.ok()) {
        std::cerr << "Failed to run TensorFlow session: " << status.ToString() << std::endl;
        session->Close();
        return results;
    }
    
    // Convert output tensors back to JTensor
    for (size_t i = 0; i < output_names.size() && i < output_tensors.size(); ++i) {
        const std::string& output_name = output_names[i];
        const tensorflow::Tensor& tf_tensor = output_tensors[i];
        
        // Convert TensorFlow tensor to JTensor
        std::shared_ptr<JTensor> result_tensor;
        if (tf_tensor.dtype() == tensorflow::DT_DOUBLE) {
            auto flat = tf_tensor.flat<double>();
            std::vector<double> data;
            data.reserve(flat.size());
            for (int64_t j = 0; j < flat.size(); ++j) {
                data.push_back(flat(j));
            }
            
            std::vector<long long> shape;
            for (int dim = 0; dim < tf_tensor.dims(); ++dim) {
                shape.push_back(tf_tensor.dim_size(dim));
            }
            
            result_tensor = JTensor::from_data(data, shape);
        } else if (tf_tensor.dtype() == tensorflow::DT_INT64) {
            auto flat = tf_tensor.flat<long long>();
            std::vector<long long> data;
            data.reserve(flat.size());
            for (int64_t j = 0; j < flat.size(); ++j) {
                data.push_back(flat(j));
            }
            
            std::vector<long long> shape;
            for (int dim = 0; dim < tf_tensor.dims(); ++dim) {
                shape.push_back(tf_tensor.dim_size(dim));
            }
            
            result_tensor = JTensor::from_data(data, shape);
        }
        
        if (result_tensor) {
            results[output_name] = result_tensor;
        }
    }
    
    session->Close();
    
#else
    // Fallback to eager execution if TensorFlow C++ API is not available
    std::cerr << "TensorFlow C++ API not available, falling back to eager execution" << std::endl;
    return execute(tf_session, inputs);
#endif
    
    return results;
}

void TFGraph::print_graph() const {
    std::cout << "TensorFlow Graph:" << std::endl;
    for (const auto& node : nodes_) {
        std::cout << "  " << node->id << " [" << static_cast<int>(node->op_type) << "]";
        if (!node->input_ids.empty()) {
            std::cout << " <- {";
            for (size_t i = 0; i < node->input_ids.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << node->input_ids[i];
            }
            std::cout << "}";
        }
        std::cout << " shape=[";
        for (size_t i = 0; i < node->shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << node->shape[i];
        }
        std::cout << "]" << std::endl;
    }
}

void TFGraph::optimize() {
    // Basic optimization: remove redundant operations
    // This is a placeholder for more sophisticated optimization
    std::cout << "Graph optimization not yet implemented." << std::endl;
}

// DeferredTensor implementation

std::shared_ptr<DeferredTensor> DeferredTensor::from_tensor(std::shared_ptr<TFGraph> graph, 
                                                           std::shared_ptr<JTensor> tensor) {
    std::string node_id = graph->add_constant(tensor);
    return std::make_shared<DeferredTensor>(graph, node_id, tensor->shape(), JTensor::dtype_to_string(tensor->dtype()));
}

std::shared_ptr<DeferredTensor> DeferredTensor::input(std::shared_ptr<TFGraph> graph,
                                                      const std::vector<long long>& shape,
                                                      const std::string& dtype) {
    std::string node_id = graph->add_input(shape, dtype);
    return std::make_shared<DeferredTensor>(graph, node_id, shape, dtype);
}

std::shared_ptr<JTensor> DeferredTensor::materialize(std::shared_ptr<TFSession> tf_session,
                                                    const std::unordered_map<std::string, std::shared_ptr<JTensor>>& inputs) {
    if (is_materialized_) {
        return materialized_value_;
    }
    
    auto results = graph_->execute(tf_session, inputs);
    auto it = results.find(node_id_);
    if (it != results.end()) {
        materialized_value_ = it->second;
        is_materialized_ = true;
        return materialized_value_;
    }
    
    return nullptr;
}

std::shared_ptr<DeferredTensor> DeferredTensor::add(std::shared_ptr<DeferredTensor> other) {
    std::string result_id = graph_->add_operation(GraphOpType::ADD, {node_id_, other->node_id_});
    
    // Determine result shape (simplified broadcasting)
    std::vector<long long> result_shape = shape_;
    if (other->shape_.size() > result_shape.size()) {
        result_shape = other->shape_;
    }
    
    return std::make_shared<DeferredTensor>(graph_, result_id, result_shape, dtype_);
}

std::shared_ptr<DeferredTensor> DeferredTensor::subtract(std::shared_ptr<DeferredTensor> other) {
    std::string result_id = graph_->add_operation(GraphOpType::SUBTRACT, {node_id_, other->node_id_});
    std::vector<long long> result_shape = shape_;
    if (other->shape_.size() > result_shape.size()) {
        result_shape = other->shape_;
    }
    return std::make_shared<DeferredTensor>(graph_, result_id, result_shape, dtype_);
}

std::shared_ptr<DeferredTensor> DeferredTensor::multiply(std::shared_ptr<DeferredTensor> other) {
    std::string result_id = graph_->add_operation(GraphOpType::MULTIPLY, {node_id_, other->node_id_});
    std::vector<long long> result_shape = shape_;
    if (other->shape_.size() > result_shape.size()) {
        result_shape = other->shape_;
    }
    return std::make_shared<DeferredTensor>(graph_, result_id, result_shape, dtype_);
}

std::shared_ptr<DeferredTensor> DeferredTensor::divide(std::shared_ptr<DeferredTensor> other) {
    std::string result_id = graph_->add_operation(GraphOpType::DIVIDE, {node_id_, other->node_id_});
    std::vector<long long> result_shape = shape_;
    if (other->shape_.size() > result_shape.size()) {
        result_shape = other->shape_;
    }
    return std::make_shared<DeferredTensor>(graph_, result_id, result_shape, dtype_);
}

std::shared_ptr<DeferredTensor> DeferredTensor::reduce_sum() {
    std::string result_id = graph_->add_operation(GraphOpType::REDUCE_SUM, {node_id_});
    return std::make_shared<DeferredTensor>(graph_, result_id, std::vector<long long>{}, dtype_); // Scalar result
}

std::shared_ptr<DeferredTensor> DeferredTensor::reduce_min() {
    std::string result_id = graph_->add_operation(GraphOpType::REDUCE_MIN, {node_id_});
    return std::make_shared<DeferredTensor>(graph_, result_id, std::vector<long long>{}, dtype_);
}

std::shared_ptr<DeferredTensor> DeferredTensor::reduce_max() {
    std::string result_id = graph_->add_operation(GraphOpType::REDUCE_MAX, {node_id_});
    return std::make_shared<DeferredTensor>(graph_, result_id, std::vector<long long>{}, dtype_);
}

std::shared_ptr<DeferredTensor> DeferredTensor::tally() {
    std::string result_id = graph_->add_operation(GraphOpType::TALLY, {node_id_});
    return std::make_shared<DeferredTensor>(graph_, result_id, std::vector<long long>{}, "int64");
}

// JGraphBuilder implementation

std::shared_ptr<DeferredTensor> JGraphBuilder::from_jvalue(const JValue& value) {
    if (std::holds_alternative<std::shared_ptr<JTensor>>(value)) {
        auto tensor = std::get<std::shared_ptr<JTensor>>(value);
        return DeferredTensor::from_tensor(graph_, tensor);
    } else if (std::holds_alternative<long long>(value)) {
        auto tensor = JTensor::scalar(std::get<long long>(value));
        return DeferredTensor::from_tensor(graph_, tensor);
    } else if (std::holds_alternative<double>(value)) {
        auto tensor = JTensor::scalar(std::get<double>(value));
        return DeferredTensor::from_tensor(graph_, tensor);
    }
    
    return nullptr;
}

std::shared_ptr<DeferredTensor> JGraphBuilder::build_fork(
    std::shared_ptr<DeferredTensor> arg,
    const std::string& left_verb,
    const std::string& middle_verb, 
    const std::string& right_verb) {
    
    // Apply left and right verbs to argument
    auto left_result = apply_monadic_verb(left_verb, arg);
    auto right_result = apply_monadic_verb(right_verb, arg);
    
    if (!left_result || !right_result) {
        return nullptr;
    }
    
    // Apply middle verb dyadically
    return apply_dyadic_verb(middle_verb, left_result, right_result);
}

std::shared_ptr<DeferredTensor> JGraphBuilder::apply_monadic_verb(const std::string& verb, 
                                                                 std::shared_ptr<DeferredTensor> operand) {
    if (verb == "+/") {
        return operand->reduce_sum();
    } else if (verb == "#") {
        return operand->tally();
    } else if (verb == "</") {
        return operand->reduce_min();
    } else if (verb == ">/") {
        return operand->reduce_max();
    } else if (verb == "$") {
        // Get the shape of the operand tensor
        const std::vector<long long>& shape = operand->shape();
        
        // Create a JTensor from the shape vector
        auto shape_tensor = JTensor::from_data(shape);
        
        // Create and return a DeferredTensor from the shape tensor
        return DeferredTensor::from_tensor(graph_, shape_tensor);
    }
    
    std::cerr << "Unknown monadic verb in graph builder: " << verb << std::endl;
    return nullptr;
}

std::shared_ptr<DeferredTensor> JGraphBuilder::apply_dyadic_verb(const std::string& verb,
                                                                std::shared_ptr<DeferredTensor> left,
                                                                std::shared_ptr<DeferredTensor> right) {
    if (verb == "+") {
        return left->add(right);
    } else if (verb == "-") {
        return left->subtract(right);
    } else if (verb == "*") {
        return left->multiply(right);
    } else if (verb == "%") {
        return left->divide(right);
    }
    
    std::cerr << "Unknown dyadic verb in graph builder: " << verb << std::endl;
    return nullptr;
}

#if HAS_TF_CC_API

tensorflow::DataType TFGraph::get_tf_data_type(const std::string& dtype) {
    if (dtype == "float64" || dtype == "double") {
        return tensorflow::DT_DOUBLE;
    } else if (dtype == "float32" || dtype == "float") {
        return tensorflow::DT_FLOAT;
    } else if (dtype == "int64" || dtype == "long long") {
        return tensorflow::DT_INT64;
    } else if (dtype == "int32" || dtype == "int") {
        return tensorflow::DT_INT32;
    }
    return tensorflow::DT_DOUBLE; // Default to double
}

tensorflow::TensorShape TFGraph::get_tf_tensor_shape(const std::vector<long long>& shape) {
    tensorflow::TensorShape tf_shape;
    for (long long dim : shape) {
        tf_shape.AppendDim(dim);
    }
    return tf_shape;
}

tensorflow::Node* TFGraph::create_tf_node(const GraphNode* node) {
    tensorflow::Node* tf_node = nullptr;
    tensorflow::Status status;
    
    switch (node->op_type) {
        case GraphOpType::INPUT: {
            // Create a placeholder node for inputs
            auto shape = get_tf_tensor_shape(node->shape);
            auto dtype = get_tf_data_type(node->dtype);
            
            status = tensorflow::NodeBuilder(node->id, "Placeholder")
                .Attr("dtype", dtype)
                .Attr("shape", shape)
                .Finalize(tf_graph_.get(), &tf_node);
            break;
        }
        
        case GraphOpType::CONSTANT: {
            // Create a constant node
            auto it = node->parameters.find("tensor_data");
            if (it != node->parameters.end()) {
                auto tensor_data = std::any_cast<std::shared_ptr<JTensor>>(it->second);
                
                // Create TensorFlow tensor from JTensor
                tensorflow::Tensor tf_tensor;
                if (tensor_data->dtype() == JTensor::DataType::FLOAT64) {
                    auto shape = get_tf_tensor_shape(tensor_data->shape());
                    tf_tensor = tensorflow::Tensor(tensorflow::DT_DOUBLE, shape);
                    auto flat = tf_tensor.flat<double>();
                    auto data = tensor_data->get_flat<double>();
                    for (size_t i = 0; i < data.size(); ++i) {
                        flat(i) = data[i];
                    }
                } else if (tensor_data->dtype() == JTensor::DataType::INT64) {
                    auto shape = get_tf_tensor_shape(tensor_data->shape());
                    tf_tensor = tensorflow::Tensor(tensorflow::DT_INT64, shape);
                    auto flat = tf_tensor.flat<long long>();
                    auto data = tensor_data->get_flat<long long>();
                    for (size_t i = 0; i < data.size(); ++i) {
                        flat(i) = data[i];
                    }
                }
                
                status = tensorflow::NodeBuilder(node->id, "Const")
                    .Attr("dtype", tf_tensor.dtype())
                    .Attr("value", tf_tensor)
                    .Finalize(tf_graph_.get(), &tf_node);
            }
            break;
        }
        
        case GraphOpType::ADD: {
            if (node->input_ids.size() >= 2) {
                auto input1 = tf_nodes_[node->input_ids[0]];
                auto input2 = tf_nodes_[node->input_ids[1]];
                
                status = tensorflow::NodeBuilder(node->id, "Add")
                    .Input(input1)
                    .Input(input2)
                    .Finalize(tf_graph_.get(), &tf_node);
            }
            break;
        }
        
        case GraphOpType::SUBTRACT: {
            if (node->input_ids.size() >= 2) {
                auto input1 = tf_nodes_[node->input_ids[0]];
                auto input2 = tf_nodes_[node->input_ids[1]];
                
                status = tensorflow::NodeBuilder(node->id, "Sub")
                    .Input(input1)
                    .Input(input2)
                    .Finalize(tf_graph_.get(), &tf_node);
            }
            break;
        }
        
        case GraphOpType::MULTIPLY: {
            if (node->input_ids.size() >= 2) {
                auto input1 = tf_nodes_[node->input_ids[0]];
                auto input2 = tf_nodes_[node->input_ids[1]];
                
                status = tensorflow::NodeBuilder(node->id, "Mul")
                    .Input(input1)
                    .Input(input2)
                    .Finalize(tf_graph_.get(), &tf_node);
            }
            break;
        }
        
        case GraphOpType::DIVIDE: {
            if (node->input_ids.size() >= 2) {
                auto input1 = tf_nodes_[node->input_ids[0]];
                auto input2 = tf_nodes_[node->input_ids[1]];
                
                status = tensorflow::NodeBuilder(node->id, "Div")
                    .Input(input1)
                    .Input(input2)
                    .Finalize(tf_graph_.get(), &tf_node);
            }
            break;
        }
        
        case GraphOpType::REDUCE_SUM: {
            if (!node->input_ids.empty()) {
                auto input = tf_nodes_[node->input_ids[0]];
                
                // Create reduction indices tensor (reduce all axes by default)
                tensorflow::Tensor indices_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
                indices_tensor.scalar<int>()() = -1; // Reduce all axes
                
                tensorflow::Node* indices_node = nullptr;
                tensorflow::NodeBuilder("reduction_indices_" + node->id, "Const")
                    .Attr("dtype", tensorflow::DT_INT32)
                    .Attr("value", indices_tensor)
                    .Finalize(tf_graph_.get(), &indices_node);
                
                status = tensorflow::NodeBuilder(node->id, "Sum")
                    .Input(input)
                    .Input(indices_node)
                    .Attr("keep_dims", false)
                    .Finalize(tf_graph_.get(), &tf_node);
            }
            break;
        }
        
        default:
            std::cerr << "TensorFlow node creation not implemented for operation type: " 
                     << static_cast<int>(node->op_type) << std::endl;
            break;
    }
    
    if (!status.ok()) {
        std::cerr << "Failed to create TensorFlow node " << node->id << ": " << status.ToString() << std::endl;
        return nullptr;
    }
    
    return tf_node;
}

void TFGraph::build_tensorflow_graph() {
    // Clear previous graph state
    tf_nodes_.clear();
    tf_graph_ = std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());
    
    // Create TensorFlow nodes in topological order
    std::vector<size_t> sorted_indices;
    topological_sort(sorted_indices);
    
    for (size_t index : sorted_indices) {
        const GraphNode* node = nodes_[index].get();
        tensorflow::Node* tf_node = create_tf_node(node);
        if (tf_node) {
            tf_nodes_[node->id] = tf_node;
        }
    }
}

#endif // HAS_TF_CC_API

} // namespace JInterpreter
