#include "src/interpreter/tf_graph.hpp"
#include "src/interpreter/tf_operations.hpp"
#include <iostream>

using namespace JInterpreter;

int main() {
    std::cout << "Testing TensorFlow GraphDef execution..." << std::endl;
    
    // Create a TensorFlow session
    auto tf_session = std::make_shared<TFSession>();
    
    // Create a graph
    auto graph = std::make_shared<TFGraph>();
    
    // Create input tensors
    auto a = JTensor::scalar(3.0);
    auto b = JTensor::scalar(4.0);
    
    // Add nodes to the graph
    std::string input_a_id = graph->add_constant(a);
    std::string input_b_id = graph->add_constant(b);
    std::string add_id = graph->add_operation(GraphOpType::ADD, {input_a_id, input_b_id});
    
    std::cout << "Graph structure:" << std::endl;
    graph->print_graph();
    
    // Execute the graph using both methods
    std::unordered_map<std::string, std::shared_ptr<JTensor>> inputs;
    
    std::cout << "\n=== Testing Eager Execution (fallback) ===" << std::endl;
    auto eager_results = graph->execute(tf_session, inputs);
    
    if (eager_results.find(add_id) != eager_results.end()) {
        auto result = eager_results[add_id];
        std::cout << "Eager result: " << result->get_scalar<double>() << std::endl;
    }
    
    std::cout << "\n=== Testing GraphDef Execution ===" << std::endl;
    auto graphdef_results = graph->execute_with_graphdef(tf_session, inputs);
    
    if (graphdef_results.find(add_id) != graphdef_results.end()) {
        auto result = graphdef_results[add_id];
        std::cout << "GraphDef result: " << result->get_scalar<double>() << std::endl;
    } else {
        std::cout << "GraphDef execution failed or no result found" << std::endl;
    }
    
    // Test more complex operations
    std::cout << "\n=== Testing Complex Graph ===" << std::endl;
    auto graph2 = std::make_shared<TFGraph>();
    
    auto x = JTensor::from_data({1.0, 2.0, 3.0}, {3});
    auto y = JTensor::from_data({4.0, 5.0, 6.0}, {3});
    
    std::string x_id = graph2->add_constant(x);
    std::string y_id = graph2->add_constant(y);
    std::string mul_id = graph2->add_operation(GraphOpType::MULTIPLY, {x_id, y_id});
    std::string sum_id = graph2->add_operation(GraphOpType::REDUCE_SUM, {mul_id});
    
    graph2->print_graph();
    
    auto complex_results = graph2->execute(tf_session, inputs);
    
    if (complex_results.find(sum_id) != complex_results.end()) {
        auto result = complex_results[sum_id];
        std::cout << "Complex graph result: " << result->get_scalar<double>() << std::endl;
        std::cout << "Expected: " << (1*4 + 2*5 + 3*6) << std::endl; // Should be 32
    }
    
    return 0;
}
