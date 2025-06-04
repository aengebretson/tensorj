#include <gtest/gtest.h>
#include "interpreter/tf_graph.hpp"
#include "interpreter/tf_operations.hpp"
#include <memory>

namespace JInterpreter {

class TFGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        tf_session = std::make_shared<TFSession>();
        graph = std::make_shared<TFGraph>();
    }

    std::shared_ptr<TFSession> tf_session;
    std::shared_ptr<TFGraph> graph;
};

// Test basic graph construction
TEST_F(TFGraphTest, EmptyGraph) {
    EXPECT_EQ(graph->node_count(), 0);
    auto outputs = graph->get_output_nodes();
    EXPECT_TRUE(outputs.empty());
}

TEST_F(TFGraphTest, AddConstant) {
    // Create a simple scalar tensor
    auto scalar_tensor = JTensor::scalar(42.0);
    ASSERT_NE(scalar_tensor, nullptr);
    
    std::string const_id = graph->add_constant(scalar_tensor);
    EXPECT_FALSE(const_id.empty());
    EXPECT_EQ(graph->node_count(), 1);
    
    const GraphNode* node = graph->get_node(const_id);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type, GraphOpType::CONSTANT);
    EXPECT_EQ(node->input_ids.size(), 0);
}

TEST_F(TFGraphTest, AddInput) {
    std::vector<long long> shape = {3, 4};
    std::string input_id = graph->add_input(shape, "float64");
    
    EXPECT_FALSE(input_id.empty());
    EXPECT_EQ(graph->node_count(), 1);
    
    const GraphNode* node = graph->get_node(input_id);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type, GraphOpType::INPUT);
    EXPECT_EQ(node->shape, shape);
    EXPECT_EQ(node->dtype, "float64");
}

TEST_F(TFGraphTest, AddOperation) {
    // Add two inputs
    std::string input1 = graph->add_input({2, 3}, "float64");
    std::string input2 = graph->add_input({2, 3}, "float64");
    
    // Add an addition operation
    std::string add_id = graph->add_operation(GraphOpType::ADD, {input1, input2});
    
    EXPECT_FALSE(add_id.empty());
    EXPECT_EQ(graph->node_count(), 3);
    
    const GraphNode* add_node = graph->get_node(add_id);
    ASSERT_NE(add_node, nullptr);
    EXPECT_EQ(add_node->op_type, GraphOpType::ADD);
    EXPECT_EQ(add_node->input_ids.size(), 2);
    EXPECT_EQ(add_node->input_ids[0], input1);
    EXPECT_EQ(add_node->input_ids[1], input2);
}

TEST_F(TFGraphTest, ComplexGraph) {
    // Build a more complex graph: (a + b) * c
    std::string a = graph->add_input({2, 2}, "float64");
    std::string b = graph->add_input({2, 2}, "float64");
    std::string c = graph->add_input({2, 2}, "float64");
    
    std::string add_result = graph->add_operation(GraphOpType::ADD, {a, b});
    std::string mul_result = graph->add_operation(GraphOpType::MULTIPLY, {add_result, c});
    
    EXPECT_EQ(graph->node_count(), 5);
    
    // Check output nodes
    auto outputs = graph->get_output_nodes();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0], mul_result);
}

TEST_F(TFGraphTest, ReductionOperations) {
    std::string input = graph->add_input({3, 4}, "float64");
    
    std::string sum_result = graph->add_operation(GraphOpType::REDUCE_SUM, {input});
    std::string min_result = graph->add_operation(GraphOpType::REDUCE_MIN, {input});
    std::string max_result = graph->add_operation(GraphOpType::REDUCE_MAX, {input});
    
    EXPECT_EQ(graph->node_count(), 4);
    
    const GraphNode* sum_node = graph->get_node(sum_result);
    ASSERT_NE(sum_node, nullptr);
    EXPECT_EQ(sum_node->op_type, GraphOpType::REDUCE_SUM);
    
    const GraphNode* min_node = graph->get_node(min_result);
    ASSERT_NE(min_node, nullptr);
    EXPECT_EQ(min_node->op_type, GraphOpType::REDUCE_MIN);
    
    const GraphNode* max_node = graph->get_node(max_result);
    ASSERT_NE(max_node, nullptr);
    EXPECT_EQ(max_node->op_type, GraphOpType::REDUCE_MAX);
}

// Test DeferredTensor operations
class DeferredTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        tf_session = std::make_shared<TFSession>();
        graph = std::make_shared<TFGraph>();
    }

    std::shared_ptr<TFSession> tf_session;
    std::shared_ptr<TFGraph> graph;
};

TEST_F(DeferredTensorTest, CreateFromTensor) {
    auto scalar_tensor = JTensor::scalar(5.0);
    ASSERT_NE(scalar_tensor, nullptr);
    
    auto deferred = DeferredTensor::from_tensor(graph, scalar_tensor);
    ASSERT_NE(deferred, nullptr);
    
    EXPECT_EQ(deferred->shape(), scalar_tensor->shape());
    EXPECT_EQ(deferred->dtype(), "FLOAT64");
    EXPECT_EQ(graph->node_count(), 1);
}

TEST_F(DeferredTensorTest, CreateInput) {
    std::vector<long long> shape = {2, 3};
    auto deferred = DeferredTensor::input(graph, shape, "float64");
    
    ASSERT_NE(deferred, nullptr);
    EXPECT_EQ(deferred->shape(), shape);
    EXPECT_EQ(deferred->dtype(), "float64");
    EXPECT_EQ(graph->node_count(), 1);
    
    const GraphNode* node = graph->get_node(deferred->node_id());
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type, GraphOpType::INPUT);
}

TEST_F(DeferredTensorTest, ArithmeticOperations) {
    // Create two input tensors
    auto a = DeferredTensor::input(graph, {2, 2}, "float64");
    auto b = DeferredTensor::input(graph, {2, 2}, "float64");
    
    // Test all arithmetic operations
    auto sum = a->add(b);
    auto diff = a->subtract(b);
    auto product = a->multiply(b);
    auto quotient = a->divide(b);
    
    EXPECT_EQ(graph->node_count(), 6); // 2 inputs + 4 operations
    
    EXPECT_EQ(sum->shape(), a->shape());
    EXPECT_EQ(diff->shape(), a->shape());
    EXPECT_EQ(product->shape(), a->shape());
    EXPECT_EQ(quotient->shape(), a->shape());
}

TEST_F(DeferredTensorTest, ReductionOperations) {
    auto input = DeferredTensor::input(graph, {3, 4}, "float64");
    
    auto sum = input->reduce_sum();
    auto min_val = input->reduce_min();
    auto max_val = input->reduce_max();
    auto count = input->tally();
    
    EXPECT_EQ(graph->node_count(), 5); // 1 input + 4 reductions
    
    // Reductions should reduce to scalars
    EXPECT_TRUE(sum->shape().empty() || (sum->shape().size() == 1 && sum->shape()[0] == 1));
    EXPECT_TRUE(min_val->shape().empty() || (min_val->shape().size() == 1 && min_val->shape()[0] == 1));
    EXPECT_TRUE(max_val->shape().empty() || (max_val->shape().size() == 1 && max_val->shape()[0] == 1));
}

TEST_F(DeferredTensorTest, ChainedOperations) {
    // Test operation chaining: ((a + b) * c) / d
    auto a = DeferredTensor::input(graph, {2, 2}, "float64");
    auto b = DeferredTensor::input(graph, {2, 2}, "float64");
    auto c = DeferredTensor::input(graph, {2, 2}, "float64");
    auto d = DeferredTensor::input(graph, {2, 2}, "float64");
    
    auto result = a->add(b)->multiply(c)->divide(d);
    
    EXPECT_EQ(graph->node_count(), 7); // 4 inputs + 3 operations
    EXPECT_EQ(result->shape(), a->shape());
    
    // Check that the graph has the correct structure
    auto outputs = graph->get_output_nodes();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0], result->node_id());
}

// Test JGraphBuilder
class JGraphBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        tf_session = std::make_shared<TFSession>();
        builder = std::make_unique<JGraphBuilder>();
    }

    std::shared_ptr<TFSession> tf_session;
    std::unique_ptr<JGraphBuilder> builder;
};

TEST_F(JGraphBuilderTest, FromJValueScalar) {
    JValue scalar_value = 42.5;
    auto deferred = builder->from_jvalue(scalar_value);
    
    ASSERT_NE(deferred, nullptr);
    EXPECT_EQ(builder->get_graph()->node_count(), 1);
    
    const GraphNode* node = builder->get_graph()->get_node(deferred->node_id());
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type, GraphOpType::CONSTANT);
}

TEST_F(JGraphBuilderTest, FromJValueTensor) {
    // Create a tensor value
    auto tensor = JTensor::from_data(std::vector<double>{1.0, 2.0, 3.0, 4.0}, {2, 2});
    ASSERT_NE(tensor, nullptr);
    
    JValue tensor_value = tensor;
    auto deferred = builder->from_jvalue(tensor_value);
    
    ASSERT_NE(deferred, nullptr);
    EXPECT_EQ(builder->get_graph()->node_count(), 1);
    EXPECT_EQ(deferred->shape(), tensor->shape());
}

TEST_F(JGraphBuilderTest, MonadicVerbApplication) {
    // Create input tensor
    auto tensor = JTensor::from_data(std::vector<double>{1.0, 2.0, 3.0, 4.0}, {2, 2});
    auto deferred = builder->from_jvalue(JValue(tensor));
    
    // Test monadic verbs
    auto sum_result = builder->apply_monadic_verb("+/", deferred);
    auto shape_result = builder->apply_monadic_verb("$", deferred);
    auto tally_result = builder->apply_monadic_verb("#", deferred);
    
    EXPECT_GT(builder->get_graph()->node_count(), 1);
    ASSERT_NE(sum_result, nullptr);
    ASSERT_NE(shape_result, nullptr);
    ASSERT_NE(tally_result, nullptr);
}

TEST_F(JGraphBuilderTest, DyadicVerbApplication) {
    // Create two input tensors
    auto tensor1 = JTensor::from_data(std::vector<double>{1.0, 2.0}, {2});
    auto tensor2 = JTensor::from_data(std::vector<double>{3.0, 4.0}, {2});
    
    auto deferred1 = builder->from_jvalue(JValue(tensor1));
    auto deferred2 = builder->from_jvalue(JValue(tensor2));
    
    // Test dyadic verbs
    auto add_result = builder->apply_dyadic_verb("+", deferred1, deferred2);
    auto mul_result = builder->apply_dyadic_verb("*", deferred1, deferred2);
    auto div_result = builder->apply_dyadic_verb("%", deferred1, deferred2);
    
    EXPECT_GT(builder->get_graph()->node_count(), 2);
    ASSERT_NE(add_result, nullptr);
    ASSERT_NE(mul_result, nullptr);
    ASSERT_NE(div_result, nullptr);
    
    EXPECT_EQ(add_result->shape(), deferred1->shape());
    EXPECT_EQ(mul_result->shape(), deferred1->shape());
    EXPECT_EQ(div_result->shape(), deferred1->shape());
}

TEST_F(JGraphBuilderTest, ForkExpression) {
    // Test fork: (+/ % #) which computes average
    auto tensor = JTensor::from_data(std::vector<double>{1.0, 2.0, 3.0, 4.0}, {4});
    auto deferred = builder->from_jvalue(JValue(tensor));
    
    // Build fork: (+/ % #) y = (+/ y) % (# y)
    auto fork_result = builder->build_fork(deferred, "+/", "%", "#");
    
    ASSERT_NE(fork_result, nullptr);
    EXPECT_GT(builder->get_graph()->node_count(), 1);
    
    // The result should be a scalar (average)
    auto result_shape = fork_result->shape();
    EXPECT_TRUE(result_shape.empty() || (result_shape.size() == 1 && result_shape[0] == 1));
}

// Test graph execution
class GraphExecutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        tf_session = std::make_shared<TFSession>();
        graph = std::make_shared<TFGraph>();
    }

    std::shared_ptr<TFSession> tf_session;
    std::shared_ptr<TFGraph> graph;
};

TEST_F(GraphExecutionTest, ExecuteSimpleAddition) {
    // Create constants
    auto tensor1 = JTensor::scalar(3.0);
    auto tensor2 = JTensor::scalar(4.0);
    
    std::string const1 = graph->add_constant(tensor1);
    std::string const2 = graph->add_constant(tensor2);
    std::string add_result = graph->add_operation(GraphOpType::ADD, {const1, const2});
    
    // Execute graph
    auto results = graph->execute(tf_session, {});
    
    ASSERT_EQ(results.size(), 3);
    ASSERT_NE(results.find(add_result), results.end());
    
    auto result_tensor = results[add_result];
    ASSERT_NE(result_tensor, nullptr);
    
    // Check result value (should be 7.0)
    auto scalar_val = result_tensor->get_scalar<double>();
    EXPECT_NEAR(scalar_val, 7.0, 1e-6);
}

TEST_F(GraphExecutionTest, ExecuteWithInputs) {
    // Create input and constant
    std::string input_id = graph->add_input({2}, "float64");
    auto const_tensor = JTensor::scalar(10.0);
    std::string const_id = graph->add_constant(const_tensor);
    
    std::string mul_result = graph->add_operation(GraphOpType::MULTIPLY, {input_id, const_id});
    
    // Prepare input data
    auto input_tensor = JTensor::from_data(std::vector<double>{2.0, 3.0}, {2});
    std::unordered_map<std::string, std::shared_ptr<JTensor>> inputs = {{input_id, input_tensor}};
    
    // Execute graph
    auto results = graph->execute(tf_session, inputs);
    
    ASSERT_NE(results.find(mul_result), results.end());
    auto result_tensor = results[mul_result];
    ASSERT_NE(result_tensor, nullptr);
    
    // Check result values (should be [20.0, 30.0])
    auto result_vec = result_tensor->get_flat<double>();
    ASSERT_EQ(result_vec.size(), 2);
    EXPECT_NEAR(result_vec[0], 20.0, 1e-6);
    EXPECT_NEAR(result_vec[1], 30.0, 1e-6);
}

TEST_F(GraphExecutionTest, ExecuteReductionOperation) {
    // Create input tensor and add reduction
    std::string input_id = graph->add_input({4}, "float64");
    std::string sum_result = graph->add_operation(GraphOpType::REDUCE_SUM, {input_id});
    
    // Prepare input data: [1, 2, 3, 4] -> sum should be 10
    auto input_tensor = JTensor::from_data(std::vector<double>{1.0, 2.0, 3.0, 4.0}, {4});
    std::unordered_map<std::string, std::shared_ptr<JTensor>> inputs = {{input_id, input_tensor}};
    
    // Execute graph
    auto results = graph->execute(tf_session, inputs);
    
    ASSERT_NE(results.find(sum_result), results.end());
    auto result_tensor = results[sum_result];
    ASSERT_NE(result_tensor, nullptr);
    
    // Check result value (should be 10.0)
    auto scalar_val = result_tensor->get_scalar<double>();
    EXPECT_NEAR(scalar_val, 10.0, 1e-6);
}

TEST_F(GraphExecutionTest, ExecuteComplexGraph) {
    // Build graph for average computation: sum / count
    std::string input_id = graph->add_input({4}, "float64");
    std::string sum_result = graph->add_operation(GraphOpType::REDUCE_SUM, {input_id});
    std::string tally_result = graph->add_operation(GraphOpType::TALLY, {input_id});
    std::string avg_result = graph->add_operation(GraphOpType::DIVIDE, {sum_result, tally_result});
    
    // Input: [2, 4, 6, 8] -> average should be 5.0
    auto input_tensor = JTensor::from_data(std::vector<double>{2.0, 4.0, 6.0, 8.0}, {4});
    std::unordered_map<std::string, std::shared_ptr<JTensor>> inputs = {{input_id, input_tensor}};
    
    // Execute graph
    auto results = graph->execute(tf_session, inputs);
    
    ASSERT_NE(results.find(avg_result), results.end());
    auto result_tensor = results[avg_result];
    ASSERT_NE(result_tensor, nullptr);
    
    // Check result value (should be 5.0)
    auto scalar_val = result_tensor->get_scalar<double>();
    EXPECT_NEAR(scalar_val, 5.0, 1e-6);
}

// Integration test for DeferredTensor materialization
TEST_F(GraphExecutionTest, DeferredTensorMaterialization) {
    // Create deferred computation: (a + b) * c
    auto a = DeferredTensor::input(graph, {2}, "float64");
    auto b = DeferredTensor::input(graph, {2}, "float64");
    auto c = DeferredTensor::input(graph, {2}, "float64");
    
    auto result = a->add(b)->multiply(c);
    
    // Prepare inputs
    auto tensor_a = JTensor::from_data(std::vector<double>{1.0, 2.0}, {2});
    auto tensor_b = JTensor::from_data(std::vector<double>{3.0, 4.0}, {2});
    auto tensor_c = JTensor::from_data(std::vector<double>{5.0, 6.0}, {2});
    
    std::unordered_map<std::string, std::shared_ptr<JTensor>> inputs = {
        {a->node_id(), tensor_a},
        {b->node_id(), tensor_b},
        {c->node_id(), tensor_c}
    };
    
    // Materialize the result
    auto materialized = result->materialize(tf_session, inputs);
    ASSERT_NE(materialized, nullptr);
    
    // Check result: (1+3)*5=20, (2+4)*6=36
    auto result_vec = materialized->get_flat<double>();
    ASSERT_EQ(result_vec.size(), 2);
    EXPECT_NEAR(result_vec[0], 20.0, 1e-6);
    EXPECT_NEAR(result_vec[1], 36.0, 1e-6);
}

TEST_F(JGraphBuilderTest, ShapeVerbSpecific) {
    // Test the "$" shape verb with different tensor shapes
    
    // Test with 2D tensor
    auto tensor_2d = JTensor::from_data(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    auto deferred_2d = builder->from_jvalue(JValue(tensor_2d));
    
    auto shape_result_2d = builder->apply_monadic_verb("$", deferred_2d);
    ASSERT_NE(shape_result_2d, nullptr);
    
    // Shape of a 2D tensor should be a 1D tensor with 2 elements: [2, 3]
    auto result_shape_2d = shape_result_2d->shape();
    EXPECT_EQ(result_shape_2d.size(), 1);
    EXPECT_EQ(result_shape_2d[0], 2); // Should contain 2 elements (dimensions)
    
    // Test with 3D tensor
    auto tensor_3d = JTensor::from_data(std::vector<double>(24, 1.0), {2, 3, 4});
    auto deferred_3d = builder->from_jvalue(JValue(tensor_3d));
    
    auto shape_result_3d = builder->apply_monadic_verb("$", deferred_3d);
    ASSERT_NE(shape_result_3d, nullptr);
    
    // Shape of a 3D tensor should be a 1D tensor with 3 elements: [2, 3, 4]
    auto result_shape_3d = shape_result_3d->shape();
    EXPECT_EQ(result_shape_3d.size(), 1);
    EXPECT_EQ(result_shape_3d[0], 3); // Should contain 3 elements (dimensions)
    
    // Test with scalar (0D tensor)
    auto scalar = JTensor::scalar(42.0);
    auto deferred_scalar = builder->from_jvalue(JValue(scalar));
    
    auto shape_result_scalar = builder->apply_monadic_verb("$", deferred_scalar);
    ASSERT_NE(shape_result_scalar, nullptr);
    
    // Shape of a scalar should be an empty tensor or tensor with 0 elements
    auto result_shape_scalar = shape_result_scalar->shape();
    EXPECT_TRUE(result_shape_scalar.empty() || 
                (result_shape_scalar.size() == 1 && result_shape_scalar[0] == 0));
    
    EXPECT_GT(builder->get_graph()->node_count(), 1);
}

// Test to verify GraphDef execution vs eager execution behavior
TEST_F(GraphExecutionTest, GraphDefExecutionBehavior) {
    // Create a simple computation graph
    auto tensor1 = JTensor::scalar(5.0);
    auto tensor2 = JTensor::scalar(3.0);
    
    std::string const1 = graph->add_constant(tensor1);
    std::string const2 = graph->add_constant(tensor2);
    std::string add_result = graph->add_operation(GraphOpType::ADD, {const1, const2});
    std::string mul_result = graph->add_operation(GraphOpType::MULTIPLY, {add_result, const2});
    
    // Execute graph
    auto results = graph->execute(tf_session, {});
    
    // Verify the computation: (5 + 3) * 3 = 24
    ASSERT_NE(results.find(mul_result), results.end());
    auto result_tensor = results[mul_result];
    ASSERT_NE(result_tensor, nullptr);
    
    auto scalar_val = result_tensor->get_scalar<double>();
    EXPECT_NEAR(scalar_val, 24.0, 1e-6);
    
    // The key difference with GraphDef execution is that operations should be
    // optimized and executed as a single graph rather than individual operations.
    // While we can't directly test the internal optimization in unit tests,
    // we can verify that the results are correct and consistent.
    
    // Verify that all intermediate results are computed correctly
    ASSERT_NE(results.find(add_result), results.end());
    auto add_tensor = results[add_result];
    ASSERT_NE(add_tensor, nullptr);
    EXPECT_NEAR(add_tensor->get_scalar<double>(), 8.0, 1e-6);
}

// Test GraphDef execution with different data types
TEST_F(GraphExecutionTest, GraphDefWithIntegerTypes) {
    // Create integer computation graph
    auto int_tensor1 = JTensor::scalar(10LL);
    auto int_tensor2 = JTensor::scalar(4LL);
    
    std::string const1 = graph->add_constant(int_tensor1);
    std::string const2 = graph->add_constant(int_tensor2);
    std::string div_result = graph->add_operation(GraphOpType::DIVIDE, {const1, const2});
    
    // Execute graph
    auto results = graph->execute(tf_session, {});
    
    // Verify the computation: 10 / 4 = 2.5
    ASSERT_NE(results.find(div_result), results.end());
    auto result_tensor = results[div_result];
    ASSERT_NE(result_tensor, nullptr);
    
    // Result should be converted to double for division
    auto scalar_val = result_tensor->get_scalar<double>();
    EXPECT_NEAR(scalar_val, 2.5, 1e-6);
}

// Test that verifies graph execution with multiple outputs
TEST_F(GraphExecutionTest, MultipleOutputNodes) {
    // Create a graph with multiple output paths
    auto input_tensor = JTensor::from_data(std::vector<double>{1.0, 2.0, 3.0, 4.0}, {4});
    std::string input_id = graph->add_constant(input_tensor);
    
    // Create multiple operations on the same input
    std::string sum_result = graph->add_operation(GraphOpType::REDUCE_SUM, {input_id});
    std::string min_result = graph->add_operation(GraphOpType::REDUCE_MIN, {input_id});
    std::string max_result = graph->add_operation(GraphOpType::REDUCE_MAX, {input_id});
    
    // Execute graph
    auto results = graph->execute(tf_session, {});
    
    // Verify all outputs are computed correctly
    ASSERT_NE(results.find(sum_result), results.end());
    ASSERT_NE(results.find(min_result), results.end());
    ASSERT_NE(results.find(max_result), results.end());
    
    EXPECT_NEAR(results[sum_result]->get_scalar<double>(), 10.0, 1e-6);  // 1+2+3+4 = 10
    EXPECT_NEAR(results[min_result]->get_scalar<double>(), 1.0, 1e-6);   // min = 1
    EXPECT_NEAR(results[max_result]->get_scalar<double>(), 4.0, 1e-6);   // max = 4
}

} // namespace JInterpreter
