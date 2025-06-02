#include <gtest/gtest.h>
#include "interpreter/tf_operations.hpp"
#include <vector>
#include <memory>

using namespace JInterpreter;

class JTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test data for various scenarios
        scalar_data = {42};
        vector_data = {1, 2, 3, 4, 5};
        matrix_data = {1, 2, 3, 4, 5, 6};
        matrix_shape = {2, 3};
        tensor_3d_data = {1, 2, 3, 4, 5, 6, 7, 8};
        tensor_3d_shape = {2, 2, 2};
    }

    std::vector<long long> scalar_data;
    std::vector<long long> vector_data;
    std::vector<long long> matrix_data;
    std::vector<long long> matrix_shape;
    std::vector<long long> tensor_3d_data;
    std::vector<long long> tensor_3d_shape;
};

// Test scalar tensor creation and properties
TEST_F(JTensorTest, ScalarTensorProperties) {
    auto tensor = JTensor::from_data(scalar_data, {});
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0);
    EXPECT_TRUE(tensor->shape().empty());
    EXPECT_EQ(tensor->size(), 1);
    EXPECT_EQ(tensor->get_scalar<long long>(), 42);
}

// Test vector tensor creation and properties
TEST_F(JTensorTest, VectorTensorProperties) {
    auto tensor = JTensor::from_data(vector_data, {5});
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape().size(), 1);
    EXPECT_EQ(tensor->shape()[0], 5);
    EXPECT_EQ(tensor->size(), 5);
}

// Test matrix tensor creation and properties
TEST_F(JTensorTest, MatrixTensorProperties) {
    auto tensor = JTensor::from_data(matrix_data, matrix_shape);
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 2);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 2);
    EXPECT_EQ(tensor->shape()[1], 3);
    EXPECT_EQ(tensor->size(), 6);
}

// Test 3D tensor creation and properties
TEST_F(JTensorTest, Tensor3DProperties) {
    auto tensor = JTensor::from_data(tensor_3d_data, tensor_3d_shape);
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 3);
    EXPECT_EQ(tensor->shape().size(), 3);
    EXPECT_EQ(tensor->shape()[0], 2);
    EXPECT_EQ(tensor->shape()[1], 2);
    EXPECT_EQ(tensor->shape()[2], 2);
    EXPECT_EQ(tensor->size(), 8);
}

// Test float tensor creation
TEST_F(JTensorTest, FloatTensorProperties) {
    std::vector<double> float_data = {1.5, 2.5, 3.5};
    auto tensor = JTensor::from_data(float_data, {3});
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape().size(), 1);
    EXPECT_EQ(tensor->shape()[0], 3);
    EXPECT_EQ(tensor->size(), 3);
}

// Test empty tensor
TEST_F(JTensorTest, EmptyTensorProperties) {
    std::vector<long long> empty_data = {};
    auto tensor = JTensor::from_data(empty_data, {0});
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape().size(), 1);
    EXPECT_EQ(tensor->shape()[0], 0);
    EXPECT_EQ(tensor->size(), 0);
}

// Test tensor with large dimensions
TEST_F(JTensorTest, LargeTensorProperties) {
    std::vector<long long> large_data(100, 1); // 100 elements of value 1
    std::vector<long long> large_shape = {10, 10};
    auto tensor = JTensor::from_data(large_data, large_shape);
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 2);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 10);
    EXPECT_EQ(tensor->shape()[1], 10);
    EXPECT_EQ(tensor->size(), 100);
}

// Test tensor shape validation
TEST_F(JTensorTest, ShapeValidation) {
    // Test mismatched data size and shape
    std::vector<long long> data = {1, 2, 3, 4}; // 4 elements
    std::vector<long long> wrong_shape = {2, 3}; // expects 6 elements
    
    // The tensor should still be created but may have inconsistencies
    auto tensor = JTensor::from_data(data, wrong_shape);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 2);
    EXPECT_EQ(tensor->shape()[0], 2);
    EXPECT_EQ(tensor->shape()[1], 3);
}

// Test rank edge cases
TEST_F(JTensorTest, RankEdgeCases) {
    // Test very high rank tensor
    std::vector<long long> high_rank_data = {1};
    std::vector<long long> high_rank_shape = {1, 1, 1, 1, 1}; // 5D tensor
    auto tensor = JTensor::from_data(high_rank_data, high_rank_shape);
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 5);
    EXPECT_EQ(tensor->size(), 1);
}

// Test tensor copy and equality (if implemented)
TEST_F(JTensorTest, TensorCopy) {
    auto original = JTensor::from_data(vector_data, {5});
    ASSERT_NE(original, nullptr);
    
    // Test that we can create multiple tensors with same data
    auto copy = JTensor::from_data(vector_data, {5});
    ASSERT_NE(copy, nullptr);
    
    EXPECT_EQ(original->rank(), copy->rank());
    EXPECT_EQ(original->shape(), copy->shape());
    EXPECT_EQ(original->size(), copy->size());
}

// Test tensor dimensions access
TEST_F(JTensorTest, DimensionsAccess) {
    auto tensor = JTensor::from_data(tensor_3d_data, tensor_3d_shape);
    
    ASSERT_NE(tensor, nullptr);
    auto shape = tensor->shape();
    
    EXPECT_EQ(shape.size(), 3);
    for (size_t i = 0; i < shape.size(); ++i) {
        EXPECT_GT(shape[i], 0);
    }
}

// Test scalar access with different data types
TEST_F(JTensorTest, ScalarAccessDifferentTypes) {
    // Test int64 scalar
    std::vector<long long> int_data = {123};
    auto int_tensor = JTensor::from_data(int_data, {});
    ASSERT_NE(int_tensor, nullptr);
    EXPECT_EQ(int_tensor->get_scalar<long long>(), 123);
    
    // Test float64 scalar
    std::vector<double> float_data = {3.14};
    auto float_tensor = JTensor::from_data(float_data, {});
    ASSERT_NE(float_tensor, nullptr);
    EXPECT_DOUBLE_EQ(float_tensor->get_scalar<double>(), 3.14);
}

// Test tensor size calculation for various shapes
TEST_F(JTensorTest, SizeCalculation) {
    // Test 1D
    std::vector<long long> data_1d = {1, 2, 3};
    auto tensor_1d = JTensor::from_data(data_1d, {3});
    EXPECT_EQ(tensor_1d->size(), 3);
    
    // Test 2D
    auto tensor_2d = JTensor::from_data(matrix_data, {2, 3});
    EXPECT_EQ(tensor_2d->size(), 6);
    
    // Test 3D
    auto tensor_3d = JTensor::from_data(tensor_3d_data, {2, 2, 2});
    EXPECT_EQ(tensor_3d->size(), 8);
    
    // Test 4D
    std::vector<long long> data_4d(24, 1);
    auto tensor_4d = JTensor::from_data(data_4d, {2, 3, 2, 2});
    EXPECT_EQ(tensor_4d->size(), 24);
}
