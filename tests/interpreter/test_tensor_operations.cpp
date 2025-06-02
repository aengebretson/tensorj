#include <gtest/gtest.h>
#include "interpreter/tf_operations.hpp"
#include "interpreter/interpreter.hpp"
#include "parser/parser.hpp"
#include "lexer/lexer.hpp"
#include <sstream>
#include <variant>
#include <cmath>

using namespace JInterpreter;

class TensorOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        interpreter = std::make_unique<Interpreter>();
    }

    JValue parseAndEvaluate(const std::string& input) {
        Lexer lexer(input);
        auto tokens = lexer.tokenize();
        
        Parser parser(tokens);
        auto ast = parser.parse();
        
        if (ast) {
            return interpreter->evaluate(ast.get());
        }
        return nullptr;
    }

    std::unique_ptr<Interpreter> interpreter;
};

// Test tensor arithmetic operations
TEST_F(TensorOperationsTest, TensorAddition) {
    // Test basic tensor addition
    auto result = parseAndEvaluate("1 2 3 + 4 5 6");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape()[0], 3);
    
    // Expected result: [5, 7, 9]
    // Note: Exact value verification depends on tensor data access methods
}

TEST_F(TensorOperationsTest, TensorSubtraction) {
    auto result = parseAndEvaluate("10 20 30 - 1 2 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape()[0], 3);
    
    // Expected result: [9, 18, 27]
}

TEST_F(TensorOperationsTest, TensorMultiplication) {
    auto result = parseAndEvaluate("2 3 4 * 5 6 7");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape()[0], 3);
    
    // Expected result: [10, 18, 28]
}

TEST_F(TensorOperationsTest, TensorDivision) {
    auto result = parseAndEvaluate("12 18 24 % 3 6 8");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape()[0], 3);
    
    // Expected result: [4, 3, 3]
}

// Test broadcasting operations
TEST_F(TensorOperationsTest, ScalarTensorBroadcasting) {
    // Scalar + Vector
    auto result1 = parseAndEvaluate("5 + 1 2 3 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result1));
    auto tensor1 = std::get<std::shared_ptr<JTensor>>(result1);
    ASSERT_NE(tensor1, nullptr);
    EXPECT_EQ(tensor1->rank(), 1);
    EXPECT_EQ(tensor1->shape()[0], 4);
    
    // Vector + Scalar
    auto result2 = parseAndEvaluate("1 2 3 4 + 10");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
    EXPECT_EQ(tensor2->shape()[0], 4);
    
    // Scalar * Vector
    auto result3 = parseAndEvaluate("3 * 2 4 6");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result3));
    auto tensor3 = std::get<std::shared_ptr<JTensor>>(result3);
    ASSERT_NE(tensor3, nullptr);
    EXPECT_EQ(tensor3->rank(), 1);
    EXPECT_EQ(tensor3->shape()[0], 3);
}

// Test iota operations with different sizes
TEST_F(TensorOperationsTest, IotaOperations) {
    // Basic iota
    auto result1 = parseAndEvaluate("i. 5");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result1));
    auto tensor1 = std::get<std::shared_ptr<JTensor>>(result1);
    ASSERT_NE(tensor1, nullptr);
    EXPECT_EQ(tensor1->rank(), 1);
    EXPECT_EQ(tensor1->shape()[0], 5);
    
    // Large iota
    auto result2 = parseAndEvaluate("i. 100");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
    EXPECT_EQ(tensor2->shape()[0], 100);
    
    // Single element iota
    auto result3 = parseAndEvaluate("i. 1");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result3));
    auto tensor3 = std::get<std::shared_ptr<JTensor>>(result3);
    ASSERT_NE(tensor3, nullptr);
    EXPECT_EQ(tensor3->rank(), 1);
    EXPECT_EQ(tensor3->shape()[0], 1);
    
    // Empty iota
    auto result4 = parseAndEvaluate("i. 0");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result4));
    auto tensor4 = std::get<std::shared_ptr<JTensor>>(result4);
    ASSERT_NE(tensor4, nullptr);
    EXPECT_EQ(tensor4->rank(), 1);
    EXPECT_EQ(tensor4->shape()[0], 0);
}

// Test reduction operations
TEST_F(TensorOperationsTest, ReductionOperations) {
    // Sum reduction
    auto sum_result = parseAndEvaluate("+/ 1 2 3 4 5");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(sum_result));
    auto sum_tensor = std::get<std::shared_ptr<JTensor>>(sum_result);
    ASSERT_NE(sum_tensor, nullptr);
    EXPECT_EQ(sum_tensor->rank(), 0); // Scalar result
    EXPECT_EQ(sum_tensor->get_scalar<long long>(), 15);
    
    // Product reduction
    auto prod_result = parseAndEvaluate("*/ 1 2 3 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(prod_result));
    auto prod_tensor = std::get<std::shared_ptr<JTensor>>(prod_result);
    ASSERT_NE(prod_tensor, nullptr);
    EXPECT_EQ(prod_tensor->rank(), 0); // Scalar result
    EXPECT_EQ(prod_tensor->get_scalar<long long>(), 24);
    
    // Sum reduction of iota
    auto iota_sum = parseAndEvaluate("+/ i. 6");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(iota_sum));
    auto iota_tensor = std::get<std::shared_ptr<JTensor>>(iota_sum);
    ASSERT_NE(iota_tensor, nullptr);
    EXPECT_EQ(iota_tensor->rank(), 0);
    EXPECT_EQ(iota_tensor->get_scalar<long long>(), 15); // 0+1+2+3+4+5 = 15
}

// Test shape operations
TEST_F(TensorOperationsTest, ShapeOperations) {
    // Shape of vector
    auto result1 = parseAndEvaluate("$ 1 2 3 4 5");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result1));
    auto tensor1 = std::get<std::shared_ptr<JTensor>>(result1);
    ASSERT_NE(tensor1, nullptr);
    EXPECT_EQ(tensor1->rank(), 1);
    
    // Shape of iota
    auto result2 = parseAndEvaluate("$ i. 10");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
    
    // Shape of scalar
    auto result3 = parseAndEvaluate("$ 42");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result3));
    auto tensor3 = std::get<std::shared_ptr<JTensor>>(result3);
    ASSERT_NE(tensor3, nullptr);
    // Shape of scalar should be empty or special representation
}

// Test chained operations
TEST_F(TensorOperationsTest, ChainedOperations) {
    // Multiple additions
    auto result1 = parseAndEvaluate("1 + 2 + 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result1));
    auto tensor1 = std::get<std::shared_ptr<JTensor>>(result1);
    ASSERT_NE(tensor1, nullptr);
    EXPECT_EQ(tensor1->rank(), 0);
    EXPECT_EQ(tensor1->get_scalar<long long>(), 6);
    
    // Mixed operations with right-to-left evaluation
    auto result2 = parseAndEvaluate("2 * 3 + 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 0);
    EXPECT_EQ(tensor2->get_scalar<long long>(), 14); // 2 * (3 + 4) = 14
    
    // Complex chain with vectors
    auto result3 = parseAndEvaluate("1 2 3 + 4 5 6 * 2");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result3));
    auto tensor3 = std::get<std::shared_ptr<JTensor>>(result3);
    ASSERT_NE(tensor3, nullptr);
    EXPECT_EQ(tensor3->rank(), 1);
    EXPECT_EQ(tensor3->shape()[0], 3);
}

// Test edge cases and error conditions
TEST_F(TensorOperationsTest, EdgeCases) {
    // Operations with zero
    auto zero_add = parseAndEvaluate("0 + 5");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(zero_add));
    auto zero_tensor = std::get<std::shared_ptr<JTensor>>(zero_add);
    ASSERT_NE(zero_tensor, nullptr);
    EXPECT_EQ(zero_tensor->get_scalar<long long>(), 5);
    
    // Operations with negative numbers
    auto neg_ops = parseAndEvaluate("_5 + 10");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(neg_ops));
    auto neg_tensor = std::get<std::shared_ptr<JTensor>>(neg_ops);
    ASSERT_NE(neg_tensor, nullptr);
    EXPECT_EQ(neg_tensor->get_scalar<long long>(), 5);
    
    // Large numbers
    auto large_ops = parseAndEvaluate("1000000 + 2000000");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(large_ops));
    auto large_tensor = std::get<std::shared_ptr<JTensor>>(large_ops);
    ASSERT_NE(large_tensor, nullptr);
    EXPECT_EQ(large_tensor->get_scalar<long long>(), 3000000);
}

// Test tensor size and memory usage
TEST_F(TensorOperationsTest, TensorSizeTests) {
    // Small vectors
    auto small_vec = parseAndEvaluate("i. 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(small_vec));
    auto small_tensor = std::get<std::shared_ptr<JTensor>>(small_vec);
    ASSERT_NE(small_tensor, nullptr);
    EXPECT_EQ(small_tensor->size(), 3);
    
    // Medium vectors
    auto med_vec = parseAndEvaluate("i. 50");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(med_vec));
    auto med_tensor = std::get<std::shared_ptr<JTensor>>(med_vec);
    ASSERT_NE(med_tensor, nullptr);
    EXPECT_EQ(med_tensor->size(), 50);
    
    // Large vectors
    auto large_vec = parseAndEvaluate("i. 1000");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(large_vec));
    auto large_tensor = std::get<std::shared_ptr<JTensor>>(large_vec);
    ASSERT_NE(large_tensor, nullptr);
    EXPECT_EQ(large_tensor->size(), 1000);
}

// Test various numeric ranges
TEST_F(TensorOperationsTest, NumericRanges) {
    // Single digit numbers
    auto single_digit = parseAndEvaluate("1 + 9");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(single_digit));
    auto sd_tensor = std::get<std::shared_ptr<JTensor>>(single_digit);
    EXPECT_EQ(sd_tensor->get_scalar<long long>(), 10);
    
    // Double digit numbers
    auto double_digit = parseAndEvaluate("25 + 75");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(double_digit));
    auto dd_tensor = std::get<std::shared_ptr<JTensor>>(double_digit);
    EXPECT_EQ(dd_tensor->get_scalar<long long>(), 100);
    
    // Triple digit numbers
    auto triple_digit = parseAndEvaluate("123 + 456");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(triple_digit));
    auto td_tensor = std::get<std::shared_ptr<JTensor>>(triple_digit);
    EXPECT_EQ(td_tensor->get_scalar<long long>(), 579);
}

// Test specific J language features
TEST_F(TensorOperationsTest, JLanguageFeatures) {
    // Test right-to-left evaluation order
    auto rtl_eval = parseAndEvaluate("2 + 3 * 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(rtl_eval));
    auto rtl_tensor = std::get<std::shared_ptr<JTensor>>(rtl_eval);
    ASSERT_NE(rtl_tensor, nullptr);
    // In J, this evaluates as 2 + (3 * 4) = 14, not (2 + 3) * 4 = 20
    EXPECT_EQ(rtl_tensor->get_scalar<long long>(), 14);
    
    // Test negative number syntax with underscore
    auto neg_syntax = parseAndEvaluate("_42 + 100");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(neg_syntax));
    auto neg_tensor = std::get<std::shared_ptr<JTensor>>(neg_syntax);
    EXPECT_EQ(neg_tensor->get_scalar<long long>(), 58);
    
    // Test array programming - element-wise operations
    auto array_prog = parseAndEvaluate("1 2 3 4 * 2 2 2 2");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(array_prog));
    auto ap_tensor = std::get<std::shared_ptr<JTensor>>(array_prog);
    ASSERT_NE(ap_tensor, nullptr);
    EXPECT_EQ(ap_tensor->rank(), 1);
    EXPECT_EQ(ap_tensor->shape()[0], 4);
}

// Test tensor rank verification
TEST_F(TensorOperationsTest, TensorRankVerification) {
    // Rank 0 - Scalars
    auto scalar_tests = {
        parseAndEvaluate("42"),
        parseAndEvaluate("5 + 7"),
        parseAndEvaluate("+/ 1 2 3")
    };
    
    for (auto& result : scalar_tests) {
        ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
        auto tensor = std::get<std::shared_ptr<JTensor>>(result);
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->rank(), 0);
    }
    
    // Rank 1 - Vectors
    auto vector_tests = {
        parseAndEvaluate("i. 5"),
        parseAndEvaluate("1 2 3 4"),
        parseAndEvaluate("$ i. 10")
    };
    
    for (auto& result : vector_tests) {
        ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
        auto tensor = std::get<std::shared_ptr<JTensor>>(result);
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->rank(), 1);
    }
}

// Test tensor dimension consistency
TEST_F(TensorOperationsTest, DimensionConsistency) {
    // Test that operations preserve expected dimensions
    auto iota_result = parseAndEvaluate("i. 7");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(iota_result));
    auto iota_tensor = std::get<std::shared_ptr<JTensor>>(iota_result);
    ASSERT_NE(iota_tensor, nullptr);
    EXPECT_EQ(iota_tensor->rank(), 1);
    EXPECT_EQ(iota_tensor->shape()[0], 7);
    EXPECT_EQ(iota_tensor->size(), 7);
    
    // Test J's right-to-left operator precedence: "i. 7 + 1" = "i. (7 + 1)" = "i. 8"
    auto arith_result = parseAndEvaluate("i. 7 + 1");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(arith_result));
    auto arith_tensor = std::get<std::shared_ptr<JTensor>>(arith_result);
    ASSERT_NE(arith_tensor, nullptr);
    EXPECT_EQ(arith_tensor->rank(), 1);
    EXPECT_EQ(arith_tensor->shape()[0], 8);  // Should be 8, not 7, due to right-to-left precedence
    EXPECT_EQ(arith_tensor->size(), 8);     // Should be 8, not 7, due to right-to-left precedence
}
