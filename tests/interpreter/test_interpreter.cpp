#include <gtest/gtest.h>
#include "interpreter/interpreter.hpp"
#include "parser/parser.hpp"
#include "lexer/lexer.hpp"
#include <sstream>
#include <variant>

using namespace JInterpreter;

class InterpreterTest : public ::testing::Test {
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

TEST_F(InterpreterTest, BasicNumericLiteral) {
    auto result = parseAndEvaluate("42");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0); // Scalar
    EXPECT_EQ(tensor->get_scalar<long long>(), 42);
}

TEST_F(InterpreterTest, FloatLiteral) {
    auto result = parseAndEvaluate("3.14");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0); // Scalar
    EXPECT_NEAR(tensor->get_scalar<double>(), 3.14, 1e-10);
}

TEST_F(InterpreterTest, IotaOperation) {
    auto result = parseAndEvaluate("i. 5");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1); // 1D vector
    EXPECT_EQ(tensor->shape()[0], 5);
}

TEST_F(InterpreterTest, BasicAddition) {
    auto result = parseAndEvaluate("3 + 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0); // Scalar result
    // Note: Exact result depends on TensorFlow implementation vs stub
    EXPECT_EQ(tensor->get_scalar<long long>(), 7);
}


TEST_F(InterpreterTest, AdditionMultiplicationPresidence) {
    auto result = parseAndEvaluate("3 + 4 * 2");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0); // Scalar result
    // Note: Exact result depends on TensorFlow implementation vs stub
    EXPECT_EQ(tensor->get_scalar<long long>(), 11);
}


TEST_F(InterpreterTest, RightToLeftAdditionMultiplicationPresidence) {
    auto result = parseAndEvaluate("2 * 3 + 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0); // Scalar result
    // Note: Exact result depends on TensorFlow implementation vs stub
    EXPECT_EQ(tensor->get_scalar<long long>(), 14);
}

TEST_F(InterpreterTest, StringLiteral) {
    auto result = parseAndEvaluate("'hello'");
    // Strings should remain as strings, not converted to tensors
    EXPECT_TRUE(std::holds_alternative<std::string>(result));
    if (std::holds_alternative<std::string>(result)) {
        EXPECT_EQ(std::get<std::string>(result), "hello");
    }
}

// Test that we can print tensor information
TEST_F(InterpreterTest, TensorPrinting) {
    auto result = parseAndEvaluate("i. 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    
    // Test that we can print without crashing
    std::ostringstream oss;
    tensor->print(oss);
    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_TRUE(output.find("JTensor") != std::string::npos);
}

TEST_F(InterpreterTest, FoldSumIota) {
    auto result = parseAndEvaluate("+/ i. 5");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0); // Scalar result
    
    // +/ i. 5 should sum 0+1+2+3+4 = 10
    auto scalar_result = tensor->get_scalar<long long>();
    EXPECT_EQ(scalar_result, 10);
}

// Test comprehensive arithmetic operations
TEST_F(InterpreterTest, Addition) {
    // Scalar addition
    auto result = parseAndEvaluate("3 + 5");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0);
    EXPECT_EQ(tensor->get_scalar<long long>(), 8);
    
    // Vector addition
    auto result2 = parseAndEvaluate("1 2 3 + 4 5 6");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
    EXPECT_EQ(tensor2->shape()[0], 3);
}

TEST_F(InterpreterTest, Subtraction) {
    // Scalar subtraction
    auto result = parseAndEvaluate("10 - 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0);
    EXPECT_EQ(tensor->get_scalar<long long>(), 7);
    
    // Vector subtraction
    auto result2 = parseAndEvaluate("10 20 30 - 1 2 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
    EXPECT_EQ(tensor2->shape()[0], 3);
}

TEST_F(InterpreterTest, Multiplication) {
    // Scalar multiplication
    auto result = parseAndEvaluate("6 * 7");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0);
    EXPECT_EQ(tensor->get_scalar<long long>(), 42);
    
    // Vector multiplication  
    auto result2 = parseAndEvaluate("2 3 4 * 5 6 7");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
    EXPECT_EQ(tensor2->shape()[0], 3);
}

TEST_F(InterpreterTest, Division) {
    // Scalar division
    auto result = parseAndEvaluate("15 % 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0);
    EXPECT_EQ(tensor->get_scalar<long long>(), 5);
    
    // Vector division
    auto result2 = parseAndEvaluate("12 18 24 % 3 6 8");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
    EXPECT_EQ(tensor2->shape()[0], 3);
}

// Test iota operation with different parameters
TEST_F(InterpreterTest, IotaVariations) {
    // Basic iota
    auto result = parseAndEvaluate("i. 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape()[0], 4);
    
    // Larger iota
    auto result2 = parseAndEvaluate("i. 10");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
    EXPECT_EQ(tensor2->shape()[0], 10);
    
    // Empty iota
    auto result3 = parseAndEvaluate("i. 0");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result3));
    auto tensor3 = std::get<std::shared_ptr<JTensor>>(result3);
    ASSERT_NE(tensor3, nullptr);
    EXPECT_EQ(tensor3->rank(), 1);
    EXPECT_EQ(tensor3->shape()[0], 0);
}

// Test shape operation
TEST_F(InterpreterTest, ShapeOperation) {
    // Test shape of vector
    auto result = parseAndEvaluate("$ 1 2 3 4 5");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape()[0], 1); // Shape should return [5] for a 5-element vector
    
    // Test shape of iota
    auto result2 = parseAndEvaluate("$ i. 7");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
}

// Test reduction operations
TEST_F(InterpreterTest, ReductionOperations) {
    // Sum reduction
    auto result = parseAndEvaluate("+/ 1 2 3 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0);
    EXPECT_EQ(tensor->get_scalar<long long>(), 10);
    
    // Product reduction
    auto result2 = parseAndEvaluate("*/ 2 3 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 0);
    EXPECT_EQ(tensor2->get_scalar<long long>(), 24);
}

// Test complex arithmetic expressions
TEST_F(InterpreterTest, ComplexArithmetic) {
    // Nested operations with precedence
    auto result = parseAndEvaluate("2 + 3 * 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 0);
    // J uses right-to-left evaluation, so this should be 2 + (3 * 4) = 14
    EXPECT_EQ(tensor->get_scalar<long long>(), 14);
    
    // Multiple operations
    auto result2 = parseAndEvaluate("10 - 2 * 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 0);
    // Should be 10 - (2 * 3) = 4
    EXPECT_EQ(tensor2->get_scalar<long long>(), 4);
}

// Test operations with different tensor shapes
TEST_F(InterpreterTest, MixedShapeOperations) {
    // Scalar with vector
    auto result = parseAndEvaluate("5 + 1 2 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape()[0], 3);
    
    // Vector with scalar
    auto result2 = parseAndEvaluate("1 2 3 * 2");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result2));
    auto tensor2 = std::get<std::shared_ptr<JTensor>>(result2);
    ASSERT_NE(tensor2, nullptr);
    EXPECT_EQ(tensor2->rank(), 1);
    EXPECT_EQ(tensor2->shape()[0], 3);
}

// Test error conditions
TEST_F(InterpreterTest, ArithmeticErrorConditions) {
    // Division by zero (should handle gracefully)
    auto result = parseAndEvaluate("5 % 0");
    // Test should not crash - specific behavior depends on implementation
    
    // Operations with empty vectors
    auto result2 = parseAndEvaluate("i. 0 + 1");
    // Should handle empty tensor operations
}

// Test tensor rank and shape properties
TEST_F(InterpreterTest, TensorRankAndShape) {
    // Test rank 0 (scalar)
    auto scalar = parseAndEvaluate("42");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(scalar));
    auto scalar_tensor = std::get<std::shared_ptr<JTensor>>(scalar);
    ASSERT_NE(scalar_tensor, nullptr);
    EXPECT_EQ(scalar_tensor->rank(), 0);
    EXPECT_TRUE(scalar_tensor->shape().empty());
    
    // Test rank 1 (vector)
    auto vector = parseAndEvaluate("i. 5");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(vector));
    auto vector_tensor = std::get<std::shared_ptr<JTensor>>(vector);
    ASSERT_NE(vector_tensor, nullptr);
    EXPECT_EQ(vector_tensor->rank(), 1);
    EXPECT_EQ(vector_tensor->shape().size(), 1);
    EXPECT_EQ(vector_tensor->shape()[0], 5);
}

// Test various numeric literal formats
TEST_F(InterpreterTest, NumericLiterals) {
    // Positive integers
    auto pos_int = parseAndEvaluate("123");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(pos_int));
    auto pos_tensor = std::get<std::shared_ptr<JTensor>>(pos_int);
    EXPECT_EQ(pos_tensor->get_scalar<long long>(), 123);
    
    // Negative integers
    auto neg_int = parseAndEvaluate("_456");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(neg_int));
    auto neg_tensor = std::get<std::shared_ptr<JTensor>>(neg_int);
    EXPECT_EQ(neg_tensor->get_scalar<long long>(), -456);
    
    // Zero
    auto zero = parseAndEvaluate("0");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(zero));
    auto zero_tensor = std::get<std::shared_ptr<JTensor>>(zero);
    EXPECT_EQ(zero_tensor->get_scalar<long long>(), 0);
}

// Test array/list operations
TEST_F(InterpreterTest, ArrayOperations) {
    // Test list creation
    auto list = parseAndEvaluate("1 2 3 4");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(list));
    auto list_tensor = std::get<std::shared_ptr<JTensor>>(list);
    ASSERT_NE(list_tensor, nullptr);
    EXPECT_EQ(list_tensor->rank(), 1);
    EXPECT_EQ(list_tensor->shape()[0], 4);
    
    // Test mixed numeric lists
    auto mixed_list = parseAndEvaluate("10 _5 0 7");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(mixed_list));
    auto mixed_tensor = std::get<std::shared_ptr<JTensor>>(mixed_list);
    ASSERT_NE(mixed_tensor, nullptr);
    EXPECT_EQ(mixed_tensor->rank(), 1);
    EXPECT_EQ(mixed_tensor->shape()[0], 4);
}

// Test dimension properties
TEST_F(InterpreterTest, DimensionProperties) {
    // Test single dimension
    auto single_dim = parseAndEvaluate("i. 8");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(single_dim));
    auto single_tensor = std::get<std::shared_ptr<JTensor>>(single_dim);
    ASSERT_NE(single_tensor, nullptr);
    EXPECT_EQ(single_tensor->rank(), 1);
    EXPECT_EQ(single_tensor->size(), 8);
    
    // Test empty dimension
    auto empty_dim = parseAndEvaluate("i. 0");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(empty_dim));
    auto empty_tensor = std::get<std::shared_ptr<JTensor>>(empty_dim);
    ASSERT_NE(empty_tensor, nullptr);
    EXPECT_EQ(empty_tensor->rank(), 1);
    EXPECT_EQ(empty_tensor->size(), 0);
    EXPECT_EQ(empty_tensor->shape()[0], 0);
}

// Test vector literal evaluation
TEST_F(InterpreterTest, VectorLiteralEvaluation) {
    auto result = parseAndEvaluate("1 2 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape()[0], 3);
}

TEST_F(InterpreterTest, EmptyVectorLiteral) {
    // Test empty vector (if supported)
    // This might not be directly parseable as empty input, but testing the interpreter method
    // We'll test via a simple expression that should create an empty result
}

TEST_F(InterpreterTest, SingleNumberVsVector) {
    // Single number should be scalar
    auto scalar = parseAndEvaluate("42");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(scalar));
    auto scalar_tensor = std::get<std::shared_ptr<JTensor>>(scalar);
    ASSERT_NE(scalar_tensor, nullptr);
    EXPECT_EQ(scalar_tensor->rank(), 0);
    
    // Multiple numbers should be vector
    auto vector = parseAndEvaluate("1 2");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(vector));
    auto vector_tensor = std::get<std::shared_ptr<JTensor>>(vector);
    ASSERT_NE(vector_tensor, nullptr);
    EXPECT_EQ(vector_tensor->rank(), 1);
    EXPECT_EQ(vector_tensor->shape()[0], 2);
}

TEST_F(InterpreterTest, MixedTypeVectorLiteral) {
    auto result = parseAndEvaluate("1 2.5 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rank(), 1);
    EXPECT_EQ(tensor->shape()[0], 3);
    // Should be converted to float tensor due to mixed types
}

TEST_F(InterpreterTest, VectorAdditionDebug) {
    // Debug the specific failing case
    std::cout << "Testing vector addition: 1 2 3 + 4 5 6" << std::endl;
    
    // First test just the left operand
    auto left = parseAndEvaluate("1 2 3");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(left));
    auto left_tensor = std::get<std::shared_ptr<JTensor>>(left);
    ASSERT_NE(left_tensor, nullptr);
    std::cout << "Left operand rank: " << left_tensor->rank() << std::endl;
    if (left_tensor->rank() > 0) {
        std::cout << "Left operand shape[0]: " << left_tensor->shape()[0] << std::endl;
    }
    EXPECT_EQ(left_tensor->rank(), 1);
    EXPECT_EQ(left_tensor->shape()[0], 3);
    
    // Test just the right operand
    auto right = parseAndEvaluate("4 5 6");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(right));
    auto right_tensor = std::get<std::shared_ptr<JTensor>>(right);
    ASSERT_NE(right_tensor, nullptr);
    std::cout << "Right operand rank: " << right_tensor->rank() << std::endl;
    if (right_tensor->rank() > 0) {
        std::cout << "Right operand shape[0]: " << right_tensor->shape()[0] << std::endl;
    }
    EXPECT_EQ(right_tensor->rank(), 1);
    EXPECT_EQ(right_tensor->shape()[0], 3);
    
    // Now test the full expression
    auto result = parseAndEvaluate("1 2 3 + 4 5 6");
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<JTensor>>(result));
    auto result_tensor = std::get<std::shared_ptr<JTensor>>(result);
    ASSERT_NE(result_tensor, nullptr);
    std::cout << "Result tensor rank: " << result_tensor->rank() << std::endl;
    if (result_tensor->rank() > 0) {
        std::cout << "Result tensor shape[0]: " << result_tensor->shape()[0] << std::endl;
    }
    EXPECT_EQ(result_tensor->rank(), 1);
    EXPECT_EQ(result_tensor->shape()[0], 3);
}

TEST_F(InterpreterTest, TensorCreationDebug) {
    // Test tensor creation directly
    auto tensor_15 = JTensor::scalar(15LL);
    auto tensor_3 = JTensor::scalar(3LL);
    
    std::cout << "tensor_15: rank=" << tensor_15->rank() << ", size=" << tensor_15->size() << std::endl;
    std::cout << "tensor_3: rank=" << tensor_3->rank() << ", size=" << tensor_3->size() << std::endl;
    
    // Get flat data
    auto data_15 = tensor_15->get_flat<long long>();
    auto data_3 = tensor_3->get_flat<long long>();
    
    std::cout << "data_15.size()=" << data_15.size() << ", data_3.size()=" << data_3.size() << std::endl;
    
    EXPECT_EQ(data_15.size(), 1);
    EXPECT_EQ(data_3.size(), 1);
    
    if (data_15.size() > 0) {
        std::cout << "data_15[0]=" << data_15[0] << std::endl;
        EXPECT_EQ(data_15[0], 15);
    }
    
    if (data_3.size() > 0) {
        std::cout << "data_3[0]=" << data_3[0] << std::endl;
        EXPECT_EQ(data_3[0], 3);
    }
}
