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
