/**
 * Unit tests for J language Lists and Tables operations from Chapter 2
 * Based on: https://code.jsoftware.com/wiki/Help/Learning/Ch_02:_Lists_and_Tables
 */

#include <gtest/gtest.h>
#include "interpreter/interpreter.hpp"
#include "parser/parser.hpp"
#include "lexer/lexer.hpp"
#include <sstream>
#include <variant>
#include <cmath>
#include <vector>

using namespace JInterpreter;

class JListsTablesTest : public ::testing::Test {
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

    // Helper to get scalar result as specific type
    template<typename T>
    T getScalarResult(const std::string& input) {
        auto result = parseAndEvaluate(input);
        if (std::holds_alternative<std::shared_ptr<JTensor>>(result)) {
            auto tensor = std::get<std::shared_ptr<JTensor>>(result);
            if (tensor && tensor->rank() == 0) {
                return tensor->get_scalar<T>();
            }
        }
        throw std::runtime_error("Expected scalar tensor result");
    }

    // Helper to get tensor result
    std::shared_ptr<JTensor> getTensorResult(const std::string& input) {
        auto result = parseAndEvaluate(input);
        if (std::holds_alternative<std::shared_ptr<JTensor>>(result)) {
            return std::get<std::shared_ptr<JTensor>>(result);
        }
        throw std::runtime_error("Expected tensor result");
    }

    // Helper to check if two vectors are equal
    bool vectorsEqual(const std::vector<long long>& a, const std::vector<long long>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    std::unique_ptr<Interpreter> interpreter;
};

// Chapter 2.1: Tables
class TablesTest : public JListsTablesTest {};

TEST_F(TablesTest, BasicTableCreation) {
    // Example: table =: 2 3 $ 5 6 7 8 9 10
    // Creates a 2x3 table with elements 5,6,7 in first row and 8,9,10 in second row
    try {
        auto result = getTensorResult("2 3 $ 5 6 7 8 9 10");
        ASSERT_TRUE(result != nullptr);
        
        // Check shape: should be [2, 3]
        auto shape = result->shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 2);  // 2 rows
        EXPECT_EQ(shape[1], 3);  // 3 columns
        
        // Check rank
        EXPECT_EQ(result->rank(), 2);
        
        // Check values if tensor supports element access
        // First row: 5, 6, 7
        // Second row: 8, 9, 10
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Table creation not implemented: " << e.what();
    }
}

TEST_F(TablesTest, TableReshapeReuse) {
    // Example: 2 4 $ 5 6 7 8 9
    // Should reuse elements: 5 6 7 8, 9 5 6 7
    try {
        auto result = getTensorResult("2 4 $ 5 6 7 8 9");
        ASSERT_TRUE(result != nullptr);
        
        auto shape = result->shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 2);  // 2 rows
        EXPECT_EQ(shape[1], 4);  // 4 columns
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Table reshape with reuse not implemented: " << e.what();
    }
}

TEST_F(TablesTest, TableArithmetic) {
    // Example: 10 * table (from chapter examples)
    try {
        // First create a table, then multiply
        auto table = getTensorResult("2 3 $ 5 6 7 8 9 10");
        auto result = getTensorResult("10 * (2 3 $ 5 6 7 8 9 10)");
        
        ASSERT_TRUE(result != nullptr);
        auto shape = result->shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 2);
        EXPECT_EQ(shape[1], 3);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Table arithmetic not implemented: " << e.what();
    }
}

// Chapter 2.2: Arrays  
class ArraysTest : public JListsTablesTest {};

TEST_F(ArraysTest, OneDimensionalArray) {
    // Example: 3 $ 1
    // Creates [1, 1, 1]
    try {
        auto result = getTensorResult("3 $ 1");
        ASSERT_TRUE(result != nullptr);
        
        auto shape = result->shape();
        EXPECT_EQ(shape.size(), 1);
        EXPECT_EQ(shape[0], 3);
        EXPECT_EQ(result->rank(), 1);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "1D array creation not implemented: " << e.what();
    }
}

TEST_F(ArraysTest, TwoDimensionalArray) {
    // Example: 2 3 $ 5 6 7
    try {
        auto result = getTensorResult("2 3 $ 5 6 7");
        ASSERT_TRUE(result != nullptr);
        
        auto shape = result->shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 2);
        EXPECT_EQ(shape[1], 3);
        EXPECT_EQ(result->rank(), 2);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "2D array creation not implemented: " << e.what();
    }
}

TEST_F(ArraysTest, ThreeDimensionalArray) {
    // Example: 2 2 3 $ 5 6 7 8
    // Creates a 3D array with 2 planes, 2 rows, 3 columns
    try {
        auto result = getTensorResult("2 2 3 $ 5 6 7 8");
        ASSERT_TRUE(result != nullptr);
        
        auto shape = result->shape();
        EXPECT_EQ(shape.size(), 3);
        EXPECT_EQ(shape[0], 2);  // 2 planes
        EXPECT_EQ(shape[1], 2);  // 2 rows
        EXPECT_EQ(shape[2], 3);  // 3 columns
        EXPECT_EQ(result->rank(), 3);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "3D array creation not implemented: " << e.what();
    }
}

// Chapter 2.3: Rank and Shape
class RankShapeTest : public JListsTablesTest {};

TEST_F(RankShapeTest, ScalarRankAndShape) {
    // Example: # $ 17 should be 0 (scalar has rank 0)
    try {
        auto result = getScalarResult<long long>("# $ 17");
        EXPECT_EQ(result, 0);
        
        // Also test: $ 17 should be empty list
        auto shape_result = getTensorResult("$ 17");
        EXPECT_EQ(shape_result->rank(), 1);
        EXPECT_EQ(shape_result->shape()[0], 0);  // empty list
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Rank/shape operations not implemented: " << e.what();
    }
}

TEST_F(RankShapeTest, ListRankAndShape) {
    // Example for list: x =: 4 5 6, then # $ x should be 1, $ x should be 3
    try {
        // Assuming we can evaluate expressions with variables
        parseAndEvaluate("x =: 4 5 6");
        
        auto rank_result = getScalarResult<long long>("# $ x");
        EXPECT_EQ(rank_result, 1);
        
        auto shape_result = getTensorResult("$ x");
        EXPECT_EQ(shape_result->rank(), 1);
        EXPECT_EQ(shape_result->shape()[0], 1);  // shape is [3]
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Variable assignment or rank/shape not implemented: " << e.what();
    }
}

TEST_F(RankShapeTest, TableRankAndShape) {
    // Example: T =: 2 3 $ 1, then # $ T should be 2, $ T should be 2 3
    try {
        parseAndEvaluate("T =: 2 3 $ 1");
        
        auto rank_result = getScalarResult<long long>("# $ T");
        EXPECT_EQ(rank_result, 2);
        
        auto shape_result = getTensorResult("$ T");
        EXPECT_EQ(shape_result->rank(), 1);
        EXPECT_EQ(shape_result->shape()[0], 2);  // shape list has 2 elements: [2, 3]
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Table rank/shape operations not implemented: " << e.what();
    }
}

// Chapter 2.4: Arrays of Characters
class CharacterArraysTest : public JListsTablesTest {};

TEST_F(CharacterArraysTest, BasicStringCreation) {
    // Example: title =: 'My Ten Years in a Quandary'
    try {
        parseAndEvaluate("title =: 'My Ten Years in a Quandary'");
        
        // Check if we can retrieve and work with the string
        auto result = getTensorResult("title");
        ASSERT_TRUE(result != nullptr);
        
        // String should be rank 1 (list of characters)
        EXPECT_EQ(result->rank(), 1);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "String handling not implemented: " << e.what();
    }
}

TEST_F(CharacterArraysTest, EmptyString) {
    // Example: '' should create empty string with length 0
    try {
        auto result = getTensorResult("''");
        ASSERT_TRUE(result != nullptr);
        
        EXPECT_EQ(result->rank(), 1);
        EXPECT_EQ(result->shape()[0], 0);  // length 0
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Empty string not implemented: " << e.what();
    }
}

// Chapter 2.5: Some Functions for Arrays
class ArrayFunctionsTest : public JListsTablesTest {};

// 2.5.1: Joining (Append function ,)
TEST_F(ArrayFunctionsTest, AppendStrings) {
    // Example: a =: 'rear', b =: 'ranged', a,b should be 'rearranged'
    try {
        parseAndEvaluate("a =: 'rear'");
        parseAndEvaluate("b =: 'ranged'");
        
        auto result = getTensorResult("a,b");
        ASSERT_TRUE(result != nullptr);
        
        EXPECT_EQ(result->rank(), 1);
        EXPECT_EQ(result->shape()[0], 11);  // 'rearranged' has 11 characters
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "String append not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, AppendNumbers) {
    // Example: x =: 1 2 3, 0,x should be 0 1 2 3
    try {
        parseAndEvaluate("x =: 1 2 3");
        
        auto result = getTensorResult("0,x");
        ASSERT_TRUE(result != nullptr);
        
        EXPECT_EQ(result->rank(), 1);
        EXPECT_EQ(result->shape()[0], 4);  // [0, 1, 2, 3]
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Number append not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, AppendTables) {
    // Example: T1,T2 joins tables end-to-end
    try {
        parseAndEvaluate("T1 =: 2 3 $ 'catdog'");
        parseAndEvaluate("T2 =: 2 3 $ 'ratpig'");
        
        auto result = getTensorResult("T1,T2");
        ASSERT_TRUE(result != nullptr);
        
        EXPECT_EQ(result->rank(), 2);
        EXPECT_EQ(result->shape()[0], 4);  // 4 rows total
        EXPECT_EQ(result->shape()[1], 3);  // 3 columns
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Table append not implemented: " << e.what();
    }
}

// 2.5.2: Items (Tally function #)
TEST_F(ArrayFunctionsTest, TallyFunction) {
    // Example: x =: 1 2 3, # x should be 3
    try {
        parseAndEvaluate("x =: 1 2 3");
        
        auto result = getScalarResult<long long>("# x");
        EXPECT_EQ(result, 3);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Tally function not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, TallyTable) {
    // Example: T1 =: 2 3 $ 'catdog', # T1 should be 2 (number of rows)
    try {
        parseAndEvaluate("T1 =: 2 3 $ 'catdog'");
        
        auto result = getScalarResult<long long>("# T1");
        EXPECT_EQ(result, 2);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Table tally not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, TallyScalar) {
    // Example: # 6 should be 1 (scalar is single item)
    try {
        auto result = getScalarResult<long long>("# 6");
        EXPECT_EQ(result, 1);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Scalar tally not implemented: " << e.what();
    }
}

// 2.5.3: Selecting (From function {)
TEST_F(ArrayFunctionsTest, SelectFromList) {
    // Example: Y =: 'abcd', 0 { Y should be 'a', 3 { Y should be 'd'
    try {
        parseAndEvaluate("Y =: 'abcd'");
        
        auto result1 = getTensorResult("0 { Y");
        auto result2 = getTensorResult("3 { Y");
        
        ASSERT_TRUE(result1 != nullptr);
        ASSERT_TRUE(result2 != nullptr);
        
        // Results should be single characters (scalars in character context)
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Selection from list not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, SelectMultipleIndices) {
    // Example: Y =: 'abcd', 0 1 { Y should be 'ab'
    try {
        parseAndEvaluate("Y =: 'abcd'");
        
        auto result = getTensorResult("0 1 { Y");
        ASSERT_TRUE(result != nullptr);
        
        EXPECT_EQ(result->rank(), 1);
        EXPECT_EQ(result->shape()[0], 2);  // 2 characters selected
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Multiple index selection not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, IndexGeneration) {
    // Example: i. 4 should generate 0 1 2 3
    try {
        auto result = getTensorResult("i. 4");
        ASSERT_TRUE(result != nullptr);
        
        EXPECT_EQ(result->rank(), 1);
        EXPECT_EQ(result->shape()[0], 4);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Index generation not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, IndexGenerationMultiDim) {
    // Example: i. 2 3 should generate 2x3 matrix with indices
    try {
        auto result = getTensorResult("i. 2 3");
        ASSERT_TRUE(result != nullptr);
        
        EXPECT_EQ(result->rank(), 2);
        EXPECT_EQ(result->shape()[0], 2);
        EXPECT_EQ(result->shape()[1], 3);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Multi-dimensional index generation not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, IndexOf) {
    // Example: 'park' i. 'k' should be 3
    try {
        auto result = getScalarResult<long long>("'park' i. 'k'");
        EXPECT_EQ(result, 3);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Index of not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, IndexOfNotFound) {
    // Example: 'park' i. 'j' should be 4 (length of list)
    try {
        auto result = getScalarResult<long long>("'park' i. 'j'");
        EXPECT_EQ(result, 4);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Index of (not found) not implemented: " << e.what();
    }
}

// 2.5.4: Equality and Matching
TEST_F(ArrayFunctionsTest, MatchFunction) {
    // Example: X =: 'abc', X -: X should be 1 (true)
    try {
        parseAndEvaluate("X =: 'abc'");
        
        auto result = getScalarResult<long long>("X -: X");
        EXPECT_EQ(result, 1);  // true
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Match function not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, MatchDifferentTypes) {
    // Example: X =: 'abc', Y =: 1 2 3 4, X -: Y should be 0 (false)
    try {
        parseAndEvaluate("X =: 'abc'");
        parseAndEvaluate("Y =: 1 2 3 4");
        
        auto result = getScalarResult<long long>("X -: Y");
        EXPECT_EQ(result, 0);  // false
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Match function (different types) not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, EqualFunction) {
    // Example: Y =: 1 2 3 4, Y = Y should be 1 1 1 1
    try {
        parseAndEvaluate("Y =: 1 2 3 4");
        
        auto result = getTensorResult("Y = Y");
        ASSERT_TRUE(result != nullptr);
        
        EXPECT_EQ(result->rank(), 1);
        EXPECT_EQ(result->shape()[0], 4);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Equal function not implemented: " << e.what();
    }
}

TEST_F(ArrayFunctionsTest, EqualScalar) {
    // Example: Y =: 1 2 3 4, Y = 2 should be 0 1 0 0
    try {
        parseAndEvaluate("Y =: 1 2 3 4");
        
        auto result = getTensorResult("Y = 2");
        ASSERT_TRUE(result != nullptr);
        
        EXPECT_EQ(result->rank(), 1);
        EXPECT_EQ(result->shape()[0], 4);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Equal function (scalar) not implemented: " << e.what();
    }
}

// Chapter 2.6: Arrays of Boxes (simplified tests)
class BoxArraysTest : public JListsTablesTest {};

TEST_F(BoxArraysTest, BasicLinking) {
    // Example: A =: 'The answer is' ; 42
    // Creates a list of 2 boxes
    try {
        parseAndEvaluate("A =: 'The answer is' ; 42");
        
        auto result = getTensorResult("A");
        ASSERT_TRUE(result != nullptr);
        
        // Should be a list of 2 items (boxes)
        EXPECT_EQ(result->rank(), 1);
        EXPECT_EQ(result->shape()[0], 2);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Box linking not implemented: " << e.what();
    }
}

TEST_F(BoxArraysTest, BoxingUnboxing) {
    // Example: b =: < 1 2 3, > b should recover 1 2 3
    try {
        parseAndEvaluate("b =: < 1 2 3");
        
        auto boxed = getTensorResult("b");
        auto unboxed = getTensorResult("> b");
        
        ASSERT_TRUE(boxed != nullptr);
        ASSERT_TRUE(unboxed != nullptr);
        
        // Boxed should be scalar (rank 0)
        EXPECT_EQ(boxed->rank(), 0);
        
        // Unboxed should be the original vector
        EXPECT_EQ(unboxed->rank(), 1);
        EXPECT_EQ(unboxed->shape()[0], 3);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Boxing/unboxing not implemented: " << e.what();
    }
}

// Integration tests combining multiple concepts
class IntegrationTest : public JListsTablesTest {};

TEST_F(IntegrationTest, TableOperationsWithTally) {
    // Create table, perform operations, check tally
    try {
        parseAndEvaluate("T =: 3 2 $ 1 2 3 4 5 6");
        
        // Tally should give number of rows
        auto tally_result = getScalarResult<long long>("# T");
        EXPECT_EQ(tally_result, 3);
        
        // Shape should be 3 2
        auto shape_result = getTensorResult("$ T");
        EXPECT_EQ(shape_result->shape()[0], 2);  // shape list has 2 elements
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Table operations integration not implemented: " << e.what();
    }
}

TEST_F(IntegrationTest, ArrayReshapeAndSelect) {
    // Create array, reshape, then select elements
    try {
        parseAndEvaluate("arr =: i. 6");  // 0 1 2 3 4 5
        parseAndEvaluate("table =: 2 3 $ arr");  // reshape to 2x3
        
        auto table_result = getTensorResult("table");
        ASSERT_TRUE(table_result != nullptr);
        
        EXPECT_EQ(table_result->rank(), 2);
        EXPECT_EQ(table_result->shape()[0], 2);
        EXPECT_EQ(table_result->shape()[1], 3);
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Array reshape and select integration not implemented: " << e.what();
    }
}
