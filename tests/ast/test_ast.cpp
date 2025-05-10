#include "gtest/gtest.h"
#include "ast/ast_nodes.hpp" // Adjust path
#include <sstream>

using namespace JInterpreter;

TEST(AstNodeTest, NounLiteralNodePrint) {
    NounLiteralNode int_node(123LL, {1,1});
    std::ostringstream oss_int;
    int_node.print(oss_int);
    // Basic check, can be more specific
    EXPECT_NE(oss_int.str().find("NounLiteralNode"), std::string::npos);
    EXPECT_NE(oss_int.str().find("123"), std::string::npos);

    NounLiteralNode str_node(std::string("hello"), {1,5});
    std::ostringstream oss_str;
    str_node.print(oss_str);
    EXPECT_NE(oss_str.str().find("NounLiteralNode"), std::string::npos);
    EXPECT_NE(oss_str.str().find("'hello'"), std::string::npos);
}

TEST(AstNodeTest, MonadicApplicationPrint) {
    auto verb = std::make_unique<VerbNode>("+", SourceLocation{1,2});
    auto arg = std::make_unique<NounLiteralNode>(5LL, SourceLocation{1,4});
    MonadicApplicationNode app_node(std::move(verb), std::move(arg), {1,1});
    
    std::ostringstream oss;
    app_node.print(oss, 0);
    
    std::string output = oss.str();
    EXPECT_NE(output.find("MonadicApplicationNode"), std::string::npos);
    EXPECT_NE(output.find("VerbNode"), std::string::npos);
    EXPECT_NE(output.find("NounLiteralNode"), std::string::npos);
    EXPECT_NE(output.find("5"), std::string::npos);
}

// Add tests for other node types and their functionality
