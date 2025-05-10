# Create a library for the J interpreter components
add_library(j_interpreter_lib)

# Add sources from subdirectories
# We'll use target_sources to add files from subdirectories
# or define them as separate object libraries.
# For simplicity here, we'll collect sources.

target_sources(j_interpreter_lib PUBLIC
    # common
    common/source_location.hpp # Header-only for now

    # AST
    ast/ast_nodes.hpp
    ast/ast_nodes.cpp

    # Lexer
    lexer/token.hpp
    lexer/token.cpp
    lexer/lexer.hpp
    lexer/lexer.cpp

    # Parser
    parser/parser.hpp
    parser/parser.cpp

    # Interpreter (future use)
    interpreter/interpreter.hpp
    interpreter/interpreter.cpp

    # Optional public umbrella header
    j_interpreter_lib.hpp
)

target_include_directories(j_interpreter_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} # So includes like "ast/ast_nodes.hpp" work
)

# Public compile features (if any, e.g., for C++20)
target_compile_features(j_interpreter_lib PUBLIC cxx_std_20)

# Link TensorFlow if/when you add it
# find_package(TensorFlow REQUIRED)
# target_link_libraries(j_interpreter_lib PUBLIC TensorFlow::TensorFlow)
