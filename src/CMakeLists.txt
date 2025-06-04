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

    # Interpreter
    interpreter/interpreter.hpp
    interpreter/interpreter.cpp
    interpreter/tf_operations.hpp
    interpreter/tf_operations.cpp
    interpreter/tf_graph.hpp
    interpreter/tf_graph.cpp

    # Optional public umbrella header
    j_interpreter_lib.hpp
)

target_include_directories(j_interpreter_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} # So includes like "ast/ast_nodes.hpp" work
)

# Public compile features (if any, e.g., for C++20)
target_compile_features(j_interpreter_lib PUBLIC cxx_std_20)

# Link Abseil C++ libraries
if(ABSEIL_FOUND)
    message(STATUS "Linking Abseil C++ to j_interpreter_lib")
    target_link_libraries(j_interpreter_lib PUBLIC ${ABSEIL_LIBRARIES})
    target_include_directories(j_interpreter_lib PUBLIC ${ABSEIL_INCLUDE_DIRS})
    message(STATUS "Abseil C++ successfully linked to j_interpreter_lib")
else()
    message(WARNING "Abseil C++ not found")
endif()

# Link TensorFlow if available
if(TENSORFLOW_FOUND)
    message(STATUS "Linking TensorFlow to j_interpreter_lib")
    target_link_libraries(j_interpreter_lib PUBLIC TensorFlow::TensorFlow)
    target_include_directories(j_interpreter_lib PUBLIC 
        "${CMAKE_CURRENT_SOURCE_DIR}/../external/tensorflow"
    )
    # Set additional flags for TensorFlow
    target_compile_definitions(j_interpreter_lib PUBLIC HAS_TF_CC_API=1)
    message(STATUS "TensorFlow successfully linked to j_interpreter_lib")
else()
    message(STATUS "Building j_interpreter_lib without TensorFlow support")
    target_compile_definitions(j_interpreter_lib PUBLIC HAS_TF_CC_API=0)
endif()

# Note: TensorFlow protobuf libraries are included in the main TensorFlow C++ library