add_executable(j_interpreter_tests
    main_test.cpp
    # Add your test files here
    lexer/test_lexer.cpp
    ast/test_ast.cpp
    parser/test_parser.cpp
    interpreter/test_interpreter.cpp
    interpreter/test_jtensor.cpp
    interpreter/test_tensor_operations.cpp
)

target_link_libraries(j_interpreter_tests PRIVATE
    j_interpreter_lib  # Link against your library
    GTest::gtest_main  # Google Test with main
    # GTest::gmock      # If using Google Mock
)

include(GoogleTest)
gtest_discover_tests(j_interpreter_tests)
