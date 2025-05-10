add_executable(j_repl main.cpp)

target_link_libraries(j_repl PRIVATE j_interpreter_lib)

# If j_interpreter_lib uses TensorFlow, j_repl might also need to link
# or have include paths if it directly interacts with TF types.
# if(TARGET TensorFlow::TensorFlow)
#   target_link_libraries(j_repl PRIVATE TensorFlow::TensorFlow)
# endif()
