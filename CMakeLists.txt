cmake_minimum_required(VERSION 3.15)
project(J_Interpreter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20) # J's complexity might benefit from C++20 features
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(TensorFlow)

# --- Google Test ---
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main # Or a specific release tag e.g., release-1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(tests)

message(STATUS "CMake CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
