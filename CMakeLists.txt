cmake_minimum_required(VERSION 3.15)

# Fix the policy warning
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(J_Interpreter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add cmake directory to module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Add external dependencies first (including Abseil)
add_subdirectory(external)

# Option to enable/disable TensorFlow
option(USE_TENSORFLOW "Enable TensorFlow support" OFF)

if(USE_TENSORFLOW)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TENSORFLOW tensorflow-cc)
        if(TENSORFLOW_FOUND)
            message(STATUS "Found TensorFlow via pkg-config")
            add_compile_definitions(HAS_TF_CC_API=1)
            message(STATUS "TensorFlow C++ API enabled (HAS_TF_CC_API=1)")
        else()
            message(STATUS "TensorFlow not found via pkg-config, using stub implementation")
            add_compile_definitions(HAS_TF_CC_API=0)
        endif()
    else()
        message(STATUS "pkg-config not found, using stub implementation")
        add_compile_definitions(HAS_TF_CC_API=0)
    endif()
else()
    message(STATUS "TensorFlow support disabled, using stub implementation")
    set(TENSORFLOW_FOUND FALSE)
    add_compile_definitions(HAS_TF_CC_API=0)
endif()

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

