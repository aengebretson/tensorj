# External dependencies CMakeLists.txt

# Add Abseil C++ subproject
message(STATUS "Adding Abseil C++ from external/abseil-cpp")

# Set Abseil options
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Use same C++ standard as parent project")
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "Disable Abseil install to avoid conflicts")

# Add Abseil subdirectory
add_subdirectory(abseil-cpp)

# Create an alias for easier linking
add_library(abseil::abseil INTERFACE IMPORTED)

# The main Abseil libraries your TensorFlow code might need
# Start with essential ones that are definitely available
set(ABSEIL_LIBRARIES
    absl::base
    absl::core_headers
    absl::strings
    absl::status
    absl::memory
    absl::utility
)

# Set variables for parent project
set(ABSEIL_FOUND TRUE PARENT_SCOPE)
set(ABSEIL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp" PARENT_SCOPE)
set(ABSEIL_LIBRARIES ${ABSEIL_LIBRARIES} PARENT_SCOPE)

message(STATUS "Abseil C++ configured successfully")
message(STATUS "  Abseil include directory: ${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp")
message(STATUS "  Abseil libraries: ${ABSEIL_LIBRARIES}")

# Note: TensorFlow protobuf files are built with Bazel, not CMake
# Use: cd external/tensorflow && bazel build //tensorflow:tensorflow_cc
