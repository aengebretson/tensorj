# FindTensorFlow.cmake - Use TensorFlow C++ libraries

# Guard against multiple inclusions
if(TARGET TensorFlow::TensorFlow)
    return()
endif()

set(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/tensorflow")

message(STATUS "Using Bazel-built TensorFlow C++ libraries")

# Clear any cached values to force fresh search
unset(TENSORFLOW_CC_LIB CACHE)
unset(TENSORFLOW_FRAMEWORK_LIB CACHE)

# Look for the libraries in the correct location: bazel-bin/tensorflow
find_library(TENSORFLOW_CC_LIB
    NAMES libtensorflow_cc.so libtensorflow_cc.dylib
    PATHS 
        "${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow"
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
)

find_library(TENSORFLOW_FRAMEWORK_LIB
    NAMES libtensorflow_framework.so libtensorflow_framework.dylib
    PATHS 
        "${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow"
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
)

# Debug: Print what we found
message(STATUS "Searching in: ${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow")
message(STATUS "Found CC lib: ${TENSORFLOW_CC_LIB}")
message(STATUS "Found Framework lib: ${TENSORFLOW_FRAMEWORK_LIB}")

if(TENSORFLOW_CC_LIB AND TENSORFLOW_FRAMEWORK_LIB)
    message(STATUS "Found TensorFlow C++ libraries")
    message(STATUS "  C++ API: ${TENSORFLOW_CC_LIB}")
    message(STATUS "  Framework: ${TENSORFLOW_FRAMEWORK_LIB}")
    
    set(TensorFlow_FOUND TRUE)
    
    # Create imported targets
    if(NOT TARGET tensorflow_cc)
        add_library(tensorflow_cc SHARED IMPORTED)
        set_target_properties(tensorflow_cc PROPERTIES
            IMPORTED_LOCATION "${TENSORFLOW_CC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOW_SOURCE_DIR}"
        )
    endif()
    
    if(NOT TARGET tensorflow_framework)
        add_library(tensorflow_framework SHARED IMPORTED)
        set_target_properties(tensorflow_framework PROPERTIES
            IMPORTED_LOCATION "${TENSORFLOW_FRAMEWORK_LIB}"
        )
    endif()
    
    # Create the main target
    if(NOT TARGET TensorFlow::TensorFlow)
        add_library(TensorFlow::TensorFlow INTERFACE IMPORTED)
        set_target_properties(TensorFlow::TensorFlow PROPERTIES
            INTERFACE_LINK_LIBRARIES "tensorflow_cc;tensorflow_framework"
            INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOW_SOURCE_DIR}"
        )
    endif()
    
    message(STATUS "TensorFlow targets created successfully")
    
else()
    message(STATUS "TensorFlow C++ libraries not found in bazel-bin/tensorflow")
    set(TensorFlow_FOUND FALSE)
endif()