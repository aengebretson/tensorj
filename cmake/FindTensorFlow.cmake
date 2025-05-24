# FindTensorFlow.cmake
# Find TensorFlow C++ installation

# Handle different installation methods and OS differences
if(APPLE)
    list(APPEND TENSORFLOW_SEARCH_PATHS
        /opt/homebrew/lib
        /opt/homebrew/include
        /usr/local/opt/tensorflow
        ~/Library/Python/*/lib/python*/site-packages/tensorflow
    )
endif()

find_path(TENSORFLOW_INCLUDE_DIR
    NAMES 
        tensorflow/core/public/session.h
        tensorflow/c/c_api.h
    PATHS
        /usr/local/include
        /usr/include
        $ENV{TENSORFLOW_ROOT}/include
        ${TENSORFLOW_ROOT}/include
        ~/tensorflow
        ~/tensorflow/bazel-bin/tensorflow
        ${TENSORFLOW_SEARCH_PATHS}
    PATH_SUFFIXES
        include
        tensorflow/include
)

# Look for different library names and extensions
find_library(TENSORFLOW_CC_LIB
    NAMES 
        tensorflow_cc
        tensorflow
        libtensorflow_cc.so
        libtensorflow_cc.so.2
        libtensorflow.so
        libtensorflow.so.2
        libtensorflow_cc.dylib
        libtensorflow.dylib
    PATHS
        /usr/local/lib
        /usr/lib
        $ENV{TENSORFLOW_ROOT}/lib
        ${TENSORFLOW_ROOT}/lib
        ~/tensorflow/bazel-bin/tensorflow
        ${TENSORFLOW_SEARCH_PATHS}
    PATH_SUFFIXES
        lib
        lib64
        tensorflow
)

find_library(TENSORFLOW_FRAMEWORK_LIB
    NAMES 
        tensorflow_framework
        libtensorflow_framework.so
        libtensorflow_framework.so.2
        libtensorflow_framework.dylib
    PATHS
        /usr/local/lib
        /usr/lib
        $ENV{TENSORFLOW_ROOT}/lib
        ${TENSORFLOW_ROOT}/lib
        ~/tensorflow/bazel-bin/tensorflow
        ${TENSORFLOW_SEARCH_PATHS}
    PATH_SUFFIXES
        lib
        lib64
        tensorflow
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow
    REQUIRED_VARS TENSORFLOW_INCLUDE_DIR TENSORFLOW_CC_LIB TENSORFLOW_FRAMEWORK_LIB
)

if(TensorFlow_FOUND)
    set(TENSORFLOW_LIBRARIES ${TENSORFLOW_CC_LIB} ${TENSORFLOW_FRAMEWORK_LIB})
    set(TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIR})
    
    if(NOT TARGET TensorFlow::TensorFlow)
        add_library(TensorFlow::TensorFlow INTERFACE IMPORTED)
        set_target_properties(TensorFlow::TensorFlow PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOW_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${TENSORFLOW_LIBRARIES}"
        )
    endif()
endif()

mark_as_advanced(TENSORFLOW_INCLUDE_DIR TENSORFLOW_CC_LIB TENSORFLOW_FRAMEWORK_LIB)