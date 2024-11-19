# Minimum required CMake version
cmake_minimum_required(VERSION 3.18)

#set(CMAKE_CUDA_ARCHITECTURES 89)
# Specify the path to the CUDA compiler
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Project name and language
project(fast-go-icp LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#set(CUDA_NVCC_FLAGS_DEBUG "-G -g")

# Find the CUDAToolkit package
find_package(CUDAToolkit REQUIRED)

# Set the source and include directories
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external/include")

# Add the external directory to the include path
include_directories(${EXTERNAL_DIR})

# Find all .cpp and .cu source files
file(GLOB_RECURSE CPP_SOURCES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${SRC_DIR}/*.cu")
file(GLOB_RECURSE HEADERS "${SRC_DIR}/*.hpp")

# Add the executable target with both .cpp and .cu files
add_executable(fgoicp ${CPP_SOURCES} ${CU_SOURCES} ${HEADERS})

# Specify include directories
target_include_directories(fgoicp
    PRIVATE ${SRC_DIR}
    PRIVATE ${EXTERNAL_DIR}
)

# Set CUDA architecture
set_target_properties(fgoicp PROPERTIES
    CUDA_ARCHITECTURES 89
)

# Link CUDA runtime library
target_link_libraries(fgoicp
    PRIVATE cuda CUDA::cudart
)
