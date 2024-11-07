# Minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name and language
#project(fast-go-icp LANGUAGES CUDA CXX)
project(fast-go-icp LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the source and include directories
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")

# Add the external directory to the include path
include_directories(${EXTERNAL_DIR})

# Find all source files
file(GLOB_RECURSE SOURCES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE HEADERS "${SRC_DIR}/*.hpp")

# Add the executable target
add_executable(fgoicp ${SOURCES})

# Specify include directories
target_include_directories(fgoicp
    PRIVATE ${SRC_DIR}
    PRIVATE ${EXTERNAL_DIR}
)
