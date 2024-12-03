set(headers
    "common.hpp"
    "fgoicp.hpp"
    "icp3d.hpp"
    "registration.hpp"
    )

set(sources
    "common.cpp"
    "fgoicp.cu"
    "icp3d.cu"
    "registration.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(fgoicp ${sources} ${headers})

target_include_directories(fgoicp PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(fgoicp PROPERTIES CUDA_ARCHITECTURES OFF)
#elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    #set_target_properties(fgoicp PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(fgoicp PROPERTIES CUDA_ARCHITECTURES all-major)
endif()

# Debug options for host code (C++)
target_compile_options(fgoicp PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g>")

# Release options for host code (C++)
target_compile_options(fgoicp PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>")

# Debug options for CUDA code
target_compile_options(fgoicp PRIVATE 
    "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--diag-suppress=20012;-lineinfo;-src-in-ptx>"
)

# Release options for CUDA code
target_compile_options(fgoicp PRIVATE 
    "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--diag-suppress=20012;-lineinfo;-src-in-ptx>"
)