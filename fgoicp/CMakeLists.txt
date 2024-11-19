set(headers
    "common.hpp"
    "fgoicp.hpp"
    "kdtree_adaptor.hpp"
    "registration.hpp"
    "../external/include/nanoflann.hpp"
    "../external/include/tinyply.h"
    "../external/include/toml.hpp"
    )

set(sources
    "common.cpp"
    "fgoicp.cu"
    "kdtree_adaptor.cu"
    "registration.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_library(fgoicp ${sources} ${headers})

target_include_directories(fgoicp PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/include
)

if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(fgoicp PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(fgoicp PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(fgoicp PROPERTIES CUDA_ARCHITECTURES native)
endif()
target_compile_options(fgoicp PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(fgoicp PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")
