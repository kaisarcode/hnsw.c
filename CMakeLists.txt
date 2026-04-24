cmake_minimum_required(VERSION 3.14)
project(hnsw C)

set(CMAKE_C_STANDARD 11)

# Output binary to the root directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Performance optimizations
option(HNSW_NATIVE "Optimize for native CPU architecture" OFF)

if(NOT MSVC)
    add_compile_options(-O3 -fno-finite-math-only)
    if(HNSW_NATIVE)
        add_compile_options(-march=native)
    endif()
    add_definitions(-D_GNU_SOURCE)
else()
    add_compile_options(/O2)
endif()

# Main executable
add_executable(hnsw 
    hnsw.c 
    libhnsw.c
)

# Include directories
target_include_directories(hnsw PRIVATE .)

# System libraries
if(UNIX)
    target_link_libraries(hnsw pthread m)
endif()
