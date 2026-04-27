## CMakeLists.txt
## Summary: Build system for hnsw — HNSW vector search library.
##
## Author:  KaisarCode
## Website: https://kaisarcode.com
## License: https://www.gnu.org/licenses/gpl-3.0.html

cmake_minimum_required(VERSION 3.14)
project(hnsw.c C)

set(CMAKE_C_STANDARD 11)

## ── Compiler flags ────────────────────────────────────────────────────────────

if(NOT MSVC)
    add_compile_options(-O3 -Wall -Wextra -Werror -fno-finite-math-only)
    add_definitions(-D_GNU_SOURCE)
else()
    add_compile_options(/O2 /W3 /WX)
endif()

## ── Include paths ─────────────────────────────────────────────────────────────

include_directories(src)

## ── Library targets ───────────────────────────────────────────────────────────

set(LIB_SOURCES src/libhnsw.c)

add_library(hnsw_static STATIC ${LIB_SOURCES})
set_target_properties(hnsw_static PROPERTIES OUTPUT_NAME "hnsw")
target_include_directories(hnsw_static PRIVATE src)

if(UNIX AND NOT ANDROID)
    target_link_libraries(hnsw_static pthread m)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm|mips")
        target_link_libraries(hnsw_static atomic)
    endif()
elseif(ANDROID)
    target_link_libraries(hnsw_static m)
endif()

add_library(hnsw_shared SHARED ${LIB_SOURCES})
set_target_properties(hnsw_shared PROPERTIES OUTPUT_NAME "hnsw")
target_include_directories(hnsw_shared PRIVATE src)

if(UNIX AND NOT ANDROID)
    target_link_libraries(hnsw_shared pthread m)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm|mips")
        target_link_libraries(hnsw_shared atomic)
    endif()
elseif(ANDROID)
    target_link_libraries(hnsw_shared m)
endif()

## ── Executable ────────────────────────────────────────────────────────────────

add_executable(hnsw src/hnsw.c)
target_include_directories(hnsw PRIVATE src)
target_link_libraries(hnsw hnsw_static)

if(UNIX AND NOT ANDROID)
    target_link_libraries(hnsw pthread m)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm|mips")
        target_link_libraries(hnsw atomic)
    endif()
elseif(ANDROID)
    target_link_libraries(hnsw m)
endif()
