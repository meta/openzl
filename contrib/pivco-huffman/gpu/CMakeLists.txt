# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.26)
project(pivco_gpu_contrib LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# A standalone build for the existing contrib tests and benchmark. The main
# OpenZL build and its production codec remain unchanged.
get_filename_component(OPENZL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
option(PIVCO_GPU_BUILD_TESTS "Build the PivCo contrib tests" ON)

set(OPENZL_BUILD_ALL OFF CACHE BOOL "")
set(OPENZL_BUILD_TESTS OFF CACHE BOOL "")
set(OPENZL_BUILD_CPP OFF CACHE BOOL "")
set(OPENZL_BUILD_CUSTOM_PARSERS OFF CACHE BOOL "")
set(OPENZL_BUILD_TOOLS OFF CACHE BOOL "")
set(OPENZL_BUILD_CLI OFF CACHE BOOL "")
set(OPENZL_BUILD_EXAMPLES OFF CACHE BOOL "")
set(OPENZL_INSTALL OFF CACHE BOOL "")
add_subdirectory("${OPENZL_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/openzl" EXCLUDE_FROM_ALL)

find_package(CUDAToolkit REQUIRED)
add_library(pivco_gpu STATIC pivco_gpu.cu pivco_gpu_context.cpp pivco_block_index.cpp)
target_compile_options(pivco_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(pivco_gpu PUBLIC openzl CUDA::cudart)
if(MSVC)
    target_compile_options(pivco_gpu PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    target_compile_options(pivco_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

option(PIVCO_GPU_BUILD_BENCHMARKS "Build the standalone GPU benchmark" ON)
if(PIVCO_GPU_BUILD_BENCHMARKS)
    add_executable(pivco_gpu_bench PivCoGpuBench.cpp)
    target_link_libraries(pivco_gpu_bench PRIVATE pivco_gpu)
endif()

if(PIVCO_GPU_BUILD_TESTS)
    # Match the version and archive identity used by the root test build.
    include(FetchContent)
    set(INSTALL_GTEST OFF)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    add_executable(pivco_gpu_test PivCoGpuTest.cpp)
    target_link_libraries(pivco_gpu_test PRIVATE pivco_gpu GTest::gtest_main)
    add_executable(pivco_block_index_test PivCoBlockIndexTest.cpp)
    target_link_libraries(pivco_block_index_test PRIVATE pivco_gpu GTest::gtest_main)
    foreach(mode IN ITEMS 0 1)
        add_test(NAME pivco_gpu_test_mode_${mode} COMMAND pivco_gpu_test)
        set_tests_properties(pivco_gpu_test_mode_${mode} PROPERTIES
            ENVIRONMENT "PIVCO_CHUNK_TD=${mode}")
    endforeach()
    add_test(NAME pivco_block_index_test COMMAND pivco_block_index_test)
endif()
