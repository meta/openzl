# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.20.2 FATAL_ERROR)
project(openzl_playground_cli CXX)

get_filename_component(OPENZL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)

# The desktop bundle only ships zli and zstd-benchmark; skip the other
# standalone deliverables.
set(OPENZL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_PYTHON_EXT OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_PYTHON_DEMO OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_WASM OFF CACHE BOOL "" FORCE)
set(OPENZL_INSTALL OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_CLI ON CACHE BOOL "" FORCE)

add_subdirectory("${OPENZL_ROOT}" openzl)

# zstd rows are measured by the same kind of native loop as zli benchmark,
# against the zstd that OpenZL vendors.
add_executable(zstd-benchmark zstd-benchmark.cpp)
target_link_libraries(zstd-benchmark PRIVATE libzstd_static)
target_compile_features(zstd-benchmark PRIVATE cxx_std_17)
# libzstd_static inherits OpenZL's static CRT, which this directory does not.
set_target_properties(
  zstd-benchmark PROPERTIES MSVC_RUNTIME_LIBRARY
                            "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Single- and multi-config generators both place the binaries where Electron
# and electron-builder look for them.
foreach(target zli zstd-benchmark)
  set_target_properties(
    ${target}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Release"
      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/Release")
  if(NOT MSVC)
    target_link_options(${target} PRIVATE "$<$<CONFIG:Release>:-s>")
  endif()
endforeach()
