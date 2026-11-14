# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.20.2 FATAL_ERROR)
project(openzl_playground_cli CXX)

get_filename_component(OPENZL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)

# The desktop bundle only ships zli; skip the other standalone deliverables.
set(OPENZL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_PYTHON_EXT OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_PYTHON_DEMO OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_WASM OFF CACHE BOOL "" FORCE)
set(OPENZL_INSTALL OFF CACHE BOOL "" FORCE)
set(OPENZL_BUILD_CLI ON CACHE BOOL "" FORCE)

add_subdirectory("${OPENZL_ROOT}" openzl)

# Single- and multi-config generators both place zli where Electron and
# electron-builder look for it.
set_target_properties(
  zli
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/Release")
if(NOT MSVC)
  target_link_options(zli PRIVATE "$<$<CONFIG:Release>:-s>")
endif()
