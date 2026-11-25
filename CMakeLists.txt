# Copyright (c) Meta Platforms, Inc. and affiliates.

# Pick a recent version of CMake to simplify our build.
# TODO: Once we go open source we'll likely need to support older versions of cmake.
cmake_minimum_required(VERSION 3.20.2 FATAL_ERROR)

cmake_policy(SET CMP0077 NEW)

set(PACKAGE_NAME "openzl")
set(PACKAGE_DESCRIPTION "A novel data compression framework")
project(${PACKAGE_NAME} CXX C ASM)

# ---- Build Options ----
option(OPENZL_BUILD_ALL "Build all optional components" OFF)
option(OPENZL_BUILD_SHARED_LIBS "Build as shared library" OFF)
option(OPENZL_BUILD_TESTS "Build tests" ${OPENZL_BUILD_ALL})
option(OPENZL_BUILD_BENCHMARKS "Build benchmarks" ${OPENZL_BUILD_ALL})
# Apache Arrow build is broken due to missing thrift download.
# See https://github.com/apache/arrow/issues/46740.
# One that is resolved, this can default to ${OPENZL_BUILD_ALL}.
option(OPENZL_BUILD_PARQUET_TOOLS "Build with parquet tools" OFF)
option(OPENZL_BUILD_PYTHON_EXT "Build Python extension" ${OPENZL_BUILD_ALL})
option(OPENZL_BUILD_PYTHON_EXT_TESTS "Build Python extension tests" ${OPENZL_BUILD_TESTS})
option(OPENZL_ALLOW_INTROSPECTION "Allow introspection" ON)
option(OPENZL_INSTALL "Install OpenZL" ON)
option(OPENZL_CPP_INSTALL "Install C++ bindings" ${OPENZL_INSTALL})
option(OPENZL_BUILD_PYTHON_DEMO "Build openzl_demo Python extension" ${OPENZL_BUILD_ALL})

# Fine-grained control of build directories that can be disabled.
# NOTE: Disabling some but not all of these may break the build, use with caution.
option(OPENZL_BUILD_CPP "Builds the cpp subdirectory" ON)
option(OPENZL_BUILD_CUSTOM_PARSERS "Builds the custom_parsers subdirectory" ON)
option(OPENZL_BUILD_TOOLS "Builds the tools subdirectory" ON)
option(OPENZL_BUILD_CLI "Build the cli subdirectory" ON)
option(OPENZL_BUILD_EXAMPLES "Build the examples subdirectory" ON)
option(OPENZL_BUILD_WASM "Builds WASM binding (requires Emscripten)" OFF)
option(OPENZL_BUILD_GPU "Builds the CUDA GPU decoder in contrib/gpu (requires the CUDA Toolkit)" OFF)

set(OPENZL_SUPPORT_SHARED_LIBRARY "${OPENZL_BUILD_SHARED_LIBS}")

# ---- Emscripten ----
# OpenZL currently requires a 64-bit platform. 32-bit is unsupported,
# so Emscripten builds must use wasm64.
if(EMSCRIPTEN)
    add_compile_options(-sMEMORY64=1)
    add_link_options(-sMEMORY64=1 -sWASM_BIGINT=1)

    # Training throws across the binding and uses std::thread, so these flags
    # must apply to every object linked into the WASM artifact.
    if(OPENZL_BUILD_WASM)
        add_compile_options(-fexceptions -pthread)
        add_link_options(-fexceptions -pthread)
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- CMake Module Path ----
set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/build-scripts/cmake"
    # For in-fbsource release builds
    "${CMAKE_CURRENT_SOURCE_DIR}/../../opensource/fbcode_builder/CMake"
    # For in-fbsource dev builds
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../opensource/fbcode_builder/CMake"
    # For shipit-transformed builds
    "${CMAKE_CURRENT_SOURCE_DIR}/build-scripts/fbcode_builder/CMake"
    ${CMAKE_MODULE_PATH})

# ---- Package Information ----
if (NOT DEFINED PACKAGE_VERSION)
    # TODO: How do we set the package version based on our headers?
    set(PACKAGE_VERSION "0.3.0")
endif()
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_TARNAME "${PACKAGE_NAME}-${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "https://github.com/facebook/openzl/issues")

# ---- Directory Settings ----
include(GNUInstallDirs)
set(OPENZL_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE STRING "Header install dir")
set(OPENZL_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}" CACHE STRING "Library install dir")
set(OPENZL_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}" CACHE STRING "Binary install dir")
set(OPENZL_INSTALL_CMAKEDIR "${OPENZL_INSTALL_LIBDIR}/cmake/openzl" CACHE STRING "CMake config dir")

# ---- Compiler Settings ----
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Platform Checks ----
MESSAGE(STATUS "current platform: ${CMAKE_SYSTEM_NAME}")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT DEFINED IS_X86_64_ARCH AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64|AMD64")
  set(IS_X86_64_ARCH TRUE)
else()
  set(IS_X86_64_ARCH FALSE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # Check target architecture
  if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "OpenZL requires a 64bit target architecture.")
  endif()

  # Check compiler compatibility
  if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    # MSVC has limited C11 support - warn users about potential issues
    if (MSVC_VERSION LESS 1910)
      message(
        FATAL_ERROR
        "This build script only supports building OpenZL on 64-bit Windows with "
        "at least Visual Studio 2017. "
        "MSVC version '${MSVC_VERSION}' is not supported."
      )
    endif()

    message(WARNING
      "MSVC has limited C11 support and may produce C2099 errors. "
      "For better C11 compatibility, consider using clang-cl instead. "
      "To use clang-cl with Visual Studio generator, use: cmake -G \"Visual Studio 17 2022\" -T ClangCL")
  endif()
endif()

set(TOP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(OPENZL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(OPENZL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")

include(OpenZLModeFlags)
include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# Select appropriate compiler configuration
set(ZL_ALLOW_INTROSPECTION ${OPENZL_ALLOW_INTROSPECTION})

if (WIN32 AND (MSVC OR
              (CMAKE_C_COMPILER_ID STREQUAL "Clang" AND CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC") OR
              CMAKE_VS_PLATFORM_TOOLSET STREQUAL "ClangCL" OR
              CMAKE_GENERATOR_TOOLSET STREQUAL "ClangCL"))
    include(OpenZLCompilerMSVC)
else()
    include(OpenZLCompilerUnix)
endif()
include(OpenZLFunctions)

include(openzl-deps)
include(OpenZLConfigChecks)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/build-scripts/cmake/zl_config.h.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/include/openzl/zl_config.h
)
set(CONFIG "${CMAKE_CURRENT_BINARY_DIR}/include/openzl/zl_config.h")

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${OPENZL_SRC_DIR}/*.c")
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${OPENZL_SRC_DIR}/*.h")
file(GLOB_RECURSE public_headers CONFIGURE_DEPENDS "${OPENZL_INCLUDE_DIR}/*.h")

if (${ZL_HAVE_X86_64_ASM})
    set_property(
        SOURCE
        ${OPENZL_SRC_DIR}/openzl/fse/decompress/huf_decompress_amd64.S
        APPEND PROPERTY COMPILE_OPTIONS "-x" "assembler-with-cpp"
    )
    list(APPEND sources ${OPENZL_SRC_DIR}/openzl/fse/decompress/huf_decompress_amd64.S)
endif()

list(APPEND openzl_files
    ${sources}
    ${headers}
)

add_library(openzl
    ${openzl_files}
    "${CONFIG}")

# We need PIC to build shared libraries. For now just enable it for all builds.
# Later we could build both PIC & non-PIC versions of the library.
set_property(TARGET openzl PROPERTY POSITION_INDEPENDENT_CODE ON)

# target_compile_options(openzl PUBLIC "-Wno-error=stringop-overflow")

source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    PREFIX "openzl"
    FILES ${sources} ${headers} ${public_headers})
source_group(
    TREE ${CMAKE_CURRENT_BINARY_DIR}
    PREFIX "openzl"
    FILES "${CONFIG}")

# TODO: pkgconfig

target_include_directories(openzl_deps
    BEFORE
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_include_directories(openzl_deps
    INTERFACE
        $<INSTALL_INTERFACE:include>
)

target_include_directories(openzl
    PUBLIC
        $<TARGET_PROPERTY:openzl_deps,INTERFACE_INCLUDE_DIRECTORIES>
)

target_include_directories(openzl
    PUBLIC
        $<TARGET_PROPERTY:openzl_deps,INTERFACE_COMPILE_DEFINITIONS>
)

set_property(TARGET openzl PROPERTY VERSION ${PACKAGE_VERSION})
apply_openzl_compile_options_to_target(openzl)

target_link_libraries(openzl PUBLIC openzl_deps)

# Shared profile graph builders used by both the CLI and wasm. Defined here
# rather than under cli/ because wasm needs them even when the CLI is off.
# Native WASM binding tests also use these builders.
if (OPENZL_BUILD_CLI
    OR OPENZL_BUILD_WASM
    OR (OPENZL_BUILD_TESTS AND NOT EMSCRIPTEN))
    if (NOT OPENZL_BUILD_CPP OR NOT OPENZL_BUILD_CUSTOM_PARSERS)
        message(FATAL_ERROR
            "Shared profile graphs for the CLI, WASM, or native WASM binding tests "
            "require OPENZL_BUILD_CPP=ON and OPENZL_BUILD_CUSTOM_PARSERS=ON.")
    endif()

    add_library(openzl_profile_graphs cli/utils/profile_graphs.cpp)
    target_include_directories(openzl_profile_graphs PUBLIC ${PROJECT_SOURCE_DIR})
    target_link_libraries(
        openzl_profile_graphs
        PUBLIC
            openzl
            openzl_cpp
            csv_parser
            parquet_graph
            shared_components
            pytorch_model_parser)
    apply_openzl_compile_options_to_target(openzl_profile_graphs)
endif()

# order matters. dependencies first
if (OPENZL_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    FetchContent_Declare(
        googletest # v1.17.0
        URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    )
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    enable_testing()

    include(CTest)
endif()

if (OPENZL_BUILD_CPP)
    add_subdirectory(cpp)
endif()

# conditional builds controlled in tools at finer granularity
add_subdirectory(tools)

if (OPENZL_BUILD_CUSTOM_PARSERS)
    add_subdirectory(custom_parsers)
endif()
if (OPENZL_BUILD_CLI)
    add_subdirectory(cli)
endif()
if (OPENZL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (OPENZL_BUILD_PYTHON_EXT)
    add_subdirectory(py)
endif()

if (OPENZL_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if (OPENZL_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if (OPENZL_BUILD_PYTHON_DEMO)
    add_subdirectory(contrib/openzl-demo)
endif()

if (OPENZL_BUILD_GPU)
    add_subdirectory(contrib/gpu)
endif()

if (OPENZL_INSTALL)
    set(OPENZL_INSTALL_TARGETS openzl openzl_deps)

    if (OPENZL_CPP_INSTALL)
        set(OPENZL_INSTALL_TARGETS ${OPENZL_INSTALL_TARGETS} openzl_cpp)
    endif()

    install(TARGETS ${OPENZL_INSTALL_TARGETS}
        EXPORT openzl
        RUNTIME DESTINATION ${OPENZL_INSTALL_BINDIR}
        LIBRARY DESTINATION ${OPENZL_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${OPENZL_INSTALL_LIBDIR})
    # Install the public headers
    install(
        DIRECTORY ${OPENZL_INCLUDE_DIR}/openzl
        DESTINATION ${OPENZL_INSTALL_INCLUDEDIR}
        COMPONENT dev)
    # Install headers generated by CMake (currently only zl_config.h)
    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/openzl
        DESTINATION ${OPENZL_INSTALL_INCLUDEDIR}
        COMPONENT dev)
    # Install C++ public headers
    if (OPENZL_CPP_INSTALL)
        install(
            DIRECTORY ${PROJECT_SOURCE_DIR}/cpp/include/openzl
            DESTINATION ${OPENZL_INSTALL_INCLUDEDIR}
            COMPONENT dev)
    endif()

    # Generate the openzl-config.cmake file for installation so that
    # downstream projects that use openzl can easily depend on it in their CMake
    # files using "find_package(openzl CONFIG)"
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        build-scripts/cmake/openzl-config.cmake.in
        openzl-config.cmake
        INSTALL_DESTINATION ${OPENZL_INSTALL_CMAKEDIR}
        PATH_VARS
            OPENZL_INSTALL_INCLUDEDIR
            OPENZL_INSTALL_CMAKEDIR
    )
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/openzl-config.cmake
        DESTINATION ${OPENZL_INSTALL_CMAKEDIR}
        COMPONENT dev
    )
    install(
        EXPORT openzl
        DESTINATION ${OPENZL_INSTALL_CMAKEDIR}
        NAMESPACE OpenZL::
        FILE openzl-targets.cmake
        COMPONENT dev
    )

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/openzl.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/openzl.pc"
        @ONLY
    )
    
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/openzl-cpp.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/openzl-cpp.pc"
        @ONLY
    )

    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/openzl.pc" "${CMAKE_CURRENT_BINARY_DIR}/openzl-cpp.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )

endif()

# ---- Trailing User Messages ----
if (OPENZL_ALLOW_INTROSPECTION)
    message(WARNING "OpenZL build with introspection enabled. Disable by re-configuring CMake with -DOPENZL_ALLOW_INTROSPECTION=OFF. For more info, see https://facebook.github.io/openzl/getting-started/advanced-build#introspection")
endif()
