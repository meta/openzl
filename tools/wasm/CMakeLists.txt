# Copyright (c) Meta Platforms, Inc. and affiliates.
# WASM binding layer - C++ with a C ABI, JS glue in js/.
# Native builds compile the binding as a library for CTest. Emscripten builds
# compile the same source directly into the WASM executable below.
set(OPENZL_WASM_SOURCE src/openzl_wasm.cpp)

if(NOT EMSCRIPTEN)
  # Native test - runs under both `buck test` and, with
  # -DOPENZL_BUILD_TESTS=ON, `ctest -R WasmBinding`. gtest_discover_tests
  # registers the gtest names, not the executable name, so the ctest filter
  # matches the suites.
  if(OPENZL_BUILD_TESTS)
    add_library(openzl_wasm_lib STATIC ${OPENZL_WASM_SOURCE})

    target_include_directories(openzl_wasm_lib PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../..
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    )

    target_link_libraries(openzl_wasm_lib
      PRIVATE
        openzl
        openzl_cpp
        openzl_profile_graphs
        tools_training
        custom_parsers
    )

    # Both live in subdirectories that can be switched off, and an unknown name
    # in the link list above is handed to the linker verbatim, so a disabled
    # subdirectory would otherwise surface as `cannot find -lcustom_parsers`
    # at link. Naming them here makes it a configure-time error instead.
    add_dependencies(openzl_wasm_lib
      openzl
      tools_training
      custom_parsers
    )
    apply_openzl_compile_options_to_target(openzl_wasm_lib)

    add_executable(test_wasm_binding tests/test_wasm_binding.cpp)
    target_include_directories(test_wasm_binding PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../..
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    )
    target_link_libraries(test_wasm_binding
      PRIVATE
        openzl_wasm_lib
        openzl
        custom_parsers
        GTest::gtest_main
    )
    if(OPENZL_BUILD_CLI)
      # Enable additional test to verify if wasm and zli are out of sync only when cli is also enabled
      target_sources(test_wasm_binding PRIVATE tests/test_wasm_profiles.cpp)
      target_link_libraries(test_wasm_binding PRIVATE utils)
    endif()
    apply_openzl_compile_options_to_target(test_wasm_binding)
    gtest_discover_tests(test_wasm_binding)
  endif()
endif()

if(NOT DEFINED OPENZL_BUILD_WASM OR NOT OPENZL_BUILD_WASM)
  return()
endif()

if(NOT EMSCRIPTEN)
  message(WARNING
    "OPENZL_BUILD_WASM=ON but Emscripten not detected, skipping WASM artifact")
  return()
endif()

add_executable(openzl_wasm ${OPENZL_WASM_SOURCE})

target_include_directories(openzl_wasm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

target_link_libraries(openzl_wasm
  PRIVATE
    openzl
    openzl_cpp
    openzl_profile_graphs
    tools_training
    custom_parsers
)

add_dependencies(openzl_wasm
  openzl
  tools_training
  custom_parsers
)

# Emscripten pre-spawns its worker pool at startup. Creating a thread beyond
# the pool has to round-trip through the main thread's event loop, which
# deadlocks when that thread is blocked, so the binding never asks for more
# than this and the pool is sized up front.
set(OPENZL_WASM_PTHREAD_POOL_SIZE 4 CACHE STRING
  "Workers pre-allocated for the WASM trainer thread pool")

# -sMEMORY64=1 / -sWASM_BIGINT=1 / -fexceptions / -pthread are set globally in
# the top-level file so every object, including the core library and the
# training tools, uses the wasm64 ABI, can unwind, and can spawn threads.
target_compile_options(openzl_wasm PRIVATE -O3)
target_compile_definitions(openzl_wasm PRIVATE
  OPENZL_WASM_TRAIN_MAX_THREADS=${OPENZL_WASM_PTHREAD_POOL_SIZE})
# The `-g` below is for speed, not debugging: emcc reads it off the link line
# and then skips binaryen's post-link pass, which costs this workload about a
# quarter of its compression throughput. DWARF in the objects plays no part --
# the same link with and without it produces a byte-identical code section --
# and whoever ships the artifact strips the result anyway, see the docs deploy.
# Stated here rather than inherited from OpenZLCompilerUnix.cmake, which adds
# `-g` to Release for native profiling: removing it there is a reasonable
# cleanup and must not silently slow this module down.
target_link_options(openzl_wasm PRIVATE
  -O3                        # optimise the linked wasm
  -g                         # speed, not debugging; see above
  -sWASM=1                   # emit wasm, not asm.js
  -sALLOW_MEMORY_GROWTH=1    # heap can grow; JS must re-read mod.HEAPU8
  -sINITIAL_MEMORY=32MB      # starting heap size
  # Growth otherwise stops at the 2GB default, which a large sample plus the
  # trainers' per-candidate working sets can reach. Raising the ceiling costs
  # nothing up front: it caps growth rather than reserving.  Only wasm64 can
  # address beyond 4GB, which the global -sMEMORY64=1 provides.
  -sMAXIMUM_MEMORY=16GB
  # Reserve extra stack space for ACE. If a future candidate goes deeper than
  # 4 MB, training will abort with a stack overflow, and the fix
  # is to raise these numbers.
  -sSTACK_SIZE=4MB
  -sDEFAULT_PTHREAD_STACK_SIZE=4MB
  -sMODULARIZE=1             # export a factory function, not globals
  -sEXPORT_ES6=1             # emit an ES module
  -sEXPORT_NAME=openzlModule # name of the factory; js/wasm_api.js imports this
  -sPTHREAD_POOL_SIZE=${OPENZL_WASM_PTHREAD_POOL_SIZE}
  # Nothing to list: every entry point, the allocator included, exports itself
  # via EMSCRIPTEN_KEEPALIVE. The empty value overrides the _main default.
  -sEXPORTED_FUNCTIONS=
  # JS helpers to keep. HEAPU64 reads the 8-byte out-params of a wasm64 build,
  # HEAPF64 the benchmark timings; UTF8ToString reads the error message.
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU64,HEAPF64,UTF8ToString
  -sENVIRONMENT=web,worker,node
)

set_target_properties(openzl_wasm PROPERTIES OUTPUT_NAME "openzl" SUFFIX ".js")

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/openzl.js
  ${CMAKE_CURRENT_BINARY_DIR}/openzl.wasm
  DESTINATION wasm
  OPTIONAL
)
