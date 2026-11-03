# Copyright (c) Meta Platforms, Inc. and affiliates.

# CMake picks the oldest architecture that nvcc supports when this is unset.
# The BF16 kernels need sm_80 or newer.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# The CUDA device calls cudaStreamGetDevice, which CUDA 12.8 added.
find_package(CUDAToolkit 12.8 REQUIRED)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Every source under src/ joins openzl_gpu, except sources in a tests/
# directory (they go into test_openzl_gpu) or a bench/ directory (each
# benchmark has its own main() and needs its own add_executable).
# RELATIVE keeps the checkout path out of the filters below.
file(GLOB_RECURSE gpu_all_sources CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")

set(gpu_sources ${gpu_all_sources})
list(FILTER gpu_sources EXCLUDE REGEX "/(tests|bench)/")
add_library(openzl_gpu ${gpu_sources})
# CMake links the static CUDA runtime into every target with .cu sources, so
# openzl_gpu must use the same one. Two runtimes in one binary do not share
# kernel registrations or device state.
target_link_libraries(openzl_gpu
    PUBLIC openzl openzl_cpp CUDA::cudart_static ${CMAKE_DL_LIBS})
apply_openzl_compile_options_to_target(openzl_gpu)

add_executable(bench_decode_bf16
    src/codecs/float_deconstruct/bench/bench_decode_bf16.cu
)
target_link_libraries(bench_decode_bf16 PRIVATE openzl_gpu)

if (OPENZL_BUILD_TESTS)
    set(gpu_test_sources ${gpu_all_sources})
    list(FILTER gpu_test_sources INCLUDE REGEX "/tests/")
    file(GLOB testkit_sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/testkit/*.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/testkit/tests/*.cpp")
    add_executable(test_openzl_gpu ${gpu_test_sources} ${testkit_sources})
    target_link_libraries(test_openzl_gpu
        PRIVATE openzl_gpu openzl_test_support GTest::gtest_main
    )
    apply_openzl_compile_options_to_target(test_openzl_gpu)
    gtest_discover_tests(test_openzl_gpu)
endif()
