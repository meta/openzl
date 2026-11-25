// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "contrib/gpu/src/common/codec_decode_execution.hpp"
#include "openzl/cpp/poly/Span.hpp"

namespace openzl::gpu {

/**
 * The CUDA runtime's zstd kernels, in order of preference.
 *
 * - `zstd.cuda` is this library's own decoder.
 * - `zstd.nvcomp` uses nvcomp's batched zstd decoder. It is listed only if
 *   @p enableNvcomp is set and the process's nvcomp (`nvcompFunctions`) has
 *   zstd functions that it can use.
 *
 * `zstd.cuda` stays first, so a decode does not depend on the libraries that a
 * machine has. Enabling nvcomp adds an alternative, and never reorders the
 * list.
 *
 * Neither kernel is implemented yet, so neither accepts a run.
 */
poly::span<const DecodeKernel> zstdCudaKernels(bool enableNvcomp);

} // namespace openzl::gpu
