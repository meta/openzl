// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "contrib/gpu/src/common/codec_decode_execution.hpp"
#include "openzl/cpp/poly/Span.hpp"

namespace openzl::gpu {

/**
 * The CUDA runtime's float_deconstruct kernels.
 *
 * `float_deconstruct.bf16.cuda` runs `bf16DeconDecode`, which balances work
 * across runs. Its scratch holds one slice descriptor per slice. It needs
 * a 2-byte-aligned destination, which the layout guarantees, and takes inputs
 * at any offset, such as inputs stored in a frame. Each slice of a run uses
 * vector loads and stores when its addresses are aligned for them.
 */
poly::span<const DecodeKernel> floatDeconstructCudaKernels();

} // namespace openzl::gpu
