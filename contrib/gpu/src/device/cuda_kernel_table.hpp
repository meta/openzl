// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "contrib/gpu/src/common/codec_decode_execution.hpp"
#include "contrib/gpu/src/device/cuda_device.hpp"

namespace openzl::gpu {

/// The CUDA runtime's kernels for each standard codec.
DecodeKernelTable cudaKernelTable(const CudaDeviceOptions& options);

} // namespace openzl::gpu
