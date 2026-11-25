// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/device/cuda_kernel_table.hpp"

#include "contrib/gpu/src/codecs/float_deconstruct/decode_float_deconstruct_cuda.hpp"
#include "contrib/gpu/src/codecs/zstd/decode_zstd_cuda.hpp"
#include "openzl/common/wire_format.h"

namespace openzl::gpu {
namespace {

void setKernels(
        std::vector<DecodeKernel>& list,
        poly::span<const DecodeKernel> kernels)
{
    list.assign(kernels.begin(), kernels.end());
}

} // namespace

DecodeKernelTable cudaKernelTable(const CudaDeviceOptions& options)
{
    DecodeKernelTable table{};
    setKernels(
            table[ZL_StandardTransformID_float_deconstruct],
            floatDeconstructCudaKernels());
    setKernels(
            table[ZL_StandardTransformID_zstd],
            zstdCudaKernels(options.enableNvcomp));
    return table;
}

} // namespace openzl::gpu
