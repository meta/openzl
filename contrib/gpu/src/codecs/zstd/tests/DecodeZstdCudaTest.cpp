// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "contrib/gpu/src/codecs/zstd/decode_zstd_cuda.hpp"
#include "contrib/gpu/src/device/cuda_kernel_table.hpp"
#include "contrib/gpu/src/device/nvcomp.hpp"
#include "openzl/common/wire_format.h"

namespace openzl::gpu {
namespace {

std::vector<std::string> zstdKernelNames(const CudaDeviceOptions& options)
{
    const DecodeKernelTable table = cudaKernelTable(options);
    std::vector<std::string> names;
    for (const DecodeKernel& kernel : table[ZL_StandardTransformID_zstd]) {
        names.emplace_back(kernel.name);
    }
    return names;
}

// Stand-ins with the signatures of nvcomp 5. Nothing calls them: no zstd
// kernel accepts a run yet.
int getTempSize(size_t, size_t, NvcompZstdDecompressOpts, size_t*, size_t)
{
    return 0;
}

int decompress(
        const void* const*,
        const size_t*,
        const size_t*,
        size_t*,
        size_t,
        void*,
        size_t,
        void* const*,
        NvcompZstdDecompressOpts,
        int*,
        void*)
{
    return 0;
}

// The alignments that nvcomp 5.0.0.6 reports for zstd.
int getAlignments(NvcompZstdDecompressOpts, NvcompAlignmentRequirements* out)
{
    *out = { .input = 1, .output = 1, .temp = 8 };
    return 0;
}

TEST(DecodeZstdCudaTest, ListsOnlyZstdCudaByDefault)
{
    EXPECT_EQ(zstdKernelNames({}), std::vector<std::string>{ "zstd.cuda" });
}

TEST(DecodeZstdCudaTest, ListsNvcompAfterZstdCudaWhenEnabled)
{
    // This fails if enabling nvcomp drops zstd.cuda or puts nvcomp first.
    NvcompFunctions nvcomp;
    nvcomp.zstd = NvcompZstdFunctions{
        .decompressGetTempSizeAsync      = getTempSize,
        .decompressAsync                 = decompress,
        .decompressGetRequiredAlignments = getAlignments,
    };
    registerNvcomp(nvcomp);
    CudaDeviceOptions options;
    options.enableNvcomp = true;

    const std::vector<std::string> expected{ "zstd.cuda", "zstd.nvcomp" };
    EXPECT_EQ(zstdKernelNames(options), expected);
}

} // namespace
} // namespace openzl::gpu
