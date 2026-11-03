// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/codecs/zstd/decode_zstd_cuda.hpp"

#include "contrib/gpu/src/device/nvcomp.hpp"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/utils.h"

namespace openzl::gpu {
namespace {

bool acceptsNoRunYet(const CodecRunInfo&)
{
    return false;
}

ZL_Report decodeWithCuda(poly::span<const BoundCodecRun>, void*)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    ZL_ERR(temporaryLibraryLimitation, "zstd.cuda is not implemented yet");
}

ZL_Report decodeWithNvcomp(poly::span<const BoundCodecRun>, void*)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    ZL_ERR(temporaryLibraryLimitation, "zstd.nvcomp is not implemented yet");
}

/// The largest input or temporary alignment requirement from nvcomp that
/// `zstd.nvcomp` accepts.
constexpr size_t kMaxNvcompAlignment = 256;

bool isAlignmentUpTo(size_t alignment, size_t limit)
{
    return alignment != 0 && alignment <= limit && ZL_isPow2(alignment);
}

/// Whether `zstd.nvcomp` can decode with @p functions.
bool nvcompZstdIsUsable(const NvcompZstdFunctions& functions)
{
    NvcompAlignmentRequirements alignments;
    if (functions.decompressGetRequiredAlignments(
                NvcompZstdDecompressOpts{}, &alignments)
        != 0) {
        return false;
    }
    // OpenZL places outputs at any byte offset. The inputs are staged copies
    // and the temporary space is scratch, so `zstd.nvcomp` can align both.
    return alignments.output == 1
            && isAlignmentUpTo(alignments.input, kMaxNvcompAlignment)
            && isAlignmentUpTo(alignments.temp, kMaxNvcompAlignment);
}

} // namespace

poly::span<const DecodeKernel> zstdCudaKernels(bool enableNvcomp)
{
    static constexpr DecodeKernel kKernels[] = {
        { "zstd.cuda", acceptsNoRunYet, decodeWithCuda },
        { "zstd.nvcomp", acceptsNoRunYet, decodeWithNvcomp },
    };
    bool listNvcomp = false;
    if (enableNvcomp) {
        const std::optional<NvcompZstdFunctions>& zstd = nvcompFunctions().zstd;
        listNvcomp = zstd.has_value() && nvcompZstdIsUsable(*zstd);
    }
    return { kKernels, listNvcomp ? 2U : 1U };
}

} // namespace openzl::gpu
