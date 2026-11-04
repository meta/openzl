// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/codecs/float_deconstruct/decode_float_deconstruct_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <vector>

#include "contrib/gpu/src/codecs/float_deconstruct/decode_float_deconstruct_bf16.cuh"
#include "contrib/gpu/src/codecs/float_deconstruct/decode_float_deconstruct_binding.hpp"
#include "openzl/common/errors_internal.h"

namespace openzl::gpu {
namespace {

bool acceptsRun(const CodecRunInfo& run)
{
    return isBf16CodecRun(run);
}

ZL_Report scratchSizeOfRuns(
        poly::span<const CodecRunInfo> runs,
        ScratchSize& size)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    ZL_ERR_IF_GT(runs.size(), std::numeric_limits<uint32_t>::max(), allocation);
    std::vector<size_t> numElts;
    numElts.reserve(runs.size());
    for (const CodecRunInfo& run : runs) {
        numElts.push_back(run.inputs[1].shape.numElts);
    }
    size = {
        .bytes = bf16DeconDecodeScratchBytes(
                numElts.data(),
                static_cast<uint32_t>(numElts.size()),
                kBf16DefaultMaxSliceElts),
        .alignment = alignof(Bf16DeconElementRange),
    };
    return ZL_returnSuccess();
}

/**
 * The host descriptors of @p runs, or no value if a run has an address
 * that the kernel cannot use.
 *
 * The kernel stores bf16 values with 2-byte stores. The layout aligns every
 * destination for them, so an odd destination is a caller's bug.
 * @throws std::bad_alloc if an allocation fails.
 */
std::optional<std::vector<Bf16DeconElementRange>> elementRangesOf(
        poly::span<const BoundCodecRun> runs)
{
    std::vector<Bf16DeconElementRange> ranges;
    ranges.reserve(runs.size());
    for (const BoundCodecRun& run : runs) {
        const std::optional<Bf16DeconElementRange> range = asBf16CodecRun(run);
        if (!range.has_value()
            || reinterpret_cast<uintptr_t>(range->dst) % alignof(uint16_t)
                    != 0) {
            return std::nullopt;
        }
        ranges.push_back(*range);
    }
    return ranges;
}

/**
 * Runs @p launch, which throws on a CUDA error as the kernel wrapper does, and
 * reports what it throws.
 */
template <typename Launch>
ZL_Report reportLaunch(const char* kernel, Launch&& launch)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    try {
        launch();
    } catch (const std::bad_alloc&) {
        ZL_ERR(allocation, "%s: allocation failed", kernel);
    } catch (const std::exception& e) {
        ZL_ERR(transform_executionFailure, "%s: %s", kernel, e.what());
    }
    return ZL_returnSuccess();
}

ZL_Report decodeRuns(
        poly::span<const BoundCodecRun> runs,
        void* scratch_d,
        size_t scratchBytes,
        void* stream)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    ZL_ERR_IF_GT(runs.size(), std::numeric_limits<uint32_t>::max(), allocation);
    std::optional<std::vector<Bf16DeconElementRange>> ranges;
    try {
        ranges = elementRangesOf(runs);
    } catch (const std::bad_alloc&) {
        ZL_ERR(allocation, "float_deconstruct.bf16.cuda: allocation failed");
    }
    ZL_ERR_IF(
            !ranges.has_value(),
            parameter_invalid,
            "float_deconstruct.bf16.cuda: a run has a null or odd address");
    return reportLaunch("float_deconstruct.bf16.cuda", [&] {
        bf16DeconDecode(
                ranges->data(),
                static_cast<uint32_t>(ranges->size()),
                kBf16DefaultMaxSliceElts,
                scratch_d,
                scratchBytes,
                static_cast<cudaStream_t>(stream));
    });
}

} // namespace

poly::span<const DecodeKernel> floatDeconstructCudaKernels()
{
    static constexpr DecodeKernel kKernels[] = {
        { "float_deconstruct.bf16.cuda",
          acceptsRun,
          scratchSizeOfRuns,
          decodeRuns },
    };
    return kKernels;
}

} // namespace openzl::gpu
