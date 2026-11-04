// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/device/decode_dispatch.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "openzl/common/errors_internal.h"

namespace openzl::gpu {

poly::span<const DecodeKernel> kernelsFor(
        const DecodeKernelTable& table,
        PublicTransformInfo transform)
{
    if (transform.trt != trt_standard || transform.trid >= table.size()) {
        return {};
    }
    return table[transform.trid];
}

const DecodeKernel* selectDecodeKernel(
        poly::span<const DecodeKernel> candidates,
        const CodecRunInfo& run)
{
    for (const DecodeKernel& candidate : candidates) {
        if (candidate.accepts(run)) {
            return &candidate;
        }
    }
    return nullptr;
}

ZL_Report chooseDecodeBatches(
        const DecodeKernelTable& table,
        poly::span<const CodecRunInfo> runs,
        std::vector<DecodeBatch>& batches)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    std::vector<DecodeBatch> chosen;
    for (size_t i = 0; i < runs.size(); ++i) {
        const DecodeKernel* const kernel = selectDecodeKernel(
                kernelsFor(table, runs[i].transform), runs[i]);
        if (kernel == nullptr) {
            ZL_ERR(temporaryLibraryLimitation,
                   "No GPU decode kernel accepts codec run %zu "
                   "(transform %u)",
                   i,
                   static_cast<unsigned>(runs[i].transform.trid));
        }
        DecodeBatch* batch = nullptr;
        for (DecodeBatch& existing : chosen) {
            if (existing.kernel == kernel) {
                batch = &existing;
                break;
            }
        }
        if (batch == nullptr) {
            batch = &chosen.emplace_back(DecodeBatch{ .kernel = kernel });
        }
        batch->runs.push_back(i);
    }
    batches = std::move(chosen);
    return ZL_returnSuccess();
}

ZL_Report decodeBatchWith(
        const DecodeKernel& kernel,
        poly::span<const BoundCodecRun> runs,
        void* scratch_d,
        size_t scratchBytes,
        void* stream)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    if (runs.empty()) {
        return ZL_returnSuccess();
    }
    ZL_Report const result =
            kernel.decode(runs, scratch_d, scratchBytes, stream);
    // The caller reads temporaryLibraryLimitation as "nothing enqueued",
    // which stops being true once a kernel runs.
    ZL_ERR_IF(
            ZL_isError(result)
                    && ZL_errorCode(result)
                            == ZL_ErrorCode_temporaryLibraryLimitation,
            transform_executionFailure,
            "%s cannot decode a run that it accepted",
            kernel.name);
    ZL_ERR_IF_ERR(result);
    return ZL_returnSuccess();
}

} // namespace openzl::gpu
