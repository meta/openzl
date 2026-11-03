// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/execution/decode_executor.hpp"

#include <cstddef>
#include <new>
#include <utility>
#include <vector>

#include "contrib/gpu/src/allocation/decode_resources.hpp"
#include "contrib/gpu/src/allocation/monotonic_layout.hpp"
#include "contrib/gpu/src/execution/decode_binding.hpp"
#include "contrib/gpu/src/scheduling/work_schedule.hpp"
#include "openzl/common/errors_internal.h"

namespace openzl::gpu {
namespace {

Result<PlannedDecode> planExecution(
        const Device& device,
        poly::span<const PreparedGpuChunkView> chunks)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    PlannedDecode decode;
    Result<DecodePlan> plan = planDecode(chunks);
    ZL_ERR_IF_ERR(plan.report());
    decode.plan = std::move(plan).value();
    ZL_ERR_IF_NE(
            decode.plan.chunks.at(0).numOutputs,
            1,
            temporaryLibraryLimitation,
            "GPU decoding supports frames with one output only");

    Result<WorkSchedule> schedule = buildDecodeWorkSchedule(decode.plan);
    ZL_ERR_IF_ERR(schedule.report());

    Result<DecodeInfo> info = buildDecodeInfo(decode.plan, chunks);
    ZL_ERR_IF_ERR(info.report());
    Result<std::vector<PlannedDecodeBatch>> batches =
            planDecodeBatches(device, schedule.value(), info.value());
    ZL_ERR_IF_ERR(batches.report());
    decode.batches = std::move(batches).value();
    for (size_t begin = 0; begin < decode.batches.size();) {
        const size_t wave = decode.batches.at(begin).wave;
        size_t end        = begin + 1;
        while (end < decode.batches.size()
               && decode.batches.at(end).wave == wave) {
            ++end;
        }
        decode.waves.push_back({ .begin = begin, .count = end - begin });
        begin = end;
    }

    Result<ExecutionRequirements> requirements =
            buildDecodeStorageRequirements(decode.plan);
    ZL_ERR_IF_ERR(requirements.report());
    decode.requirements = std::move(requirements).value();

    Result<AllocationLayout> layout =
            buildMonotonicAllocationLayout(decode.requirements);
    ZL_ERR_IF_ERR(layout.report());
    decode.layout           = std::move(layout).value();
    decode.decompressedSize = outputSizeOf(decode.layout);
    return decode;
}

/// Enqueues @p batch on @p stream.
ZL_Report enqueueBatch(const Stream& stream, const BoundBatch& batch)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    // NULLSAFECLANG does not see the check in ZL_ERR_IF_NULL, because that
    // macro compares copies of its operands.
    if (batch.kernel == nullptr) {
        ZL_ERR(logicError, "A planned batch has no kernel");
    }
    return stream.decodeBatch(*batch.kernel, batch.runs);
}

/**
 * Enqueues batches @p first to @p end - 1, which belong to one wave, each on
 * its own branch of @p stream, and joins the branches. It joins even after a
 * batch fails or throws, so no branch work outlives the frees that follow on
 * @p stream.
 */
ZL_Report enqueueWave(
        const Stream& stream,
        const BoundDecode& bound,
        size_t first,
        size_t end)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    std::vector<Stream> branches;
    ZL_ERR_IF_ERR(stream.fork(end - first, branches));
    ZL_Report result = ZL_returnSuccess();
    try {
        for (size_t b = first; b < end && !ZL_isError(result); ++b) {
            result = enqueueBatch(branches.at(b - first), bound.batches.at(b));
        }
    } catch (...) {
        // The caller frees the workspace while this exception unwinds.
        (void)stream.join(branches);
        throw;
    }
    ZL_Report const joined = stream.join(branches);
    ZL_ERR_IF_ERR(result);
    ZL_ERR_IF_ERR(joined);
    return ZL_returnSuccess();
}

ZL_Report enqueue(
        const Stream& stream,
        const PlannedDecode& decode,
        poly::span<const PreparedGpuChunkView> chunks,
        void* dst_d,
        size_t dstCapacity)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    // The resources free the workspace in stream order, after every use
    // enqueued below, including on an error path.
    Result<DecodeResources> created = DecodeResources::create(
            stream,
            decode.layout,
            { .output_d = dst_d, .outputCapacity = dstCapacity });
    ZL_ERR_IF_ERR(created.report());
    const DecodeResources resources = std::move(created).value();

    Result<BoundDecode> boundResult = bindDecode(
            decode.plan,
            decode.requirements,
            resources.addresses(),
            chunks,
            decode.batches);
    ZL_ERR_IF_ERR(boundResult.report());
    const BoundDecode bound = std::move(boundResult).value();

    // The batches of one wave are independent, so each one gets its own
    // branch of the stream when the wave has several.
    for (const IndexRange wave : decode.waves) {
        if (wave.count == 1) {
            ZL_ERR_IF_ERR(enqueueBatch(stream, bound.batches.at(wave.begin)));
        } else {
            ZL_ERR_IF_ERR(enqueueWave(
                    stream, bound, wave.begin, wave.begin + wave.count));
        }
    }
    ZL_ERR_IF_ERR(stream.copyDeviceToDevice(bound.terminalCopies));
    return ZL_returnValue(decode.decompressedSize);
}

} // namespace

Result<PlannedDecode> planDecodeExecution(
        const Device& device,
        poly::span<const PreparedGpuChunkView> chunks)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    try {
        return planExecution(device, chunks);
    } catch (const std::bad_alloc&) {
        ZL_ERR(allocation);
    }
}

ZL_Report enqueueDecode(
        const Stream& stream,
        const PlannedDecode& decode,
        poly::span<const PreparedGpuChunkView> chunks,
        void* dst_d,
        size_t dstCapacity)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    try {
        return enqueue(stream, decode, chunks, dst_d, dstCapacity);
    } catch (const std::bad_alloc&) {
        ZL_ERR(allocation);
    }
}

} // namespace openzl::gpu
