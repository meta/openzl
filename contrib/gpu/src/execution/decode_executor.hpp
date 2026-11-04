// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <vector>

#include "contrib/gpu/src/allocation/allocation_layout.hpp"
#include "contrib/gpu/src/allocation/storage_requirements.hpp"
#include "contrib/gpu/src/common/result.hpp"
#include "contrib/gpu/src/device/device.hpp"
#include "contrib/gpu/src/execution/decode_binding.hpp"
#include "contrib/gpu/src/planning/decode_plan.hpp"
#include "contrib/gpu/src/planning/decode_planner.hpp"
#include "openzl/cpp/poly/Span.hpp"
#include "openzl/zl_errors.h"

namespace openzl::gpu {

/**
 * Everything a decode needs before it has device addresses.
 *
 * It holds no device address and reads no device memory, so a caller can keep
 * it and enqueue it more than once while the chunks it was planned from and
 * the device it was planned for stay valid.
 */
struct PlannedDecode {
    DecodePlan plan;
    /// The kernel and scratch of each batch, in wave order.
    std::vector<PlannedDecodeBatch> batches;
    /// The range of @c batches that each wave holds, in wave order.
    std::vector<IndexRange> waves;
    /// Stream regions, then the scratch of each batch.
    ExecutionRequirements requirements;
    AllocationLayout layout;
    /// Bytes that the decode writes to the caller's output.
    size_t decompressedSize = 0;
};

/**
 * Plans, schedules, and lays out the decode of @p chunks on @p device.
 *
 * Chooses the kernel of every codec run and sizes the scratch of each batch
 * (binding, part 1), so the layout covers all the memory that the decode
 * needs. Allocates no device memory.
 *
 * @param device The device that will decode. It must outlive the result.
 * @param chunks Prepared chunks of one frame, as @c planDecode takes them.
 * @returns The planned decode; @c temporaryLibraryLimitation if the frame has
 * more than one output, because the caller provides one output buffer, or if no
 * kernel on @p device accepts a codec run; @c allocation if a host allocation
 * fails; or the first error of planning, scheduling, binding, storage, or
 * layout.
 */
Result<PlannedDecode> planDecodeExecution(
        const Device& device,
        poly::span<const PreparedGpuChunkView> chunks);

/**
 * Enqueues @p decode on @p stream and returns without waiting for the device.
 *
 * Allocates the workspace on the device of @p stream and binds every codec
 * run and batch scratch to an address (binding, part 2). It then enqueues the
 * batches in wave order, and last the copies of final outputs that the frame
 * stores or that refer to another stream. The batches of one wave are
 * independent: when a wave has several, each runs on its own branch of
 * @p stream (@c Stream::fork), and @p stream waits for all of them before the
 * next wave (@c Stream::join).
 *
 * @param stream A stream of the device that @p decode was planned for.
 * @param chunks The chunks that @p decode was planned from. They and their
 * frame must stay valid until the enqueued work completes.
 * @param dst_d Device buffer for the decompressed output.
 * @returns The decompressed size; @c dstCapacity_tooSmall or
 * @c parameter_invalid (@p dst_d is not aligned as core requires for the
 * output) before anything is enqueued; or a device error, after which earlier
 * work may still be running, so the caller must synchronize @p stream before
 * it frees @p dst_d.
 */
ZL_Report enqueueDecode(
        const Stream& stream,
        const PlannedDecode& decode,
        poly::span<const PreparedGpuChunkView> chunks,
        void* dst_d,
        size_t dstCapacity);

} // namespace openzl::gpu
