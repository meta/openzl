// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <vector>

#include "contrib/gpu/src/allocation/arena_addresses.hpp"
#include "contrib/gpu/src/allocation/storage_requirements.hpp"
#include "contrib/gpu/src/common/codec_decode_execution.hpp"
#include "contrib/gpu/src/common/result.hpp"
#include "contrib/gpu/src/device/device.hpp"
#include "contrib/gpu/src/planning/decode_plan.hpp"
#include "contrib/gpu/src/planning/decode_planner.hpp"
#include "contrib/gpu/src/scheduling/work_schedule.hpp"
#include "openzl/cpp/poly/Span.hpp"

namespace openzl::gpu {

/**
 * The info of every codec run of a planned decode, without addresses.
 *
 * The spans in @c codecRuns point into @c inputs, @c outputs, and the chunks'
 * staged host data. A move keeps them valid. A copy would point
 * into the original, so copying is deleted.
 */
struct DecodeInfo {
    DecodeInfo()                             = default;
    DecodeInfo(DecodeInfo&&)                 = default;
    DecodeInfo& operator=(DecodeInfo&&)      = default;
    DecodeInfo(const DecodeInfo&)            = delete;
    DecodeInfo& operator=(const DecodeInfo&) = delete;
    ~DecodeInfo()                            = default;

    /// One entry per planned codec run, indexed by @c PlannedCodecRunIndex.
    std::vector<CodecRunInfo> codecRuns;
    /// Storage for the input spans of @c codecRuns.
    std::vector<InputStreamInfo> inputs;
    /// Storage for the output spans of @c codecRuns.
    std::vector<OutputStreamInfo> outputs;
};

/**
 * Describes every codec run of @p plan for its kernel.
 *
 * @param chunks The chunks that @c planDecode received. Their staged transform
 * headers must outlive the result.
 * @returns The info; @c temporaryLibraryLimitation if the plan has a string
 * stream; or @c allocation if an allocation fails.
 */
Result<DecodeInfo> buildDecodeInfo(
        const DecodePlan& plan,
        poly::span<const PreparedGpuChunkView> chunks);

/**
 * Codec runs of one wave that one kernel decodes together, and the scratch that
 * they need.
 */
struct PlannedDecodeBatch {
    /// Index of the batch's wave in @c WorkSchedule::waves.
    size_t wave = 0;
    /// Kernel in the device's table. It lives as long as the device.
    const DecodeKernel* kernel = nullptr;
    /// The batch's codec runs, in their order in the wave.
    std::vector<PlannedCodecRunIndex> codecRuns;
    ScratchSize scratch;
};

/**
 * Binding, part 1: chooses a kernel for every codec run that a wave of
 * @p schedule runs, and sizes the scratch of each batch.
 *
 * Within each wave, runs that choose the same kernel form one batch, as
 * @c Device::chooseDecodeBatches groups them. Allocates no device memory, so a
 * decode that cannot run on @p device fails before any allocation.
 *
 * @param info Info from @c buildDecodeInfo, for the plan that @p schedule
 * schedules.
 * @returns The batches in wave order; @c temporaryLibraryLimitation if no
 * kernel accepts a run; a kernel's @c scratchSize error; or @c allocation if an
 * allocation fails.
 */
Result<std::vector<PlannedDecodeBatch>> planDecodeBatches(
        const Device& device,
        const WorkSchedule& schedule,
        const DecodeInfo& info);

/**
 * Adds one @c StorageContent::BatchScratch requirement to @p requirements for
 * each batch that needs scratch, in the workspace arena, and records its index
 * in @c ExecutionRequirements::batchScratch.
 *
 * The runs of a batch use its scratch at the same time, and the dependency
 * graph orders none of them before another, so no interval of codec-run
 * stages covers them. The scratch therefore lives for the whole decode, as
 * storage shared by independent uses does.
 *
 * @throws std::bad_alloc if an allocation fails.
 */
void addScratchRequirements(
        poly::span<const PlannedDecodeBatch> batches,
        ExecutionRequirements& requirements);

/** One batch with addresses: everything that @c Stream::decodeBatch takes. */
struct BoundBatch {
    /// Kernel in the device's table. It lives as long as the device.
    const DecodeKernel* kernel = nullptr;
    /// The batch's runs with addresses, in the batch's order.
    poly::span<const BoundCodecRun> runs;
    /// The batch's scratch, or null if the batch needs none.
    std::byte* scratch_d = nullptr;
    /// Bytes at @c scratch_d.
    size_t scratchBytes = 0;
};

/**
 * Device addresses for every batch of a planned decode.
 *
 * The spans in @c batches point into @c codecRuns, and the runs point into
 * @c info, @c inputs, and @c outputs. A move keeps them valid. A copy would
 * point into the original, so copying is deleted.
 */
struct BoundDecode {
    BoundDecode()                              = default;
    BoundDecode(BoundDecode&&)                 = default;
    BoundDecode& operator=(BoundDecode&&)      = default;
    BoundDecode(const BoundDecode&)            = delete;
    BoundDecode& operator=(const BoundDecode&) = delete;
    ~BoundDecode()                             = default;

    /// One entry per batch, in batch order.
    std::vector<BoundBatch> batches;
    /// Storage for the run spans of @c batches, in batch order. A run that no
    /// batch holds, such as an aliasing run, has no entry.
    std::vector<BoundCodecRun> codecRuns;
    /**
     * Copies that move each final @c Stored or @c StreamRef output into the
     * caller's output, in chunk order and then output order.
     */
    std::vector<DeviceToDeviceCopy> terminalCopies;
    /// The info that @c codecRuns refer to.
    DecodeInfo info;
    /// Storage for the input spans of @c codecRuns.
    std::vector<BoundInputStream> inputs;
    /// Storage for the output spans of @c codecRuns.
    std::vector<BoundOutputStream> outputs;
};

/**
 * Binding, part 2: gives every stream of @p plan, and every run and scratch of
 * @p batches, its device address.
 *
 * A @c Stored stream uses its address in the frame. A @c Destination or
 * @c StreamArena stream, and a batch's scratch, use the base address of their
 * arena plus their offset in the layout. A @c StreamRef uses its backing
 * stream's address plus its offset.
 *
 * @param plan Decode graph from @c planDecode.
 * @param requirements Requirements from @c buildDecodeStorageRequirements of
 * @p plan, with @c addScratchRequirements of @p batches.
 * @param arenas The layout of @p requirements, with an address for each
 * arena.
 * @param chunks The chunks that @c planDecode received. Their contexts and
 * staged headers must outlive the result.
 * @param batches The batches from @c planDecodeBatches.
 * This function trusts its inputs, as storage trusts the plan.
 * @returns The addresses, @c temporaryLibraryLimitation if the plan has a
 * string stream, or @c allocation if an allocation fails.
 */
Result<BoundDecode> bindDecode(
        const DecodePlan& plan,
        const ExecutionRequirements& requirements,
        const ArenaAddresses& arenas,
        poly::span<const PreparedGpuChunkView> chunks,
        poly::span<const PlannedDecodeBatch> batches);

} // namespace openzl::gpu
