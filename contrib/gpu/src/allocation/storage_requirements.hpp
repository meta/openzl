// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "contrib/gpu/src/common/result.hpp"
#include "contrib/gpu/src/planning/decode_plan.hpp"

namespace openzl::gpu {

/// Allocation arena in which a storage requirement must be placed.
enum class ArenaClass : uint32_t {
    /// Caller-provided storage receiving the decoded output.
    Destination,
    /// Workspace holding materialized intermediate streams and the kernel
    /// scratch of each batch.
    StreamArena,
};

/// What a storage requirement holds.
enum class StorageContent : uint32_t {
    /// One region of a planned stream, named by @c StorageRequirement::region.
    Stream,
    /// The kernel scratch of one decode batch. It belongs to no stream, and no
    /// stream aliases it.
    BatchScratch,
};

/// Addressable region of a stream covered by a storage requirement.
enum class StreamRegion : uint32_t {
    /// Primary stream data.
    Data,
    /// String-length table for a string stream.
    StringLengths,
};

/// Kind of decode execution stage that bounds a storage lifetime.
enum class DecodeExecutionStageKind : uint32_t {
    /// Boundary before any codec run starts.
    DecodeStart,
    /// Execution of one codec run in the decode graph.
    CodecRun,
    /// Boundary after all codec runs finish.
    DecodeEnd,
};

/** A stage in the decode dependency graph, independent of a work schedule. */
struct DecodeExecutionStage {
    /// Type of the execution stage.
    DecodeExecutionStageKind kind{ DecodeExecutionStageKind::DecodeStart };
    /// Codec run identified by this stage, set only when @c kind is
    /// @c CodecRun.
    std::optional<PlannedCodecRunIndex> codecRunIndex;

    /// @returns The boundary before any codec run starts.
    static constexpr DecodeExecutionStage decodeStart() noexcept
    {
        return {};
    }

    /// @returns The execution stage for @p codecRunIndex.
    static constexpr DecodeExecutionStage codecRun(
            PlannedCodecRunIndex codecRunIndex) noexcept
    {
        return {
            .kind          = DecodeExecutionStageKind::CodecRun,
            .codecRunIndex = codecRunIndex,
        };
    }

    /// @returns The boundary after all codec runs finish.
    static constexpr DecodeExecutionStage decodeEnd() noexcept
    {
        return { .kind = DecodeExecutionStageKind::DecodeEnd };
    }

    bool operator==(const DecodeExecutionStage& other) const
    {
        return kind == other.kind && codecRunIndex == other.codecRunIndex;
    }
};

/** Inclusive execution stages over which storage must remain valid. */
struct StorageLifetime {
    /// First stage that can access the storage.
    DecodeExecutionStage firstUse;
    /// Last stage that can access the storage.
    DecodeExecutionStage lastUse;

    bool operator==(const StorageLifetime& other) const
    {
        return firstUse == other.firstUse && lastUse == other.lastUse;
    }
};

/**
 * One stream region, or one batch's kernel scratch, that needs storage,
 * described independently of its owner and its final address.
 * @c ExecutionRequirements::streamRequirements and
 * @c ExecutionRequirements::batchScratch find the requirements of each owner.
 *
 * A workspace region may be placed anywhere in its arena. A destination region
 * follows the core output layout instead: each final output has its own
 * buffer, chunks follow each other in chunk order, and string lengths go in the
 * output's separate lengths buffer. A layout therefore groups destination
 * regions by @c finalOutput and @c region, and places each group's regions one
 * after another in list order.
 */
struct StorageRequirement {
    /// What the region holds.
    StorageContent content{ StorageContent::Stream };
    /// Addressable region of the stream. Unused for
    /// @c StorageContent::BatchScratch.
    StreamRegion region{};
    /// Arena in which the region must be placed.
    ArenaClass arena{};
    /// Final output that a destination region belongs to.
    std::optional<size_t> finalOutput;
    /**
     * Execution stages during which the region must remain valid.
     *
     * Two workspace regions may share storage only when one region's
     * @c lastUse happens before the other's @c firstUse. Only the dependency
     * graph proves this order. A lower codec run index does not, because
     * codec runs in different chunks are independent.
     */
    StorageLifetime lifetime{};
    /// Number of bytes required.
    size_t size{};
    /**
     * Minimum address alignment in bytes.
     *
     * For a destination data region, this is the alignment that core requires
     * of a caller's output buffer. For a workspace data region, it also covers
     * every alias that reads the region.
     */
    size_t alignment{ 1 };

    bool operator==(const StorageRequirement& other) const
    {
        return content == other.content && region == other.region
                && arena == other.arena && finalOutput == other.finalOutput
                && lifetime == other.lifetime && size == other.size
                && alignment == other.alignment;
    }
};

/**
 * The requirements that place one stream's regions, as indexes into
 * @c ExecutionRequirements::requirements.
 *
 * A region has no requirement when it holds zero bytes, when its data is
 * stored in the frame, or when it views another stream's data. For a final
 * output, the data index names its destination region.
 */
struct StreamRequirements {
    /// Requirement of the stream's data.
    std::optional<size_t> data;
    /// Requirement of a string stream's lengths.
    std::optional<size_t> stringLengths;

    /// The requirement of @p region: @c data or @c stringLengths.
    std::optional<size_t>& of(StreamRegion region);

    bool operator==(const StreamRequirements& other) const
    {
        return data == other.data && stringLengths == other.stringLengths;
    }
};

/** Every storage requirement of a planned decode, before placement. */
struct ExecutionRequirements {
    /**
     * Requirements in deterministic order: destination regions ordered by final
     * output and then by chunk, followed by workspace regions in planned stream
     * order, followed by scratch in batch order.
     */
    std::vector<StorageRequirement> requirements;
    /// The requirements that place each stream's regions, indexed by
    /// @c PlannedStreamIndex.
    std::vector<StreamRequirements> streamRequirements;
    /// The requirement that places each batch's kernel scratch, indexed by
    /// batch. Empty for a batch that needs no scratch.
    std::vector<std::optional<size_t>> batchScratch;
};

/**
 * Derives every storage requirement of a planned decode.
 *
 * Frame-resident data and aliased data need no allocation. String-length
 * tables remain separate because a data alias does not provide their storage.
 * A final alias's codec run writes its string lengths directly into the
 * destination length region.
 * Every final output receives destination storage, including stored and
 * referenced outputs that the executor copies when decoding finishes. A region
 * of zero bytes needs no storage and gets no requirement.
 *
 * The function derives requirements in four steps:
 *
 * 1. It derives each produced stream's lifetime.
 * 2. It adds destination regions for final outputs.
 * 3. It extends and aligns the workspace that stream aliases read.
 * 4. It adds workspace regions for materialized intermediate streams.
 *
 * @param plan Decode graph from @c planDecode, which checks the frame. This
 * function trusts the plan and does not check its structure again.
 * @returns Storage requirements without their addresses, or @c allocation if
 * an allocation fails.
 */
Result<ExecutionRequirements> buildDecodeStorageRequirements(
        const DecodePlan& plan);

} // namespace openzl::gpu
