// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/execution/decode_binding.hpp"

#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "openzl/common/errors_internal.h"
#include "openzl/decompress/dctx2.h"
#include "openzl/zl_data.h"

namespace openzl::gpu {
namespace {

/**
 * Builds the @c DecodeInfo of a plan: one @c CodecRunInfo for each planned
 * codec run, with the shape and storage size of each of its streams, its codec
 * header, and its frame's format version.
 *
 * It reads no device memory and needs no addresses. So planning can choose a
 * kernel for every run before anything is allocated (binding, part 1). It also
 * rejects string streams, so such a decode fails before any allocation.
 */
Result<DecodeInfo> buildInfo(
        const DecodePlan& plan,
        poly::span<const PreparedGpuChunkView> chunks)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    for (const PlannedStream& stream : plan.streams) {
        if (stream.shape.type == ZL_Type_string) {
            ZL_ERR(temporaryLibraryLimitation,
                   "GPU decoding does not support string streams yet");
        }
    }
    DecodeInfo info;
    size_t numInputs  = 0;
    size_t numOutputs = 0;
    for (const PlannedCodecRun& run : plan.codecRuns) {
        numInputs += run.inputs.count;
        numOutputs += run.outputs.count;
    }
    // The spans below point into these arrays, so they must not grow
    // after this reservation.
    info.inputs.reserve(numInputs);
    info.outputs.reserve(numOutputs);
    info.codecRuns.reserve(plan.codecRuns.size());

    // Each chunk's DCtx holds its frame header. A kernel can need the
    // format version to read a codec header, as zstd does.
    std::vector<uint32_t> formatVersions;
    formatVersions.reserve(chunks.size());
    for (const PreparedGpuChunkView& chunk : chunks) {
        const DFH_Struct* const header = DCtx_getFrameHeader(chunk.dctx);
        if (header == nullptr) {
            ZL_ERR(logicError, "Prepared chunk has no frame header");
        }
        formatVersions.push_back(header->formatVersion);
    }

    for (const PlannedCodecRun& run : plan.codecRuns) {
        const size_t firstInput = info.inputs.size();
        for (const PlannedStreamIndex s : plan.inputsOf(run)) {
            const PlannedStream& stream = plan.streams.at(s);
            info.inputs.push_back(
                    {
                            .shape            = stream.shape,
                            .availableStorage = stream.storageSize,
                    });
        }
        const size_t firstOutput = info.outputs.size();
        for (const PlannedStreamIndex s : plan.outputsOf(run)) {
            const PlannedStream& stream = plan.streams.at(s);
            info.outputs.push_back(
                    {
                            .shape       = stream.shape,
                            .storageSize = stream.storageSize,
                    });
        }
        ZL_ASSERT_LT(run.chunkNode.chunkIndex, chunks.size());
        const PreparedGpuChunkView& chunk = chunks[run.chunkNode.chunkIndex];
        info.codecRuns.push_back(
                {
                        .transform = run.transform,
                        .frameFormatVersion =
                                formatVersions.at(run.chunkNode.chunkIndex),
                        .codecHeader_h = chunk.transformHeaders_h.subspan(
                                run.codecHeader.offset, run.codecHeader.size),
                        .inputs =
                                poly::span<const InputStreamInfo>{ info.inputs }
                                        .subspan(firstInput, run.inputs.count),
                        .outputs =
                                poly::span<const OutputStreamInfo>{
                                        info.outputs }
                                        .subspan(
                                                firstOutput, run.outputs.count),
                });
    }
    return info;
}

/**
 * For each wave, gives the info of its runs to the device in plan order,
 * and turns the batches that the device chooses into plan indexes.
 *
 * @c Device::chooseDecodeBatches returns positions in the list it received, so
 * @c waveRuns keeps the plan index of each position.
 */
Result<std::vector<PlannedDecodeBatch>> planBatches(
        const Device& device,
        const WorkSchedule& schedule,
        const DecodeInfo& info)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    std::vector<PlannedDecodeBatch> planned;
    std::vector<PlannedCodecRunIndex> waveRuns;
    std::vector<CodecRunInfo> waveInfo;
    for (size_t w = 0; w < schedule.waves.size(); ++w) {
        waveRuns.clear();
        waveInfo.clear();
        for (const PlannedCodecRunIndex index :
             schedule.waves.at(w).codecRunIndices) {
            waveRuns.push_back(index);
            waveInfo.push_back(info.codecRuns.at(index));
        }
        std::vector<DecodeBatch> chosen;
        ZL_ERR_IF_ERR(device.chooseDecodeBatches(waveInfo, chosen));
        for (const DecodeBatch& batch : chosen) {
            PlannedDecodeBatch plannedBatch{ .wave   = w,
                                             .kernel = batch.kernel };
            for (const size_t i : batch.runs) {
                plannedBatch.codecRuns.push_back(waveRuns.at(i));
            }
            planned.push_back(std::move(plannedBatch));
        }
    }
    return planned;
}

/**
 * Binding, part 2, for one plan. First it finds the device address of every
 * stream. Then it builds one @c BoundCodecRun for each planned run, and the
 * copies of final outputs that no kernel writes.
 */
class Binder {
   public:
    Binder(const DecodePlan& plan,
           const ExecutionRequirements& requirements,
           const ArenaAddresses& arenas,
           poly::span<const PreparedGpuChunkView> chunks,
           poly::span<const PlannedDecodeBatch> batches)
            : plan_(plan),
              requirements_(requirements),
              arenas_(arenas),
              chunks_(chunks),
              batches_(batches)
    {
    }

    Result<BoundDecode> bind()
    {
        ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
        Result<DecodeInfo> info = buildInfo(plan_, chunks_);
        ZL_ERR_IF_ERR(info.report());
        bound_.info = std::move(info).value();
        resolveStreamData();
        bindBatches();
        addTerminalCopies();
        return std::move(bound_);
    }

   private:
    /**
     * The address of the bytes that @p stream owns: a @c Stored stream's bytes
     * in the frame, which its chunk's DCtx finds, or else its data placement.
     * A stream that is not @c Stored and has zero data bytes has no placement,
     * so its address is nullptr. A @c StreamRef stream owns no bytes, so
     * @c resolveStreamData never asks for its address.
     */
    std::byte* ownedData(PlannedStreamIndex stream) const
    {
        const PlannedStream& planned = plan_.streams.at(stream);
        if (planned.storageClass == StorageClass::Stored) {
            const size_t chunkIndex = plan_.chunkOf(stream);
            ZL_ASSERT_LT(chunkIndex, chunks_.size());
            const ZL_Data* const data = ZL_DCtx_getConstStream(
                    chunks_[chunkIndex].dctx, plan_.slotOf(chunkIndex, stream));
            ZL_ASSERT_NN(data);
            // Codecs never write frame bytes: the planner aliases an output
            // onto a stored stream only when no codec writes that output.
            return const_cast<std::byte*>(
                    static_cast<const std::byte*>(ZL_Data_rPtr(data)));
        }
        const std::optional<size_t> requirement =
                requirements_.streamRequirements.at(stream).data;
        return requirement.has_value() ? arenas_.addressOf(*requirement)
                                       : nullptr;
    }

    /**
     * Sets the device address of every stream. A @c StreamRef stream views
     * part of its backing stream, so its address is the backing stream's
     * address plus its offset.
     *
     * The planner points each alias at an input of the run that produces it,
     * and gives a run's inputs lower slots than its outputs. So the backing
     * stream has a lower index, and its address is already set.
     */
    void resolveStreamData()
    {
        streamData_d_.assign(plan_.streams.size(), nullptr);
        for (PlannedStreamIndex s = 0; s < plan_.streams.size(); ++s) {
            const PlannedStream& stream = plan_.streams.at(s);
            if (stream.storageClass != StorageClass::StreamRef) {
                streamData_d_.at(s) = ownedData(s);
                continue;
            }
            ZL_ASSERT(stream.aliasStream.has_value());
            ZL_ASSERT_LT(*stream.aliasStream, s);
            std::byte* const backing_d = streamData_d_.at(*stream.aliasStream);
            streamData_d_.at(s)        = backing_d == nullptr
                           ? nullptr
                           : backing_d + stream.aliasOffset;
        }
    }

    /**
     * Builds one @c BoundBatch for each batch: its kernel, and its runs, in the
     * batch's order, with the addresses of their streams.
     */
    void bindBatches()
    {
        size_t numRuns = 0;
        for (const PlannedDecodeBatch& batch : batches_) {
            numRuns += batch.codecRuns.size();
        }
        // The spans below point into these arrays, so they must not grow
        // after this reservation. Each run is in at most one batch.
        bound_.inputs.reserve(bound_.info.inputs.size());
        bound_.outputs.reserve(bound_.info.outputs.size());
        bound_.codecRuns.reserve(numRuns);
        bound_.batches.reserve(batches_.size());
        for (const PlannedDecodeBatch& batch : batches_) {
            const size_t firstRun = bound_.codecRuns.size();
            for (const PlannedCodecRunIndex r : batch.codecRuns) {
                bindCodecRun(r);
            }
            bound_.batches.push_back(
                    {
                            .kernel = batch.kernel,
                            .runs =
                                    poly::span<const BoundCodecRun>{
                                            bound_.codecRuns }
                                            .subspan(
                                                    firstRun,
                                                    batch.codecRuns.size()),
                    });
        }
    }

    /// Appends planned run @p r, with the addresses of its streams, to
    /// @c BoundDecode::codecRuns.
    void bindCodecRun(PlannedCodecRunIndex r)
    {
        const PlannedCodecRun& run = plan_.codecRuns.at(r);
        const size_t firstInput    = bound_.inputs.size();
        for (const PlannedStreamIndex s : plan_.inputsOf(run)) {
            bound_.inputs.push_back({ .data_d = streamData_d_.at(s) });
        }
        const size_t firstOutput = bound_.outputs.size();
        for (const PlannedStreamIndex s : plan_.outputsOf(run)) {
            bound_.outputs.push_back({ .data_d = streamData_d_.at(s) });
        }
        const poly::span<const BoundInputStream> inputs{ bound_.inputs };
        const poly::span<const BoundOutputStream> outputs{ bound_.outputs };
        bound_.codecRuns.push_back(
                {
                        .info   = bound_.info.codecRuns.at(r),
                        .inputs = inputs.subspan(firstInput, run.inputs.count),
                        .outputs =
                                outputs.subspan(firstOutput, run.outputs.count),
                });
    }

    /**
     * Adds a copy for each final output that the frame stores, or that views
     * another stream. No kernel writes such an output, so its bytes must move
     * to its place in the caller's buffer. A codec writes every other final
     * output into the caller's buffer directly.
     */
    void addTerminalCopies()
    {
        for (size_t c = 0; c < plan_.chunks.size(); ++c) {
            for (size_t o = 0; o < plan_.numFinalOutputsOf(c); ++o) {
                const PlannedStreamIndex s  = plan_.finalOutputOf(c, o);
                const PlannedStream& stream = plan_.streams.at(s);
                if (stream.storageClass != StorageClass::Stored
                    && stream.storageClass != StorageClass::StreamRef) {
                    continue;
                }
                const std::optional<size_t> destination =
                        requirements_.streamRequirements.at(s).data;
                if (!destination.has_value()) {
                    continue;
                }
                bound_.terminalCopies.push_back(
                        {
                                .dst_d = arenas_.addressOf(*destination),
                                .src_d = streamData_d_.at(s),
                                .size  = stream.storageSize.dataBytes,
                        });
            }
        }
    }

    const DecodePlan& plan_;
    const ExecutionRequirements& requirements_;
    const ArenaAddresses& arenas_;
    poly::span<const PreparedGpuChunkView> chunks_;
    poly::span<const PlannedDecodeBatch> batches_;

    /// The device address of each stream, indexed by stream.
    std::vector<std::byte*> streamData_d_;
    BoundDecode bound_;
};

} // namespace

// The builders grow vectors, which can throw std::bad_alloc. Each entry point
// turns that into the allocation error, so no exception leaves binding.
Result<DecodeInfo> buildDecodeInfo(
        const DecodePlan& plan,
        poly::span<const PreparedGpuChunkView> chunks)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    try {
        return buildInfo(plan, chunks);
    } catch (const std::bad_alloc&) {
        ZL_ERR(allocation);
    }
}

Result<std::vector<PlannedDecodeBatch>> planDecodeBatches(
        const Device& device,
        const WorkSchedule& schedule,
        const DecodeInfo& info)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    try {
        return planBatches(device, schedule, info);
    } catch (const std::bad_alloc&) {
        ZL_ERR(allocation);
    }
}

Result<BoundDecode> bindDecode(
        const DecodePlan& plan,
        const ExecutionRequirements& requirements,
        const ArenaAddresses& arenas,
        poly::span<const PreparedGpuChunkView> chunks,
        poly::span<const PlannedDecodeBatch> batches)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    try {
        return Binder{ plan, requirements, arenas, chunks, batches }.bind();
    } catch (const std::bad_alloc&) {
        ZL_ERR(allocation);
    }
}

} // namespace openzl::gpu
