// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "contrib/gpu/src/allocation/monotonic_layout.hpp"
#include "contrib/gpu/src/codecs/float_deconstruct/tests/bf16_test_frames.h"
#include "contrib/gpu/src/decompress/gpu_decompress.hpp"
#include "contrib/gpu/src/execution/decode_binding.hpp"
#include "contrib/gpu/testkit/frame_factory.h"
#include "contrib/gpu/testkit/multichunk_frame.h"
#include "contrib/gpu/testkit/staged_host_frame.h"
#include "openzl/common/wire_format.h"
#include "openzl/openzl.hpp"

namespace openzl::gpu {
namespace {

template <typename T>
T valueOf(Result<T> result)
{
    if (result.isError()) {
        throw std::runtime_error(ZL_E_str(ZL_RES_error(result.report())));
    }
    return std::move(result).value();
}

/**
 * A frame that the host memory device stages and the planner plans, with one
 * host buffer per arena of its layout.
 *
 * Binding ignores each batch's kernel. Without @p batches, one batch holds
 * every planned run, in plan order.
 */
class PlannedHostFrame {
   public:
    explicit PlannedHostFrame(
            std::string frame,
            std::optional<std::vector<PlannedDecodeBatch>> batches =
                    std::nullopt)
            : staged_(std::move(frame))
    {
        plan_ = valueOf(planDecode(staged_.views()));
        if (batches.has_value()) {
            batches_ = std::move(*batches);
        } else {
            batches_.resize(1);
            for (PlannedCodecRunIndex r = 0; r < plan_.codecRuns.size(); ++r) {
                batches_[0].codecRuns.push_back(r);
            }
        }
        requirements_ = valueOf(buildDecodeStorageRequirements(plan_));
        addScratchRequirements(batches_, requirements_);
        layout_ = valueOf(buildMonotonicAllocationLayout(requirements_));
        for (const AllocationArena& arena : layout_.arenas) {
            // Operator new aligns to at least 16 bytes, which covers every
            // stream arena alignment of these frames. Scratch tests check
            // offsets from the workspace base instead.
            arenas_.emplace_back(arena.size);
            arenaBases_d_.push_back(arenas_.back().data());
        }
    }

    BoundDecode bind() const
    {
        return valueOf(bindDecode(
                plan_,
                requirements_,
                ArenaAddresses{ layout_, arenaBases_d_ },
                staged_.views(),
                batches_));
    }

    const DecodePlan& plan() const
    {
        return plan_;
    }

    /// Base address of the workspace arena, or nullptr if there is none.
    const std::byte* workspace() const
    {
        for (size_t i = 0; i < layout_.arenas.size(); ++i) {
            if (layout_.arenas.at(i).arenaClass == ArenaClass::StreamArena) {
                return arenaBases_d_.at(i);
            }
        }
        return nullptr;
    }

    /// Base address of the arena that holds final output 0's data.
    const std::byte* destination_d() const
    {
        for (size_t i = 0; i < layout_.arenas.size(); ++i) {
            const AllocationArena& arena = layout_.arenas.at(i);
            if (arena.arenaClass == ArenaClass::Destination
                && arena.finalOutput == 0
                && arena.region == StreamRegion::Data) {
                return arenaBases_d_.at(i);
            }
        }
        return nullptr;
    }

   private:
    testkit::StagedHostFrame staged_;
    std::vector<PlannedDecodeBatch> batches_;
    DecodePlan plan_;
    ExecutionRequirements requirements_;
    AllocationLayout layout_;
    std::vector<std::vector<std::byte>> arenas_;
    std::vector<std::byte*> arenaBases_d_;
};

std::vector<std::byte> bytesAt(const void* address, size_t size)
{
    const auto* const begin = static_cast<const std::byte*>(address);
    return { begin, begin + size };
}

template <typename T>
std::vector<std::byte> bytesOf(const std::vector<T>& values)
{
    return bytesAt(values.data(), values.size() * sizeof(T));
}

TEST(DecodeBindingTest, BindsFloatDeconstructToTheFrameAndTheDestination)
{
    // This fails if a stored input is bound to the wrong frame bytes, or the
    // output to anything but the caller's destination.
    const std::vector<uint16_t> values = test::makeBf16Values(1000, 1);
    const test::Bf16Streams encoded    = test::encodeBf16(values);
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const PlannedHostFrame frame{ testkit::makeFrameNodeToStore(
            compressor,
            nodes::BFloat16Deconstruct::node,
            Input::refNumeric<uint16_t>(values.data(), values.size())) };

    const BoundDecode bound = frame.bind();

    const auto found = std::find_if(
            bound.codecRuns.begin(),
            bound.codecRuns.end(),
            [](const BoundCodecRun& run) {
                return run.info.transform.trid
                        == ZL_StandardTransformID_float_deconstruct;
            });
    ASSERT_NE(found, bound.codecRuns.end());
    const BoundCodecRun& run = *found;
    EXPECT_EQ(
            bytesAt(run.info.codecHeader_h.data(),
                    run.info.codecHeader_h.size()),
            bytesAt(test::kBf16Header, sizeof(test::kBf16Header)));
    ASSERT_EQ(run.inputs.size(), 2);
    EXPECT_EQ(
            bytesAt(run.inputs[0].data_d, values.size()),
            bytesOf(encoded.signFrac));
    EXPECT_EQ(
            bytesAt(run.inputs[1].data_d, values.size()),
            bytesOf(encoded.exponent));
    ASSERT_EQ(run.outputs.size(), 1);
    EXPECT_EQ(run.outputs[0].data_d, frame.destination_d());
    EXPECT_TRUE(bound.terminalCopies.empty());
}

TEST(DecodeBindingTest, CopiesAStoredFinalOutputFromTheFrame)
{
    // This fails if a stored final output is not copied, or is copied from
    // anywhere but its bytes in the frame.
    const std::string values(1000, '\x2a');
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const PlannedHostFrame frame{ testkit::makeFrameWithGraph(
            compressor,
            graphs::Store{}(),
            Input::refSerial(values.data(), values.size())) };

    const BoundDecode bound = frame.bind();

    EXPECT_TRUE(bound.codecRuns.empty());
    ASSERT_EQ(bound.terminalCopies.size(), 1);
    const DeviceToDeviceCopy& copy = bound.terminalCopies.front();
    EXPECT_EQ(copy.dst_d, frame.destination_d());
    ASSERT_EQ(copy.size, values.size());
    EXPECT_EQ(
            bytesAt(copy.src_d, copy.size),
            bytesAt(values.data(), values.size()));
}

TEST(DecodeBindingTest, RejectsAStringStreamBeforeAnyAllocation)
{
    // The executor builds the info before it allocates. This fails if only
    // binding, part 2, which needs the allocated arenas, rejects strings. No
    // codec with a GPU planner makes a string stream yet, so the plan is built
    // by hand.
    DecodePlan plan;
    plan.streams.push_back(
            {
                    .shape       = { .type = ZL_Type_string, .numElts = 3 },
                    .storageSize = { .dataBytes = 6, .stringLengthsBytes = 12 },
                    .storageClass = StorageClass::Stored,
            });
    plan.chunks.push_back(
            { .streams = { .begin = 0, .count = 1 }, .numOutputs = 1 });

    EXPECT_EQ(
            ZL_errorCode(buildDecodeInfo(plan, {}).report()),
            ZL_ErrorCode_temporaryLibraryLimitation);
}

TEST(DecodeBindingTest, AddsTheOffsetOfEveryAliasInAChain)
{
    // Real conversions alias whole streams at offset 0, so the plan is built
    // by hand. This fails if an alias of an alias drops either offset.
    DecodePlan plan;
    const auto addSerial = [&plan](StorageClass storageClass,
                                   size_t bytes,
                                   std::optional<PlannedStreamIndex> backing,
                                   size_t offset) {
        plan.streams.push_back(
                {
                        .shape        = { .type     = ZL_Type_serial,
                                          .numElts  = bytes,
                                          .eltWidth = 1 },
                        .storageSize  = { .dataBytes = bytes },
                        .storageClass = storageClass,
                        .aliasStream  = backing,
                        .aliasOffset  = offset,
                });
        return plan.streams.size() - 1;
    };
    const PlannedStreamIndex owner =
            addSerial(StorageClass::StreamArena, 64, std::nullopt, 0);
    const PlannedStreamIndex view =
            addSerial(StorageClass::StreamRef, 32, owner, 8);
    addSerial(StorageClass::StreamRef, 16, view, 4);
    plan.chunks.push_back(
            { .streams = { .begin = 0, .count = 3 }, .numOutputs = 1 });
    const ExecutionRequirements requirements =
            valueOf(buildDecodeStorageRequirements(plan));
    const AllocationLayout layout =
            valueOf(buildMonotonicAllocationLayout(requirements));
    std::vector<std::vector<std::byte>> arenas;
    std::vector<std::byte*> arenaBases_d;
    for (const AllocationArena& arena : layout.arenas) {
        arenas.emplace_back(arena.size);
        arenaBases_d.push_back(arenas.back().data());
    }

    const BoundDecode bound = valueOf(bindDecode(
            plan,
            requirements,
            ArenaAddresses{ layout, arenaBases_d },
            {},
            {}));

    const AllocationPlacement& ownerPlacement = layout.placements.at(
            *requirements.streamRequirements.at(owner).data);
    ASSERT_EQ(bound.terminalCopies.size(), 1);
    EXPECT_EQ(
            bound.terminalCopies.front().src_d,
            arenaBases_d.at(ownerPlacement.arenaIndex) + ownerPlacement.offset
                    + 8 + 4);
}

TEST(DecodeBindingTest, BindsEachBatchInItsOrderAndSkipsRunsOutsideBatches)
{
    // The batches list the two chunks' float_deconstruct runs in reverse. This
    // fails if a batch's span holds another batch's runs, if runs follow plan
    // order, or if binding binds a run that no batch holds.
    const std::vector<uint16_t> values = test::makeBf16Values(30000, 1);
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const GraphID graph = compressor.buildStaticGraph(
            nodes::BFloat16Deconstruct::node,
            { graphs::Store{}(), graphs::Store{}() });
    const std::string bytes = testkit::makeMultiChunkFrame(
            compressor,
            {
                    testkit::ChunkSpec{ 10000, graph },
                    testkit::ChunkSpec{ 20000, graph },
            },
            Input::refNumeric<uint16_t>(values.data(), values.size()));
    std::vector<PlannedCodecRunIndex> floatRuns;
    {
        const PlannedHostFrame planned{ bytes };
        for (PlannedCodecRunIndex r = 0; r < planned.plan().codecRuns.size();
             ++r) {
            if (planned.plan().codecRuns[r].transform.trid
                == ZL_StandardTransformID_float_deconstruct) {
                floatRuns.push_back(r);
            }
        }
    }
    ASSERT_EQ(floatRuns.size(), 2);
    std::vector<PlannedDecodeBatch> batches(2);
    batches[0].codecRuns = { floatRuns[1] };
    batches[1].codecRuns = { floatRuns[0] };
    const PlannedHostFrame frame{ bytes, batches };

    const BoundDecode bound = frame.bind();

    ASSERT_EQ(bound.batches.size(), 2);
    ASSERT_EQ(bound.batches[0].runs.size(), 1);
    ASSERT_EQ(bound.batches[1].runs.size(), 1);
    EXPECT_EQ(
            bound.batches[0].runs[0].outputs[0].data_d,
            frame.destination_d() + 10000 * sizeof(uint16_t));
    EXPECT_EQ(
            bound.batches[1].runs[0].outputs[0].data_d, frame.destination_d());
    EXPECT_EQ(bound.codecRuns.size(), 2);
}

TEST(DecodeBindingTest, CopiesAReferencedFinalOutputThroughItsAlias)
{
    // A numeric input is stored as serial bytes, and a conversion views them
    // as numbers. This fails if the alias does not resolve to the stored bytes.
    const std::vector<float> values(1000, 1.0F);
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const PlannedHostFrame frame{ testkit::makeFrameWithGraph(
            compressor,
            graphs::Store{}(),
            Input::refNumeric<float>(values.data(), values.size())) };

    const BoundDecode bound = frame.bind();

    ASSERT_EQ(bound.terminalCopies.size(), 1);
    const DeviceToDeviceCopy& copy = bound.terminalCopies.front();
    EXPECT_EQ(copy.dst_d, frame.destination_d());
    ASSERT_EQ(copy.size, values.size() * sizeof(float));
    EXPECT_EQ(bytesAt(copy.src_d, copy.size), bytesOf(values));
}

TEST(DecodeBindingTest, CopiesEachChunkAfterThePreviousOne)
{
    // Core appends chunk outputs in chunk order. This fails if the second
    // chunk's copy does not start where the first one ends.
    std::string values(30000, '\0');
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<char>(i * 13);
    }
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const PlannedHostFrame frame{ testkit::makeMultiChunkFrame(
            compressor,
            {
                    testkit::ChunkSpec{ 10000, graphs::Store{}() },
                    testkit::ChunkSpec{ 20000, graphs::Store{}() },
            },
            Input::refSerial(values.data(), values.size())) };

    const BoundDecode bound = frame.bind();

    ASSERT_EQ(bound.terminalCopies.size(), 2);
    const DeviceToDeviceCopy& first  = bound.terminalCopies.at(0);
    const DeviceToDeviceCopy& second = bound.terminalCopies.at(1);
    EXPECT_EQ(first.dst_d, frame.destination_d());
    EXPECT_EQ(second.dst_d, frame.destination_d() + first.size);
    ASSERT_EQ(first.size + second.size, values.size());
    EXPECT_EQ(
            bytesAt(first.src_d, first.size),
            bytesAt(values.data(), first.size));
    EXPECT_EQ(
            bytesAt(second.src_d, second.size),
            bytesAt(values.data() + first.size, second.size));
}

TEST(DecodeBindingTest, PlacesEachBatchScratchInTheWorkspace)
{
    // Each batch that needs scratch gets its own region of the workspace. This
    // fails if a region is misaligned from the workspace base, if two regions
    // overlap, or if a batch without scratch gets an address.
    const std::string values(1000, '\x2a');
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    std::vector<PlannedDecodeBatch> batches(3);
    batches[0].scratch = { .bytes = 100, .alignment = 512 };
    batches[2].scratch = { .bytes = 40, .alignment = 64 };
    const PlannedHostFrame frame{
        testkit::makeFrameWithGraph(
                compressor,
                graphs::Store{}(),
                Input::refSerial(values.data(), values.size())),
        batches
    };

    const BoundDecode bound = frame.bind();

    ASSERT_EQ(bound.batches.size(), 3);
    ASSERT_NE(frame.workspace(), nullptr);
    const size_t first =
            static_cast<size_t>(bound.batches[0].scratch_d - frame.workspace());
    const size_t third =
            static_cast<size_t>(bound.batches[2].scratch_d - frame.workspace());
    EXPECT_EQ(first % 512, 0);
    EXPECT_EQ(bound.batches[1].scratch_d, nullptr);
    EXPECT_EQ(third % 64, 0);
    EXPECT_GE(third, first + 100);
}

TEST(DecodeBindingTest, AddsEachScratchAsBatchScratchForTheWholeDecode)
{
    // A layout tells scratch from stream data by its content, and the runs of a
    // batch have no order in the dependency graph. This fails if a scratch
    // requirement claims to be a stream region, leaves the workspace, or lives
    // for less than the whole decode.
    const StorageRequirement stream{
        .arena = ArenaClass::StreamArena,
        .size  = 8,
    };
    ExecutionRequirements requirements;
    requirements.requirements.push_back(stream);
    std::vector<PlannedDecodeBatch> batches(3);
    batches[0].scratch = { .bytes = 100, .alignment = 512 };
    batches[2].scratch = { .bytes = 40, .alignment = 64 };

    addScratchRequirements(batches, requirements);

    const StorageLifetime wholeDecode{
        .firstUse = DecodeExecutionStage::decodeStart(),
        .lastUse  = DecodeExecutionStage::decodeEnd(),
    };
    const std::vector<StorageRequirement> expected{
        stream,
        {
                .content   = StorageContent::BatchScratch,
                .arena     = ArenaClass::StreamArena,
                .lifetime  = wholeDecode,
                .size      = 100,
                .alignment = 512,
        },
        {
                .content   = StorageContent::BatchScratch,
                .arena     = ArenaClass::StreamArena,
                .lifetime  = wholeDecode,
                .size      = 40,
                .alignment = 64,
        },
    };
    EXPECT_EQ(requirements.requirements, expected);
    const std::vector<std::optional<size_t>> expectedIndexes{ 1,
                                                              std::nullopt,
                                                              2 };
    EXPECT_EQ(requirements.batchScratch, expectedIndexes);
}

} // namespace
} // namespace openzl::gpu
