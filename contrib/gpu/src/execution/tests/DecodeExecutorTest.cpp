// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "contrib/gpu/src/decompress/gpu_decompress.hpp"
#include "contrib/gpu/src/execution/decode_executor.hpp"
#include "contrib/gpu/testkit/frame_factory.h"
#include "contrib/gpu/testkit/host_memory_device.h"
#include "contrib/gpu/testkit/multichunk_frame.h"
#include "contrib/gpu/testkit/staged_host_frame.h"
#include "openzl/common/wire_format.h"
#include "openzl/openzl.hpp"

namespace openzl::gpu {
namespace {

std::vector<std::string>& recordedEvents()
{
    static std::vector<std::string> events;
    return events;
}

/// What the recording kernel answers.
struct RecordingAnswers {
    bool accepts = true;
    /// Whether the "small" kernel fails.
    bool failSmall = false;
    /// Whether the "small" kernel throws std::bad_alloc.
    bool throwSmall = false;
};

/// The native streams that the recording device gives its branches.
std::array<int, 8>& branchMarkers()
{
    static std::array<int, 8> markers{};
    return markers;
}

/// " on branch N" for a branch stream, or "" for the caller's stream.
std::string streamName(void* stream)
{
    for (size_t i = 0; i < branchMarkers().size(); ++i) {
        if (stream == &branchMarkers()[i]) {
            return " on branch " + std::to_string(i);
        }
    }
    return "";
}

RecordingAnswers& recordingAnswers()
{
    static RecordingAnswers answers;
    return answers;
}

bool acceptsRecorded(const CodecRunInfo&)
{
    return recordingAnswers().accepts;
}

ZL_Report record(
        const char* eventPrefix,
        poly::span<const BoundCodecRun> runs,
        void* stream)
{
    std::string event = eventPrefix;
    for (const BoundCodecRun& run : runs) {
        event += " " + std::to_string(run.info.transform.trid);
    }
    event += streamName(stream);
    recordedEvents().push_back(std::move(event));
    return ZL_returnSuccess();
}

ZL_Report decodeRecorded(poly::span<const BoundCodecRun> runs, void* stream)
{
    return record("decode", runs, stream);
}

/// Accepts runs whose first output has fewer than 15000 elements.
bool acceptsSmall(const CodecRunInfo& run)
{
    return run.outputs.size() == 1 && run.outputs[0].shape.numElts < 15000;
}

ZL_Report decodeSmall(poly::span<const BoundCodecRun> runs, void* stream)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    if (recordingAnswers().throwSmall) {
        throw std::bad_alloc();
    }
    ZL_ERR_IF(
            recordingAnswers().failSmall,
            transform_executionFailure,
            "injected failure");
    return record("decode small", runs, stream);
}

constexpr DecodeKernel kRecorded{ "recorded", acceptsRecorded, decodeRecorded };
constexpr DecodeKernel kSmall{ "small", acceptsSmall, decodeSmall };

/// The recording kernel for every codec of the float frame below, except
/// @p missing.
DecodeKernelTable recordingTable(std::optional<ZL_IDType> missing)
{
    DecodeKernelTable table{};
    for (const ZL_IDType trid : { ZL_StandardTransformID_constant_serial,
                                  ZL_StandardTransformID_float_deconstruct }) {
        if (trid != missing) {
            table[trid] = { kRecorded };
        }
    }
    return table;
}

/**
 * The table of @c recordingTable, with a second constant kernel that takes
 * small runs first, as a variant of a codec would.
 */
DecodeKernelTable variantTable()
{
    DecodeKernelTable table = recordingTable(std::nullopt);
    table[ZL_StandardTransformID_constant_serial] = { kSmall, kRecorded };
    return table;
}

/**
 * A device over host memory that records each call, and whose kernels record
 * each call and decode nothing.
 *
 * Device-to-device copies run with memcpy, so a test can read what reached
 * the destination. Device-to-host copies are only recorded.
 */
class RecordingDevice final : public Device {
   public:
    explicit RecordingDevice(
            DecodeKernelTable table = recordingTable(std::nullopt))
            : Device(std::move(table)), events(recordedEvents())
    {
        events.clear();
        recordingAnswers() = {};
    }

    std::vector<std::string>& events;

   protected:
    ZL_Report doFork(void*, size_t count, std::vector<void*>& branches) override
    {
        events.push_back("fork " + std::to_string(count));
        branches.clear();
        for (size_t i = 0; i < count; ++i) {
            branches.push_back(&branchMarkers().at(i));
        }
        return ZL_returnSuccess();
    }

    ZL_Report doJoin(void*, poly::span<const Stream> branches) noexcept override
    {
        events.push_back("join " + std::to_string(branches.size()));
        return ZL_returnSuccess();
    }

    ZL_Report doCopyToHostAndSynchronize(
            poly::span<const DeviceToHostCopy>,
            void*) override
    {
        events.emplace_back("copyToHost");
        return ZL_returnSuccess();
    }

    ZL_Report doAllocate(size_t size, size_t alignment, void*, void*& ptr_d)
            override
    {
        events.emplace_back("allocate");
        size_t const rounded = (size + alignment - 1) / alignment * alignment;
        ptr_d                = std::aligned_alloc(alignment, rounded);
        return ZL_returnSuccess();
    }

    void doDeallocate(void* ptr_d, void*) noexcept override
    {
        events.emplace_back(
                ptr_d == nullptr ? "deallocate null" : "deallocate");
        std::free(ptr_d);
    }

    ZL_Report doCopyDeviceToDevice(
            poly::span<const DeviceToDeviceCopy> copies,
            void*) override
    {
        events.emplace_back("copy " + std::to_string(copies.size()));
        for (const DeviceToDeviceCopy& copy : copies) {
            std::memcpy(copy.dst_d, copy.src_d, copy.size);
        }
        return ZL_returnSuccess();
    }
};

template <typename T>
T valueOf(Result<T> result)
{
    if (result.isError()) {
        throw std::runtime_error(ZL_E_str(ZL_RES_error(result.report())));
    }
    return std::move(result).value();
}

using testkit::StagedHostFrame;

/// A float frame in two chunks whose exponents go through the constant codec.
std::string makeConstantExponentFloatFrame(std::vector<float>& values)
{
    values.resize(30000);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 1.0F + static_cast<float>(i % 1024) / 2048.0F;
    }
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const GraphID graph = compressor.buildStaticGraph(
            nodes::Float32Deconstruct::node,
            { graphs::Store{}(), graphs::Constant{}() });
    return testkit::makeMultiChunkFrame(
            compressor,
            {
                    testkit::ChunkSpec{ 10000, graph },
                    testkit::ChunkSpec{ 20000, graph },
            },
            Input::refNumeric<float>(values.data(), values.size()));
}

std::string makeStoredSerialFrame(const std::string& values)
{
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    return testkit::makeFrameWithGraph(
            compressor,
            graphs::Store{}(),
            Input::refSerial(values.data(), values.size()));
}

TEST(DecodeExecutorTest, ChoosesEveryKernelWhilePlanning)
{
    // Each chunk decodes its exponents with the constant codec, then
    // float_deconstruct, so each wave is one batch of two runs. This fails if
    // planning groups the runs differently, or touches the device.
    std::vector<float> values;
    const StagedHostFrame frame{ makeConstantExponentFloatFrame(values) };
    RecordingDevice device;

    const PlannedDecode decode =
            valueOf(planDecodeExecution(device, frame.views()));

    ASSERT_EQ(decode.batches.size(), 2);
    EXPECT_EQ(decode.batches[0].codecRuns.size(), 2);
    EXPECT_EQ(decode.batches[1].codecRuns.size(), 2);
    EXPECT_TRUE(device.events.empty());
}

TEST(DecodeExecutorTest, EnqueuesEachBatchInWaveOrder)
{
    // This fails if the batches change order, or if the workspace outlives
    // them.
    std::vector<float> values;
    const StagedHostFrame frame{ makeConstantExponentFloatFrame(values) };
    RecordingDevice device;
    const PlannedDecode decode =
            valueOf(planDecodeExecution(device, frame.views()));
    std::vector<std::byte> dst(values.size() * sizeof(float));
    device.events.clear();

    ZL_Report const result = enqueueDecode(
            device.defaultStream(),
            decode,
            frame.views(),
            dst.data(),
            dst.size());

    ASSERT_FALSE(ZL_isError(result));
    EXPECT_EQ(ZL_validResult(result), dst.size());
    const std::string constant =
            std::to_string(ZL_StandardTransformID_constant_serial);
    const std::string floatDeconstruct =
            std::to_string(ZL_StandardTransformID_float_deconstruct);
    const std::vector<std::string> expected{
        "allocate",
        "decode " + constant + " " + constant,
        "decode " + floatDeconstruct + " " + floatDeconstruct,
        "copy 0",
        "deallocate",
    };
    EXPECT_EQ(device.events, expected);
}

TEST(DecodeExecutorTest, AllocatesAndFreesEveryWorkspaceArena)
{
    // Another allocator can lay out more than one workspace arena. This adds
    // an arena that no requirement uses. It fails if enqueue allocates only
    // one workspace arena, or frees fewer than it allocated.
    std::vector<float> values;
    const StagedHostFrame frame{ makeConstantExponentFloatFrame(values) };
    RecordingDevice device;
    PlannedDecode decode = valueOf(planDecodeExecution(device, frame.views()));
    decode.layout.arenas.push_back(
            { .arenaClass = ArenaClass::StreamArena,
              .size       = 64,
              .alignment  = 256 });
    std::vector<std::byte> dst(values.size() * sizeof(float));
    device.events.clear();

    ZL_Report const result = enqueueDecode(
            device.defaultStream(),
            decode,
            frame.views(),
            dst.data(),
            dst.size());

    ASSERT_FALSE(ZL_isError(result));
    const auto count = [&](const std::string& event) {
        return std::count(device.events.begin(), device.events.end(), event);
    };
    EXPECT_EQ(count("allocate"), 2);
    EXPECT_EQ(count("deallocate"), 2);
}

TEST(DecodeExecutorTest, GivesEachVariantOfACodecItsOwnBatch)
{
    // A codec's variants can need different kernels. The two constant runs
    // differ only in size, so a size rule stands in for a variant in the codec
    // header. This fails if the variants share a batch, if the batches of the
    // wave do not get branches of their own, or if the next wave starts before
    // the join.
    std::vector<float> values;
    const StagedHostFrame frame{ makeConstantExponentFloatFrame(values) };
    RecordingDevice device{ variantTable() };
    const PlannedDecode decode =
            valueOf(planDecodeExecution(device, frame.views()));
    std::vector<std::byte> dst(values.size() * sizeof(float));
    device.events.clear();

    ZL_Report const result = enqueueDecode(
            device.defaultStream(),
            decode,
            frame.views(),
            dst.data(),
            dst.size());

    ASSERT_FALSE(ZL_isError(result));
    const std::string constant =
            std::to_string(ZL_StandardTransformID_constant_serial);
    const std::string floatDeconstruct =
            std::to_string(ZL_StandardTransformID_float_deconstruct);
    const std::vector<std::string> expected{
        "allocate",
        "fork 2",
        "decode small " + constant + " on branch 0",
        "decode " + constant + " on branch 1",
        "join 2",
        "decode " + floatDeconstruct + " " + floatDeconstruct,
        "copy 0",
        "deallocate",
    };
    EXPECT_EQ(device.events, expected);
}

TEST(DecodeExecutorTest, JoinsTheBranchesWhenABatchThrows)
{
    // A kernel's decode can throw. This fails if the exception skips the
    // join, so the workspace is freed while branch work can still run.
    std::vector<float> values;
    const StagedHostFrame frame{ makeConstantExponentFloatFrame(values) };
    RecordingDevice device{ variantTable() };
    const PlannedDecode decode =
            valueOf(planDecodeExecution(device, frame.views()));
    std::vector<std::byte> dst(values.size() * sizeof(float));
    recordingAnswers().throwSmall = true;
    device.events.clear();

    ZL_Report const result = enqueueDecode(
            device.defaultStream(),
            decode,
            frame.views(),
            dst.data(),
            dst.size());

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_allocation);
    const std::vector<std::string> expected{
        "allocate",
        "fork 2",
        "join 2",
        "deallocate",
    };
    EXPECT_EQ(device.events, expected);
}

TEST(DecodeExecutorTest, JoinsTheBranchesWhenABatchFails)
{
    // The first batch of a two-batch wave fails. This fails if the executor
    // returns, or frees the workspace, before it joins the branches, or if it
    // enqueues more work after the failure.
    std::vector<float> values;
    const StagedHostFrame frame{ makeConstantExponentFloatFrame(values) };
    RecordingDevice device{ variantTable() };
    const PlannedDecode decode =
            valueOf(planDecodeExecution(device, frame.views()));
    std::vector<std::byte> dst(values.size() * sizeof(float));
    recordingAnswers().failSmall = true;
    device.events.clear();

    ZL_Report const result = enqueueDecode(
            device.defaultStream(),
            decode,
            frame.views(),
            dst.data(),
            dst.size());

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_transform_executionFailure);
    const std::vector<std::string> expected{
        "allocate",
        "fork 2",
        "join 2",
        "deallocate",
    };
    EXPECT_EQ(device.events, expected);
}

TEST(DecodeExecutorTest, PlanningRejectsARunThatNoKernelAccepts)
{
    // A caller falls back to the CPU on this error, before it allocates. This
    // fails if planning succeeds, or touches the device.
    std::vector<float> values;
    const StagedHostFrame frame{ makeConstantExponentFloatFrame(values) };
    RecordingDevice device;
    recordingAnswers().accepts = false;

    Result<PlannedDecode> const decode =
            planDecodeExecution(device, frame.views());

    EXPECT_EQ(
            ZL_errorCode(decode.report()),
            ZL_ErrorCode_temporaryLibraryLimitation);
    EXPECT_TRUE(device.events.empty());
}

TEST(DecodeExecutorTest, PlanningRejectsACodecThatTheDeviceLacks)
{
    // float_deconstruct runs in the second wave. This fails if planning checks
    // only the first wave, or leaves the check to enqueueDecode.
    std::vector<float> values;
    const StagedHostFrame frame{ makeConstantExponentFloatFrame(values) };
    RecordingDevice device{ recordingTable(
            ZL_StandardTransformID_float_deconstruct) };

    Result<PlannedDecode> const decode =
            planDecodeExecution(device, frame.views());

    EXPECT_EQ(
            ZL_errorCode(decode.report()),
            ZL_ErrorCode_temporaryLibraryLimitation);
}

TEST(DecodeExecutorTest, PlansAStoredFrameOnADeviceWithNoKernels)
{
    // A stored frame runs no codec. This fails if planning asks the device for
    // a kernel that the decode does not run.
    const std::string values(5000, 'x');
    const StagedHostFrame frame{ makeStoredSerialFrame(values) };
    const std::unique_ptr<Device> device = testkit::makeHostMemoryDevice();

    Result<PlannedDecode> const decode =
            planDecodeExecution(*device, frame.views());

    EXPECT_FALSE(decode.isError());
}

TEST(DecodeExecutorTest, CopiesAStoredOutputIntoTheDestination)
{
    // This fails if a stored final output does not reach the caller's buffer.
    std::string values(5000, '\0');
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<char>(i * 31);
    }
    const StagedHostFrame frame{ makeStoredSerialFrame(values) };
    RecordingDevice device;
    const PlannedDecode decode =
            valueOf(planDecodeExecution(device, frame.views()));
    std::string dst(values.size(), '\0');

    ZL_Report const result = enqueueDecode(
            device.defaultStream(),
            decode,
            frame.views(),
            dst.data(),
            dst.size());

    ASSERT_FALSE(ZL_isError(result));
    EXPECT_EQ(ZL_validResult(result), values.size());
    EXPECT_EQ(dst, values);
}

TEST(DecodeExecutorTest, RejectsADestinationThatIsTooSmall)
{
    // This fails if a decode starts that cannot fit in the caller's buffer.
    const std::string values(5000, 'x');
    const StagedHostFrame frame{ makeStoredSerialFrame(values) };
    RecordingDevice device;
    const PlannedDecode decode =
            valueOf(planDecodeExecution(device, frame.views()));
    std::string dst(values.size() - 1, '\0');

    ZL_Report const result = enqueueDecode(
            device.defaultStream(),
            decode,
            frame.views(),
            dst.data(),
            dst.size());

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_dstCapacity_tooSmall);
    EXPECT_TRUE(device.events.empty());
}

TEST(DecodeExecutorTest, RejectsAMisalignedDestination)
{
    // Core requires a float output buffer to be 4-byte aligned. This fails if a
    // misaligned buffer reaches the device, or if the error names a missing
    // kernel instead of the caller's buffer.
    std::vector<float> values;
    const StagedHostFrame frame{ makeConstantExponentFloatFrame(values) };
    RecordingDevice device;
    const PlannedDecode decode =
            valueOf(planDecodeExecution(device, frame.views()));
    std::vector<std::byte> dst(values.size() * sizeof(float) + 1);
    device.events.clear();

    ZL_Report const result = enqueueDecode(
            device.defaultStream(),
            decode,
            frame.views(),
            dst.data() + 1,
            dst.size() - 1);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_parameter_invalid);
    EXPECT_TRUE(device.events.empty());
}

} // namespace
} // namespace openzl::gpu
