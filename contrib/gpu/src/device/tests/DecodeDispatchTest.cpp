// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "contrib/gpu/src/device/decode_dispatch.hpp"
#include "openzl/common/errors_internal.h"

namespace openzl::gpu {
namespace {

// A run's frameFormatVersion is its ID in these tests, and a one-byte codec
// header marks a run that only the "fallback" kernel accepts.
constexpr ZL_IDType kSupportedTransform = 1;
constexpr ZL_IDType kUnknownTransform   = 3;

struct Enqueued {
    std::string kernel;
    std::vector<uint32_t> runIds;
    void* scratch_d     = nullptr;
    size_t scratchBytes = 0;
    void* stream        = nullptr;

    bool operator==(const Enqueued& other) const
    {
        return kernel == other.kernel && runIds == other.runIds
                && scratch_d == other.scratch_d
                && scratchBytes == other.scratchBytes && stream == other.stream;
    }
};

std::vector<Enqueued>& enqueued()
{
    static std::vector<Enqueued> calls;
    return calls;
}

ZL_Report record(
        const char* kernel,
        poly::span<const BoundCodecRun> runs,
        void* scratch_d,
        size_t scratchBytes,
        void* stream)
{
    Enqueued call{ kernel, {}, scratch_d, scratchBytes, stream };
    for (const BoundCodecRun& run : runs) {
        call.runIds.push_back(run.info.frameFormatVersion);
    }
    enqueued().push_back(call);
    return ZL_returnSuccess();
}

bool acceptsAll(const CodecRunInfo&)
{
    return true;
}

bool acceptsUnmarked(const CodecRunInfo& run)
{
    return run.codecHeader_h.empty();
}

ZL_Report noScratch(poly::span<const CodecRunInfo>, ScratchSize& size)
{
    size = {};
    return ZL_returnSuccess();
}

ZL_Report decodeAsFast(
        poly::span<const BoundCodecRun> runs,
        void* scratch_d,
        size_t scratchBytes,
        void* stream)
{
    return record("fast", runs, scratch_d, scratchBytes, stream);
}

ZL_Report decodeAsFallback(
        poly::span<const BoundCodecRun> runs,
        void* scratch_d,
        size_t scratchBytes,
        void* stream)
{
    return record("fallback", runs, scratch_d, scratchBytes, stream);
}

ZL_Report fail(poly::span<const BoundCodecRun>, void*, size_t, void*)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    ZL_ERR(transform_executionFailure, "injected failure");
}

ZL_Report limit(poly::span<const BoundCodecRun>, void*, size_t, void*)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    ZL_ERR(temporaryLibraryLimitation, "injected limitation");
}

constexpr DecodeKernel kFast{ "fast",
                              acceptsUnmarked,
                              noScratch,
                              decodeAsFast };
constexpr DecodeKernel kFallback{ "fallback",
                                  acceptsAll,
                                  noScratch,
                                  decodeAsFallback };
constexpr DecodeKernel kFailing{ "failing", acceptsAll, noScratch, fail };
constexpr DecodeKernel kLimited{ "limited", acceptsAll, noScratch, limit };

const DecodeKernelTable& testTable()
{
    static const DecodeKernelTable table = [] {
        DecodeKernelTable kernels{};
        kernels[kSupportedTransform] = { kFast, kFallback };
        return kernels;
    }();
    return table;
}

const std::byte kMark[1] = {};

CodecRunInfo makeRun(uint32_t id, ZL_IDType trid, bool marked)
{
    CodecRunInfo run;
    run.transform          = { trt_standard, trid };
    run.frameFormatVersion = id;
    if (marked) {
        run.codecHeader_h = kMark;
    }
    return run;
}

/// Each batch's kernel name and run indexes.
std::vector<std::pair<std::string, std::vector<size_t>>> describe(
        const std::vector<DecodeBatch>& batches)
{
    std::vector<std::pair<std::string, std::vector<size_t>>> described;
    described.reserve(batches.size());
    for (const DecodeBatch& batch : batches) {
        described.emplace_back(batch.kernel->name, batch.runs);
    }
    return described;
}

class DecodeDispatchTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        enqueued().clear();
    }
};

TEST_F(DecodeDispatchTest, ListsKernelsOfStandardTransformsOnly)
{
    // Custom transforms share IDs with standard ones. This fails if a custom
    // transform finds a standard transform's kernels, or if an ID past the
    // table reads outside it.
    EXPECT_EQ(
            kernelsFor(testTable(), { trt_standard, kSupportedTransform })
                    .size(),
            2);
    EXPECT_TRUE(kernelsFor(testTable(), { trt_custom, kSupportedTransform })
                        .empty());
    EXPECT_TRUE(
            kernelsFor(
                    testTable(), { trt_standard, ZL_StandardTransformID_end })
                    .empty());
}

TEST_F(DecodeDispatchTest, SelectsFirstAcceptingKernel)
{
    // This fails if selection ignores preference order, or picks a kernel that
    // rejects the run.
    const poly::span<const DecodeKernel> candidates =
            kernelsFor(testTable(), { trt_standard, kSupportedTransform });

    EXPECT_EQ(
            selectDecodeKernel(
                    candidates, makeRun(0, kSupportedTransform, false))
                    ->name,
            std::string{ "fast" });
    EXPECT_EQ(
            selectDecodeKernel(
                    candidates, makeRun(0, kSupportedTransform, true))
                    ->name,
            std::string{ "fallback" });
    EXPECT_EQ(
            selectDecodeKernel({}, makeRun(0, kSupportedTransform, false)),
            nullptr);
    EXPECT_EQ(
            selectDecodeKernel(
                    poly::span<const DecodeKernel>{ &kFast, 1 },
                    makeRun(0, kSupportedTransform, true)),
            nullptr);
}

TEST_F(DecodeDispatchTest, BatchesRunsByKernel)
{
    // This fails if runs that choose the same kernel are split across batches,
    // if a batch reorders its runs, or if batches are not in the order their
    // kernels are first chosen.
    const std::vector<CodecRunInfo> runs{
        makeRun(0, kSupportedTransform, true),
        makeRun(1, kSupportedTransform, false),
        makeRun(2, kSupportedTransform, true),
        makeRun(3, kSupportedTransform, false),
    };
    std::vector<DecodeBatch> batches;

    ZL_Report const result = chooseDecodeBatches(testTable(), runs, batches);

    ASSERT_FALSE(ZL_isError(result));
    const std::vector<std::pair<std::string, std::vector<size_t>>> expected{
        { "fallback", { 0, 2 } },
        { "fast", { 1, 3 } },
    };
    EXPECT_EQ(describe(batches), expected);
}

TEST_F(DecodeDispatchTest, RejectsARunThatNoKernelAccepts)
{
    // A caller falls back to the CPU on this error. This fails if the error
    // differs, or if the batches of the runs before it are returned.
    const std::vector<CodecRunInfo> runs{
        makeRun(0, kSupportedTransform, false),
        makeRun(1, kUnknownTransform, false),
    };
    std::vector<DecodeBatch> batches;

    ZL_Report const result = chooseDecodeBatches(testTable(), runs, batches);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_temporaryLibraryLimitation);
    EXPECT_TRUE(batches.empty());
}

TEST_F(DecodeDispatchTest, PassesTheScratchAndTheStreamToTheKernel)
{
    // This fails if the kernel receives a scratch or a stream other than the
    // caller's.
    int marker[2]{};
    std::vector<BoundCodecRun> runs(2);
    runs[0].info.frameFormatVersion = 0;
    runs[1].info.frameFormatVersion = 1;

    ZL_Report const result =
            decodeBatchWith(kFast, runs, &marker[0], 64, &marker[1]);

    ASSERT_FALSE(ZL_isError(result));
    const std::vector<Enqueued> expected{
        { "fast", { 0, 1 }, &marker[0], 64, &marker[1] },
    };
    EXPECT_EQ(enqueued(), expected);
}

TEST_F(DecodeDispatchTest, ForwardsAKernelError)
{
    // This fails if a kernel's error is replaced.
    const std::vector<BoundCodecRun> runs(1);

    ZL_Report const result =
            decodeBatchWith(kFailing, runs, nullptr, 0, nullptr);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_transform_executionFailure);
}

TEST_F(DecodeDispatchTest, ReportsALateLimitationAsAFailure)
{
    // A caller decodes on the CPU after temporaryLibraryLimitation. This fails
    // if a kernel can return that code after earlier batches may have enqueued
    // work.
    const std::vector<BoundCodecRun> runs(1);

    ZL_Report const result =
            decodeBatchWith(kLimited, runs, nullptr, 0, nullptr);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_transform_executionFailure);
}

TEST_F(DecodeDispatchTest, EnqueuesNothingForAnEmptyBatch)
{
    // This fails if an empty batch reaches a kernel.
    ZL_Report const result = decodeBatchWith(
            kFast, poly::span<const BoundCodecRun>{}, nullptr, 0, nullptr);

    ASSERT_FALSE(ZL_isError(result));
    EXPECT_TRUE(enqueued().empty());
}

} // namespace
} // namespace openzl::gpu
