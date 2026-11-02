// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "contrib/gpu/src/decompress/gpu_decompress.hpp"
#include "contrib/gpu/testkit/frame_verifier.h"
#include "contrib/gpu/testkit/host_memory_device.h"
#include "contrib/gpu/testkit/multichunk_frame.h"
#include "openzl/common/errors_internal.h"
#include "openzl/cpp/poly/Span.hpp"
#include "openzl/openzl.hpp"
#include "openzl/zl_decompress.h"
#include "openzl/zl_version.h"

namespace openzl {
namespace tests {
namespace {

constexpr size_t kEltsPerChunk = 20000;

/** A test device backed by host memory; these tests run without a GPU. */
gpu::Device& hostDevice()
{
    static const std::unique_ptr<gpu::Device> device =
            gpu::testkit::makeHostMemoryDevice();
    return *device;
}

gpu::Stream hostStream()
{
    return hostDevice().defaultStream();
}

std::string makeTwoChunkFrameWithMultiCodecChunk()
{
    std::vector<int32_t> values(2 * kEltsPerChunk);
    for (size_t i = 0; i < kEltsPerChunk; ++i) {
        values[i]                 = static_cast<int32_t>(i % 8);
        values[kEltsPerChunk + i] = static_cast<int32_t>(i) * 3;
    }

    const Input input =
            Input::refNumeric<int32_t>(values.data(), values.size());
    Compressor compressor;
    const GraphID bitpack      = graphs::Bitpack{}();
    const GraphID deltaToStore = ZL_Compressor_registerStaticGraph_fromNode1o(
            compressor.get(), nodes::DeltaInt::node, graphs::Store{}());
    return gpu::testkit::makeMultiChunkFrame(
            compressor,
            {
                    gpu::testkit::ChunkSpec{ kEltsPerChunk, bitpack },
                    gpu::testkit::ChunkSpec{ kEltsPerChunk, deltaToStore },
            },
            input);
}

std::string makeLegacyFrame()
{
    const std::string input(1000, 'x');
    Compressor compressor;
    compressor.selectStartingGraph(graphs::Store{}());

    CCtx cctx;
    cctx.refCompressor(compressor);
    cctx.setParameter(CParam::FormatVersion, ZL_CHUNK_VERSION_MIN - 1);
    return cctx.compressOne(Input::refSerial(input));
}

poly::span<const std::byte> bytes(const std::string& frame)
{
    return {
        reinterpret_cast<const std::byte*>(frame.data()),
        frame.size(),
    };
}

TEST(GPUDecompressTest, ViewOfGivesThePlannerEveryFieldOfTheChunk)
{
    // The planner rejects chunks of more than one frame by frameIndex. This
    // fails if a view leaves out the chunk's frame, or points at other data.
    gpu::PreparedGPUChunk chunk{
        .chunk              = { .frameHeaderIdx = 2 },
        .transformHeaders_h = std::vector<std::byte>(3),
    };

    const gpu::PreparedGpuChunkView view = gpu::viewOf(chunk);

    EXPECT_EQ(view.dctx, chunk.dctx.get());
    EXPECT_EQ(view.transformHeaders_h.data(), chunk.transformHeaders_h.data());
    EXPECT_EQ(view.transformHeaders_h.size(), 3);
    EXPECT_EQ(view.frameIndex, 2);
}

TEST(GPUDecompressTest, CollectGPUChunksFromFrameEnumeratesEveryChunk)
{
    // This test verifies that one frame produces an ordered descriptor for
    // every chunk; it fails if enumeration skips a chunk or miscomputes a
    // device pointer, frame-header association, or chunk boundary. The second
    // chunk uses two codecs so its formal header exercises multiple decoder
    // nodes.
    const std::string frame = makeTwoChunkFrameWithMultiCodecChunk();
    const std::vector<std::vector<uint32_t>> codecsPerChunk =
            gpu::testkit::standardCodecsPerChunk(frame);
    ASSERT_EQ(codecsPerChunk.size(), 2);
    EXPECT_EQ(codecsPerChunk[0].size(), 1);
    EXPECT_EQ(codecsPerChunk[1].size(), 2);
    std::vector<std::byte> deviceFrame(frame.size());
    std::vector<gpu::GPUChunk> chunks;
    std::vector<gpu::GPUFrameHeaderForChunks> frameHeaders;

    ZL_Report const result = gpu::collectGPUChunksFromFrame(
            deviceFrame.data(), bytes(frame), chunks, frameHeaders);
    ASSERT_FALSE(ZL_isError(result));
    EXPECT_EQ(ZL_validResult(result), frame.size());
    ASSERT_EQ(chunks.size(), 2);
    ASSERT_EQ(frameHeaders.size(), 1);

    ZL_Report const headerResult = ZL_getHeaderSize(frame.data(), frame.size());
    ASSERT_FALSE(ZL_isError(headerResult));
    size_t const frameHeaderSize = ZL_validResult(headerResult);
    EXPECT_EQ(frameHeaders[0].frameHeader_d, deviceFrame.data());
    EXPECT_EQ(frameHeaders[0].frameHeaderSize, frameHeaderSize);

    size_t chunkOffset = frameHeaderSize;
    for (const gpu::GPUChunk& chunk : chunks) {
        EXPECT_EQ(chunk.frameHeaderIdx, 0);
        EXPECT_EQ(chunk.chunk_d, deviceFrame.data() + chunkOffset);
        EXPECT_GT(chunk.chunkHeaderSize, 0);
        EXPECT_LE(chunk.chunkHeaderSize, chunk.chunkSize);
        chunkOffset += chunk.chunkSize;
    }
    ASSERT_EQ(chunkOffset + 1, frame.size());
    EXPECT_EQ(frame[chunkOffset], '\0');
}

TEST(GPUDecompressTest, CollectGPUChunksFromFrameAppendsToExistingBatch)
{
    // This test verifies that frames can be collected into one batch across
    // repeated calls; it fails if a later frame replaces earlier descriptors
    // or its chunks reference the wrong frame-header index.
    const std::string frame = makeTwoChunkFrameWithMultiCodecChunk();
    std::vector<std::byte> firstDeviceFrame(frame.size());
    std::vector<std::byte> secondDeviceFrame(frame.size());
    std::vector<gpu::GPUChunk> chunks;
    std::vector<gpu::GPUFrameHeaderForChunks> frameHeaders;

    ZL_Report const firstResult = gpu::collectGPUChunksFromFrame(
            firstDeviceFrame.data(), bytes(frame), chunks, frameHeaders);
    ASSERT_FALSE(ZL_isError(firstResult));
    ZL_Report const secondResult = gpu::collectGPUChunksFromFrame(
            secondDeviceFrame.data(), bytes(frame), chunks, frameHeaders);
    ASSERT_FALSE(ZL_isError(secondResult));

    ASSERT_EQ(frameHeaders.size(), 2);
    EXPECT_EQ(frameHeaders[0].frameHeader_d, firstDeviceFrame.data());
    EXPECT_EQ(frameHeaders[1].frameHeader_d, secondDeviceFrame.data());
    ASSERT_EQ(chunks.size(), 4);
    EXPECT_EQ(chunks[0].frameHeaderIdx, 0);
    EXPECT_EQ(chunks[1].frameHeaderIdx, 0);
    EXPECT_EQ(chunks[2].frameHeaderIdx, 1);
    EXPECT_EQ(chunks[3].frameHeaderIdx, 1);

    ZL_Report const headerResult = ZL_getHeaderSize(frame.data(), frame.size());
    ASSERT_FALSE(ZL_isError(headerResult));
    size_t const frameHeaderSize = ZL_validResult(headerResult);
    EXPECT_EQ(chunks[0].chunk_d, firstDeviceFrame.data() + frameHeaderSize);
    EXPECT_EQ(chunks[2].chunk_d, secondDeviceFrame.data() + frameHeaderSize);
}

TEST(GPUDecompressTest, DecompressChunksRejectsUnknownFrameHeader)
{
    // This test verifies that every chunk names a supplied frame header; it
    // fails if an out-of-range index reaches device-to-host staging.
    std::byte sentinel{};
    const std::vector<gpu::GPUFrameHeaderForChunks> frameHeaders{
        {
                .frameHeader_d   = &sentinel,
                .frameHeaderSize = 1,
        },
    };
    const std::vector<gpu::GPUChunk> chunks{
        {
                .frameHeaderIdx  = frameHeaders.size(),
                .chunk_d         = &sentinel,
                .chunkSize       = 1,
                .chunkHeaderSize = 1,
        },
    };

    ZL_Report const result = gpu::decompressChunks(
            hostStream(), nullptr, 0, frameHeaders, chunks);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_parameter_invalid);
}

TEST(GPUDecompressTest, CApiRejectsImpossibleSourceSize)
{
    // This test verifies that a source size no host allocation can represent
    // produces an error report; it fails if the input is accepted or terminates
    // the process.
    ZL_Report const result = ZL_GPU_decompress(
            nullptr, 0, nullptr, std::numeric_limits<size_t>::max(), nullptr);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_srcSize_tooLarge);
}

TEST(GPUDecompressTest, PlansFrameOnHostDevice)
{
    // Chunk 0 uses Bitpack, which has no GPU planner, so decoding stops at
    // planning. This test runs without a GPU, so it fails if any step before
    // planning calls the CUDA runtime instead of the device it receives.
    const std::string frame = makeTwoChunkFrameWithMultiCodecChunk();
    std::vector<gpu::GPUChunk> chunks;
    std::vector<gpu::GPUFrameHeaderForChunks> frameHeaders;
    ASSERT_FALSE(ZL_isError(
            gpu::collectGPUChunks(
                    hostStream(), bytes(frame), chunks, frameHeaders)));

    ZL_Report const result = gpu::decompressChunks(
            hostStream(), nullptr, 0, frameHeaders, chunks);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_temporaryLibraryLimitation);
}

TEST(GPUDecompressTest, CollectGPUChunksFromFrameRejectsLegacyFormat)
{
    // This test verifies the documented v21 format floor; it fails if a legacy
    // frame reaches chunk enumeration even though it has no separate chunks.
    const std::string frame = makeLegacyFrame();
    std::vector<std::byte> deviceFrame(frame.size());
    std::vector<gpu::GPUChunk> chunks;
    std::vector<gpu::GPUFrameHeaderForChunks> frameHeaders;

    ZL_Report const result = gpu::collectGPUChunksFromFrame(
            deviceFrame.data(), bytes(frame), chunks, frameHeaders);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_formatVersion_unsupported);
    EXPECT_TRUE(chunks.empty());
    EXPECT_TRUE(frameHeaders.empty());
}

TEST(GPUDecompressTest, CollectGPUChunksFromFrameRejectsFrameWithoutChunks)
{
    // This test verifies that a frame with no chunks is rejected; it fails if
    // a frame whose header declares output becomes an empty batch that
    // decodes nothing and reports success.
    const std::string fullFrame = makeTwoChunkFrameWithMultiCodecChunk();
    ZL_Report const headerResult =
            ZL_getHeaderSize(fullFrame.data(), fullFrame.size());
    ASSERT_FALSE(ZL_isError(headerResult));
    std::string frame = fullFrame.substr(0, ZL_validResult(headerResult));
    frame.push_back('\0');
    std::vector<std::byte> deviceFrame(frame.size());
    std::vector<gpu::GPUChunk> chunks;
    std::vector<gpu::GPUFrameHeaderForChunks> frameHeaders;

    ZL_Report const result = gpu::collectGPUChunksFromFrame(
            deviceFrame.data(), bytes(frame), chunks, frameHeaders);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_corruption);
    EXPECT_TRUE(chunks.empty());
    EXPECT_TRUE(frameHeaders.empty());
}

TEST(GPUDecompressTest, CollectGPUChunksFromFrameRejectsTrailingBytes)
{
    // This test verifies that enumeration consumes exactly one frame and only
    // appends to its outputs on success; it fails if trailing bytes are
    // accepted or either output is partially modified on error.
    std::string frame = makeTwoChunkFrameWithMultiCodecChunk();
    frame.push_back('\x7f');
    std::vector<std::byte> deviceFrame(frame.size());
    std::byte sentinel{};
    const gpu::GPUChunk expectedChunk{
        .frameHeaderIdx  = 7,
        .chunk_d         = &sentinel,
        .chunkSize       = 11,
        .chunkHeaderSize = 3,
    };
    const gpu::GPUFrameHeaderForChunks expectedFrameHeader{
        .frameHeader_d   = &sentinel,
        .frameHeaderSize = 5,
    };
    std::vector<gpu::GPUChunk> chunks{ expectedChunk };
    std::vector<gpu::GPUFrameHeaderForChunks> frameHeaders{
        expectedFrameHeader
    };

    ZL_Report const result = gpu::collectGPUChunksFromFrame(
            deviceFrame.data(), bytes(frame), chunks, frameHeaders);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_srcSize_tooLarge);
    ASSERT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0].frameHeaderIdx, expectedChunk.frameHeaderIdx);
    EXPECT_EQ(chunks[0].chunk_d, expectedChunk.chunk_d);
    EXPECT_EQ(chunks[0].chunkSize, expectedChunk.chunkSize);
    EXPECT_EQ(chunks[0].chunkHeaderSize, expectedChunk.chunkHeaderSize);
    ASSERT_EQ(frameHeaders.size(), 1);
    EXPECT_EQ(frameHeaders[0].frameHeader_d, expectedFrameHeader.frameHeader_d);
    EXPECT_EQ(
            frameHeaders[0].frameHeaderSize,
            expectedFrameHeader.frameHeaderSize);
}

TEST(GPUDecompressTest, CollectGPUChunksFromFrameRejectsMissingEndMarker)
{
    // This test verifies that a frame must include its terminating marker; it
    // fails if a truncated frame is reported as successfully enumerated.
    std::string frame = makeTwoChunkFrameWithMultiCodecChunk();
    ASSERT_EQ(frame.back(), '\0');
    frame.pop_back();
    std::vector<std::byte> deviceFrame(frame.size());
    std::vector<gpu::GPUChunk> chunks;
    std::vector<gpu::GPUFrameHeaderForChunks> frameHeaders;

    ZL_Report const result = gpu::collectGPUChunksFromFrame(
            deviceFrame.data(), bytes(frame), chunks, frameHeaders);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_srcSize_tooSmall);
    EXPECT_TRUE(chunks.empty());
    EXPECT_TRUE(frameHeaders.empty());
}

TEST(GPUDecompressTest, CollectGPUChunksFromFrameDropsChunkParserErrorInfo)
{
    // The chunk parser keeps an error's info in its DCTX, which
    // collectGPUChunksFromFrame frees before it returns. This fails if the
    // returned error still points into that DCTX.
    std::string frame            = makeTwoChunkFrameWithMultiCodecChunk();
    ZL_Report const headerResult = ZL_getHeaderSize(frame.data(), frame.size());
    ASSERT_FALSE(ZL_isError(headerResult));
    frame.resize(ZL_validResult(headerResult) + 2);
    std::vector<std::byte> deviceFrame(frame.size());
    std::vector<gpu::GPUChunk> chunks;
    std::vector<gpu::GPUFrameHeaderForChunks> frameHeaders;

    ZL_Report const result = gpu::collectGPUChunksFromFrame(
            deviceFrame.data(), bytes(frame), chunks, frameHeaders);

    ASSERT_TRUE(ZL_isError(result));
    EXPECT_EQ(ZL_E_dy(ZL_RES_error(result)), nullptr);
}

TEST(GPUDecompressTest, PrepareChunksRejectsImpossibleStagingSize)
{
    // This fails if a header size that no host buffer can hold throws instead
    // of returning an error.
    std::byte sentinel{};
    const gpu::GPUFrameHeaderForChunks frameHeader{
        .frameHeader_d   = &sentinel,
        .frameHeaderSize = std::numeric_limits<size_t>::max(),
    };

    const gpu::Result<std::vector<gpu::PreparedGPUChunk>> result =
            gpu::prepareChunksForPlanning(
                    hostStream(),
                    poly::span<const gpu::GPUFrameHeaderForChunks>{
                            &frameHeader, 1 },
                    poly::span<const gpu::GPUChunk>{});

    ASSERT_TRUE(result.isError());
    EXPECT_EQ(ZL_errorCode(result.report()), ZL_ErrorCode_allocation);
}

} // namespace
} // namespace tests
} // namespace openzl
