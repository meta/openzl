// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "contrib/gpu/src/codecs/float_deconstruct/tests/bf16_test_frames.h"
#include "contrib/gpu/src/common/cuda_error.cuh"
#include "contrib/gpu/src/common/cuda_raii.cuh"
#include "contrib/gpu/src/decompress/gpu_decompress.h"
#include "contrib/gpu/testkit/frame_factory.h"
#include "contrib/gpu/testkit/multichunk_frame.h"
#include "openzl/openzl.hpp"

namespace openzl::gpu {
namespace {

/**
 * Decompresses @p frame with @c ZL_GPU_decompress from device memory.
 *
 * @returns The decompressed bytes, resized to the reported size, or empty on
 * error, with the error code in @p code.
 */
std::vector<std::byte> decompressOnGpu(
        const std::string& frame,
        size_t dstCapacity,
        ZL_ErrorCode& code)
{
    const DevicePtr<std::byte> src_d = deviceAlloc<std::byte>(frame.size());
    ZL_CUDA_CHECK(cudaMemcpy(
            src_d.get(), frame.data(), frame.size(), cudaMemcpyHostToDevice));
    const DevicePtr<std::byte> dst_d =
            deviceAlloc<std::byte>(dstCapacity == 0 ? 1 : dstCapacity);

    ZL_Report const result = ZL_GPU_decompress(
            dst_d.get(), dstCapacity, src_d.get(), frame.size(), nullptr);
    ZL_CUDA_CHECK(cudaDeviceSynchronize());

    code = ZL_errorCode(result);
    if (ZL_isError(result)) {
        return {};
    }
    std::vector<std::byte> dst_h(ZL_validResult(result));
    ZL_CUDA_CHECK(cudaMemcpy(
            dst_h.data(), dst_d.get(), dst_h.size(), cudaMemcpyDeviceToHost));
    return dst_h;
}

template <typename T>
std::vector<std::byte> bytesOf(const std::vector<T>& values)
{
    const auto* const begin = reinterpret_cast<const std::byte*>(values.data());
    return { begin, begin + values.size() * sizeof(T) };
}

TEST(GpuDecompressEndToEndTest, DecodesBf16FloatDeconstructInTwoChunks)
{
    // Both chunks decode in one batch. This fails if a codec run reads the
    // wrong frame bytes or writes outside its chunk's part of the output.
    const std::vector<uint16_t> values = test::makeBf16Values(150000, 11);
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const GraphID bf16 = compressor.buildStaticGraph(
            nodes::BFloat16Deconstruct::node,
            { graphs::Store{}(), graphs::Store{}() });
    // GPU decoding supports only a serial output, so the frame converts serial
    // bytes to 16-bit numbers first.
    const GraphID graph     = nodes::ConvertSerialToNumLE16{}(compressor, bf16);
    const std::string frame = testkit::makeMultiChunkFrame(
            compressor,
            {
                    testkit::ChunkSpec{ 50000 * sizeof(uint16_t), graph },
                    testkit::ChunkSpec{ 100000 * sizeof(uint16_t), graph },
            },
            Input::refSerial(values.data(), values.size() * sizeof(uint16_t)));
    ZL_ErrorCode code = ZL_ErrorCode_no_error;

    const std::vector<std::byte> decoded =
            decompressOnGpu(frame, values.size() * sizeof(uint16_t), code);

    ASSERT_EQ(code, ZL_ErrorCode_no_error) << ZL_ErrorCode_toString(code);
    EXPECT_EQ(decoded, bytesOf(values));
}

TEST(GpuDecompressEndToEndTest, RejectsANumericOutput)
{
    // The output buffer has no type, so GPU decoding does not support a typed
    // output yet. This fails if the frame decodes, or if the error is not the
    // library limitation that tells the caller to decode on the CPU.
    std::vector<uint32_t> values(20000);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<uint32_t>(i * 3);
    }
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const std::string frame = testkit::makeFrameWithGraph(
            compressor,
            graphs::Store{}(),
            Input::refNumeric<uint32_t>(values.data(), values.size()));
    ZL_ErrorCode code = ZL_ErrorCode_no_error;

    (void)decompressOnGpu(frame, values.size() * sizeof(uint32_t), code);

    EXPECT_EQ(code, ZL_ErrorCode_temporaryLibraryLimitation)
            << ZL_ErrorCode_toString(code);
}

TEST(GpuDecompressEndToEndTest, CopiesAStoredFrame)
{
    // No codec runs. This fails if the terminal copy of a stored output does
    // not reach the caller's buffer.
    std::string values(70000, '\0');
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<char>(i * 7);
    }
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const std::string frame = testkit::makeFrameWithGraph(
            compressor,
            graphs::Store{}(),
            Input::refSerial(values.data(), values.size()));
    ZL_ErrorCode code = ZL_ErrorCode_no_error;

    const std::vector<std::byte> decoded =
            decompressOnGpu(frame, values.size(), code);

    ASSERT_EQ(code, ZL_ErrorCode_no_error) << ZL_ErrorCode_toString(code);
    const auto* const begin = reinterpret_cast<const std::byte*>(values.data());
    EXPECT_EQ(decoded, std::vector<std::byte>(begin, begin + values.size()));
}

TEST(GpuDecompressEndToEndTest, ReportsACodecWithoutAGpuKernel)
{
    // The planner sizes float32 float_deconstruct, but no CUDA kernel decodes
    // it. This fails if the decode starts anyway or reports success.
    std::vector<float> values(20000);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i) * 0.5F;
    }
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    const std::string frame = testkit::makeFrameNodeToStore(
            compressor,
            nodes::Float32Deconstruct::node,
            Input::refNumeric<float>(values.data(), values.size()));
    ZL_ErrorCode code = ZL_ErrorCode_no_error;

    (void)decompressOnGpu(frame, values.size() * sizeof(float), code);

    EXPECT_EQ(code, ZL_ErrorCode_temporaryLibraryLimitation);
}

} // namespace
} // namespace openzl::gpu
