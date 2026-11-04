// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "contrib/gpu/src/codecs/float_deconstruct/decode_float_deconstruct_cuda.hpp"
#include "contrib/gpu/src/codecs/float_deconstruct/tests/bf16_test_frames.h"
#include "contrib/gpu/src/common/cuda_error.cuh"
#include "contrib/gpu/src/common/cuda_raii.cuh"
#include "contrib/gpu/src/device/cuda_device.hpp"
#include "openzl/common/errors_internal.h"
#include "openzl/common/wire_format.h"
#include "openzl/zl_version.h"

namespace openzl::gpu {
namespace {

constexpr std::byte kSentinel{ 0xA5 };

/**
 * One bf16 run whose encoded streams and output live on the device, each at a
 * chosen byte offset from an allocation, so that tests control alignment.
 * Not movable, because the run points into its own members.
 */
struct DeviceBf16Run {
    DeviceBf16Run(
            const std::vector<uint16_t>& values,
            size_t inputOffset,
            size_t outputOffset)
            : numElts{ values.size() }, outputOffset{ outputOffset }
    {
        const test::Bf16Streams streams = test::encodeBf16(values);
        signFrac_d                      = upload(streams.signFrac, inputOffset);
        exponent_d                      = upload(streams.exponent, inputOffset);
        output_d = deviceAlloc<std::byte>(outputBytes() + outputOffset);
        ZL_CUDA_CHECK(cudaMemset(
                output_d.get(),
                std::to_integer<int>(kSentinel),
                outputBytes() + outputOffset));

        inputInfo[0] = {
            .shape            = { ZL_Type_struct, numElts, 1 },
            .availableStorage = { numElts, 0 },
        };
        inputInfo[1] = {
            .shape            = { ZL_Type_serial, numElts, 1 },
            .availableStorage = { numElts, 0 },
        };
        outputInfo[0] = {
            .shape       = { ZL_Type_numeric, numElts, sizeof(uint16_t) },
            .storageSize = { outputBytes(), 0 },
        };
        inputs[0]  = { .data_d = signFrac_d.get() + inputOffset };
        inputs[1]  = { .data_d = exponent_d.get() + inputOffset };
        outputs[0] = { .data_d = output_d.get() + outputOffset };
        run        = {
                   .info    = { .transform          = { trt_standard,
                                                        ZL_StandardTransformID_float_deconstruct },
                                .frameFormatVersion = ZL_MAX_FORMAT_VERSION,
                                .codecHeader_h      = test::kBf16Header,
                                .inputs             = inputInfo,
                                .outputs            = outputInfo },
                   .inputs  = inputs,
                   .outputs = outputs,
        };
    }

    DeviceBf16Run(const DeviceBf16Run&)            = delete;
    DeviceBf16Run& operator=(const DeviceBf16Run&) = delete;

    std::vector<uint16_t> download() const
    {
        std::vector<uint16_t> output(numElts);
        ZL_CUDA_CHECK(cudaMemcpy(
                output.data(),
                output_d.get() + outputOffset,
                outputBytes(),
                cudaMemcpyDeviceToHost));
        return output;
    }

    size_t numElts;
    size_t outputOffset;
    DevicePtr<std::byte> signFrac_d;
    DevicePtr<std::byte> exponent_d;
    DevicePtr<std::byte> output_d;
    std::array<InputStreamInfo, 2> inputInfo{};
    std::array<OutputStreamInfo, 1> outputInfo{};
    std::array<BoundInputStream, 2> inputs{};
    std::array<BoundOutputStream, 1> outputs{};
    BoundCodecRun run{};

   private:
    size_t outputBytes() const
    {
        return numElts * sizeof(uint16_t);
    }

    static DevicePtr<std::byte> upload(
            const std::vector<uint8_t>& bytes,
            size_t offset)
    {
        DevicePtr<std::byte> buffer_d =
                deviceAlloc<std::byte>(bytes.size() + offset);
        ZL_CUDA_CHECK(cudaMemcpy(
                buffer_d.get() + offset,
                bytes.data(),
                bytes.size(),
                cudaMemcpyHostToDevice));
        return buffer_d;
    }
};

const DecodeKernel& kernelNamed(const std::string& name)
{
    for (const DecodeKernel& kernel : floatDeconstructCudaKernels()) {
        if (kernel.name == name) {
            return kernel;
        }
    }
    throw std::runtime_error("no kernel named " + name);
}

/**
 * Decodes @p runs as one batch with @p kernel on the default stream, with
 * scratch of the size that @c DecodeKernel::scratchSize reports, and waits.
 */
ZL_Report decodeBatch(
        const DecodeKernel& kernel,
        poly::span<const BoundCodecRun> runs)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    std::vector<CodecRunInfo> info;
    for (const BoundCodecRun& run : runs) {
        info.push_back(run.info);
    }
    ScratchSize size;
    ZL_ERR_IF_ERR(kernel.scratchSize(info, size));
    // cudaMalloc aligns to 256 bytes, which covers every scratch alignment.
    const DevicePtr<std::byte> scratch_d = deviceAlloc<std::byte>(size.bytes);
    ZL_Report const result =
            kernel.decode(runs, scratch_d.get(), size.bytes, nullptr);
    ZL_CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

/** Runs of the given sizes and offsets, and the values each must decode to. */
struct RunBatch {
    RunBatch(
            const std::vector<size_t>& sizes,
            const std::vector<size_t>& inputOffsets,
            size_t outputOffset)
    {
        for (size_t i = 0; i < sizes.size(); ++i) {
            values.push_back(
                    test::makeBf16Values(
                            sizes[i], static_cast<unsigned>(i + 20)));
            devices.push_back(
                    std::make_unique<DeviceBf16Run>(
                            values.back(), inputOffsets[i], outputOffset));
            runs.push_back(devices.back()->run);
        }
    }

    void expectDecoded() const
    {
        for (size_t i = 0; i < values.size(); ++i) {
            EXPECT_EQ(devices[i]->download(), values[i]) << "run " << i;
        }
    }

    std::vector<std::vector<uint16_t>> values;
    std::vector<std::unique_ptr<DeviceBf16Run>> devices;
    std::vector<BoundCodecRun> runs;
};

const std::vector<size_t> kUnevenSizes{ 50000, 1, 200000, 12345, 7 };

TEST(DecodeFloatDeconstructCudaTest, DecodesAnAlignedBatch)
{
    // Runs on fresh allocations take the vector path. This fails if the kernel
    // rejects such a run, or decodes a value that differs from the one
    // compressed.
    const DecodeKernel& kernel = kernelNamed("float_deconstruct.bf16.cuda");
    const RunBatch batch{ kUnevenSizes, { 0, 0, 0, 0, 0 }, 0 };
    for (const BoundCodecRun& run : batch.runs) {
        ASSERT_TRUE(kernel.accepts(run.info));
    }

    ASSERT_FALSE(ZL_isError(decodeBatch(kernel, batch.runs)));
    ZL_CUDA_CHECK(cudaDeviceSynchronize());

    batch.expectDecoded();
}

TEST(DecodeFloatDeconstructCudaTest, DecodesMisalignedRuns)
{
    // Inputs stored in a frame start at any offset, and a later chunk's output
    // can start at any 2-byte boundary. This fails if the kernel rejects such a
    // run, or if a misaligned slice takes the vector path.
    const DecodeKernel& kernel = kernelNamed("float_deconstruct.bf16.cuda");
    const RunBatch batch{ kUnevenSizes, { 1, 3, 2, 1, 3 }, 2 };
    for (const BoundCodecRun& run : batch.runs) {
        ASSERT_TRUE(kernel.accepts(run.info));
    }

    ASSERT_FALSE(ZL_isError(decodeBatch(kernel, batch.runs)));
    ZL_CUDA_CHECK(cudaDeviceSynchronize());

    batch.expectDecoded();
}

TEST(DecodeFloatDeconstructCudaTest, DecodesAMixedBatchInOneCall)
{
    // Aligned runs take the vector path and misaligned ones the element path,
    // inside one launch. This fails if a run is dropped or decoded into another
    // run's output.
    const DecodeKernel& kernel = kernelNamed("float_deconstruct.bf16.cuda");
    const RunBatch batch{ kUnevenSizes, { 0, 3, 0, 1, 2 }, 0 };

    ASSERT_FALSE(ZL_isError(decodeBatch(kernel, batch.runs)));
    ZL_CUDA_CHECK(cudaDeviceSynchronize());

    batch.expectDecoded();
}

TEST(DecodeFloatDeconstructCudaTest,
     DecodesRunsWhoseOnlyMisalignedAddressIsTheDestination)
{
    // The inputs are aligned, and each output starts 2 bytes into its
    // allocation. This fails if the destination is not checked before the
    // vector path, whose 8-byte stores would then be misaligned.
    const DecodeKernel& kernel = kernelNamed("float_deconstruct.bf16.cuda");
    const RunBatch batch{ kUnevenSizes, { 0, 0, 0, 0, 0 }, 2 };

    ASSERT_FALSE(ZL_isError(decodeBatch(kernel, batch.runs)));
    ZL_CUDA_CHECK(cudaDeviceSynchronize());

    batch.expectDecoded();
}

TEST(DecodeFloatDeconstructCudaTest, IgnoresAnErrorThatTheCallerLeftPending)
{
    // This fails if the decode reports an error that the caller left pending
    // as its own, or clears it. A dev build links a separate CUDA runtime, with
    // its own pending error, into each library, so only a build with one
    // runtime, such as opt, can catch that.
    const DecodeKernel& kernel = kernelNamed("float_deconstruct.bf16.cuda");
    const RunBatch batch{ { 1000 }, { 0 }, 0 };
    ScratchSize size;
    ASSERT_FALSE(
            ZL_isError(kernel.scratchSize({ &batch.runs[0].info, 1 }, size)));
    const DevicePtr<std::byte> scratch_d = deviceAlloc<std::byte>(size.bytes);
    ASSERT_EQ(cudaSetDevice(-1), cudaErrorInvalidDevice);

    ZL_Report const result =
            kernel.decode(batch.runs, scratch_d.get(), size.bytes, nullptr);

    EXPECT_FALSE(ZL_isError(result));
    EXPECT_EQ(cudaGetLastError(), cudaErrorInvalidDevice);
    ZL_CUDA_CHECK(cudaDeviceSynchronize());
    batch.expectDecoded();
}

TEST(DecodeFloatDeconstructCudaTest, RejectsAnOddDestinationBeforeLaunching)
{
    // The kernel stores bf16 values with 2-byte stores, which need an even
    // address. The layout never gives an odd one. This fails if a caller that
    // breaks that rule gets a launch, or anything is written to its output.
    const DecodeKernel& kernel = kernelNamed("float_deconstruct.bf16.cuda");
    const RunBatch batch{ { 1000 }, { 0 }, 1 };

    ZL_Report const result = decodeBatch(kernel, batch.runs);
    ZL_CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_parameter_invalid);
    std::vector<std::byte> output(1000 * sizeof(uint16_t));
    ZL_CUDA_CHECK(cudaMemcpy(
            output.data(),
            batch.devices[0]->output_d.get() + 1,
            output.size(),
            cudaMemcpyDeviceToHost));
    EXPECT_EQ(output, std::vector<std::byte>(output.size(), kSentinel));
}

} // namespace
} // namespace openzl::gpu
