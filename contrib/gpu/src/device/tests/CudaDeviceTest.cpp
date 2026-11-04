// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "contrib/gpu/src/common/cuda_error.cuh"
#include "contrib/gpu/src/common/cuda_raii.cuh"
#include "contrib/gpu/src/device/cuda_device.hpp"

namespace openzl::gpu {
namespace {

struct PinnedFree {
    void operator()(std::byte* ptr_h) const noexcept
    {
        (void)cudaFreeHost(ptr_h);
    }
};

struct StreamDestroy {
    void operator()(cudaStream_t stream) const noexcept
    {
        (void)cudaStreamDestroy(stream);
    }
};

/// Keeps a stream busy, so work enqueued behind it has not run when the call
/// that enqueued it returns.
void CUDART_CB holdStream(void*)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

int currentDevice()
{
    int device = 0;
    ZL_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

TEST(CudaDeviceTest, CopiesEveryNonEmptyRegion)
{
    // The destination is pinned and the stream is busy, so each copy runs
    // after the call enqueues it. This fails if a region is skipped or copied
    // from the wrong offset, or if the call returns before the copies finish.
    const std::vector<std::byte> expected{
        std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 }, std::byte{ 4 }
    };
    const DevicePtr<std::byte> source_d = deviceAlloc<std::byte>(4);
    ZL_CUDA_CHECK(cudaMemcpy(
            source_d.get(), expected.data(), 4, cudaMemcpyHostToDevice));
    std::byte* pinned_h = nullptr;
    ZL_CUDA_CHECK(cudaMallocHost(&pinned_h, 4));
    const std::unique_ptr<std::byte, PinnedFree> actual_h{ pinned_h };
    std::memset(actual_h.get(), 0, 4);
    const poly::span<std::byte> actual{ actual_h.get(), 4 };
    const std::vector<DeviceToHostCopy> copies{
        { .dst_h = actual.subspan(0, 1), .src_d = source_d.get() },
        { .dst_h = {}, .src_d = nullptr },
        { .dst_h = actual.subspan(1), .src_d = source_d.get() + 1 },
    };
    cudaStream_t native = nullptr;
    ZL_CUDA_CHECK(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking));
    const std::unique_ptr<CUstream_st, StreamDestroy> streamOwner{ native };
    const std::unique_ptr<Device> device = makeCurrentCudaDevice();
    ASSERT_NE(device, nullptr);
    std::optional<Stream> stream;
    ASSERT_FALSE(ZL_isError(device->wrapStream(native, stream)));
    ZL_CUDA_CHECK(cudaLaunchHostFunc(native, holdStream, nullptr));

    ZL_Report const result = stream->copyToHostAndSynchronize(copies);

    ASSERT_FALSE(ZL_isError(result));
    EXPECT_EQ(std::vector<std::byte>(actual.begin(), actual.end()), expected);
}

TEST(CudaDeviceTest, RejectsCodecsThatItDoesNotImplement)
{
    // No runtime implements custom transforms, and the CUDA device has no
    // constant decoder. This fails if the device chooses a kernel for a run
    // that it cannot decode, or changes the caller's current device.
    const int callerDevice               = currentDevice();
    const std::unique_ptr<Device> device = makeCurrentCudaDevice();
    ASSERT_NE(device, nullptr);
    for (const PublicTransformInfo transform :
         { PublicTransformInfo{ trt_custom, 0 },
           PublicTransformInfo{ trt_standard,
                                ZL_StandardTransformID_constant_serial } }) {
        CodecRunInfo run;
        run.transform = transform;
        std::vector<DecodeBatch> batches;

        ZL_Report const result = device->chooseDecodeBatches(
                poly::span<const CodecRunInfo>{ &run, 1 }, batches);

        EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_temporaryLibraryLimitation)
                << transform.trid;
    }
    EXPECT_EQ(currentDevice(), callerDevice);
}

ZL_Report
failWithoutCudaCall(poly::span<const BoundCodecRun>, void*, size_t, void*)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    ZL_ERR(transform_executionFailure, "injected failure");
}

TEST(CudaDeviceTest, KeepsAnErrorThatTheCallerLeftPending)
{
    // The kernel fails without a CUDA call. This fails if the device clears an
    // error that it did not cause. A dev build links a separate CUDA runtime,
    // with its own pending error, into each library, so only a build with one
    // runtime, such as opt, can catch that.
    const DecodeKernel failing{
        "failing", nullptr, nullptr, failWithoutCudaCall
    };
    const BoundCodecRun run{};
    const std::unique_ptr<Device> device = makeCurrentCudaDevice();
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(cudaSetDevice(-1), cudaErrorInvalidDevice);

    ZL_Report const result = device->defaultStream().decodeBatch(
            failing, poly::span<const BoundCodecRun>{ &run, 1 }, nullptr, 0);

    EXPECT_EQ(ZL_errorCode(result), ZL_ErrorCode_transform_executionFailure);
    EXPECT_EQ(cudaGetLastError(), cudaErrorInvalidDevice);
}

/// Copies 4 bytes at @p source_d to host memory on @p native and waits.
std::vector<uint8_t> readFourBytes(
        const std::byte* source_d,
        cudaStream_t native)
{
    std::vector<uint8_t> bytes(4);
    ZL_CUDA_CHECK(cudaMemcpyAsync(
            bytes.data(), source_d, 4, cudaMemcpyDeviceToHost, native));
    ZL_CUDA_CHECK(cudaStreamSynchronize(native));
    return bytes;
}

TEST(CudaDeviceTest, BranchesStartAfterTheWorkOfTheirStream)
{
    // The stream is busy when the memset and the fork are enqueued. This fails
    // if a branch copies the buffer before the memset that came before the
    // fork.
    cudaStream_t native = nullptr;
    ZL_CUDA_CHECK(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking));
    const std::unique_ptr<CUstream_st, StreamDestroy> streamOwner{ native };
    const std::unique_ptr<Device> device = makeCurrentCudaDevice();
    ASSERT_NE(device, nullptr);
    std::optional<Stream> stream;
    ASSERT_FALSE(ZL_isError(device->wrapStream(native, stream)));
    const DevicePtr<std::byte> source_d = deviceAlloc<std::byte>(4);
    const DevicePtr<std::byte> copies_d = deviceAlloc<std::byte>(8);
    ZL_CUDA_CHECK(cudaMemset(source_d.get(), 0, 4));
    ZL_CUDA_CHECK(cudaMemset(copies_d.get(), 0, 8));
    ZL_CUDA_CHECK(cudaLaunchHostFunc(native, holdStream, nullptr));
    ZL_CUDA_CHECK(cudaMemsetAsync(source_d.get(), 7, 4, native));
    std::vector<Stream> branches;

    ASSERT_FALSE(ZL_isError(stream->fork(2, branches)));
    for (size_t i = 0; i < branches.size(); ++i) {
        ZL_CUDA_CHECK(cudaMemcpyAsync(
                copies_d.get() + 4 * i,
                source_d.get(),
                4,
                cudaMemcpyDeviceToDevice,
                static_cast<cudaStream_t>(branches[i].native())));
    }
    ASSERT_FALSE(ZL_isError(stream->join(branches)));

    EXPECT_EQ(
            readFourBytes(copies_d.get(), native), std::vector<uint8_t>(4, 7));
    EXPECT_EQ(
            readFourBytes(copies_d.get() + 4, native),
            std::vector<uint8_t>(4, 7));
}

TEST(CudaDeviceTest, StreamWaitsForEveryBranchAfterJoin)
{
    // The second branch is busy when its memset is enqueued. This fails if
    // work on the stream after the join runs before that memset.
    cudaStream_t native = nullptr;
    ZL_CUDA_CHECK(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking));
    const std::unique_ptr<CUstream_st, StreamDestroy> streamOwner{ native };
    const std::unique_ptr<Device> device = makeCurrentCudaDevice();
    ASSERT_NE(device, nullptr);
    std::optional<Stream> stream;
    ASSERT_FALSE(ZL_isError(device->wrapStream(native, stream)));
    const DevicePtr<std::byte> outputs_d = deviceAlloc<std::byte>(8);
    ZL_CUDA_CHECK(cudaMemset(outputs_d.get(), 0, 8));
    std::vector<Stream> branches;

    ASSERT_FALSE(ZL_isError(stream->fork(2, branches)));
    const auto first  = static_cast<cudaStream_t>(branches[0].native());
    const auto second = static_cast<cudaStream_t>(branches[1].native());
    ZL_CUDA_CHECK(cudaMemsetAsync(outputs_d.get(), 8, 4, first));
    ZL_CUDA_CHECK(cudaLaunchHostFunc(second, holdStream, nullptr));
    ZL_CUDA_CHECK(cudaMemsetAsync(outputs_d.get() + 4, 9, 4, second));
    ASSERT_FALSE(ZL_isError(stream->join(branches)));

    EXPECT_EQ(
            readFourBytes(outputs_d.get(), native), std::vector<uint8_t>(4, 8));
    EXPECT_EQ(
            readFourBytes(outputs_d.get() + 4, native),
            std::vector<uint8_t>(4, 9));
}

TEST(CudaDeviceTest, AllocatesCopiesAndFreesInStreamOrder)
{
    // Every operation is enqueued while the stream is busy, and both frees
    // are enqueued before the only wait. This fails if an allocation is
    // misaligned, if a device-to-device copy moves the wrong bytes, if a copy
    // of 0 bytes reaches the runtime, or if an operation leaves stream order.
    constexpr size_t kSize      = 4096;
    constexpr size_t kAlignment = 16;
    std::byte* pinned_h         = nullptr;
    ZL_CUDA_CHECK(cudaMallocHost(&pinned_h, 2 * kSize));
    const std::unique_ptr<std::byte, PinnedFree> host{ pinned_h };
    const poly::span<std::byte> expected{ host.get(), kSize };
    const poly::span<std::byte> actual{ host.get() + kSize, kSize };
    for (size_t i = 0; i < kSize; ++i) {
        expected[i] = static_cast<std::byte>(i * 7);
        actual[i]   = std::byte{ 0 };
    }
    cudaStream_t native = nullptr;
    ZL_CUDA_CHECK(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking));
    const std::unique_ptr<CUstream_st, StreamDestroy> streamOwner{ native };
    const std::unique_ptr<Device> device = makeCurrentCudaDevice();
    ASSERT_NE(device, nullptr);
    std::optional<Stream> stream;
    ASSERT_FALSE(ZL_isError(device->wrapStream(native, stream)));
    ZL_CUDA_CHECK(cudaLaunchHostFunc(native, holdStream, nullptr));
    void* source_d      = nullptr;
    void* destination_d = nullptr;
    ASSERT_FALSE(ZL_isError(stream->allocate(kSize, kAlignment, source_d)));
    ASSERT_FALSE(
            ZL_isError(stream->allocate(kSize, kAlignment, destination_d)));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(source_d) % kAlignment, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(destination_d) % kAlignment, 0);
    ZL_CUDA_CHECK(cudaMemcpyAsync(
            source_d, expected.data(), kSize, cudaMemcpyHostToDevice, native));
    const std::vector<DeviceToDeviceCopy> copies{
        { .dst_d = static_cast<std::byte*>(destination_d),
          .src_d = static_cast<const std::byte*>(source_d),
          .size  = kSize },
        { .dst_d = nullptr, .src_d = nullptr, .size = 0 },
    };

    ASSERT_FALSE(ZL_isError(stream->copyDeviceToDevice(copies)));
    stream->deallocate(source_d);
    ZL_CUDA_CHECK(cudaMemcpyAsync(
            actual.data(),
            destination_d,
            kSize,
            cudaMemcpyDeviceToHost,
            native));
    stream->deallocate(destination_d);
    ZL_CUDA_CHECK(cudaStreamSynchronize(native));

    EXPECT_EQ(
            std::vector<std::byte>(actual.begin(), actual.end()),
            std::vector<std::byte>(expected.begin(), expected.end()));
}

TEST(CudaDeviceTest, AllocatesNothingForZeroBytes)
{
    // This fails if a 0-byte allocation reaches the runtime or leaves a
    // non-null address that a later free would pass to the runtime.
    const std::unique_ptr<Device> device = makeCurrentCudaDevice();
    ASSERT_NE(device, nullptr);
    const Stream stream = device->defaultStream();
    int sentinel        = 0;
    void* ptr_d         = &sentinel;

    ASSERT_FALSE(ZL_isError(stream.allocate(0, 16, ptr_d)));

    EXPECT_EQ(ptr_d, nullptr);
    stream.deallocate(ptr_d);
}

TEST(CudaDeviceTest, WrapsAStreamOfTheCurrentDevice)
{
    // This fails if the device rejects a stream that belongs to the current
    // device, or wraps a different handle.
    const std::unique_ptr<Device> device = makeCurrentCudaDevice();
    ASSERT_NE(device, nullptr);
    cudaStream_t native = nullptr;
    ZL_CUDA_CHECK(cudaStreamCreate(&native));
    std::optional<Stream> stream;

    ZL_Report const result = device->wrapStream(native, stream);

    EXPECT_FALSE(ZL_isError(result));
    ASSERT_TRUE(stream.has_value());
    EXPECT_EQ(stream->native(), static_cast<void*>(native));
    EXPECT_EQ(&stream->device(), device.get());
    ZL_CUDA_CHECK(cudaStreamDestroy(native));
}

TEST(CudaDeviceTest, WrapsTheSpecialDefaultStreams)
{
    // Callers may pass the per-thread or the legacy default stream. This fails
    // if the device check rejects either handle.
    const std::unique_ptr<Device> device = makeCurrentCudaDevice();
    ASSERT_NE(device, nullptr);

    for (cudaStream_t const native :
         { cudaStreamPerThread, cudaStreamLegacy }) {
        std::optional<Stream> stream;
        ZL_Report const result = device->wrapStream(native, stream);
        EXPECT_FALSE(ZL_isError(result))
                << ZL_ErrorCode_toString(ZL_errorCode(result));
        EXPECT_TRUE(stream.has_value());
    }
}

} // namespace
} // namespace openzl::gpu
