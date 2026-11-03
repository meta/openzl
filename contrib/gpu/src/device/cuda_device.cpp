// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/device/cuda_device.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include "contrib/gpu/src/device/cuda_kernel_table.hpp"
#include "openzl/common/errors_internal.h"

static_assert(
        CUDART_VERSION >= 12080,
        "The CUDA device requires CUDA 12.8 or newer");

namespace openzl::gpu {
namespace {

/**
 * Clears the runtime's per-thread last error after a failed call. The failure
 * reaches the caller through a @c ZL_Report, and a stale last error would make
 * the caller's own @c cudaGetLastError check report it again.
 */
void clearLastError()
{
    (void)cudaGetLastError();
}

ZL_Report copyToHostOnCurrentDevice(
        poly::span<const DeviceToHostCopy> copies,
        cudaStream_t stream)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);

    bool copyEnqueued = false;
    for (const DeviceToHostCopy& copy : copies) {
        if (copy.dst_h.empty()) {
            continue;
        }
        cudaError_t const result = cudaMemcpyAsync(
                copy.dst_h.data(),
                copy.src_d,
                copy.dst_h.size(),
                cudaMemcpyDeviceToHost,
                stream);
        if (result != cudaSuccess) {
            if (copyEnqueued) {
                // Destinations must outlive any copies queued before failure.
                (void)cudaStreamSynchronize(stream);
            }
            clearLastError();
            ZL_ERR(GENERIC,
                   "cudaMemcpyAsync failed: %s",
                   cudaGetErrorString(result));
        }
        copyEnqueued = true;
    }
    if (copyEnqueued) {
        cudaError_t const result = cudaStreamSynchronize(stream);
        if (result != cudaSuccess) {
            clearLastError();
            ZL_ERR(GENERIC,
                   "cudaStreamSynchronize failed: %s",
                   cudaGetErrorString(result));
        }
    }
    return ZL_returnSuccess();
}

ZL_Report allocateOnCurrentDevice(
        size_t size,
        size_t alignment,
        cudaStream_t stream,
        void*& ptr_d)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    ZL_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
    if (size == 0) {
        ptr_d = nullptr;
        return ZL_returnSuccess();
    }
    void* allocation_d       = nullptr;
    cudaError_t const result = cudaMallocAsync(&allocation_d, size, stream);
    if (result == cudaErrorNotSupported) {
        // The device has no stream-ordered memory pool.
        clearLastError();
        ZL_ERR(temporaryLibraryLimitation,
               "cudaMallocAsync is not supported on this device");
    }
    if (result != cudaSuccess) {
        clearLastError();
        ZL_ERR(allocation,
               "cudaMallocAsync(%zu) failed: %s",
               size,
               cudaGetErrorString(result));
    }
    // Stream-ordered pools align to at least 256 bytes, which covers every
    // alignment a layout asks for. Check it rather than rely on it.
    if (reinterpret_cast<uintptr_t>(allocation_d) % alignment != 0) {
        if (cudaFreeAsync(allocation_d, stream) != cudaSuccess) {
            clearLastError();
        }
        ZL_ERR(allocation,
               "cudaMallocAsync returned an address not aligned to %zu",
               alignment);
    }
    ptr_d = allocation_d;
    return ZL_returnSuccess();
}

ZL_Report copyDeviceToDeviceOnCurrentDevice(
        poly::span<const DeviceToDeviceCopy> copies,
        cudaStream_t stream)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    for (const DeviceToDeviceCopy& copy : copies) {
        if (copy.size == 0) {
            continue;
        }
        cudaError_t const result = cudaMemcpyAsync(
                copy.dst_d,
                copy.src_d,
                copy.size,
                cudaMemcpyDeviceToDevice,
                stream);
        if (result != cudaSuccess) {
            clearLastError();
            ZL_ERR(GENERIC,
                   "cudaMemcpyAsync failed: %s",
                   cudaGetErrorString(result));
        }
    }
    return ZL_returnSuccess();
}

/**
 * Creates @p count non-blocking streams on the current device that wait for
 * the work enqueued on @p stream so far. On error, destroys the streams that
 * it created and leaves @p branches unchanged.
 */
ZL_Report forkOnCurrentDevice(
        cudaStream_t stream,
        size_t count,
        std::vector<void*>& branches)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    // Reserved before the event exists, so that nothing throws while it does.
    std::vector<void*> created;
    created.reserve(count);
    cudaEvent_t start = nullptr;
    cudaError_t result =
            cudaEventCreateWithFlags(&start, cudaEventDisableTiming);
    if (result == cudaSuccess) {
        result = cudaEventRecord(start, stream);
    }
    while (result == cudaSuccess && created.size() < count) {
        cudaStream_t branch = nullptr;
        result = cudaStreamCreateWithFlags(&branch, cudaStreamNonBlocking);
        if (result != cudaSuccess) {
            break;
        }
        created.push_back(branch);
        result = cudaStreamWaitEvent(branch, start, 0);
    }
    // Destroying an event or a stream with pending work is safe: the runtime
    // releases it when the work completes.
    if (start != nullptr) {
        (void)cudaEventDestroy(start);
    }
    if (result != cudaSuccess) {
        for (void* const branch : created) {
            (void)cudaStreamDestroy(static_cast<cudaStream_t>(branch));
        }
        clearLastError();
        ZL_ERR(GENERIC,
               "Creating %zu concurrent streams failed: %s",
               count,
               cudaGetErrorString(result));
    }
    branches = std::move(created);
    return ZL_returnSuccess();
}

/**
 * Makes @p stream wait for the work enqueued on each of @p branches so far,
 * then destroys the branches. Every branch is destroyed, even on error. If
 * @p stream cannot wait for a branch, the host waits for it first.
 */
ZL_Report joinOnCurrentDevice(
        cudaStream_t stream,
        poly::span<const Stream> branches) noexcept
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    cudaError_t first = cudaSuccess;
    for (const Stream& forked : branches) {
        const auto branch = static_cast<cudaStream_t>(forked.native());
        // A branch that is the parent stream needs no wait, and destroying
        // it would destroy the caller's stream.
        if (branch == stream) {
            continue;
        }
        cudaEvent_t done = nullptr;
        cudaError_t result =
                cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
        if (result == cudaSuccess) {
            result = cudaEventRecord(done, branch);
        }
        if (result == cudaSuccess) {
            result = cudaStreamWaitEvent(stream, done, 0);
        }
        if (done != nullptr) {
            (void)cudaEventDestroy(done);
        }
        if (result != cudaSuccess) {
            // Destroying a stream does not wait for its work, and nothing
            // orders that work before later frees on `stream`.
            (void)cudaStreamSynchronize(branch);
            if (first == cudaSuccess) {
                first = result;
            }
        }
        (void)cudaStreamDestroy(branch);
    }
    if (first != cudaSuccess) {
        clearLastError();
        ZL_ERR(GENERIC,
               "Joining concurrent streams failed: %s",
               cudaGetErrorString(first));
    }
    return ZL_returnSuccess();
}

class CudaDevice final : public Device {
   public:
    explicit CudaDevice(const CudaDeviceOptions& options)
            : Device(cudaKernelTable(options))
    {
    }

   protected:
    ZL_Report doCheckStream(void* native) override
    {
        ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
        int streamDevice         = 0;
        int currentDevice        = 0;
        cudaError_t const result = cudaStreamGetDevice(
                static_cast<cudaStream_t>(native), &streamDevice);
        if (result != cudaSuccess) {
            clearLastError();
            // Other failures, such as a driver older than CUDA 12.8, are not
            // the caller's stream.
            ZL_ERR_IF(
                    result == cudaErrorInvalidResourceHandle
                            || result == cudaErrorInvalidValue,
                    parameter_invalid,
                    "The stream is not a valid CUDA stream: %s",
                    cudaGetErrorString(result));
            ZL_ERR(GENERIC,
                   "cudaStreamGetDevice failed: %s",
                   cudaGetErrorString(result));
        }
        if (cudaGetDevice(&currentDevice) != cudaSuccess) {
            clearLastError();
            ZL_ERR(GENERIC, "cudaGetDevice failed");
        }
        ZL_ERR_IF_NE(
                streamDevice,
                currentDevice,
                parameter_invalid,
                "The stream belongs to device %d, but device %d is current",
                streamDevice,
                currentDevice);
        return ZL_returnSuccess();
    }

    ZL_Report doCopyToHostAndSynchronize(
            poly::span<const DeviceToHostCopy> copies,
            void* native) override
    {
        return copyToHostOnCurrentDevice(
                copies, static_cast<cudaStream_t>(native));
    }

    ZL_Report doDecodeBatch(
            const DecodeKernel& kernel,
            poly::span<const BoundCodecRun> runs,
            void* native) override
    {
        // Clears only an error that the batch set. An error that the caller
        // left pending before the call stays for the caller.
        cudaError_t const pending = cudaPeekAtLastError();
        ZL_Report const result    = Device::doDecodeBatch(kernel, runs, native);
        if (cudaPeekAtLastError() != pending) {
            clearLastError();
        }
        return result;
    }

    ZL_Report doAllocate(
            size_t size,
            size_t alignment,
            void* native,
            void*& ptr_d) override
    {
        return allocateOnCurrentDevice(
                size, alignment, static_cast<cudaStream_t>(native), ptr_d);
    }

    void doDeallocate(void* ptr_d, void* native) noexcept override
    {
        if (ptr_d == nullptr) {
            return;
        }
        if (cudaFreeAsync(ptr_d, static_cast<cudaStream_t>(native))
            != cudaSuccess) {
            clearLastError();
        }
    }

    ZL_Report doCopyDeviceToDevice(
            poly::span<const DeviceToDeviceCopy> copies,
            void* native) override
    {
        return copyDeviceToDeviceOnCurrentDevice(
                copies, static_cast<cudaStream_t>(native));
    }

    ZL_Report doFork(void* native, size_t count, std::vector<void*>& branches)
            override
    {
        return forkOnCurrentDevice(
                static_cast<cudaStream_t>(native), count, branches);
    }

    ZL_Report doJoin(void* native, poly::span<const Stream> branches) noexcept
            override
    {
        return joinOnCurrentDevice(static_cast<cudaStream_t>(native), branches);
    }
};

} // namespace

std::unique_ptr<Device> makeCurrentCudaDevice(const CudaDeviceOptions& options)
{
    // The constructor builds the kernel table, which can throw.
    try {
        return std::make_unique<CudaDevice>(options);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

} // namespace openzl::gpu
