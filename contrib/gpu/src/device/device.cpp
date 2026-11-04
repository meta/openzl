// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/src/device/device.hpp"

#include <utility>
#include <vector>

#include "contrib/gpu/src/device/decode_dispatch.hpp"
#include "openzl/common/errors_internal.h"

namespace openzl::gpu {

ZL_Report Stream::copyToHostAndSynchronize(
        poly::span<const DeviceToHostCopy> copies) const
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    for (const DeviceToHostCopy& copy : copies) {
        if (copy.dst_h.empty()) {
            continue;
        }
        ZL_ERR_IF_NULL(copy.dst_h.data(), parameter_invalid);
        ZL_ERR_IF_NULL(copy.src_d, parameter_invalid);
    }
    return device().doCopyToHostAndSynchronize(copies, native_);
}

ZL_Report Stream::decodeBatch(
        const DecodeKernel& kernel,
        poly::span<const BoundCodecRun> runs,
        void* scratch_d,
        size_t scratchBytes) const
{
    return device().doDecodeBatch(
            kernel, runs, scratch_d, scratchBytes, native_);
}

ZL_Report Stream::allocate(size_t size, size_t alignment, void*& ptr_d) const
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    ZL_ERR_IF(
            alignment == 0 || (alignment & (alignment - 1)) != 0,
            parameter_invalid,
            "The alignment %zu is not a power of two",
            alignment);
    return device().doAllocate(size, alignment, native_, ptr_d);
}

void Stream::deallocate(void* ptr_d) const noexcept
{
    device().doDeallocate(ptr_d, native_);
}

ZL_Report Stream::copyDeviceToDevice(
        poly::span<const DeviceToDeviceCopy> copies) const
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    for (const DeviceToDeviceCopy& copy : copies) {
        if (copy.size == 0) {
            continue;
        }
        ZL_ERR_IF_NULL(copy.dst_d, parameter_invalid);
        ZL_ERR_IF_NULL(copy.src_d, parameter_invalid);
    }
    return device().doCopyDeviceToDevice(copies, native_);
}

ZL_Report Stream::fork(size_t count, std::vector<Stream>& branches) const
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    // Nothing after doFork may throw: the caller cannot join branches that it
    // never received, so their streams would leak.
    std::vector<Stream> forked;
    ZL_ERR_IF_GT(count, forked.max_size(), parameter_invalid);
    forked.reserve(count);
    std::vector<void*> natives;
    ZL_ERR_IF_ERR(device().doFork(native_, count, natives));
    ZL_ASSERT_EQ(natives.size(), count);
    for (void* const native : natives) {
        forked.push_back(Stream{ device(), native });
    }
    branches = std::move(forked);
    return ZL_returnSuccess();
}

ZL_Report Stream::join(poly::span<const Stream> branches) const noexcept
{
    return device().doJoin(native_, branches);
}

Device::Device(DecodeKernelTable decodeKernels) noexcept
        : decodeKernels_{ std::move(decodeKernels) }
{
}

ZL_Report Device::wrapStream(void* native, std::optional<Stream>& stream)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(nullptr);
    if (native != nullptr) {
        ZL_ERR_IF_ERR(doCheckStream(native));
    }
    stream = Stream{ *this, native };
    return ZL_returnSuccess();
}

ZL_Report Device::chooseDecodeBatches(
        poly::span<const CodecRunInfo> runs,
        std::vector<DecodeBatch>& batches) const
{
    return openzl::gpu::chooseDecodeBatches(decodeKernels_, runs, batches);
}

ZL_Report Device::doCheckStream(void*)
{
    return ZL_returnSuccess();
}

ZL_Report
Device::doFork(void* native, size_t count, std::vector<void*>& branches)
{
    branches.assign(count, native);
    return ZL_returnSuccess();
}

ZL_Report Device::doJoin(void*, poly::span<const Stream>) noexcept
{
    return ZL_returnSuccess();
}

ZL_Report Device::doDecodeBatch(
        const DecodeKernel& kernel,
        poly::span<const BoundCodecRun> runs,
        void* scratch_d,
        size_t scratchBytes,
        void* native)
{
    return decodeBatchWith(kernel, runs, scratch_d, scratchBytes, native);
}

} // namespace openzl::gpu
