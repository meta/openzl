// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "contrib/gpu/src/common/codec_decode_execution.hpp"
#include "openzl/cpp/poly/Span.hpp"
#include "openzl/zl_errors.h"

namespace openzl::gpu {

/** One device-to-host copy of @c dst_h.size() bytes starting at @c src_d. */
struct DeviceToHostCopy {
    poly::span<std::byte> dst_h;
    const std::byte* src_d;
};

/** One device-to-device copy of @c size bytes from @c src_d to @c dst_d. */
struct DeviceToDeviceCopy {
    std::byte* dst_d;
    const std::byte* src_d;
    size_t size;
};

class Device;

/**
 * A native stream of one device's runtime, such as a CUDA stream, and the
 * operations that host code enqueues on it.
 *
 * Only a @c Device creates a stream, so a stream cannot reach a device of
 * another runtime or another GPU. A stream owns neither its device nor its
 * native handle, and it lives for one call.
 *
 * Each operation checks what every runtime relies on, such as non-null copy
 * addresses, before it calls the device's runtime.
 */
class Stream {
   public:
    Device& device() const noexcept
    {
        return *device_;
    }

    /// The runtime's native stream handle, or NULL for the default stream.
    void* native() const noexcept
    {
        return native_;
    }

    /**
     * Copies every region to host memory and waits for the copies to finish.
     *
     * Empty regions are skipped. When this returns, no copy is still running,
     * even on error, so the caller may free every destination.
     *
     * @returns Success, @c parameter_invalid if a non-empty region has a null
     * address, or an error if the runtime fails.
     */
    ZL_Report copyToHostAndSynchronize(
            poly::span<const DeviceToHostCopy> copies) const;

    /**
     * Enqueues the decode of @p runs with @p kernel, and returns without
     * waiting for the device.
     *
     * @p kernel comes from this device's @c chooseDecodeBatches, and @p runs
     * are the runs of that batch with addresses. Host memory that @p runs
     * references may be reused when this returns. Device buffers must stay
     * valid until the work on this stream completes.
     *
     * @returns Success, or a kernel's error, after which earlier work may still
     * be running, so the caller must synchronize this stream before freeing its
     * buffers. A kernel's @c temporaryLibraryLimitation becomes
     * @c transform_executionFailure.
     */
    ZL_Report decodeBatch(
            const DecodeKernel& kernel,
            poly::span<const BoundCodecRun> runs) const;

    /**
     * Allocates @p size bytes of device memory in the order of this stream.
     *
     * Work enqueued on this stream after this call can use the memory. A size
     * of 0 stores NULL in @p ptr_d.
     *
     * @param alignment Power of two that the address must be a multiple of.
     * @param ptr_d Receives the address on success, and is unchanged on error.
     * @returns Success, @c parameter_invalid if @p alignment is not a power of
     * two, @c allocation if the runtime cannot allocate the memory with this
     * alignment, or @c temporaryLibraryLimitation if the runtime cannot
     * allocate at all, so the caller must provide the memory.
     */
    ZL_Report allocate(size_t size, size_t alignment, void*& ptr_d) const;

    /**
     * Frees @p ptr_d after the work enqueued on this stream so far finishes.
     *
     * @p ptr_d must come from @c allocate on a stream of the same device, or
     * be NULL. The runtime orders the free only after work on this stream, so
     * any use of @p ptr_d on another stream must be ordered before this call.
     * Errors are ignored, because the caller cannot act on them.
     */
    void deallocate(void* ptr_d) const noexcept;

    /**
     * Enqueues every copy, and returns without waiting for the device.
     *
     * Copies of 0 bytes are skipped. Regions must not overlap.
     *
     * @returns Success, @c parameter_invalid if a copy of more than 0 bytes
     * has a null address, or an error if the runtime fails, after which
     * earlier copies may still be running.
     */
    ZL_Report copyDeviceToDevice(
            poly::span<const DeviceToDeviceCopy> copies) const;

    /**
     * Sets @p branches to @p count streams of this device, whose work starts
     * after the work enqueued on this stream so far.
     *
     * Work on different branches can run at the same time. A runtime without
     * concurrent streams makes every branch this stream, so the work runs in
     * order. Call @c join with the same branches before work on this stream
     * needs their results; @c join ends the branches.
     *
     * @returns Success, @c parameter_invalid if @p count is more streams than
     * a vector can hold, or an error if the runtime fails. After an error,
     * @p branches is unchanged and no branch exists.
     * @throws std::bad_alloc if an allocation fails, before any branch exists.
     */
    ZL_Report fork(size_t count, std::vector<Stream>& branches) const;

    /**
     * Makes work enqueued on this stream from now on wait for the work
     * enqueued on each of @p branches so far, and ends the branches.
     *
     * Call it after every @c fork, even after an error on a branch or while an
     * exception unwinds, so that no branch work outlives the frees that follow
     * on this stream. It never throws, so it is safe on every path.
     *
     * @returns Success, or the first error of the runtime. Every branch ends,
     * even on error. If the runtime cannot make this stream wait for a branch,
     * the host waits for that branch, so its work still finishes before the
     * frees that follow.
     */
    ZL_Report join(poly::span<const Stream> branches) const noexcept;

   private:
    friend class Device;

    Stream(Device& device, void* native) noexcept
            : device_{ &device }, native_{ native }
    {
    }

    Device* device_;
    void* native_;
};

/**
 * Settings that every device takes, whatever its runtime. A runtime's own
 * options extend this struct, as `CudaDeviceOptions` does. It holds no setting
 * yet.
 */
struct DeviceOptions {};

/**
 * The device-runtime operations that the host decode code needs.
 *
 * The host code calls no device runtime directly. Each runtime implements the
 * protected @c do* methods and passes its kernel table to the constructor, so
 * supporting another runtime does not change the host code. Host code enqueues
 * work through a @c Stream of the device.
 */
class Device {
   public:
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&)                 = delete;
    Device& operator=(Device&&)      = delete;
    virtual ~Device()                = default;

    /// The runtime's default stream, which always belongs to this device.
    Stream defaultStream() noexcept
    {
        return Stream{ *this, nullptr };
    }

    /**
     * Wraps @p native, a stream of this device's runtime, in @p stream.
     *
     * @param native The runtime's native stream handle, or NULL for the
     * default stream.
     * @param stream Receives the stream on success, and is unchanged on error.
     * @returns Success, or @c parameter_invalid if the runtime reports that
     * @p native belongs to another device.
     */
    ZL_Report wrapStream(void* native, std::optional<Stream>& stream);

    /**
     * Chooses the first kernel in this device's table that accepts each run,
     * and groups the runs that choose the same kernel into one batch, as the
     * free function @c chooseDecodeBatches does.
     *
     * Reads no device memory, so binding can choose every kernel before any
     * memory is allocated. The chosen kernels live as long as this device.
     *
     * @param batches Receives the batches on success, and is unchanged on
     * error.
     * @returns Success, or @c temporaryLibraryLimitation if no kernel accepts a
     * run, so the caller can decode on the CPU instead.
     * @throws std::bad_alloc if an allocation fails.
     */
    ZL_Report chooseDecodeBatches(
            poly::span<const CodecRunInfo> runs,
            std::vector<DecodeBatch>& batches) const;

   protected:
    /// @p decodeKernels lists this runtime's kernels for each codec. It does
    /// not change after construction.
    explicit Device(DecodeKernelTable decodeKernels) noexcept;

    /**
     * Checks that @p native, a non-null native stream, belongs to this device.
     * The default accepts every stream, for runtimes that cannot tell.
     */
    virtual ZL_Report doCheckStream(void* native);

    // The runtime's operations. The public method of the same name, on
    // @c Stream or on this class, checks the arguments and then calls it.
    virtual ZL_Report doCopyToHostAndSynchronize(
            poly::span<const DeviceToHostCopy> copies,
            void* native) = 0;
    virtual ZL_Report
    doAllocate(size_t size, size_t alignment, void* native, void*& ptr_d) = 0;
    virtual void doDeallocate(void* ptr_d, void* native) noexcept         = 0;
    virtual ZL_Report doCopyDeviceToDevice(
            poly::span<const DeviceToDeviceCopy> copies,
            void* native) = 0;

    // Concurrent streams. The default has none: every branch is the parent's
    // native stream, and joining does nothing. @c doJoin must not throw, and
    // it skips a branch whose native stream is @p native.
    virtual ZL_Report
    doFork(void* native, size_t count, std::vector<void*>& branches);
    virtual ZL_Report doJoin(
            void* native,
            poly::span<const Stream> branches) noexcept;

    // Runs the kernel. A runtime can override it to add work around each batch.
    virtual ZL_Report doDecodeBatch(
            const DecodeKernel& kernel,
            poly::span<const BoundCodecRun> runs,
            void* native);

   private:
    friend class Stream;

    DecodeKernelTable decodeKernels_;
};

} // namespace openzl::gpu
