// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <vector>

#include "contrib/gpu/src/common/result.hpp"
#include "contrib/gpu/src/decompress/gpu_decompress.h"
#include "contrib/gpu/src/device/device.hpp"
#include "contrib/gpu/src/planning/decode_planner.hpp"
#include "openzl/cpp/DCtx.hpp"
#include "openzl/cpp/poly/Span.hpp"
#include "openzl/zl_errors.h"

namespace openzl::gpu {

/** Non-owning metadata for one compressed frame header in device memory. */
struct GPUFrameHeaderForChunks {
    /// Beginning of the encoded frame header.
    const void* frameHeader_d;
    /// Number of bytes in the frame header.
    size_t frameHeaderSize;
};

/**
 * Non-owning metadata for one compressed chunk in device memory.
 *
 * @p chunk_d refers into a device allocation owned by the caller. Its storage
 * must remain valid until work using this descriptor has completed on its
 * stream. @p frameHeaderIdx identifies the frame-header descriptor supplied
 * alongside this chunk.
 */
struct GPUChunk {
    /// Index of this chunk's frame header in the accompanying descriptor list.
    size_t frameHeaderIdx;
    /// Beginning of the encoded chunk, including its formal header.
    const void* chunk_d;
    /// Total encoded chunk size, including headers and checksums.
    size_t chunkSize;
    /// Number of bytes in the formal chunk header.
    size_t chunkHeaderSize;
};

/** One device chunk prepared for GPU decode planning. */
struct PreparedGPUChunk {
    GPUChunk chunk;
    DCtx dctx;
    /// Concatenated private transform headers copied from the device chunk.
    std::vector<std::byte> transformHeaders_h;
};

/**
 * Copies frame, formal chunk, and transform headers to host and prepares one
 * DCTX per chunk.
 * Stored-stream references remain bound to the device addresses in @p chunks.
 * @param stream Stream of the device whose memory holds @p frameHeaders and
 * @p chunks. The copies run on it, and they are synchronized before this
 * function returns.
 * @returns One prepared chunk per entry of @p chunks, in the same order, or an
 * error.
 * @throws std::bad_alloc or openzl::Exception if an allocation fails.
 */
Result<std::vector<PreparedGPUChunk>> prepareChunksForPlanning(
        const Stream& stream,
        poly::span<const GPUFrameHeaderForChunks> frameHeaders,
        poly::span<const GPUChunk> chunks);

/// The planner's view of @p chunk, valid while @p chunk is alive and unchanged.
PreparedGpuChunkView viewOf(PreparedGPUChunk& chunk);

/**
 * Enumerates the chunks in one host-readable frame. A temporary solution until
 * we have a jump table supported by ZL frame format footer.
 *
 * @param src_d Beginning of the device frame. This function does not read the
 * device allocation; it uses the address as the base for each descriptor.
 * @param src_h Byte-identical host view of the complete frame at @p src_d.
 * @param chunks Output descriptors. Appended on success and unchanged on
 * error.
 * @param frameHeaders Output containing the frame-header descriptor referenced
 * by the appended chunks. Appended on success and unchanged on error.
 * @returns The compressed frame size on success, or an error.
 * @throws std::bad_alloc or openzl::Exception if an allocation fails.
 *
 * @note Only frame format version 21 and newer is supported.
 */
ZL_Report collectGPUChunksFromFrame(
        const void* src_d,
        poly::span<const std::byte> src_h,
        std::vector<GPUChunk>& chunks,
        std::vector<GPUFrameHeaderForChunks>& frameHeaders);

/**
 * Copies one device frame to host and enumerates its chunks.
 *
 * @param stream Stream of the device whose memory holds @p src_d. The
 * device-to-host copy runs on it and is synchronized before this function
 * returns.
 * @param src_d Device allocation containing one complete compressed frame.
 * @param chunks Output descriptors. Appended on success and unchanged on
 * error. Their pointers remain valid only while @p src_d remains valid.
 * @param frameHeaders Output containing the frame-header descriptor referenced
 * by the appended chunks. Appended on success and unchanged on error. Its
 * pointer remains valid only while @p src_d remains valid.
 * @returns The compressed frame size on success, or an error.
 * @throws std::bad_alloc or openzl::Exception if an allocation fails.
 */
ZL_Report collectGPUChunks(
        const Stream& stream,
        poly::span<const std::byte> src_d,
        std::vector<GPUChunk>& chunks,
        std::vector<GPUFrameHeaderForChunks>& frameHeaders);

/**
 * Decompresses independently prepared chunks on @p stream.
 *
 * @param stream Stream of the device whose memory holds @p dst_d and every
 * chunk. Decompression is enqueued on it.
 * @param dst_d Device allocation that receives decompressed output.
 * @param dstCapacity Capacity of @p dst_d in bytes.
 * @param frameHeaders Non-owning frame-header descriptors referenced by
 * @p chunks.
 * @param chunks Non-owning descriptors for the chunks to decompress. All
 * frame-header indices must refer to @p frameHeaders, and all referenced device
 * storage must remain valid until the enqueued work has completed on @p stream.
 * @returns The decompressed size on success, or an error. An empty batch
 * returns @c parameter_invalid. The planner cannot yet combine frames, so a
 * batch with chunks of more than one frame returns
 * @c temporaryLibraryLimitation.
 * @throws std::bad_alloc or openzl::Exception if an allocation fails.
 */
ZL_Report decompressChunks(
        const Stream& stream,
        void* dst_d,
        size_t dstCapacity,
        poly::span<const GPUFrameHeaderForChunks> frameHeaders,
        poly::span<const GPUChunk> chunks);

} // namespace openzl::gpu
