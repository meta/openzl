// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>

#include "openzl/zl_errors.h" // ZL_Report

// Device stream handle. Structurally identical to cudaStream_t; opaque here so
// this header stays CUDA-free.
typedef struct CUstream_st* ZL_GPU_Stream;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Decompress a single OpenZL frame that already resides in GPU memory.
 *
 * GPU analogue of core `ZL_decompress`: decodes exactly ONE frame -- which may
 * itself contain multiple chunks -- from @p src_d into @p dst_d. Both buffers
 * are device pointers; the decode is enqueued on @p stream and may run
 * asynchronously with respect to the host.
 *
 * It decodes only a frame with one serial output. Any other frame returns
 * `temporaryLibraryLimitation`.
 *
 * @note Only frame format version 21 and newer is supported.
 *
 * @param dst_d Device buffer that receives the decompressed output.
 * @param dstCapacity Capacity of @p dst_d in bytes; at least the frame's
 * decompressed size.
 * @param src_d Device buffer holding the compressed frame.
 * @param srcSize Size of the compressed frame, in bytes.
 * @param stream CUDA stream the decode is enqueued on (opaque `ZL_GPU_Stream`
 * so this header stays CUDA-free).
 * @returns A `ZL_Report`: the decompressed size on success, or an error.
 */
ZL_Report ZL_GPU_decompress(
        void* dst_d,
        size_t dstCapacity,
        const void* src_d,
        size_t srcSize,
        ZL_GPU_Stream stream);

#ifdef __cplusplus
}
#endif
