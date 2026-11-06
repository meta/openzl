// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "openzl/zl_errors.h"

typedef struct CUstream_st* ZL_GPU_Stream;

/**
 * The decoder writes output in aligned 8-byte groups, so the destination buffer
 * must reserve this many extra trailing bytes past `dstSize` for the final
 * group's over-write (the trailing bytes are written with unspecified values).
 */
#define PIVCO_GPU_DECODE_DST_SLOP 8

/**
 * The chunk-in-shared decoder reads bitmap words with a loop-free over-read of
 * a few bytes past a node's bitmap (the extra bytes are always masked off). For
 * interior blocks that lands in the next block's slice; for the last block the
 * input buffer must reserve this many readable trailing bytes past
 * `bitstreamSize` (their contents are irrelevant). Callers should allocate the
 * compressed input with this much slop.
 */
#define PIVCO_GPU_DECODE_SRC_SLOP 8

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PivCoGpuContext PivCoGpuContext;

ZL_Report pivcoGpuContextCreate(
        PivCoGpuContext** context,
        const uint8_t* weights,
        size_t weightsSize,
        int tableLog);

void pivcoGpuContextDestroy(PivCoGpuContext* context);

/** Conservative budget, independent of the selected context. */
size_t pivcoGpuEncodeWorkspaceBytes(size_t srcSize, size_t blockSize);

/**
 * Budget for the path selected by this immutable context and input shape.
 * Returns 0 for empty/constant input and SIZE_MAX for invalid parameters.
 * Cooperatively encoded input uses a bounded batch pool. Other shapes retain
 * their existing path. Does not include input/output or completion buffers.
 */
size_t pivcoGpuEncodeWorkspaceBytesForContext(
        const PivCoGpuContext* context,
        size_t srcSize,
        size_t blockSize);

size_t pivcoGpuDecodeWorkspaceBytes(size_t dstSize, size_t blockSize);

/**
 * Blocking device-buffer encoder. On success, this call's submitted work
 * is complete before return.
 * On error, output bytes and offsets are invalid and may be partially written.
 */
ZL_Report pivcoGpuEncode(
        const PivCoGpuContext* context,
        void* dst_d,
        size_t dstCapacity,
        uint64_t* offsets_d,
        size_t offsetsCapacity,
        const void* src_d,
        size_t srcSize,
        size_t blockSize,
        void* workspace_d,
        size_t workspaceBytes,
        ZL_GPU_Stream stream);

ZL_Report pivcoGpuDecode(
        const PivCoGpuContext* context,
        void* dst_d,
        size_t dstSize,
        const void* bitstream_d,
        size_t bitstreamSize,
        const uint64_t* offsets_d,
        size_t offsetsCount,
        size_t blockSize,
        void* workspace_d,
        size_t workspaceBytes,
        ZL_GPU_Stream stream);

#ifdef __cplusplus
}
#endif
