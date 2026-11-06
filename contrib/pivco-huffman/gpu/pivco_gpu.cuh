// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "contrib/pivco-huffman/gpu/pivco_gpu.h"

struct PivCoGpuStatus {
    unsigned code;
    uint64_t detail;
};

enum PivCoGpuStatusCode : unsigned {
    PIVCO_GPU_STATUS_OK             = 0,
    PIVCO_GPU_STATUS_PARAMETER      = 1,
    PIVCO_GPU_STATUS_CORRUPTION     = 2,
    PIVCO_GPU_STATUS_CAPACITY       = 3,
    PIVCO_GPU_STATUS_MISSING_SYMBOL = 4,
};

/** Caller-owned device completion storage, reusable after stream completion. */
struct PivCoGpuEncodeResult {
    PivCoGpuStatus status;
    uint64_t totalSize;
};

/**
 * Initialize result_d and enqueue encoding, without allocation or a host fence.
 * result_d must not overlap source, destination, offsets or workspace. Keep the
 * host context and all device buffers alive until stream completion, including
 * after an enqueue error. Concurrent calls require disjoint mutable buffers;
 * cross-stream reuse requires an event dependency or explicit synchronization.
 *
 * To retrieve completion, copy this one result to caller-owned pinned host
 * memory and record an event on the same stream. Read it only after completion,
 * then use pivcoGpuEncodeResultReport(). CUDA success means enqueue success;
 * output bytes/offsets are valid only when the completed result reports
 * success.
 */
cudaError_t pivcoGpuEncodeWithResultAsync(
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
        PivCoGpuEncodeResult* result_d,
        cudaStream_t stream);

/** Convert a completed host result to encoded size or an OpenZL error. */
ZL_Report pivcoGpuEncodeResultReport(const PivCoGpuEncodeResult* result);

/**
 * Enqueue encoding on stream. Initialize status_d to OK before each call.
 * CUDA success reports enqueue success; read status_d and totalSize_d only
 * after stream completion. On any error, output bytes, offsets and total size
 * are invalid; completed batches may already have written output.
 *
 * The host context and all device buffers must remain alive until completion.
 * Concurrent calls may share an immutable context, but must use disjoint
 * mutable buffers. Reuse across streams requires an event dependency or
 * explicit synchronization. Nonempty cooperative calls require nonoverlapping
 * source, destination, offsets, workspace, status and total-size ranges.
 */
cudaError_t pivcoGpuEncodeAsync(
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
        PivCoGpuStatus* status_d,
        uint64_t* totalSize_d,
        cudaStream_t stream);

cudaError_t pivcoGpuDecodeAsync(
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
        PivCoGpuStatus* status_d,
        cudaStream_t stream);
