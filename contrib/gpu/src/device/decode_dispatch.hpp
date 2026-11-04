// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <vector>

#include "contrib/gpu/src/common/codec_decode_execution.hpp"
#include "openzl/common/wire_format.h"
#include "openzl/cpp/poly/Span.hpp"
#include "openzl/zl_errors.h"

namespace openzl::gpu {

/**
 * @returns The kernels of @p transform in @p table, in order of preference, or
 * none if @p transform is not a standard transform.
 */
poly::span<const DecodeKernel> kernelsFor(
        const DecodeKernelTable& table,
        PublicTransformInfo transform);

/**
 * @returns The first kernel in @p candidates that accepts @p run, or nullptr if
 * there is none.
 */
const DecodeKernel* selectDecodeKernel(
        poly::span<const DecodeKernel> candidates,
        const CodecRunInfo& run);

/**
 * Chooses a kernel for every run from @p table, and groups the runs that choose
 * the same kernel into one batch.
 *
 * Batches follow the order in which their kernels are first chosen, and keep
 * their runs in input order. Runs must be independent of each other, as within
 * one wave.
 *
 * @param batches Receives the batches on success, and is unchanged on error.
 * @returns Success, or @c temporaryLibraryLimitation if no kernel accepts a
 * run.
 * @throws std::bad_alloc if an allocation fails.
 */
ZL_Report chooseDecodeBatches(
        const DecodeKernelTable& table,
        poly::span<const CodecRunInfo> runs,
        std::vector<DecodeBatch>& batches);

/**
 * Enqueues the decode of @p runs with @p kernel on @p stream. An empty batch
 * enqueues nothing.
 *
 * @returns Success, or @p kernel's error. A kernel's
 * @c temporaryLibraryLimitation becomes @c transform_executionFailure: earlier
 * batches may be running, so the caller cannot fall back to the CPU.
 */
ZL_Report decodeBatchWith(
        const DecodeKernel& kernel,
        poly::span<const BoundCodecRun> runs,
        void* scratch_d,
        size_t scratchBytes,
        void* stream);

} // namespace openzl::gpu
