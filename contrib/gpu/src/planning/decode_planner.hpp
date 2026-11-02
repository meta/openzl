// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

#include "contrib/gpu/src/common/result.hpp"
#include "contrib/gpu/src/planning/decode_plan.hpp"
#include "openzl/cpp/poly/Span.hpp"

namespace openzl::gpu {

/**
 * Host data needed to plan one prepared device chunk.
 *
 * The decode context contains the frame graph, stream producers, and stored
 * streams. @c transformHeaders_h contains the codec headers used during
 * planning. It must remain valid until planning returns, and while a
 * @c DecodeInfo or @c BoundDecode built from the chunk is in use, because their
 * codec headers point into it. The decode context and its frame storage must
 * remain valid until the plan finishes executing.
 */
struct PreparedGpuChunkView {
    /// Non-null prepared context with frame, producer, and stream metadata.
    ZL_DCtx* dctx;
    /// Host-staged codec headers indexed by each node's offset and size.
    poly::span<const std::byte> transformHeaders_h;
    /// Frame that the chunk belongs to. Every chunk in one plan must belong to
    /// the same frame.
    size_t frameIndex = 0;
};

/**
 * Builds a decode plan without allocating device buffers.
 *
 * Each prepared context and its frame storage must remain valid until the plan
 * finishes executing because they own stored stream bytes. The plan does not
 * point into the staged transform headers or the caller's output buffers.
 *
 * A final @c StreamArena stream becomes a @c Destination so its decoder writes
 * directly to the caller's output. A final @c Stored or @c StreamRef stream
 * keeps its class and must be copied to the caller's output.
 *
 * Checks each final output against the frame header, as core does when it
 * appends chunk outputs: every chunk must have the frame's number of outputs,
 * every chunk's contribution must have the frame's output type and one element
 * width, a numeric output must have a width that core accepts, and the
 * contributions must add up to the declared size. Also rejects a codec run that
 * reads a final output, as core does.
 *
 * @param chunks Prepared chunks of one frame, in batch order.
 * @returns The plan for every chunk in @p chunks, or the first validation,
 * allocation, or unsupported-codec error encountered while planning the batch.
 * An empty chunk list returns @c parameter_invalid. Chunks of more than one
 * frame return @c temporaryLibraryLimitation. A final
 * output that disagrees with the frame header returns @c corruption, or
 * @c integerOverflow if the chunk sizes overflow. A numeric final output with
 * a width that core rejects returns @c streamParameter_invalid, as in core. A
 * codec run that reads a final output returns @c graph_invalid.
 */
Result<DecodePlan> planDecode(poly::span<const PreparedGpuChunkView> chunks);

} // namespace openzl::gpu
