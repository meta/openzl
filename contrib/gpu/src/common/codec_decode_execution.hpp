// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "contrib/gpu/src/common/codec_decode_planning.hpp"
#include "openzl/common/wire_format.h"
#include "openzl/cpp/poly/Span.hpp"
#include "openzl/zl_errors.h"

namespace openzl::gpu {

/** One decoder input, as a kernel sees it before it has an address. */
struct InputStreamInfo {
    /// Logical type and element layout of the input.
    StreamShape shape{};
    /// Bytes readable in the input's data and string-length regions.
    StreamStorageSize availableStorage{};
};

/** One decoder output, as a kernel sees it before it has an address. */
struct OutputStreamInfo {
    /// Logical type and element layout the decoder must produce.
    StreamShape shape{};
    /// Bytes writable in the output's data and string-length regions.
    StreamStorageSize storageSize{};
};

/**
 * One planned codec run, without addresses.
 *
 * Binding chooses each run's kernel and sizes its scratch from this info,
 * before any memory is allocated. Every span is non-owning host memory. Inputs
 * are in decoder argument order, and outputs are in regenerated-stream order,
 * as in @c PlannedCodecRun.
 */
struct CodecRunInfo {
    /// Exact transform namespace and ID decoded from the frame node.
    PublicTransformInfo transform{};
    /// Frame format version governing the transform and header semantics.
    uint32_t frameFormatVersion = 0;
    /// Host-readable codec-private transform header for this node.
    poly::span<const std::byte> codecHeader_h;
    poly::span<const InputStreamInfo> inputs;
    poly::span<const OutputStreamInfo> outputs;
};

/**
 * The device addresses of one decoder input.
 *
 * The address space is the device's: a host device's addresses are host
 * addresses.
 */
struct BoundInputStream {
    /// Beginning of the primary data region.
    const void* data_d = nullptr;
    /// Beginning of the string-length table, or nullptr for non-string streams.
    const uint32_t* stringLengths_d = nullptr;
};

/** The device addresses of one decoder output. */
struct BoundOutputStream {
    /// Beginning of the primary data region.
    void* data_d = nullptr;
    /// Beginning of the string-length table, or nullptr for non-string streams.
    uint32_t* stringLengths_d = nullptr;
};

/**
 * One codec run with device addresses for all its streams.
 *
 * @c inputs[i] and @c outputs[i] hold the addresses of @c info.inputs[i] and
 * @c info.outputs[i]. A bound run holds all the info that binding chose its
 * kernel by.
 */
struct BoundCodecRun {
    CodecRunInfo info;
    poly::span<const BoundInputStream> inputs;
    poly::span<const BoundOutputStream> outputs;
};

/** Device scratch that a kernel needs for one batch of runs. */
struct ScratchSize {
    size_t bytes = 0;
    /// Power of two that the scratch address must be a multiple of.
    size_t alignment = 1;
};

/**
 * One implementation of a codec on one device runtime.
 *
 * A device lists the kernels of each codec in its table, in order of
 * preference. A codec can have several: kernels for different variants, such
 * as element types, or alternatives for the same runs, such as a vendor
 * library. The first kernel that accepts a run decodes it.
 */
struct DecodeKernel {
    /// Stable name for errors, benchmarks, and test output.
    const char* name;
    /**
     * Whether this kernel can decode @p run, from its header, shapes, sizes,
     * and staged bytes. Binding calls it before any memory exists.
     */
    bool (*accepts)(const CodecRunInfo& run);
    /**
     * Sets @p size to the scratch that decoding @p runs as one batch needs.
     * The size may be an upper bound. Every run was accepted. Must not access
     * device memory or wait for it.
     *
     * @returns Success, or an error that stops the decode before any memory
     * is allocated. A kernel that cannot compute a maximum scratch size from
     * @p runs returns @c temporaryLibraryLimitation, so the caller can decode
     * on the CPU instead.
     */
    ZL_Report (*scratchSize)(
            poly::span<const CodecRunInfo> runs,
            ScratchSize& size);
    /**
     * Enqueues the decode of every run in @p runs on the runtime's native
     * @p stream, which is current, and returns without waiting for the device.
     *
     * @p runs are the runs that @c scratchSize sized, with addresses.
     * @p scratch_d holds @p scratchBytes bytes at the reported alignment, at
     * least the reported size, and is null when that size is 0. It stays valid
     * until the work completes. @p runs may be reused when this returns.
     *
     * Must not allocate device memory. The layout covers only the memory that
     * planning reports, so a kernel keeps its temporary data in @p scratch_d.
     */
    ZL_Report (*decode)(
            poly::span<const BoundCodecRun> runs,
            void* scratch_d,
            size_t scratchBytes,
            void* stream);
};

/**
 * Runs of one wave that one kernel decodes in one call.
 *
 * The work schedule puts each run in a wave after its producers, so the runs of
 * one wave are independent. A kernel decodes one codec, so a batch never mixes
 * codecs. Runs of one codec that need different kernels, such as two element
 * types, go to different batches.
 */
struct DecodeBatch {
    const DecodeKernel* kernel = nullptr;
    /// Indexes of the batch's runs among the runs that were batched, in
    /// their order.
    std::vector<size_t> runs;
};

/**
 * One device's kernels for each standard codec, indexed by standard transform
 * ID. Each list is in the device's order of preference. A device leaves out
 * the kernels that cannot run in this process.
 */
using DecodeKernelTable =
        std::array<std::vector<DecodeKernel>, ZL_StandardTransformID_end>;

} // namespace openzl::gpu
