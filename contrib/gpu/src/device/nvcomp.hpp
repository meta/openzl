// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <optional>

namespace openzl::gpu {

/// nvcompBatchedZstdDecompressOpts_t of nvcomp 5.
struct NvcompZstdDecompressOpts {
    /// nvcompDecompressBackend_t. 0 lets nvcomp choose the backend.
    int backend = 0;
    /// nvcomp requires zeros here.
    char reserved[60] = {};
};
static_assert(sizeof(NvcompZstdDecompressOpts) == 64);

/// nvcompAlignmentRequirements_t of nvcomp 5: the minimum alignment of each
/// buffer kind, in bytes.
struct NvcompAlignmentRequirements {
    size_t input  = 0;
    size_t output = 0;
    size_t temp   = 0;
};

/**
 * nvcomp's batched zstd decompression functions, with the signatures of
 * nvcomp 5 (nvcomp/zstd.h). Status values are @c nvcompStatus_t, where 0 is
 * success, and @c stream is a @c cudaStream_t.
 */
struct NvcompZstdFunctions {
    /// nvcompBatchedZstdDecompressGetTempSizeAsync: an upper bound of the
    /// temporary space, without device work.
    int (*decompressGetTempSizeAsync)(
            size_t numChunks,
            size_t maxUncompressedChunkBytes,
            NvcompZstdDecompressOpts opts,
            size_t* tempBytes,
            size_t maxTotalUncompressedBytes) = nullptr;
    /// nvcompBatchedZstdDecompressAsync.
    int (*decompressAsync)(
            const void* const* compressedChunks_d,
            const size_t* compressedBytes_d,
            const size_t* uncompressedBufferBytes_d,
            size_t* uncompressedBytes_d,
            size_t numChunks,
            void* temp_d,
            size_t tempBytes,
            void* const* uncompressedChunks_d,
            NvcompZstdDecompressOpts opts,
            int* statuses_d,
            void* stream) = nullptr;
    /// nvcompBatchedZstdDecompressGetRequiredAlignments.
    int (*decompressGetRequiredAlignments)(
            NvcompZstdDecompressOpts opts,
            NvcompAlignmentRequirements* alignments) = nullptr;
};

/**
 * The nvcomp 5 functions that the CUDA device's kernels use, with one part for
 * each codec. A part has a value only if all of its functions are set.
 */
struct NvcompFunctions {
    std::optional<NvcompZstdFunctions> zstd;
};

/**
 * Loads nvcomp from @p library with dlopen, and looks up the functions of each
 * codec.
 *
 * @returns nothing if @p library does not load or provides no codec. A library
 * that this returns functions for stays loaded until the process exits.
 */
std::optional<NvcompFunctions> loadNvcomp(const char* library);

/**
 * Makes the CUDA device's nvcomp kernels use @p functions, for a process that
 * links nvcomp itself or loads it from its own path. A part that lacks one of
 * its functions becomes empty. Later calls replace earlier ones, and a
 * replaced copy stays valid, because a kernel may still read it. Call it
 * before creating a device that enables nvcomp.
 *
 * @throws std::bad_alloc if memory runs out.
 */
void registerNvcomp(const NvcompFunctions& functions);

/**
 * The process's nvcomp: the registered functions, or else the functions of
 * `libnvcomp.so.5`. Every part is empty if nothing is registered and the
 * library does not load.
 *
 * Only the first call can load the library, and only if nothing is
 * registered. Later calls, for any codec or device, return the same object
 * with one atomic load until the next `registerNvcomp()`, so a kernel can call
 * this on every run. Each object stays valid until the process exits, so a
 * reference from an earlier call keeps the functions from before that
 * registration.
 *
 * @throws std::bad_alloc if memory runs out during the first call.
 */
const NvcompFunctions& nvcompFunctions();

} // namespace openzl::gpu
