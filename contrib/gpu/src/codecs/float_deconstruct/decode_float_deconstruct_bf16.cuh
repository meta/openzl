// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "contrib/gpu/src/codecs/float_deconstruct/float_deconstruct_bf16.hpp"
#include "contrib/gpu/src/common/cuda_launch.cuh"
#include "contrib/gpu/src/common/slice_plan.cuh"

namespace openzl::gpu {

// Default slice size, in elements. From the benchmark's granularity sweep,
// kernel throughput keeps improving down to ~4Ki-element slices; 16Ki stays
// within a few percent of that peak on every shape while producing 4x fewer
// slices (less descriptor upload for the one-shot path). Tunable; must be a
// positive multiple of 4 (see Bf16DecodePlan).
constexpr size_t kBf16DefaultMaxSliceElts = 16384;

// Prepared bf16 float_deconstruct decode. Construction splits every range into
// slices of at most maxSliceElts on the host (advancing device pointers, zero
// copy) and stages the slice descriptors on the device once; launch() then
// runs one kernel over them. Equal-sized slices balance the work across
// ranges of any size. Splitting once and launching repeatedly keeps the host
// split and the descriptor upload out of the per-launch cost.
//
// Takes HOST-side descriptors `ranges_h` (device pointers + sizes), since the
// split runs on the host. Each dst must be 2-byte aligned. The kernel decodes
// a slice with vector loads and stores (uchar4 in, ushort4 out) when its
// exponent and signFrac are 4-byte aligned and its dst is 8-byte aligned, and
// with element loads and stores otherwise. maxSliceElts must be a positive
// multiple of 4, so every slice of a range has the alignment of the range's
// start; the constructor throws otherwise.
class Bf16DecodePlan {
   public:
    Bf16DecodePlan(
            const Bf16DeconElementRange* ranges_h,
            uint32_t numRanges,
            size_t maxSliceElts);

    // Launches the decode on `stream`. This object must stay alive until the
    // stream work completes (it owns the device slice descriptors).
    void launch(cudaStream_t stream) const;

    uint32_t numSlices() const
    {
        return plan_.numSlices();
    }

   private:
    SlicePlan<Bf16DeconElementRange> plan_;
};

// One-shot convenience: prepare a Bf16DecodePlan and launch it on `stream`.
// Prefer Bf16DecodePlan directly when decoding the same shape repeatedly.
void bf16DeconDecode(
        const Bf16DeconElementRange* ranges_h,
        uint32_t numRanges,
        size_t maxSliceElts,
        cudaStream_t stream);

// Device bytes of the slice descriptors that bf16DeconDecode stages for
// numRanges ranges of nbElts_h[c] elements each. Throws as bf16DeconDecode
// does for an invalid maxSliceElts.
size_t bf16DeconDecodeScratchBytes(
        const size_t* nbElts_h,
        uint32_t numRanges,
        size_t maxSliceElts);

// Like the one-shot bf16DeconDecode, but stages the slice descriptors in
// scratch_d, which the caller allocates and keeps until the stream work
// completes. scratch_d holds scratchBytes bytes, at least
// bf16DeconDecodeScratchBytes of the same ranges, aligned for
// Bf16DeconElementRange. Throws if scratchBytes is too small.
void bf16DeconDecode(
        const Bf16DeconElementRange* ranges_h,
        uint32_t numRanges,
        size_t maxSliceElts,
        void* scratch_d,
        size_t scratchBytes,
        cudaStream_t stream);

// Occupancy/launch metadata for the decode kernel, so a benchmark harness can
// report theoretical occupancy without seeing the kernel internals.
// KernelLaunchInfo is the shared descriptor from common/cuda_launch.cuh.
KernelLaunchInfo bf16DeconDecodeLaunchInfo();

} // namespace openzl::gpu
