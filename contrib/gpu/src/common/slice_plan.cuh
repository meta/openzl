// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "contrib/gpu/src/common/cuda_error.cuh"
#include "contrib/gpu/src/common/cuda_raii.cuh"

// Splits a batch of ranges into equal-work slices on the host (advancing
// device pointers only, no copy) and stages the slice descriptors on the
// device once; the codec launches its own kernel over deviceSlices().
//
// A Range must expose:
//   size_t nbElts;         // elements left in this range
//   Range  peel(size_t n); // take the first min(n, nbElts) elements as a new
//                          // slice and advance *this past them (device
//                          // pointers only, no copy)

namespace openzl::gpu {

// Number of slices of at most maxSliceElts that a range of nbElts splits into.
// nbElts + maxSliceElts can wrap for a maxSliceElts near SIZE_MAX, which a
// caller can pass to mean "no limit", so the division rounds up without it.
inline size_t sliceCount(size_t nbElts, size_t maxSliceElts)
{
    return nbElts / maxSliceElts + (nbElts % maxSliceElts != 0 ? 1 : 0);
}

// Splits each range into slices of at most maxSliceElts.
template <typename Range>
std::vector<Range>
splitIntoSlices(const Range* ranges_h, uint32_t numRanges, size_t maxSliceElts)
{
    size_t numSlices = 0;
    for (uint32_t c = 0; c < numRanges; ++c) {
        numSlices += sliceCount(ranges_h[c].nbElts, maxSliceElts);
    }
    if (numSlices > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
                "SlicePlan: slice count " + std::to_string(numSlices)
                + " exceeds uint32_t limit; use a larger maxSliceElts");
    }
    std::vector<Range> out;
    out.reserve(numSlices);
    for (uint32_t c = 0; c < numRanges; ++c) {
        Range cur = ranges_h[c];
        while (cur.nbElts > 0) {
            out.push_back(cur.peel(maxSliceElts));
        }
    }
    return out;
}

// Splits `ranges_h` into slices and stages the descriptors on the device
// once; owns that memory. The codec launches its kernel over
// deviceSlices()/numSlices().
template <typename Range>
class SlicePlan {
   public:
    // sliceAlignElts: slice sizes must be a positive multiple of this (e.g. the
    // codec's vector width, so slice starts stay aligned). 1 = no constraint.
    SlicePlan(
            const Range* ranges_h,
            uint32_t numRanges,
            size_t maxSliceElts,
            size_t sliceAlignElts = 1)
    {
        if (maxSliceElts == 0 || sliceAlignElts == 0
            || maxSliceElts % sliceAlignElts != 0) {
            throw std::runtime_error(
                    "SlicePlan: maxSliceElts " + std::to_string(maxSliceElts)
                    + " must be a positive multiple of "
                    + std::to_string(sliceAlignElts));
        }
        if (numRanges == 0) {
            return;
        }
        const std::vector<Range> slices =
                splitIntoSlices(ranges_h, numRanges, maxSliceElts);
        if (slices.empty()) {
            return;
        }
        numSlices_ = (uint32_t)slices.size();
        slices_d_  = deviceAlloc<Range>(numSlices_);
        ZL_CUDA_CHECK(cudaMemcpy(
                slices_d_.get(),
                slices.data(),
                (size_t)numSlices_ * sizeof(Range),
                cudaMemcpyHostToDevice));
    }

    // Device array of slice descriptors, numSlices() long. Valid until *this is
    // destroyed.
    const Range* deviceSlices() const
    {
        return slices_d_.get();
    }
    uint32_t numSlices() const
    {
        return numSlices_;
    }

   private:
    DevicePtr<Range> slices_d_;
    uint32_t numSlices_ = 0;
};

} // namespace openzl::gpu
