// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "decode_float_deconstruct_bf16.cuh"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "contrib/gpu/src/common/cuda_error.cuh"
#include "contrib/gpu/src/common/cuda_launch.cuh"
#include "contrib/gpu/src/common/cuda_raii.cuh"
#include "contrib/gpu/src/common/slice_plan.cuh"

namespace openzl::gpu {

namespace {

constexpr int kThreads = 256;
constexpr int kVec     = 4; // elements per thread step (uchar4 in, ushort4 out)
constexpr int kUnroll  = 2; // independent uchar4 loads/stream in flight (MLP)

// Reassembles one uchar4 of exponent + one uchar4 of signFrac into a ushort4.
__device__ __forceinline__ ushort4 joinVec4(uchar4 e, uchar4 s)
{
    ushort4 out;
    out.x = decodeBf16Elt(e.x, s.x);
    out.y = decodeBf16Elt(e.y, s.y);
    out.z = decodeBf16Elt(e.z, s.z);
    out.w = decodeBf16Elt(e.w, s.w);
    return out;
}

// Whether every address of `c` is aligned for the vector loads and stores of
// decodeSliceVec.
__device__ __forceinline__ bool isVecAligned(const Bf16DeconElementRange& c)
{
    const uintptr_t inputs = reinterpret_cast<uintptr_t>(c.exponent)
            | reinterpret_cast<uintptr_t>(c.signFrac);
    return inputs % alignof(uchar4) == 0
            && reinterpret_cast<uintptr_t>(c.dst) % alignof(ushort4) == 0;
}

// Vectorized decode of one slice: the threads [start, start+stride, ...)
// cover the slice, each handling kVec elements per step via wide transactions
// (uchar4 in, ushort4 out) so both input streams and the output move in wide,
// fully coalesced transactions. The main loop is unrolled by kUnroll so
// several independent loads from each stream are in flight before the stores,
// hiding DRAM latency (memory-level parallelism). A remainder loop and then a
// scalar tail (last nbElts % kVec) finish the slice. Requires
// isVecAligned(c).
__device__ __forceinline__ void
decodeSliceVec(const Bf16DeconElementRange& c, size_t start, size_t stride)
{
    // Misaligned vector loads and stores are UB on CUDA.
    assert(isVecAligned(c));
    const size_t nVec = c.nbElts / kVec;
    size_t v          = start;
    for (; v + (size_t)(kUnroll - 1) * stride < nVec;
         v += (size_t)kUnroll * stride) {
        uchar4 e[kUnroll];
        uchar4 s[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const size_t base = (v + (size_t)u * stride) * kVec;
            e[u] = *reinterpret_cast<const uchar4*>(c.exponent + base);
            s[u] = *reinterpret_cast<const uchar4*>(c.signFrac + base);
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const size_t base = (v + (size_t)u * stride) * kVec;
            *reinterpret_cast<ushort4*>(c.dst + base) = joinVec4(e[u], s[u]);
        }
    }
    for (; v < nVec; v += stride) {
        const size_t base = v * kVec;
        const uchar4 e    = *reinterpret_cast<const uchar4*>(c.exponent + base);
        const uchar4 s    = *reinterpret_cast<const uchar4*>(c.signFrac + base);
        *reinterpret_cast<ushort4*>(c.dst + base) = joinVec4(e, s);
    }
    for (size_t i = nVec * kVec + start; i < c.nbElts; i += stride) {
        c.dst[i] = decodeBf16Elt(c.exponent[i], c.signFrac[i]);
    }
}

// Element-by-element decode of one slice, for addresses that decodeSliceVec
// cannot use. Consecutive threads touch consecutive elements, so the accesses
// stay coalesced.
__device__ __forceinline__ void
decodeSliceScalar(const Bf16DeconElementRange& c, size_t start, size_t stride)
{
    for (size_t i = start; i < c.nbElts; i += stride) {
        c.dst[i] = decodeBf16Elt(c.exponent[i], c.signFrac[i]);
    }
}

// The host has split the input into equal-sized slices (each a contiguous
// part of one range); one block owns a slice and grid-strides over the
// slice array, so equal slices balance the work. A 1D grid means numSlices
// has no gridDim.y cap.
__global__ void decodeSlicesKernel(
        const Bf16DeconElementRange* __restrict__ slices,
        uint32_t numSlices)
{
    for (uint32_t s = blockIdx.x; s < numSlices; s += gridDim.x) {
        // Every thread of the block reads the same slice, so the whole block
        // takes the same branch.
        const Bf16DeconElementRange slice = slices[s];
        if (isVecAligned(slice)) {
            decodeSliceVec(slice, threadIdx.x, blockDim.x);
        } else {
            decodeSliceScalar(slice, threadIdx.x, blockDim.x);
        }
    }
}

// Throws unless maxSliceElts is a positive multiple of kVec, so that every
// slice of a range keeps the alignment of the range's start.
void validateMaxSliceElts(size_t maxSliceElts)
{
    if (maxSliceElts == 0 || maxSliceElts % kVec != 0) {
        throw std::runtime_error(
                "bf16 decode: maxSliceElts " + std::to_string(maxSliceElts)
                + " must be a positive multiple of " + std::to_string(kVec));
    }
}

// Sizes a 1D grid that fills the GPU (capped at numSlices) and launches the
// decode kernel over the staged slice array.
void launchSlices(
        const Bf16DeconElementRange* slices_d,
        uint32_t numSlices,
        cudaStream_t stream)
{
    const uint32_t target =
            fillGpuGrid((const void*)decodeSlicesKernel, kThreads);
    const uint32_t grid = (uint32_t)std::min<size_t>(numSlices, target);
    ZL_CUDA_CHECK(launchKernel(
            decodeSlicesKernel,
            grid,
            kThreads,
            0,
            stream,
            slices_d,
            numSlices));
}

} // namespace

// sliceAlignElts = kVec keeps every slice start at the alignment of its
// range's start.
Bf16DecodePlan::Bf16DecodePlan(
        const Bf16DeconElementRange* ranges_h,
        uint32_t numRanges,
        size_t maxSliceElts)
        : plan_(ranges_h, numRanges, maxSliceElts, kVec)
{
}

void Bf16DecodePlan::launch(cudaStream_t stream) const
{
    if (plan_.numSlices() == 0) {
        return;
    }
    launchSlices(plan_.deviceSlices(), plan_.numSlices(), stream);
}

void bf16DeconDecode(
        const Bf16DeconElementRange* ranges_h,
        uint32_t numRanges,
        size_t maxSliceElts,
        cudaStream_t stream)
{
    if (numRanges == 0) {
        return;
    }
    validateMaxSliceElts(maxSliceElts);
    const std::vector<Bf16DeconElementRange> slices =
            splitIntoSlices(ranges_h, numRanges, maxSliceElts);
    if (slices.empty()) {
        return;
    }
    const uint32_t numSlices = (uint32_t)slices.size();

    // Stream-ordered scratch so the one-shot path stays async and leak-free.
    StreamScratch<Bf16DeconElementRange> slices_d(numSlices, stream);
    ZL_CUDA_CHECK(cudaMemcpyAsync(
            slices_d.get(),
            slices.data(),
            (size_t)numSlices * sizeof(Bf16DeconElementRange),
            cudaMemcpyHostToDevice,
            stream));
    launchSlices(slices_d.get(), numSlices, stream);
}

size_t bf16DeconDecodeScratchBytes(
        const size_t* nbElts_h,
        uint32_t numRanges,
        size_t maxSliceElts)
{
    validateMaxSliceElts(maxSliceElts);
    size_t numSlices = 0;
    for (uint32_t c = 0; c < numRanges; ++c) {
        numSlices += sliceCount(nbElts_h[c], maxSliceElts);
    }
    return numSlices * sizeof(Bf16DeconElementRange);
}

void bf16DeconDecode(
        const Bf16DeconElementRange* ranges_h,
        uint32_t numRanges,
        size_t maxSliceElts,
        void* scratch_d,
        size_t scratchBytes,
        cudaStream_t stream)
{
    if (numRanges == 0) {
        return;
    }
    validateMaxSliceElts(maxSliceElts);
    const std::vector<Bf16DeconElementRange> slices =
            splitIntoSlices(ranges_h, numRanges, maxSliceElts);
    if (slices.empty()) {
        return;
    }
    const size_t segBytes = slices.size() * sizeof(Bf16DeconElementRange);
    if (scratchBytes < segBytes) {
        throw std::runtime_error(
                "bf16 decode: scratch has " + std::to_string(scratchBytes)
                + " bytes, needs " + std::to_string(segBytes));
    }
    auto* const slices_d = static_cast<Bf16DeconElementRange*>(scratch_d);
    ZL_CUDA_CHECK(cudaMemcpyAsync(
            slices_d, slices.data(), segBytes, cudaMemcpyHostToDevice, stream));
    launchSlices(slices_d, (uint32_t)slices.size(), stream);
}

KernelLaunchInfo bf16DeconDecodeLaunchInfo()
{
    return launchInfoFor((const void*)decodeSlicesKernel, kThreads);
}

} // namespace openzl::gpu
