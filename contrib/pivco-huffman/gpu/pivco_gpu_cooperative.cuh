// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "contrib/pivco-huffman/gpu/pivco_gpu.cuh"
#include "contrib/pivco-huffman/gpu/pivco_gpu_tree.h"

namespace {

bool encodeBuffersOverlap(
        const void* first,
        size_t firstBytes,
        const void* second,
        size_t secondBytes)
{
    const uintptr_t a = reinterpret_cast<uintptr_t>(first);
    const uintptr_t b = reinterpret_cast<uintptr_t>(second);
    return a > UINTPTR_MAX - firstBytes || b > UINTPTR_MAX - secondBytes
            || (firstBytes != 0 && secondBytes != 0 && a < b + secondBytes
                && b < a + firstBytes);
}

} // namespace

// The CUDA implementation uses 32-lane ballots and CUB. HIP builds retain the
// existing encoder without adding a hipCUB dependency or assuming wave size.
#if !defined(__HIPCC__)

#    include <cub/block/block_scan.cuh>

namespace {

constexpr unsigned kCooperativeThreads = 256;
constexpr unsigned kCooperativeWarps   = kCooperativeThreads / 32;
constexpr size_t kCooperativeMaxBlocks = kCooperativeThreads;
// tableLog <= 12 costs at most 1.5 bytes per symbol. Even allowing 24 bits
// for every tree node's count/alignment overhead, both supported block sizes
// fit in 2 * blockSize bytes. Fixed slices keep CTA writes disjoint until
// compaction. Scratch is reused across stream-ordered batches of 256 blocks.
using CooperativeScan = cub::BlockScan<unsigned, kCooperativeThreads>;

__device__ void
setStatus(PivCoGpuStatus* status, PivCoGpuStatusCode code, uint64_t detail);

size_t cooperativeEncodeWorkspaceBytes(size_t srcSize, size_t blockSize)
{
    if (srcSize == 0 || (blockSize != 32 * 1024 && blockSize != 64 * 1024)) {
        return SIZE_MAX;
    }
    const size_t totalBlocks = srcSize / blockSize + (srcSize % blockSize != 0);
    const size_t blocks      = totalBlocks < kCooperativeMaxBlocks
                 ? totalBlocks
                 : kCooperativeMaxBlocks;
    const size_t contextBytes = (sizeof(PivCoGpuContext) + 7) & ~size_t{ 7 };
    // All arithmetic is bounded by 256 blocks of 64 KiB before multiplying.
    return contextBytes + blocks * (4 * blockSize + sizeof(uint64_t));
}

bool canUseCooperativeEncode(
        const PivCoGpuContext& context,
        size_t srcSize,
        size_t blockSize,
        size_t workspaceBytes)
{
    const size_t required = cooperativeEncodeWorkspaceBytes(srcSize, blockSize);
    const auto& tree      = context.hostTree;
    const auto& schedule  = context.decodeSchedule;
    if (required == SIZE_MAX || workspaceBytes < required || tree.tableLog <= 0
        || tree.tableLog > 12 || tree.numRanks < 2
        || tree.numRanks > PIVCO_GPU_MAX_SYMBOLS || !schedule.enabled
        || schedule.nodeCount == 0
        || schedule.nodeCount > PIVCO_GPU_MAX_TREE_NODES) {
        return false;
    }
    if (schedule.nodes[0].firstRank != 0
        || schedule.nodes[0].rankEnd != tree.numRanks
        || schedule.nodes[0].level != 0) {
        return false;
    }
    // A context produced by pivcoGpuContextCreate already satisfies this.
    // Keep the dispatcher conservative if another schedule shape is added.
    for (unsigned i = 0; i < schedule.nodeCount; ++i) {
        const auto& node = schedule.nodes[i];
        if (node.firstRank >= node.rankEnd || node.rankEnd > tree.numRanks
            || node.level >= PIVCO_GPU_MAX_LEVELS) {
            return false;
        }
        if (node.kind == PIVCO_GPU_SCHEDULE_CONSTANT) {
            if (node.rankEnd != node.firstRank + 1) {
                return false;
            }
        } else if (node.kind == PIVCO_GPU_SCHEDULE_FLAT) {
            if (node.flatDepth == 0 || node.flatDepth > 8
                || node.rankEnd - node.firstRank != (1u << node.flatDepth)) {
                return false;
            }
        } else if (node.kind == PIVCO_GPU_SCHEDULE_INTERNAL) {
            if (node.leftChild <= i || node.leftChild >= schedule.nodeCount
                || node.rightChild <= node.leftChild
                || node.rightChild >= schedule.nodeCount) {
                return false;
            }
            const auto& left  = schedule.nodes[node.leftChild];
            const auto& right = schedule.nodes[node.rightChild];
            if (left.firstRank != node.firstRank
                || left.rankEnd != right.firstRank
                || right.rankEnd != node.rankEnd || left.level != node.level + 1
                || right.level != node.level + 1) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

template <unsigned Depth>
__device__ uint8_t cooperativePackedByte(
        const uint8_t* ranks,
        unsigned count,
        unsigned firstRank,
        unsigned byte)
{
    unsigned result = 0;
#    pragma unroll
    for (unsigned bit = 0; bit < 8; ++bit) {
        const unsigned position = byte * 8 + bit;
        if (position < count * Depth) {
            const unsigned rank = ranks[position / Depth] - firstRank;
            result |= ((rank >> (position % Depth)) & 1u) << bit;
        }
    }
    return static_cast<uint8_t>(result);
}

__device__ uint8_t cooperativeLeafByte(
        const uint8_t* ranks,
        unsigned count,
        unsigned firstRank,
        unsigned byte,
        unsigned depth)
{
    switch (depth) {
        case 1:
            return cooperativePackedByte<1>(ranks, count, firstRank, byte);
        case 2:
            return cooperativePackedByte<2>(ranks, count, firstRank, byte);
        case 3:
            return cooperativePackedByte<3>(ranks, count, firstRank, byte);
        case 4:
            return cooperativePackedByte<4>(ranks, count, firstRank, byte);
        case 5:
            return cooperativePackedByte<5>(ranks, count, firstRank, byte);
        case 6:
            return cooperativePackedByte<6>(ranks, count, firstRank, byte);
        case 7:
            return cooperativePackedByte<7>(ranks, count, firstRank, byte);
        case 8:
            return cooperativePackedByte<8>(ranks, count, firstRank, byte);
        default:
            return 0;
    }
}

__global__ void cooperativeEncodeBlocksKernel(
        const PivCoGpuContext* context,
        const uint8_t* src,
        size_t srcSize,
        size_t blockSize,
        size_t firstBlock,
        uint8_t* rankA,
        uint8_t* rankB,
        uint8_t* blockBitstreams,
        uint64_t* blockSizes,
        PivCoGpuStatus* status)
{
    __shared__ unsigned histogram[kCooperativeWarps][PIVCO_GPU_MAX_SYMBOLS];
    __shared__ unsigned prefix[PIVCO_GPU_MAX_SYMBOLS + 1];
    __shared__ CooperativeScan::TempStorage scan;
    __shared__ unsigned nodeBegin, nodeCount, nodeZeros, nodeOnes;
    __shared__ unsigned nodeByte, countBits, runningOnes, invalid;
    __shared__ uint64_t bitPosition;
    const auto& tree      = context->hostTree;
    const auto& schedule  = context->decodeSchedule;
    const size_t block    = blockIdx.x;
    const size_t start    = block * blockSize;
    const unsigned length = static_cast<unsigned>(
            srcSize - start < blockSize ? srcSize - start : blockSize);
    uint8_t* const a      = rankA + start;
    uint8_t* const b      = rankB + start;
    uint8_t* const output = blockBitstreams + block * (2 * blockSize);
    for (unsigned x = threadIdx.x; x < kCooperativeWarps * 256;
         x += kCooperativeThreads) {
        histogram[x / 256][x % 256] = 0;
    }
    if (threadIdx.x == 0) {
        bitPosition = 0;
        invalid     = 0;
    }
    __syncthreads();
    for (unsigned tile = 0; tile < length; tile += kCooperativeThreads) {
        const unsigned x = tile + threadIdx.x;
        const bool valid = x < length;
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        const unsigned active = __ballot_sync(0xffffffffu, valid);
#    endif
        if (valid) {
            const uint8_t symbol = src[start + x];
            const bool present   = tree.symbolPresent[symbol] != 0;
            if (!present) {
                atomicExch(&invalid, 1u);
            }
            const unsigned rank = present ? tree.symbolToRank[symbol] : 0;
            a[x]                = static_cast<uint8_t>(rank);
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            // Warp-private bins and match-any avoid one shared atomic per
            // repeated symbol. Older CUDA architectures keep scalar atomics.
            const unsigned equal = __match_any_sync(active, rank);
            if ((threadIdx.x & 31u) == unsigned(__ffs(equal) - 1)) {
                atomicAdd(
                        &histogram[threadIdx.x / 32][rank],
                        unsigned(__popc(equal)));
            }
#    else
            atomicAdd(&histogram[threadIdx.x / 32][rank], 1u);
#    endif
        }
    }
    __syncthreads();
    if (invalid) {
        if (threadIdx.x == 0) {
            blockSizes[block] = 0;
            setStatus(
                    status,
                    PIVCO_GPU_STATUS_MISSING_SYMBOL,
                    firstBlock + block);
        }
        return;
    }
    if (threadIdx.x == 0) {
        prefix[0] = 0;
        for (unsigned rank = 0; rank < 256; ++rank) {
            unsigned count = 0;
            for (unsigned warp = 0; warp < kCooperativeWarps; ++warp) {
                count += histogram[warp][rank];
            }
            prefix[rank + 1] = prefix[rank] + count;
        }
    }
    __syncthreads();
    // Preorder and stable partition order are wire-format invariants. Child
    // rank intervals are disjoint, so depth-parity buffers preserve unvisited
    // sibling streams even though nodes are visited in depth-first order.
    for (unsigned index = 0; index < schedule.nodeCount; ++index) {
        const auto node = schedule.nodes[index];
        if (node.kind == PIVCO_GPU_SCHEDULE_CONSTANT) {
            continue;
        }
        if (threadIdx.x == 0) {
            nodeBegin   = prefix[node.firstRank];
            nodeCount   = prefix[node.rankEnd] - nodeBegin;
            bitPosition = (bitPosition + 7) & ~uint64_t{ 7 };
            nodeByte    = static_cast<unsigned>(bitPosition / 8);
            if (node.kind == PIVCO_GPU_SCHEDULE_FLAT) {
                bitPosition += uint64_t(nodeCount) * node.flatDepth;
            } else {
                const auto& left  = schedule.nodes[node.leftChild];
                const auto& right = schedule.nodes[node.rightChild];
                nodeZeros         = prefix[right.firstRank] - nodeBegin;
                nodeOnes          = nodeCount - nodeZeros;
                const bool bothConstant =
                        left.kind == PIVCO_GPU_SCHEDULE_CONSTANT
                        && right.kind == PIVCO_GPU_SCHEDULE_CONSTANT;
                countBits = bothConstant || nodeCount == 0
                        ? 0
                        : 32u - __clz(nodeCount);
                bitPosition += nodeCount + countBits;
                runningOnes = 0;
            }
            if ((bitPosition + 7) / 8 > 2 * blockSize) {
                invalid = 1;
            }
        }
        __syncthreads();
        if (invalid) {
            if (threadIdx.x == 0) {
                blockSizes[block] = 0;
                setStatus(
                        status, PIVCO_GPU_STATUS_CAPACITY, firstBlock + block);
            }
            return;
        }
        const uint8_t* const ranks = (node.level & 1u) ? b : a;
        uint8_t* const partitioned = (node.level & 1u) ? a : b;
        if (node.kind == PIVCO_GPU_SCHEDULE_FLAT) {
            const unsigned bytes = (nodeCount * node.flatDepth + 7) / 8;
            for (unsigned x = threadIdx.x; x < bytes;
                 x += kCooperativeThreads) {
                output[nodeByte + x] = cooperativeLeafByte(
                        ranks + nodeBegin,
                        nodeCount,
                        node.firstRank,
                        x,
                        node.flatDepth);
            }
            __syncthreads();
            continue;
        }
        const unsigned split = schedule.nodes[node.rightChild].firstRank;
        for (unsigned tile = 0; tile < nodeCount; tile += kCooperativeThreads) {
            const unsigned x      = tile + threadIdx.x;
            const bool valid      = x < nodeCount;
            const unsigned rank   = valid ? ranks[nodeBegin + x] : 0;
            const unsigned right  = valid && rank >= split;
            const unsigned ballot = __ballot_sync(0xffffffffu, right);
            if ((threadIdx.x & 31u) == 0) {
                const unsigned first = tile + threadIdx.x;
                const unsigned count = first < nodeCount
                        ? ((nodeCount - first < 32 ? nodeCount - first : 32)
                           + 7) / 8
                        : 0;
                for (unsigned byte = 0; byte < count; ++byte) {
                    output[nodeByte + first / 8 + byte] =
                            static_cast<uint8_t>(ballot >> (8 * byte));
                }
            }
            unsigned before = 0, total = 0;
            CooperativeScan(scan).ExclusiveSum(right, before, total);
            if (valid) {
                const unsigned position = right
                        ? nodeBegin + nodeZeros + runningOnes + before
                        : nodeBegin + tile - runningOnes + threadIdx.x - before;
                partitioned[position]   = static_cast<uint8_t>(rank);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                runningOnes += total;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const unsigned tail = nodeCount & 7u;
            if (tail) {
                output[nodeByte + nodeCount / 8] &=
                        static_cast<uint8_t>((1u << tail) - 1u);
            }
            // The count may share the bitmap's last byte. All bitmap writers
            // have finished; untouched bits start at zero in the staging area.
            for (unsigned bit = 0; bit < countBits; ++bit) {
                if ((nodeOnes >> bit) & 1u) {
                    const unsigned position = nodeCount + bit;
                    output[nodeByte + position / 8] |=
                            static_cast<uint8_t>(1u << (position & 7u));
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        blockSizes[block] = (bitPosition + 7) / 8;
    }
}

__global__ void cooperativeScanSizesKernel(
        const uint64_t* sizes,
        unsigned blocks,
        uint64_t* offsets,
        uint64_t* totalSize,
        size_t capacity,
        PivCoGpuStatus* status,
        bool firstBatch)
{
    if (status->code != PIVCO_GPU_STATUS_OK) {
        return;
    }
    __shared__ cub::BlockScan<uint64_t, kCooperativeThreads>::TempStorage scan;
    __shared__ uint64_t base;
    if (threadIdx.x == 0) {
        base = firstBatch ? 0 : offsets[0];
    }
    __syncthreads();
    const uint64_t value = threadIdx.x < blocks ? sizes[threadIdx.x] : 0;
    uint64_t before = 0, total = 0;
    cub::BlockScan<uint64_t, kCooperativeThreads>(scan).ExclusiveSum(
            value, before, total);
    // The API caps the call at 2^31 - 1 blocks of at most 64 KiB;
    // even the 2 * blockSize staging bound keeps this sum below 2^48.
    if (threadIdx.x < blocks) {
        offsets[threadIdx.x] = base + before;
    }
    if (threadIdx.x == 0) {
        offsets[blocks] = base + total;
        *totalSize      = base + total;
        if (base > capacity || total > capacity - base) {
            setStatus(status, PIVCO_GPU_STATUS_CAPACITY, base + total);
        }
    }
}

__global__ void cooperativeCopyBlocksKernel(
        const uint8_t* blockBitstreams,
        const uint64_t* sizes,
        const uint64_t* offsets,
        size_t blockSize,
        uint8_t* dst,
        const PivCoGpuStatus* status)
{
    if (status->code != PIVCO_GPU_STATUS_OK) {
        return;
    }
    const unsigned block = blockIdx.x;
    for (size_t x = threadIdx.x; x < sizes[block]; x += kCooperativeThreads) {
        dst[offsets[block] + x] = blockBitstreams[block * (2 * blockSize) + x];
    }
}

cudaError_t launchCooperativeEncode(
        const PivCoGpuContext* context,
        void* dst,
        size_t dstCapacity,
        uint64_t* offsets,
        const void* src,
        size_t srcSize,
        size_t blockSize,
        void* workspace,
        PivCoGpuStatus* status,
        uint64_t* totalSize,
        cudaStream_t stream)
{
    const size_t blocks = srcSize / blockSize + (srcSize % blockSize != 0);
    const size_t batchCapacity =
            blocks < kCooperativeMaxBlocks ? blocks : kCooperativeMaxBlocks;
    const size_t required = cooperativeEncodeWorkspaceBytes(srcSize, blockSize);
    const size_t offsetsBytes = (blocks + 1) * sizeof(uint64_t);
    const void* buffers[] = { src, dst, offsets, workspace, status, totalSize };
    const size_t sizes[]  = { srcSize,  dstCapacity,     offsetsBytes,
                              required, sizeof(*status), sizeof(*totalSize) };
    if (reinterpret_cast<uintptr_t>(workspace) % alignof(uint64_t) != 0
        || reinterpret_cast<uintptr_t>(offsets) % alignof(uint64_t) != 0
        || reinterpret_cast<uintptr_t>(totalSize) % alignof(uint64_t) != 0
        || reinterpret_cast<uintptr_t>(status) % alignof(PivCoGpuStatus) != 0) {
        return cudaErrorInvalidValue;
    }
    for (unsigned i = 0; i < 6; ++i) {
        for (unsigned j = i + 1; j < 6; ++j) {
            if (encodeBuffersOverlap(
                        buffers[i], sizes[i], buffers[j], sizes[j])) {
                return cudaErrorInvalidValue;
            }
        }
    }
    auto* const context_d = static_cast<PivCoGpuContext*>(workspace);
    auto* const rankA     = static_cast<uint8_t*>(workspace)
            + ((sizeof(PivCoGpuContext) + 7) & ~size_t{ 7 });
    auto* const rankB           = rankA + batchCapacity * blockSize;
    auto* const blockBitstreams = rankB + batchCapacity * blockSize;
    auto* const blockSizes      = reinterpret_cast<uint64_t*>(
            blockBitstreams + batchCapacity * (2 * blockSize));
    cudaError_t err = cudaMemcpyAsync(
            context_d,
            context,
            sizeof(*context),
            cudaMemcpyHostToDevice,
            stream);
    if (err != cudaSuccess) {
        return err;
    }
    for (size_t firstBlock = 0; firstBlock < blocks;
         firstBlock += batchCapacity) {
        const size_t batchBlocks = blocks - firstBlock < batchCapacity
                ? blocks - firstBlock
                : batchCapacity;
        const size_t start       = firstBlock * blockSize;
        const size_t batchBytes  = srcSize - start < batchBlocks * blockSize
                 ? srcSize - start
                 : batchBlocks * blockSize;
        err                      = cudaMemsetAsync(
                blockBitstreams, 0, batchBlocks * (2 * blockSize), stream);
        if (err != cudaSuccess) {
            return err;
        }
        cooperativeEncodeBlocksKernel<<<
                batchBlocks,
                kCooperativeThreads,
                0,
                stream>>>(
                context_d,
                static_cast<const uint8_t*>(src) + start,
                batchBytes,
                blockSize,
                firstBlock,
                rankA,
                rankB,
                blockBitstreams,
                blockSizes,
                status);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        cooperativeScanSizesKernel<<<1, kCooperativeThreads, 0, stream>>>(
                blockSizes,
                static_cast<unsigned>(batchBlocks),
                offsets + firstBlock,
                totalSize,
                dstCapacity,
                status,
                firstBlock == 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        cooperativeCopyBlocksKernel<<<
                batchBlocks,
                kCooperativeThreads,
                0,
                stream>>>(
                blockBitstreams,
                blockSizes,
                offsets + firstBlock,
                blockSize,
                static_cast<uint8_t*>(dst),
                status);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
    }
    return cudaSuccess;
}

} // namespace

#endif // !defined(__HIPCC__)
