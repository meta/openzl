// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/pivco-huffman/gpu/pivco_gpu.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "contrib/pivco-huffman/gpu/pivco_gpu_tree.h"

namespace {

constexpr size_t kMaxGridX           = 2147483647u;
constexpr size_t kTreeWorkspaceBytes = ((sizeof(PivCoGpuTree) + 7) / 8) * 8;
constexpr size_t kScheduleWorkspaceBytes =
        ((sizeof(PivCoGpuDecodeSchedule) + 7) / 8) * 8;
constexpr size_t kDecodeStaticWorkspaceBytes =
        kTreeWorkspaceBytes + kScheduleWorkspaceBytes;
constexpr size_t kFastMaxBlockSize       = 64 * 1024;
constexpr size_t kFastMaxRootBytes       = (kFastMaxBlockSize + 7) / 8;
constexpr int kFastBlockThreads          = 256;
constexpr int kMaxRankSelectNodes        = PIVCO_GPU_MAX_TREE_NODES;
constexpr int kRankSelectThreads         = 256;
constexpr size_t kRankSelectMaxBlockSize = 64 * 1024;
constexpr size_t kRankSelectMaxTableLog  = 12;
constexpr uint32_t kRankSelectMaxBitmapWords =
        (kRankSelectMaxBlockSize + 31) / 32;
constexpr uint32_t kBottomUpChunkedMergeThreshold = 1024;
constexpr int kFastThreadLeafWords =
        ((((kFastMaxRootBytes + kFastBlockThreads - 1) / kFastBlockThreads) * 8)
         + 31)
        / 32;
constexpr int kScanItemsPerBlock = 256;
// Trailing pad on each scheduled ping-pong buffer. Node streams are placed at
// 8-byte-aligned offsets (up to 7 bytes of padding each, at most one per
// materialized node per level, i.e. <= numRanks <= 256) and the byte merge
// always stores a full aligned 8-byte group, so the buffer must hold blockSize
// + 7*256 + 8 rounded up.
constexpr size_t kMergeBufferPad = 2048;

struct ScheduledNodeState {
    uint32_t bitmapByteBase;
    uint32_t leafBitBase;
    uint32_t count;
    uint32_t dirBase;
    uint32_t streamBase;
};

ZL_Report cudaReport(cudaError_t err)
{
    return err == cudaSuccess ? ZL_returnSuccess()
                              : ZL_returnError(ZL_ErrorCode_GENERIC);
}

ZL_Report statusReport(const PivCoGpuStatus& status)
{
    switch (status.code) {
        case PIVCO_GPU_STATUS_OK:
            return ZL_returnSuccess();
        case PIVCO_GPU_STATUS_PARAMETER:
            return ZL_returnError(ZL_ErrorCode_parameter_invalid);
        case PIVCO_GPU_STATUS_CORRUPTION:
            return ZL_returnError(ZL_ErrorCode_corruption);
        case PIVCO_GPU_STATUS_CAPACITY:
            return ZL_returnError(ZL_ErrorCode_dstCapacity_tooSmall);
        case PIVCO_GPU_STATUS_MISSING_SYMBOL:
            return ZL_returnError(ZL_ErrorCode_node_invalid_input);
        default:
            return ZL_returnError(ZL_ErrorCode_GENERIC);
    }
}

bool addOverflows(size_t a, size_t b)
{
    return a > SIZE_MAX - b;
}

size_t numBlocksFor(size_t size, size_t blockSize)
{
    return size == 0 ? 0 : (size + blockSize - 1) / blockSize;
}

size_t alignUpSize(size_t value, size_t alignment);
size_t rankSelectWorkspaceBytes(size_t blockSize, int tableLog);
size_t scheduledDecodeBlockWorkspaceBytes(size_t blockSize);

size_t workspaceBytesFor(size_t size, size_t blockSize)
{
    if (size == 0) {
        return 0;
    }
    if (blockSize == 0 || addOverflows(size, blockSize - 1)) {
        return SIZE_MAX;
    }
    const size_t numBlocks = numBlocksFor(size, blockSize);
    if (numBlocks > SIZE_MAX / blockSize / 2) {
        return SIZE_MAX;
    }
    const size_t blockWorkspaceBytes = 2 * numBlocks * blockSize;
    if (addOverflows(kTreeWorkspaceBytes, blockWorkspaceBytes)) {
        return SIZE_MAX;
    }
    return kTreeWorkspaceBytes + blockWorkspaceBytes;
}

// Per-block bytes for the main decode workspace (node states + rank directory +
// the scheduled/chunk decoders' stream buffers).
size_t decodeBlockWorkspaceBytes(size_t blockSize)
{
    const size_t scheduledWorkspace =
            scheduledDecodeBlockWorkspaceBytes(blockSize);
    return scheduledWorkspace == SIZE_MAX ? 2 * blockSize : scheduledWorkspace;
}

size_t decodeWorkspaceBytesFor(size_t size, size_t blockSize)
{
    if (size == 0) {
        return 0;
    }
    if (blockSize == 0 || addOverflows(size, blockSize - 1)) {
        return SIZE_MAX;
    }
    const size_t numBlocks      = numBlocksFor(size, blockSize);
    const size_t blockWorkspace = decodeBlockWorkspaceBytes(blockSize);
    if (numBlocks > SIZE_MAX / blockWorkspace) {
        return SIZE_MAX;
    }
    const size_t blockWorkspaceBytes = numBlocks * blockWorkspace;
    if (addOverflows(kDecodeStaticWorkspaceBytes, blockWorkspaceBytes)) {
        return SIZE_MAX;
    }
    return kDecodeStaticWorkspaceBytes + blockWorkspaceBytes;
}

// The fast root-const-flat1 DECODE kernel handles any block size up to the max.
bool canUseFastRootConstFlat1Decode(const PivCoGpuTree& tree, size_t blockSize)
{
    return tree.fastMode == PIVCO_GPU_FAST_ROOT_CONST_FLAT1
            && blockSize <= kFastMaxBlockSize;
}

// The fast ENCODE path additionally requires blockSize >= 1024: its offset-scan
// is only exercised at that scale, so smaller block counts use the general
// encode path.
bool canUseFastRootConstFlat1Encode(const PivCoGpuTree& tree, size_t blockSize)
{
    return canUseFastRootConstFlat1Decode(tree, blockSize) && blockSize >= 1024;
}

bool canUseFastFlatRoot(const PivCoGpuTree& tree)
{
    return tree.fastMode == PIVCO_GPU_FAST_FLAT_ROOT;
}

size_t alignUpSize(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

size_t rankSelectWorkspaceBytes(size_t blockSize, int tableLog)
{
    if (tableLog <= 0 || blockSize > kRankSelectMaxBlockSize) {
        return SIZE_MAX;
    }
    const size_t dirEntries =
            ((blockSize * static_cast<size_t>(tableLog) + 31) / 32)
            + kMaxRankSelectNodes;
    return dirEntries * sizeof(uint16_t);
}

size_t scheduledDecodeBlockWorkspaceBytes(size_t blockSize)
{
    if (blockSize > kRankSelectMaxBlockSize) {
        return SIZE_MAX;
    }
    const size_t nodeStateBytes = alignUpSize(
            sizeof(ScheduledNodeState) * PIVCO_GPU_MAX_TREE_NODES,
            alignof(ScheduledNodeState));
    const size_t directoryBytes =
            rankSelectWorkspaceBytes(blockSize, kRankSelectMaxTableLog);
    if (directoryBytes == SIZE_MAX) {
        return SIZE_MAX;
    }
    const size_t prefixBytes = alignUpSize(
            nodeStateBytes + alignUpSize(directoryBytes, alignof(uint16_t)), 8);
    const size_t bufBytes = blockSize + kMergeBufferPad;
    if (bufBytes > (SIZE_MAX - prefixBytes) / 2) {
        return SIZE_MAX;
    }
    return alignUpSize(prefixBytes + 2 * bufBytes, alignof(ScheduledNodeState));
}

bool canUseScheduledDecode(
        const PivCoGpuTree& tree,
        const PivCoGpuDecodeSchedule& schedule,
        size_t blockSize)
{
    return tree.fastMode == PIVCO_GPU_FAST_NONE && schedule.enabled != 0
            && tree.tableLog > 0
            && static_cast<size_t>(tree.tableLog) <= kRankSelectMaxTableLog
            && scheduledDecodeBlockWorkspaceBytes(blockSize) != SIZE_MAX;
}

size_t scheduleStageIndex(uint16_t level, uint8_t op)
{
    return static_cast<size_t>(level) * PIVCO_GPU_SCHEDULE_OP_COUNT + op;
}

__device__ void
setStatus(PivCoGpuStatus* status, PivCoGpuStatusCode code, uint64_t detail)
{
    if (atomicCAS(&status->code, PIVCO_GPU_STATUS_OK, code)
        == PIVCO_GPU_STATUS_OK) {
        status->detail = detail;
    }
}

__device__ size_t dMin(size_t a, size_t b)
{
    return a < b ? a : b;
}

__device__ size_t dBitmapBytes(size_t bits)
{
    return (bits + 7) / 8;
}

__device__ size_t dAlignUpSize(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__device__ size_t dScheduledDirectoryEntries(size_t blockSize)
{
    return ((blockSize * kRankSelectMaxTableLog + 31) / 32)
            + kMaxRankSelectNodes;
}

__device__ size_t dScheduledBlockWorkspaceBytes(size_t blockSize)
{
    const size_t nodeStateBytes = dAlignUpSize(
            sizeof(ScheduledNodeState) * PIVCO_GPU_MAX_TREE_NODES,
            alignof(ScheduledNodeState));
    const size_t directoryBytes =
            dScheduledDirectoryEntries(blockSize) * sizeof(uint16_t);
    const size_t prefixBytes = dAlignUpSize(
            nodeStateBytes + dAlignUpSize(directoryBytes, alignof(uint16_t)),
            8);
    return dAlignUpSize(
            prefixBytes + 2 * (blockSize + kMergeBufferPad),
            alignof(ScheduledNodeState));
}

__device__ size_t dNextPow2(uint64_t upperBound)
{
    size_t bits    = 0;
    uint64_t value = 1;
    while (value < upperBound) {
        value <<= 1;
        ++bits;
    }
    return bits;
}

__device__ void scheduledDecodeWorkspace(
        uint8_t* workspace,
        size_t block,
        size_t blockSize,
        ScheduledNodeState** states,
        uint16_t** directory,
        uint32_t* directoryCapacity,
        uint8_t** bufferA,
        uint8_t** bufferB)
{
    uint8_t* const blockWorkspace =
            workspace + block * dScheduledBlockWorkspaceBytes(blockSize);
    const size_t nodeStateBytes = dAlignUpSize(
            sizeof(ScheduledNodeState) * PIVCO_GPU_MAX_TREE_NODES,
            alignof(ScheduledNodeState));
    const size_t directoryEntries = dScheduledDirectoryEntries(blockSize);
    const size_t directoryBytes   = directoryEntries * sizeof(uint16_t);
    const size_t alignedDirectoryBytes =
            dAlignUpSize(directoryBytes, alignof(uint16_t));
    // 8-byte-align the buffers so the byte merge can store aligned 8-byte
    // groups.
    const size_t prefixBytes =
            dAlignUpSize(nodeStateBytes + alignedDirectoryBytes, 8);

    *states    = reinterpret_cast<ScheduledNodeState*>(blockWorkspace);
    *directory = reinterpret_cast<uint16_t*>(blockWorkspace + nodeStateBytes);
    *directoryCapacity = static_cast<uint32_t>(directoryEntries);
    *bufferA           = blockWorkspace + prefixBytes;
    *bufferB           = *bufferA + (blockSize + kMergeBufferPad);
}

__device__ bool
dRangeIsLeaf(const PivCoGpuTree* tree, size_t firstRank, size_t rankEnd)
{
    if (firstRank >= rankEnd || rankEnd > tree->numRanks) {
        return false;
    }
    return (static_cast<size_t>(1) << tree->rankToFlatDepth[firstRank])
            == rankEnd - firstRank;
}

__device__ size_t dLeafFlatDepth(const PivCoGpuTree* tree, size_t firstRank)
{
    return tree->rankToFlatDepth[firstRank];
}

__device__ bool
dRangeIsConstantLeaf(const PivCoGpuTree* tree, size_t firstRank, size_t rankEnd)
{
    return dRangeIsLeaf(tree, firstRank, rankEnd)
            && dLeafFlatDepth(tree, firstRank) == 0;
}

__device__ size_t dSplitRank(
        const PivCoGpuTree* tree,
        size_t level,
        size_t firstRank,
        size_t rankEnd)
{
    const uint16_t mask = static_cast<uint16_t>(0x8000u >> level);
    size_t splitRank    = firstRank + 1;
    while (splitRank < rankEnd
           && (tree->rankToCodeword[splitRank] & mask) == 0) {
        ++splitRank;
    }
    return splitRank;
}

struct DeviceBitCounter {
    uint64_t bitPos{ 0 };
    bool ok{ true };

    __device__ void reserveAlignedBits(uint64_t bits)
    {
        bitPos = (bitPos + 7u) & ~uint64_t{ 7 };
        if (bits > UINT64_MAX - bitPos) {
            ok = false;
            return;
        }
        bitPos += bits;
    }

    __device__ void writeBits(uint64_t, size_t bits)
    {
        if (bits > UINT64_MAX - bitPos) {
            ok = false;
            return;
        }
        bitPos += bits;
    }

    __device__ uint64_t bytes() const
    {
        return (bitPos + 7u) / 8u;
    }
};

struct DeviceBitWriter {
    uint8_t* data;
    uint64_t capacity;
    uint64_t bitPos{ 0 };
    bool ok{ true };

    __device__ uint8_t* reserveAlignedBits(uint64_t bits)
    {
        bitPos = (bitPos + 7u) & ~uint64_t{ 7 };
        if (bits > UINT64_MAX - bitPos || bitPos + bits > capacity * 8u) {
            ok = false;
            return nullptr;
        }
        uint8_t* out = data + bitPos / 8u;
        bitPos += bits;
        return out;
    }

    __device__ void writeBits(uint64_t value, size_t bits)
    {
        if (bits > UINT64_MAX - bitPos || bitPos + bits > capacity * 8u) {
            ok = false;
            return;
        }
        for (size_t i = 0; i < bits; ++i) {
            if (((value >> i) & 1u) != 0) {
                const uint64_t pos = bitPos + i;
                data[pos / 8u] |= static_cast<uint8_t>(1u << (pos & 7u));
            }
        }
        bitPos += bits;
    }

    __device__ uint64_t bytes() const
    {
        return (bitPos + 7u) / 8u;
    }
};

__device__ size_t partitionRanks(
        uint8_t* bitmap,
        uint8_t* lhs,
        uint8_t* rhs,
        const uint8_t* ranks,
        size_t count,
        uint8_t rightRank)
{
    size_t zeros = 0;
    size_t ones  = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint8_t rank = ranks[i];
        const bool isRight = rank >= rightRank;
        if (bitmap != nullptr && isRight) {
            bitmap[i / 8] |= static_cast<uint8_t>(1u << (i & 7));
        }
        if (isRight) {
            if (rhs != nullptr) {
                rhs[ones] = rank;
            }
            ++ones;
        } else {
            if (lhs != nullptr) {
                lhs[zeros] = rank;
            }
            ++zeros;
        }
    }
    return ones;
}

__device__ bool measureNode(
        const PivCoGpuTree* tree,
        DeviceBitCounter& writer,
        uint8_t* nodeRanks,
        uint8_t* nodeScratch,
        size_t count,
        size_t level,
        size_t firstRank,
        size_t rankEnd)
{
    if (dRangeIsLeaf(tree, firstRank, rankEnd)) {
        const size_t depth = dLeafFlatDepth(tree, firstRank);
        if (depth != 0) {
            writer.reserveAlignedBits(static_cast<uint64_t>(count) * depth);
        }
        return writer.ok;
    }
    if (level >= tree->numLevels) {
        return false;
    }

    const size_t splitRank   = dSplitRank(tree, level, firstRank, rankEnd);
    const bool lhsIsConstant = dRangeIsConstantLeaf(tree, firstRank, splitRank);
    const bool rhsIsConstant = dRangeIsConstantLeaf(tree, splitRank, rankEnd);

    uint8_t* const lhsRanks = nodeScratch;
    uint8_t* const rhsRanks = lhsIsConstant ? nodeScratch : nodeRanks;

    writer.reserveAlignedBits(count);
    const size_t numOnes = partitionRanks(
            nullptr,
            lhsIsConstant ? nullptr : lhsRanks,
            rhsIsConstant ? nullptr : rhsRanks,
            nodeRanks,
            count,
            static_cast<uint8_t>(splitRank));
    if (!(lhsIsConstant && rhsIsConstant)) {
        writer.writeBits(0, dNextPow2(count + 1));
    }

    const size_t numZeros     = count - numOnes;
    uint8_t* const lhsScratch = rhsIsConstant ? nodeRanks : rhsRanks + numOnes;
    uint8_t* const rhsScratch = lhsIsConstant ? nodeRanks : lhsRanks + numZeros;

    if (!lhsIsConstant
        && !measureNode(
                tree,
                writer,
                lhsRanks,
                lhsScratch,
                numZeros,
                level + 1,
                firstRank,
                splitRank)) {
        return false;
    }
    if (!rhsIsConstant
        && !measureNode(
                tree,
                writer,
                rhsRanks,
                rhsScratch,
                numOnes,
                level + 1,
                splitRank,
                rankEnd)) {
        return false;
    }
    return writer.ok;
}

__device__ void packFlatDepth(
        uint8_t* bitmap,
        size_t depth,
        const uint8_t* ranks,
        size_t count,
        uint8_t firstRank)
{
    for (size_t i = 0; i < count; ++i) {
        const uint8_t index  = static_cast<uint8_t>(ranks[i] - firstRank);
        const size_t bitBase = i * depth;
        for (size_t bit = 0; bit < depth; ++bit) {
            if (((index >> bit) & 1u) != 0) {
                const size_t pos = bitBase + bit;
                bitmap[pos / 8] |= static_cast<uint8_t>(1u << (pos & 7));
            }
        }
    }
}

__device__ bool emitNode(
        const PivCoGpuTree* tree,
        DeviceBitWriter& writer,
        uint8_t* nodeRanks,
        uint8_t* nodeScratch,
        size_t count,
        size_t level,
        size_t firstRank,
        size_t rankEnd)
{
    if (dRangeIsLeaf(tree, firstRank, rankEnd)) {
        const size_t depth = dLeafFlatDepth(tree, firstRank);
        if (depth != 0) {
            uint8_t* const bitmap = writer.reserveAlignedBits(
                    static_cast<uint64_t>(count) * depth);
            if (bitmap == nullptr && count * depth != 0) {
                return false;
            }
            packFlatDepth(
                    bitmap,
                    depth,
                    nodeRanks,
                    count,
                    static_cast<uint8_t>(firstRank));
        }
        return writer.ok;
    }
    if (level >= tree->numLevels) {
        return false;
    }

    const size_t splitRank   = dSplitRank(tree, level, firstRank, rankEnd);
    const bool lhsIsConstant = dRangeIsConstantLeaf(tree, firstRank, splitRank);
    const bool rhsIsConstant = dRangeIsConstantLeaf(tree, splitRank, rankEnd);

    uint8_t* const lhsRanks = nodeScratch;
    uint8_t* const rhsRanks = lhsIsConstant ? nodeScratch : nodeRanks;

    uint8_t* const bitmap = writer.reserveAlignedBits(count);
    if (bitmap == nullptr && count != 0) {
        return false;
    }
    const size_t numOnes = partitionRanks(
            bitmap,
            lhsIsConstant ? nullptr : lhsRanks,
            rhsIsConstant ? nullptr : rhsRanks,
            nodeRanks,
            count,
            static_cast<uint8_t>(splitRank));
    if (!(lhsIsConstant && rhsIsConstant)) {
        writer.writeBits(numOnes, dNextPow2(count + 1));
    }

    const size_t numZeros     = count - numOnes;
    uint8_t* const lhsScratch = rhsIsConstant ? nodeRanks : rhsRanks + numOnes;
    uint8_t* const rhsScratch = lhsIsConstant ? nodeRanks : lhsRanks + numZeros;

    if (!lhsIsConstant
        && !emitNode(
                tree,
                writer,
                lhsRanks,
                lhsScratch,
                numZeros,
                level + 1,
                firstRank,
                splitRank)) {
        return false;
    }
    if (!rhsIsConstant
        && !emitNode(
                tree,
                writer,
                rhsRanks,
                rhsScratch,
                numOnes,
                level + 1,
                splitRank,
                rankEnd)) {
        return false;
    }
    return writer.ok;
}

struct DeviceBitReader {
    const uint8_t* data;
    uint64_t size;
    uint64_t bitPos{ 0 };

    __device__ void byteAlign()
    {
        bitPos = (bitPos + 7u) & ~uint64_t{ 7 };
    }

    __device__ bool
    popAlignedBits(uint64_t bits, const uint8_t** out, uint64_t* outBytes)
    {
        byteAlign();
        if (bits > size * 8u || bitPos > size * 8u - bits) {
            return false;
        }
        *out      = data + bitPos / 8u;
        *outBytes = (bits + 7u) / 8u;
        bitPos += bits;
        return true;
    }

    __device__ bool readBits(size_t bits, size_t* value)
    {
        if (bits == 0) {
            *value = 0;
            return true;
        }
        if (bits >= sizeof(size_t) * 8 || bits > size * 8u
            || bitPos > size * 8u - bits) {
            return false;
        }
        // Read the byte span holding the field into a chunk and shift/mask,
        // instead of a bit-by-bit loop (this runs on the serial 1-thread
        // parse).
        const uint64_t bytePos = bitPos >> 3u;
        const uint32_t bitOff  = static_cast<uint32_t>(bitPos & 7u);
        const uint32_t nbytes =
                static_cast<uint32_t>((bitOff + bits + 7u) >> 3u);
        uint64_t chunk = 0;
        for (uint32_t i = 0; i < nbytes; ++i) {
            chunk |= static_cast<uint64_t>(data[bytePos + i]) << (i * 8u);
        }
        *value = static_cast<size_t>(
                (chunk >> bitOff) & ((uint64_t{ 1 } << bits) - 1u));
        bitPos += bits;
        return true;
    }

    __device__ uint64_t consumedBytes() const
    {
        return (bitPos + 7u) / 8u;
    }
};

__device__ uint8_t dGetBit(const uint8_t* bitmap, uint64_t bitBase)
{
    return static_cast<uint8_t>((bitmap[bitBase / 8] >> (bitBase & 7)) & 1u);
}

__device__ uint32_t
dLoadLe32Masked(const uint8_t* data, size_t bitBase, size_t count)
{
    uint32_t value        = 0;
    const size_t byteBase = bitBase / 8;
    for (size_t i = 0; i < 4 && byteBase + i < dBitmapBytes(count); ++i) {
        value |= static_cast<uint32_t>(data[byteBase + i]) << (i * 8);
    }

    const size_t remainingBits = count > bitBase ? count - bitBase : 0;
    if (remainingBits < 32) {
        const uint32_t mask =
                remainingBits == 0 ? 0 : ((uint64_t{ 1 } << remainingBits) - 1);
        value &= mask;
    }
    return value;
}

__device__ uint32_t dRankSelectOnesBefore(
        const uint8_t* bitmap,
        const uint16_t* directory,
        uint32_t dirBase,
        uint32_t bitPos)
{
    const uint32_t wordIndex = bitPos / 32u;
    const uint32_t bit       = bitPos & 31u;
    uint32_t count           = directory[dirBase + wordIndex];
    if (bit != 0) {
        const uint32_t word = dLoadLe32Masked(bitmap, wordIndex * 32u, bitPos);
        count += static_cast<uint32_t>(__popc(word & ((1u << bit) - 1u)));
    }
    return count;
}

// Branchless/loop-free rank used inside the chunk decoder's hot descent: reads
// the enclosing 32-bit word directly (over-reading a few bytes into the
// bitstream's trailing slop, all masked off) instead of dLoadLe32Masked's
// bounded byte loop.
__device__ __forceinline__ uint32_t dRankOnesBeforeFast(
        const uint8_t* bitmap,
        const uint16_t* directory,
        uint32_t dirBase,
        uint32_t bitPos)
{
    const uint32_t wordIndex = bitPos >> 5u;
    const uint32_t bit       = bitPos & 31u;
    // A full block's terminal rank can reach 65536, beyond uint16. Reconstruct
    // it from the preceding prefix (at most 65504) and final word.
    if (bitPos == kRankSelectMaxBlockSize) {
        const uint32_t previous = wordIndex - 1u;
        const uint32_t b        = previous << 2u;
        const uint32_t word     = static_cast<uint32_t>(bitmap[b])
                | (static_cast<uint32_t>(bitmap[b + 1u]) << 8u)
                | (static_cast<uint32_t>(bitmap[b + 2u]) << 16u)
                | (static_cast<uint32_t>(bitmap[b + 3u]) << 24u);
        return uint32_t(directory[dirBase + previous])
                + static_cast<uint32_t>(__popc(word));
    }
    uint32_t count = directory[dirBase + wordIndex];
    if (bit != 0u) {
        const uint32_t b = wordIndex << 2u;
        const uint32_t w = static_cast<uint32_t>(bitmap[b])
                | (static_cast<uint32_t>(bitmap[b + 1u]) << 8u)
                | (static_cast<uint32_t>(bitmap[b + 2u]) << 16u)
                | (static_cast<uint32_t>(bitmap[b + 3u]) << 24u);
        count += static_cast<uint32_t>(__popc(w & ((1u << bit) - 1u)));
    }
    return count;
}

// Rank/select for a byte-aligned bit position (the merge kernels always query
// at a byte boundary). The sub-word remainder is then 0-3 whole bytes, so the
// partial popcount reads exactly those bytes -- no bit masking and, unlike a
// 32-bit load, no read past the bitmap's last byte.
__device__ __forceinline__ uint32_t dRankSelectOnesBeforeAligned(
        const uint8_t* bitmap,
        const uint16_t* directory,
        uint32_t dirBase,
        uint32_t bitPos)
{
    const uint32_t bytePos   = bitPos >> 3u;
    const uint32_t wordIndex = bytePos >> 2u;
    const uint32_t base      = wordIndex << 2u;
    const uint32_t rem       = bytePos & 3u;
    uint32_t count           = directory[dirBase + wordIndex];
    if (rem > 0u) {
        count += static_cast<uint32_t>(__popc(bitmap[base]));
    }
    if (rem > 1u) {
        count += static_cast<uint32_t>(__popc(bitmap[base + 1u]));
    }
    if (rem > 2u) {
        count += static_cast<uint32_t>(__popc(bitmap[base + 2u]));
    }
    return count;
}

template <bool ComputeOnes = true>
__device__ bool buildRankSelectDirectoryCooperative(
        const uint8_t* bitmap,
        uint32_t count,
        uint16_t* directory,
        uint32_t directoryCapacity,
        uint32_t dirBase,
        uint16_t* prefixA,
        uint16_t* prefixB,
        uint32_t* ones)
{
    const uint32_t words = (count + 31u) / 32u;
    if (dirBase > directoryCapacity || words + 1u > directoryCapacity - dirBase
        || words > kRankSelectMaxBitmapWords) {
        return false;
    }

    // Only prefixA[0] needs initializing: the popcount loop below writes every
    // prefixA[1..words], and nothing reads prefixA before the scan (after the
    // next barrier), so the old full-array zeroing pass + its barrier were
    // redundant.
    if (threadIdx.x == 0) {
        prefixA[0] = 0;
    }

    // Two words per thread with both loads issued before either popcount, so
    // the (latency-bound) bitmap loads overlap.
    for (uint32_t word = threadIdx.x; word < words; word += blockDim.x * 2u) {
        const uint32_t w1    = word + blockDim.x;
        const uint32_t load0 = dLoadLe32Masked(bitmap, word * 32u, count);
        const uint32_t load1 =
                w1 < words ? dLoadLe32Masked(bitmap, w1 * 32u, count) : 0u;
        prefixA[word + 1] = static_cast<uint16_t>(__popc(load0));
        if (w1 < words) {
            prefixA[w1 + 1] = static_cast<uint16_t>(__popc(load1));
        }
    }
    __syncthreads();

    // Work-efficient inclusive scan of prefixA[0..words] into the directory:
    // each thread sums a contiguous tile, an exclusive scan of the per-thread
    // tile sums (warp-scan + one warp of warp-totals) gives each tile's base,
    // then each thread writes its tile's running prefix. O(entries) work with
    // two barriers, versus the O(entries log entries) Hillis-Steele it
    // replaces.
    (void)prefixB;
    const uint32_t entries = words + 1u;

    // Small node: a single warp scans all entries with shuffles and no block
    // barriers (the caller reads *ones only on thread 0). This avoids the
    // two-barrier fixed cost of the block scan for the many tiny nodes.
    if (entries <= 32u) {
        if (threadIdx.x < 32u) {
            const uint32_t lane = threadIdx.x;
            uint32_t v          = lane < entries ? prefixA[lane] : 0u;
#pragma unroll
            for (uint32_t o = 1; o < 32u; o <<= 1u) {
                const uint32_t n = __shfl_up_sync(0xFFFFFFFFu, v, o);
                if (lane >= o) {
                    v += n;
                }
            }
            if (lane < entries) {
                directory[dirBase + lane] = static_cast<uint16_t>(v);
            }
            if constexpr (ComputeOnes) {
                const uint32_t total = __shfl_sync(0xFFFFFFFFu, v, words);
                if (lane == 0u) {
                    *ones = total;
                }
            }
        }
        return true;
    }

    const uint32_t tile  = (entries + blockDim.x - 1u) / blockDim.x;
    const uint32_t start = threadIdx.x * tile;
    const uint32_t stop  = start + tile < entries ? start + tile : entries;

    uint32_t tileSum = 0;
    for (uint32_t i = start; i < stop; ++i) {
        tileSum += prefixA[i];
    }

    __shared__ uint32_t s_warpTotals[32];
    const uint32_t lane     = threadIdx.x & 31u;
    const uint32_t warp     = threadIdx.x >> 5u;
    const uint32_t numWarps = blockDim.x >> 5u;
    uint32_t incl           = tileSum;
#pragma unroll
    for (uint32_t o = 1; o < 32u; o <<= 1u) {
        const uint32_t n = __shfl_up_sync(0xFFFFFFFFu, incl, o);
        if (lane >= o) {
            incl += n;
        }
    }
    if (lane == 31u) {
        s_warpTotals[warp] = incl;
    }
    __syncthreads();
    if (warp == 0u) {
        uint32_t w = lane < numWarps ? s_warpTotals[lane] : 0u;
#pragma unroll
        for (uint32_t o = 1; o < 32u; o <<= 1u) {
            const uint32_t n = __shfl_up_sync(0xFFFFFFFFu, w, o);
            if (lane >= o) {
                w += n;
            }
        }
        if (lane < numWarps) {
            s_warpTotals[lane] = w; // inclusive warp totals
        }
    }
    __syncthreads();

    const uint32_t warpBase = warp == 0u ? 0u : s_warpTotals[warp - 1u];
    uint32_t running        = warpBase + incl - tileSum; // exclusive tile base
    for (uint32_t i = start; i < stop; ++i) {
        running += prefixA[i];
        directory[dirBase + i] = static_cast<uint16_t>(running);
    }
    // The caller (buildSharedDirectory / directory kernel) issues the barrier
    // that publishes the directory to readers. `*ones` (a corruption check) is
    // only needed by the top-down path, so its extra read + two barriers are
    // skipped when ComputeOnes is false (the fused merge never reads it).
    if constexpr (ComputeOnes) {
        __syncthreads();
        if (threadIdx.x == 0) {
            // The terminal directory entry may wrap at 65536 ones. The
            // inclusive warp totals retain the full 32-bit aggregate.
            *ones = s_warpTotals[numWarps - 1u];
        }
        __syncthreads();
    }
    return true;
}

// Builds a node's rank directory directly into shared memory at the top of a
// merge kernel (fusing the former standalone directory kernel): keeps the
// directory off global memory, lets the merge's rank reads hit shared, and
// reuses the bitmap that is already resident from this build for the merge's
// own bitmap reads. All threads of the block must call this; it ends with a
// barrier so the merge can read `s_dir` immediately.
__device__ __forceinline__ void buildSharedDirectory(
        const uint8_t* bitmap,
        uint32_t count,
        uint16_t* s_dir,
        uint16_t* s_prefix,
        uint32_t* s_ones)
{
    buildRankSelectDirectoryCooperative<false>(
            bitmap,
            count,
            s_dir,
            kRankSelectMaxBitmapWords + 1u,
            0u,
            s_prefix,
            s_prefix,
            s_ones);
    __syncthreads();
}

__device__ void
dOrBits(uint8_t* bitmap, uint64_t bitBase, uint64_t value, size_t bits)
{
    for (size_t i = 0; i < bits; ++i) {
        if (((value >> i) & 1u) != 0) {
            const uint64_t pos = bitBase + i;
            bitmap[pos / 8] |= static_cast<uint8_t>(1u << (pos & 7));
        }
    }
}

__device__ void dAtomicOrSharedBits(
        uint32_t* words,
        uint32_t bitBase,
        uint32_t value,
        uint32_t bits)
{
    if (bits == 0) {
        return;
    }
    const uint32_t wordIndex = bitBase / 32u;
    const uint32_t shift     = bitBase & 31u;
    atomicOr(&words[wordIndex], value << shift);
    if (shift != 0 && shift + bits > 32u) {
        atomicOr(&words[wordIndex + 1], value >> (32u - shift));
    }
}

__device__ uint8_t dEqMask4(uint32_t word, uint8_t value)
{
    const uint32_t repeated = static_cast<uint32_t>(value) * 0x01010101u;
    const uint32_t eq       = __vcmpeq4(word, repeated) & 0x01010101u;
    return static_cast<uint8_t>(
            (eq & 1u) | ((eq >> 7) & 2u) | ((eq >> 14) & 4u)
            | ((eq >> 21) & 8u));
}

__device__ uint8_t dEqMask8(const uint8_t* ptr, uint8_t value)
{
    const auto* const words = reinterpret_cast<const uint32_t*>(ptr);
    return static_cast<uint8_t>(
            dEqMask4(words[0], value) | (dEqMask4(words[1], value) << 4));
}

__device__ size_t dFastCountBits(size_t blockLen)
{
    return dNextPow2(blockLen + 1);
}

__device__ uint64_t dFastLeafBitBase(size_t blockLen)
{
    return (static_cast<uint64_t>(blockLen) + dFastCountBits(blockLen) + 7u)
            & ~uint64_t{ 7 };
}

__device__ uint64_t dFastEncodedBytes(size_t blockLen, uint32_t numOnes)
{
    return (dFastLeafBitBase(blockLen) + numOnes + 7u) / 8u;
}

__device__ uint8_t dRootByteMask(size_t byteIndex, size_t blockLen)
{
    const size_t remainingBits = blockLen - byteIndex * 8;
    if (remainingBits >= 8) {
        return 0xFF;
    }
    return static_cast<uint8_t>((1u << remainingBits) - 1u);
}

__device__ void fastScanRootPrefix(
        uint32_t* prefix,
        const uint8_t* rootBytes,
        size_t rootBytesCount,
        size_t blockLen)
{
    for (size_t i = threadIdx.x; i < rootBytesCount; i += blockDim.x) {
        const uint8_t rootByte = rootBytes[i] & dRootByteMask(i, blockLen);
        prefix[i + 1]          = static_cast<uint32_t>(__popc(rootByte));
    }
    if (threadIdx.x == 0) {
        prefix[0] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t running = 0;
        for (size_t i = 0; i < rootBytesCount; ++i) {
            const uint32_t count = prefix[i + 1];
            prefix[i]            = running;
            running += count;
        }
        prefix[rootBytesCount] = running;
    }
    __syncthreads();
}

__global__ void encodeLayoutKernel(
        const PivCoGpuTree* tree,
        const uint8_t* src,
        size_t srcSize,
        size_t blockSize,
        uint8_t* workspace,
        uint64_t* offsets,
        PivCoGpuStatus* status)
{
    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= srcSize) {
        return;
    }
    const size_t blockLen  = dMin(blockSize, srcSize - blockOff);
    uint8_t* const ranks   = workspace + block * 2 * blockSize;
    uint8_t* const scratch = ranks + blockSize;

    for (size_t i = 0; i < blockLen; ++i) {
        const uint8_t symbol = src[blockOff + i];
        if (tree->symbolPresent[symbol] == 0) {
            setStatus(status, PIVCO_GPU_STATUS_MISSING_SYMBOL, symbol);
            offsets[block + 1] = 0;
            return;
        }
        ranks[i] = tree->symbolToRank[symbol];
    }

    DeviceBitCounter writer;
    if (!measureNode(
                tree, writer, ranks, scratch, blockLen, 0, 0, tree->numRanks)) {
        setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
        offsets[block + 1] = 0;
        return;
    }
    offsets[block + 1] = writer.bytes();
}

__global__ void scanOffsetsKernel(
        uint64_t* offsets,
        size_t numBlocks,
        uint64_t* totalSize,
        PivCoGpuStatus* status)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    uint64_t running = 0;
    for (size_t i = 0; i < numBlocks; ++i) {
        const uint64_t blockSize = offsets[i + 1];
        offsets[i]               = running;
        if (UINT64_MAX - running < blockSize) {
            setStatus(status, PIVCO_GPU_STATUS_CAPACITY, i);
            *totalSize = running;
            return;
        }
        running += blockSize;
    }
    offsets[numBlocks] = running;
    *totalSize         = running;
}

__global__ void
scanBlockSizesKernel(uint64_t* offsets, size_t numBlocks, uint64_t* chunkSums)
{
    __shared__ uint64_t values[kScanItemsPerBlock];

    const size_t index =
            blockIdx.x * static_cast<size_t>(kScanItemsPerBlock) + threadIdx.x;
    const uint64_t value = index < numBlocks ? offsets[index + 1] : 0;
    values[threadIdx.x]  = value;
    __syncthreads();

    for (int step = 1; step < kScanItemsPerBlock; step <<= 1) {
        const uint64_t addend =
                threadIdx.x >= step ? values[threadIdx.x - step] : 0;
        __syncthreads();
        values[threadIdx.x] += addend;
        __syncthreads();
    }

    if (index < numBlocks) {
        offsets[index] = values[threadIdx.x] - value;
    }
    if (threadIdx.x == kScanItemsPerBlock - 1) {
        chunkSums[blockIdx.x] = values[threadIdx.x];
    }
}

__global__ void scanChunkSumsKernel(
        const uint64_t* chunkSums,
        uint64_t* chunkOffsets,
        size_t numChunks,
        uint64_t* totalSize)
{
    __shared__ uint64_t values[kScanItemsPerBlock];

    const size_t index   = threadIdx.x;
    const uint64_t value = index < numChunks ? chunkSums[index] : 0;
    values[threadIdx.x]  = value;
    __syncthreads();

    for (int step = 1; step < kScanItemsPerBlock; step <<= 1) {
        const uint64_t addend =
                threadIdx.x >= step ? values[threadIdx.x - step] : 0;
        __syncthreads();
        values[threadIdx.x] += addend;
        __syncthreads();
    }

    if (index < numChunks) {
        chunkOffsets[index] = values[threadIdx.x] - value;
    }
    if (threadIdx.x == kScanItemsPerBlock - 1) {
        *totalSize = values[threadIdx.x];
    }
}

__global__ void addChunkOffsetsKernel(
        uint64_t* offsets,
        size_t numBlocks,
        const uint64_t* chunkOffsets,
        const uint64_t* totalSize)
{
    const size_t index =
            blockIdx.x * static_cast<size_t>(kScanItemsPerBlock) + threadIdx.x;
    if (index < numBlocks) {
        offsets[index] += chunkOffsets[blockIdx.x];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        offsets[numBlocks] = *totalSize;
    }
}

__global__ void encodeEmitKernel(
        const PivCoGpuTree* tree,
        uint8_t* dst,
        size_t dstCapacity,
        const uint8_t* src,
        size_t srcSize,
        size_t blockSize,
        uint8_t* workspace,
        const uint64_t* offsets,
        PivCoGpuStatus* status)
{
    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= srcSize) {
        return;
    }

    const uint64_t outBegin = offsets[block];
    const uint64_t outEnd   = offsets[block + 1];
    if (outBegin > outEnd || outEnd > dstCapacity) {
        setStatus(status, PIVCO_GPU_STATUS_CAPACITY, block);
        return;
    }

    uint8_t* const out     = dst + outBegin;
    const uint64_t outSize = outEnd - outBegin;
    for (uint64_t i = 0; i < outSize; ++i) {
        out[i] = 0;
    }

    const size_t blockLen  = dMin(blockSize, srcSize - blockOff);
    uint8_t* const ranks   = workspace + block * 2 * blockSize;
    uint8_t* const scratch = ranks + blockSize;
    for (size_t i = 0; i < blockLen; ++i) {
        ranks[i] = tree->symbolToRank[src[blockOff + i]];
    }

    DeviceBitWriter writer;
    writer.data     = out;
    writer.capacity = outSize;
    if (!emitNode(tree, writer, ranks, scratch, blockLen, 0, 0, tree->numRanks)
        || writer.bytes() != outSize) {
        setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
    }
}

__device__ uint8_t* scheduledNodeOutput(
        const PivCoGpuScheduleNode& node,
        const ScheduledNodeState& state,
        uint32_t maxLevel,
        uint8_t* dst,
        size_t blockOff,
        uint8_t* bufferA,
        uint8_t* bufferB)
{
    if (node.level == 0) {
        return dst + blockOff;
    }
    const bool writeToA = ((maxLevel - node.level) & 1u) == 0;
    return (writeToA ? bufferA : bufferB) + state.streamBase;
}

__device__ const uint8_t* scheduledChildOutput(
        const PivCoGpuScheduleNode& parent,
        const ScheduledNodeState& childState,
        uint32_t maxLevel,
        const uint8_t* bufferA,
        const uint8_t* bufferB)
{
    const bool parentWritesToA = ((maxLevel - parent.level) & 1u) == 0;
    return (parentWritesToA ? bufferB : bufferA) + childState.streamBase;
}

__global__ void scheduledParseKernel(
        const PivCoGpuDecodeSchedule* schedule,
        const uint8_t* bitstream,
        size_t bitstreamSize,
        const uint64_t* offsets,
        size_t dstSize,
        size_t blockSize,
        uint8_t* workspace,
        PivCoGpuStatus* status)
{
    if (threadIdx.x != 0) {
        return;
    }
    if (status->code != PIVCO_GPU_STATUS_OK) {
        return;
    }

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }

    const uint64_t sliceBegin = offsets[block];
    const uint64_t sliceEnd   = offsets[block + 1];
    if (sliceBegin > sliceEnd || sliceEnd > bitstreamSize) {
        setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
        return;
    }

    ScheduledNodeState* states = nullptr;
    uint16_t* directory        = nullptr;
    uint32_t dirCapacity       = 0;
    uint8_t* bufferA           = nullptr;
    uint8_t* bufferB           = nullptr;
    scheduledDecodeWorkspace(
            workspace,
            block,
            blockSize,
            &states,
            &directory,
            &dirCapacity,
            &bufferA,
            &bufferB);
    (void)bufferA;
    (void)bufferB;

    const size_t blockLen = dMin(blockSize, dstSize - blockOff);
    if (blockLen > UINT32_MAX) {
        setStatus(status, PIVCO_GPU_STATUS_CAPACITY, block);
        return;
    }
    // Only `count` is read before being written (root set here, every other
    // node by its parent), so the state array does not need zeroing; unread
    // fields of constant/CC nodes are simply never touched.
    states[0].count = static_cast<uint32_t>(blockLen);

    const uint8_t* const slice = bitstream + sliceBegin;
    const uint64_t sliceBytes  = sliceEnd - sliceBegin;
    DeviceBitReader reader{ slice, sliceBytes, 0 };
    uint32_t dirCursor = 0;
    // Per-level stream cursors, so each materializing node's output offset is
    // assigned in this single pass (8-byte aligned; `levelData` tracks the true
    // byte total for the integrity check) instead of a second O(levels*nodes)
    // pass over the schedule.
    uint32_t levelCursor[PIVCO_GPU_MAX_LEVELS] = {};
    uint32_t levelData[PIVCO_GPU_MAX_LEVELS]   = {};

    for (uint32_t nodeIndex = 0; nodeIndex < schedule->nodeCount; ++nodeIndex) {
        const PivCoGpuScheduleNode node = schedule->nodes[nodeIndex];
        ScheduledNodeState& state       = states[nodeIndex];
        const uint32_t count            = state.count;

        if (node.kind == PIVCO_GPU_SCHEDULE_CONSTANT) {
            continue;
        }

        const uint32_t level = node.level;
        state.streamBase     = levelCursor[level];
        levelData[level] += count;
        if (levelData[level] > blockLen) {
            setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
            return;
        }
        levelCursor[level] += (count + 7u) & ~uint32_t{ 7 };

        if (node.kind == PIVCO_GPU_SCHEDULE_FLAT) {
            const uint8_t* bitmap = nullptr;
            uint64_t bitmapBytes  = 0;
            if (!reader.popAlignedBits(
                        static_cast<uint64_t>(count) * node.flatDepth,
                        &bitmap,
                        &bitmapBytes)) {
                setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
                return;
            }
            (void)bitmapBytes;
            state.leafBitBase = static_cast<uint32_t>(
                    static_cast<uint64_t>(bitmap - slice) * 8u);
            continue;
        }

        const uint8_t* bitmap = nullptr;
        uint64_t bitmapBytes  = 0;
        if (!reader.popAlignedBits(count, &bitmap, &bitmapBytes)) {
            setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
            return;
        }
        (void)bitmapBytes;
        state.bitmapByteBase = static_cast<uint32_t>(bitmap - slice);

        if (node.op != PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_CONSTANT) {
            const uint32_t words = (count + 31u) / 32u;
            if (dirCursor > dirCapacity
                || words + 1u > dirCapacity - dirCursor) {
                setStatus(status, PIVCO_GPU_STATUS_CAPACITY, block);
                return;
            }
            state.dirBase = dirCursor;
            dirCursor += words + 1u;

            size_t storedNumOnes = 0;
            if (!reader.readBits(dNextPow2(count + 1u), &storedNumOnes)
                || storedNumOnes > count) {
                setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
                return;
            }
            const uint32_t numOnes       = static_cast<uint32_t>(storedNumOnes);
            states[node.leftChild].count = count - numOnes;
            states[node.rightChild].count = numOnes;
        }
    }

    if (reader.consumedBytes() != sliceBytes) {
        setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
        return;
    }
}

__global__ void __launch_bounds__(256, 8) scheduledDirectoryKernel(
        const PivCoGpuDecodeSchedule* schedule,
        const uint8_t* bitstream,
        const uint64_t* offsets,
        size_t dstSize,
        size_t blockSize,
        uint8_t* workspace,
        PivCoGpuStatus* status)
{
    __shared__ uint16_t prefixA[kRankSelectMaxBitmapWords + 1];
    __shared__ uint16_t prefixB[kRankSelectMaxBitmapWords + 1];
    __shared__ uint32_t numOnes;

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }

    ScheduledNodeState* states = nullptr;
    uint16_t* directory        = nullptr;
    uint32_t dirCapacity       = 0;
    uint8_t* bufferA           = nullptr;
    uint8_t* bufferB           = nullptr;
    scheduledDecodeWorkspace(
            workspace,
            block,
            blockSize,
            &states,
            &directory,
            &dirCapacity,
            &bufferA,
            &bufferB);
    (void)bufferA;
    (void)bufferB;

    const uint16_t nodeIndex        = schedule->internalItems[blockIdx.y].node;
    const PivCoGpuScheduleNode node = schedule->nodes[nodeIndex];
    if (node.op == PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_CONSTANT) {
        return;
    }

    const ScheduledNodeState state = states[nodeIndex];
    const uint8_t* const bitmap =
            bitstream + offsets[block] + state.bitmapByteBase;
    if (!buildRankSelectDirectoryCooperative(
                bitmap,
                state.count,
                directory,
                dirCapacity,
                state.dirBase,
                prefixA,
                prefixB,
                &numOnes)) {
        if (threadIdx.x == 0) {
            setStatus(status, PIVCO_GPU_STATUS_CAPACITY, block);
        }
        return;
    }

    if (threadIdx.x == 0 && numOnes != states[node.rightChild].count) {
        setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
    }
}

__device__ __forceinline__ uint64_t
dLoad8Bounded(const uint8_t* p, uint32_t cursor, uint32_t total);

template <int Depth>
__global__ void scheduledFlatKernel(
        const PivCoGpuTree* tree,
        const PivCoGpuDecodeSchedule* schedule,
        uint8_t* dst,
        size_t dstSize,
        const uint8_t* bitstream,
        const uint64_t* offsets,
        size_t blockSize,
        uint8_t* workspace,
        PivCoGpuStatus* status,
        uint16_t stageOffset)
{
    if (status->code != PIVCO_GPU_STATUS_OK) {
        return;
    }

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }

    ScheduledNodeState* states = nullptr;
    uint16_t* directory        = nullptr;
    uint32_t dirCapacity       = 0;
    uint8_t* bufferA           = nullptr;
    uint8_t* bufferB           = nullptr;
    scheduledDecodeWorkspace(
            workspace,
            block,
            blockSize,
            &states,
            &directory,
            &dirCapacity,
            &bufferA,
            &bufferB);
    (void)directory;
    (void)dirCapacity;

    const uint16_t nodeIndex =
            schedule->stageItems[stageOffset + blockIdx.y].node;
    const PivCoGpuScheduleNode node = schedule->nodes[nodeIndex];
    const ScheduledNodeState state  = states[nodeIndex];
    uint8_t* const out              = scheduledNodeOutput(
            node, state, schedule->maxLevel, dst, blockOff, bufferA, bufferB);
    const uint8_t* const bitmap =
            bitstream + offsets[block] + state.leafBitBase / 8u;

    // Cache the leaf's 2^Depth symbols in shared memory once, so the per-output
    // table lookup hits shared instead of a dependent global load (this kernel
    // is latency bound on that lookup).
    constexpr uint32_t kNumSyms = 1u << Depth;
    __shared__ uint8_t s_syms[kNumSyms];
    for (uint32_t i = threadIdx.x; i < kNumSyms; i += blockDim.x) {
        s_syms[i] = tree->rankToSymbol[node.firstRank + i];
    }
    __syncthreads();

    // Eight outputs per thread. A group of 8 symbols starts at output byte
    // `g*8`, i.e. bit `g*8*Depth` = byte `g*Depth` (always byte-aligned), and
    // spans exactly `Depth` contiguous bitmap bytes. Load those bytes once,
    // unpack the eight `Depth`-bit indices from the register, look each up in
    // the shared symbol table, and write one coalesced 8-byte group -- instead
    // of one dependent 1-2 byte load + byte store per output (which re-reads
    // shared bytes across neighbouring threads for small depths). `out` is
    // 8-byte aligned with trailing slop, matching the merge kernels' store
    // contract.
    constexpr uint32_t kMask = (1u << Depth) - 1u;
    const uint32_t numGroups = (state.count + 7u) / 8u;
    const uint32_t bitmapBytes =
            (state.count * static_cast<uint32_t>(Depth) + 7u) / 8u;
    // Depth-2 memory-level parallelism on the packed-index load (this kernel is
    // latency bound on that dependent load): fetch two groups' packed words
    // before unpacking either, so one load hides the other's latency.
    for (uint32_t gg = threadIdx.x; gg < numGroups; gg += blockDim.x * 2u) {
        const uint32_t g1 = gg + blockDim.x;
        const bool has1   = g1 < numGroups;
        const uint64_t p0 = dLoad8Bounded(
                bitmap, gg * static_cast<uint32_t>(Depth), bitmapBytes);
        const uint64_t p1 = has1 ? dLoad8Bounded(
                                           bitmap,
                                           g1 * static_cast<uint32_t>(Depth),
                                           bitmapBytes)
                                 : 0u;
#pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t g      = half == 0u ? gg : g1;
            const uint64_t packed = half == 0u ? p0 : p1;
            if (half == 1u && !has1) {
                break;
            }
            uint64_t outWord = 0;
#pragma unroll
            for (uint32_t k = 0; k < 8u; ++k) {
                const uint32_t index =
                        static_cast<uint32_t>(
                                packed >> (k * static_cast<uint32_t>(Depth)))
                        & kMask;
                outWord |= static_cast<uint64_t>(s_syms[index]) << (k * 8u);
            }
            *reinterpret_cast<uint64_t*>(out + g * 8u) = outWord;
        }
    }
}

__global__ void scheduledMergeConstantConstantKernel(
        const PivCoGpuDecodeSchedule* schedule,
        uint8_t* dst,
        size_t dstSize,
        const uint8_t* bitstream,
        const uint64_t* offsets,
        size_t blockSize,
        uint8_t* workspace,
        PivCoGpuStatus* status,
        uint16_t stageOffset)
{
    if (status->code != PIVCO_GPU_STATUS_OK) {
        return;
    }

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }

    ScheduledNodeState* states = nullptr;
    uint16_t* directory        = nullptr;
    uint32_t dirCapacity       = 0;
    uint8_t* bufferA           = nullptr;
    uint8_t* bufferB           = nullptr;
    scheduledDecodeWorkspace(
            workspace,
            block,
            blockSize,
            &states,
            &directory,
            &dirCapacity,
            &bufferA,
            &bufferB);
    (void)directory;
    (void)dirCapacity;

    const uint16_t nodeIndex =
            schedule->stageItems[stageOffset + blockIdx.y].node;
    const PivCoGpuScheduleNode node = schedule->nodes[nodeIndex];
    const ScheduledNodeState state  = states[nodeIndex];
    uint8_t* const out              = scheduledNodeOutput(
            node, state, schedule->maxLevel, dst, blockOff, bufferA, bufferB);
    const uint8_t* const bitmap =
            bitstream + offsets[block] + state.bitmapByteBase;

    // Eight outputs per thread: load one bitmap byte, select left/right per bit
    // from a 2-entry register table, and write one coalesced 8-byte group
    // (instead of one dependent bit read + byte store per output, which
    // re-reads the same bitmap byte across neighbouring threads). `out` is
    // 8-byte aligned with trailing slop, matching the merge kernels' store
    // contract.
    const uint32_t regSyms =
            node.leftSymbol | (static_cast<uint32_t>(node.rightSymbol) << 8u);
    const uint32_t numGroups   = (state.count + 7u) / 8u;
    const uint32_t bitmapBytes = numGroups;
    for (uint32_t g = threadIdx.x; g < numGroups; g += blockDim.x) {
        const uint32_t bits = g < bitmapBytes ? bitmap[g] : 0u;
        uint64_t outWord    = 0;
#pragma unroll
        for (uint32_t k = 0; k < 8u; ++k) {
            const uint32_t sym = (regSyms >> (((bits >> k) & 1u) * 8u)) & 0xFFu;
            outWord |= static_cast<uint64_t>(sym) << (k * 8u);
        }
        *reinterpret_cast<uint64_t*>(out + g * 8u) = outWord;
    }
}

// Reads 8 bytes starting at p[cursor] into a little-endian u64 (p[cursor] is
// the low byte). Bounded so it never reads past the `total`-byte child stream:
// the fast path (a full 8 bytes remain) is a single contiguous load, and only
// the final partial group of the last node takes the byte tail.
__device__ __forceinline__ uint64_t
dLoad8Bounded(const uint8_t* p, uint32_t cursor, uint32_t total)
{
    uint64_t v = 0;
    if (cursor + 8u <= total) {
        __builtin_memcpy(&v, p + cursor, 8);
    } else {
        for (uint32_t i = cursor; i < total; ++i) {
            v |= static_cast<uint64_t>(p[i]) << ((i - cursor) * 8u);
        }
    }
    return v;
}

// Reads 8 little-endian bytes at p[cursor] using two *aligned* 8-byte read-only
// (`__ldg`) loads of the enclosing 16-byte window plus a funnel shift. Aligned
// loads let the overlapping windows of neighbouring threads share L2/read-only
// sectors (killing the ~66% excess sectors of per-thread unaligned loads), and
// the read-only path spares the L1 the rank loads use. Requires `p` 8-byte
// aligned (child stream bases are alignUp8) and up to 16 bytes of trailing pad
// past the stream (workspace buffers reserve kMergeBufferPad), so no bounds
// branch is needed on the interior fast path.
__device__ __forceinline__ uint64_t
dLoad8Aligned(const uint8_t* p, uint32_t cursor)
{
    const uint32_t off = cursor & 7u;
    const uint64_t* const pw =
            reinterpret_cast<const uint64_t*>(p + (cursor & ~7u));
    const uint64_t w0 = __ldg(pw);
    if (off == 0u) {
        return w0;
    }
    const uint64_t w1 = __ldg(pw + 1);
    return (w0 >> (off * 8u)) | (w1 << ((8u - off) * 8u));
}

// `__byte_perm` selector per 4-bit partition mask: it merges 4 outputs from the
// 8 bytes of {left(a):right(b)} in one instruction. Output nibble i selects a
// left byte (index = zeros before i, in a's bytes 0-3) when mask bit i is
// clear, or a right byte (index = ones before i, in b's bytes 4-7) when set.
__device__ const uint32_t kMergeSel[16] = {
    0x3210u, 0x2104u, 0x2140u, 0x1054u, 0x2410u, 0x1504u, 0x1540u, 0x0654u,
    0x4210u, 0x5104u, 0x5140u, 0x6054u, 0x5410u, 0x6504u, 0x6540u, 0x7654u
};

// Merge 8 outputs from two contiguous child windows (each holding up to 8
// little-endian bytes at its cursor) driven by `mask8`: output byte i takes the
// next byte of `rightWin` if bit i is set, else of `leftWin`. Two `__byte_perm`
// (one per 4-bit nibble), advancing each window by the nibble popcount. A
// constant child is passed as a byte-replicated window.
__device__ __forceinline__ uint64_t
byteMerge8(uint64_t leftWin, uint64_t rightWin, uint32_t mask8)
{
    const uint32_t m0 = mask8 & 0xFu;
    const uint32_t m1 = (mask8 >> 4u) & 0xFu;
    const uint32_t o0 = __byte_perm(
            static_cast<uint32_t>(leftWin),
            static_cast<uint32_t>(rightWin),
            kMergeSel[m0]);
    leftWin >>= (4u - __popc(m0)) * 8u;
    rightWin >>= __popc(m0) * 8u;
    const uint32_t o1 = __byte_perm(
            static_cast<uint32_t>(leftWin),
            static_cast<uint32_t>(rightWin),
            kMergeSel[m1]);
    return static_cast<uint64_t>(o0) | (static_cast<uint64_t>(o1) << 32u);
}

// Reads 8 little-endian bytes at sp[off] from shared via two aligned 8-byte
// shared loads + a funnel shift. `sp` must be 8-byte aligned (chunk stream
// buffers are 16-aligned) and hold up to 15 bytes of readable slop past `off`
// (the following metadata region provides it).
__device__ __forceinline__ uint64_t
dLoad8Shared(const uint8_t* sp, uint32_t off)
{
    const uint32_t a = off & 7u;
    const uint64_t* const pw =
            reinterpret_cast<const uint64_t*>(sp + (off & ~7u));
    const uint64_t w0 = pw[0];
    if (a == 0u) {
        return w0;
    }
    const uint64_t w1 = pw[1];
    return (w0 >> (a * 8u)) | (w1 << ((8u - a) * 8u));
}

// Thread-per-bitmap-byte merge shared by the vector/vector, constant/vector,
// and vector/constant large-node paths. Each thread owns 8 outputs (one bitmap
// byte). It reads the rank once (giving each child's cursor), pulls up to 8
// contiguous bytes from each child stream into a register (the most either
// child can supply for 8 outputs), then selects per bit by shifting the bottom
// byte out of the chosen child -- all the branchiness stays in registers.
// Result: two contiguous child loads + one coalesced 8-byte store per 8
// outputs, instead of eight scattered, dependent child gathers. Constant
// children need no load.
template <bool LeftConst, bool RightConst, uint32_t Unroll>
__device__ __forceinline__ void byteMergeThread(
        uint8_t* out,
        uint32_t count,
        const uint8_t* bitmap,
        const uint16_t* directory,
        uint32_t dirBase,
        const uint8_t* lhs,
        const uint8_t* rhs,
        uint8_t leftSym,
        uint8_t rightSym)
{
    const uint32_t numBytes  = (count + 7u) / 8u;
    const uint32_t stepBytes = blockDim.x * Unroll;
    for (uint32_t bb = threadIdx.x; bb < numBytes; bb += stepBytes) {
#pragma unroll
        for (uint32_t u = 0; u < Unroll; ++u) {
            const uint32_t b = bb + u * blockDim.x;
            if (b >= numBytes) {
                continue;
            }
            const uint32_t j = b * 8u;
            const uint32_t rank =
                    dRankSelectOnesBeforeAligned(bitmap, directory, dirBase, j);
            // No tail masking: for the final partial group the high bits route
            // only the outputs past `count`, which are over-stored into the
            // buffer's 8-byte padding / dst slop (harmless). The rank -- and
            // thus both child cursors -- is exact regardless.
            const uint32_t rootBits = bitmap[b];
            // A constant child is a byte-replicated window; a vector child is
            // up to 8 contiguous bytes at its cursor, read via aligned
            // read-only loads (see dLoad8Aligned; relies on the buffers'
            // trailing pad).
            const uint64_t leftWin  = LeftConst
                     ? static_cast<uint64_t>(leftSym) * 0x0101010101010101ull
                     : dLoad8Aligned(lhs, j - rank);
            const uint64_t rightWin = RightConst
                    ? static_cast<uint64_t>(rightSym) * 0x0101010101010101ull
                    : dLoad8Aligned(rhs, rank);
            // `out` is 8-byte aligned with trailing slop (workspace buffers are
            // padded; dst reserves PIVCO_GPU_DECODE_DST_SLOP), so always store
            // a full aligned 8-byte group -- any over-store lands in the slop.
            *reinterpret_cast<uint64_t*>(out + j) =
                    byteMerge8(leftWin, rightWin, rootBits);
        }
    }
}

constexpr uint32_t kMergeUnroll = 2;

__global__ void scheduledMergeConstantVectorKernel(
        const PivCoGpuDecodeSchedule* schedule,
        uint8_t* dst,
        size_t dstSize,
        const uint8_t* bitstream,
        const uint64_t* offsets,
        size_t blockSize,
        uint8_t* workspace,
        PivCoGpuStatus* status,
        uint16_t stageOffset)
{
    if (status->code != PIVCO_GPU_STATUS_OK) {
        return;
    }

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }

    ScheduledNodeState* states = nullptr;
    uint16_t* directory        = nullptr;
    uint32_t dirCapacity       = 0;
    uint8_t* bufferA           = nullptr;
    uint8_t* bufferB           = nullptr;
    scheduledDecodeWorkspace(
            workspace,
            block,
            blockSize,
            &states,
            &directory,
            &dirCapacity,
            &bufferA,
            &bufferB);
    (void)dirCapacity;

    const uint16_t nodeIndex =
            schedule->stageItems[stageOffset + blockIdx.y].node;
    const PivCoGpuScheduleNode node     = schedule->nodes[nodeIndex];
    const ScheduledNodeState state      = states[nodeIndex];
    const ScheduledNodeState rightState = states[node.rightChild];
    uint8_t* const out                  = scheduledNodeOutput(
            node, state, schedule->maxLevel, dst, blockOff, bufferA, bufferB);
    const uint8_t* const rhs = scheduledChildOutput(
            node, rightState, schedule->maxLevel, bufferA, bufferB);
    const uint8_t* const bitmap =
            bitstream + offsets[block] + state.bitmapByteBase;
    const uint8_t leftSym = node.leftSymbol;
    (void)directory;

    __shared__ uint16_t s_dir[kRankSelectMaxBitmapWords + 1];
    __shared__ uint16_t s_prefix[kRankSelectMaxBitmapWords + 1];
    __shared__ uint32_t s_ones;
    buildSharedDirectory(bitmap, state.count, s_dir, s_prefix, &s_ones);

    if (state.count < kBottomUpChunkedMergeThreshold) {
        for (uint32_t j = threadIdx.x; j < state.count; j += blockDim.x) {
            const bool isRight = dGetBit(bitmap, j) != 0;
            const uint32_t onesBefore =
                    dRankSelectOnesBefore(bitmap, s_dir, 0u, j);
            out[j] = isRight ? rhs[onesBefore] : leftSym;
        }
        return;
    }

    byteMergeThread<true, false, kMergeUnroll>(
            out, state.count, bitmap, s_dir, 0u, nullptr, rhs, leftSym, 0u);
}

__global__ void scheduledMergeVectorConstantKernel(
        const PivCoGpuDecodeSchedule* schedule,
        uint8_t* dst,
        size_t dstSize,
        const uint8_t* bitstream,
        const uint64_t* offsets,
        size_t blockSize,
        uint8_t* workspace,
        PivCoGpuStatus* status,
        uint16_t stageOffset)
{
    if (status->code != PIVCO_GPU_STATUS_OK) {
        return;
    }

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }

    ScheduledNodeState* states = nullptr;
    uint16_t* directory        = nullptr;
    uint32_t dirCapacity       = 0;
    uint8_t* bufferA           = nullptr;
    uint8_t* bufferB           = nullptr;
    scheduledDecodeWorkspace(
            workspace,
            block,
            blockSize,
            &states,
            &directory,
            &dirCapacity,
            &bufferA,
            &bufferB);
    (void)dirCapacity;

    const uint16_t nodeIndex =
            schedule->stageItems[stageOffset + blockIdx.y].node;
    const PivCoGpuScheduleNode node    = schedule->nodes[nodeIndex];
    const ScheduledNodeState state     = states[nodeIndex];
    const ScheduledNodeState leftState = states[node.leftChild];
    uint8_t* const out                 = scheduledNodeOutput(
            node, state, schedule->maxLevel, dst, blockOff, bufferA, bufferB);
    const uint8_t* const lhs = scheduledChildOutput(
            node, leftState, schedule->maxLevel, bufferA, bufferB);
    const uint8_t* const bitmap =
            bitstream + offsets[block] + state.bitmapByteBase;
    const uint8_t rightSym = node.rightSymbol;
    (void)directory;

    __shared__ uint16_t s_dir[kRankSelectMaxBitmapWords + 1];
    __shared__ uint16_t s_prefix[kRankSelectMaxBitmapWords + 1];
    __shared__ uint32_t s_ones;
    buildSharedDirectory(bitmap, state.count, s_dir, s_prefix, &s_ones);

    if (state.count < kBottomUpChunkedMergeThreshold) {
        for (uint32_t j = threadIdx.x; j < state.count; j += blockDim.x) {
            const bool isRight = dGetBit(bitmap, j) != 0;
            const uint32_t onesBefore =
                    dRankSelectOnesBefore(bitmap, s_dir, 0u, j);
            out[j] = isRight ? rightSym : lhs[j - onesBefore];
        }
        return;
    }

    byteMergeThread<false, true, kMergeUnroll>(
            out, state.count, bitmap, s_dir, 0u, lhs, nullptr, 0u, rightSym);
}

__global__ void scheduledMergeVectorVectorKernel(
        const PivCoGpuDecodeSchedule* schedule,
        uint8_t* dst,
        size_t dstSize,
        const uint8_t* bitstream,
        const uint64_t* offsets,
        size_t blockSize,
        uint8_t* workspace,
        PivCoGpuStatus* status,
        uint16_t stageOffset)
{
    if (status->code != PIVCO_GPU_STATUS_OK) {
        return;
    }

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }

    ScheduledNodeState* states = nullptr;
    uint16_t* directory        = nullptr;
    uint32_t dirCapacity       = 0;
    uint8_t* bufferA           = nullptr;
    uint8_t* bufferB           = nullptr;
    scheduledDecodeWorkspace(
            workspace,
            block,
            blockSize,
            &states,
            &directory,
            &dirCapacity,
            &bufferA,
            &bufferB);
    (void)dirCapacity;

    const uint16_t nodeIndex =
            schedule->stageItems[stageOffset + blockIdx.y].node;
    const PivCoGpuScheduleNode node     = schedule->nodes[nodeIndex];
    const ScheduledNodeState state      = states[nodeIndex];
    const ScheduledNodeState leftState  = states[node.leftChild];
    const ScheduledNodeState rightState = states[node.rightChild];
    uint8_t* const out                  = scheduledNodeOutput(
            node, state, schedule->maxLevel, dst, blockOff, bufferA, bufferB);
    const uint8_t* const lhs = scheduledChildOutput(
            node, leftState, schedule->maxLevel, bufferA, bufferB);
    const uint8_t* const rhs = scheduledChildOutput(
            node, rightState, schedule->maxLevel, bufferA, bufferB);
    const uint8_t* const bitmap =
            bitstream + offsets[block] + state.bitmapByteBase;
    (void)directory;

    __shared__ uint16_t s_dir[kRankSelectMaxBitmapWords + 1];
    __shared__ uint16_t s_prefix[kRankSelectMaxBitmapWords + 1];
    __shared__ uint32_t s_ones;
    buildSharedDirectory(bitmap, state.count, s_dir, s_prefix, &s_ones);

    if (state.count < kBottomUpChunkedMergeThreshold) {
        for (uint32_t j = threadIdx.x; j < state.count; j += blockDim.x) {
            const bool isRight = dGetBit(bitmap, j) != 0;
            const uint32_t onesBefore =
                    dRankSelectOnesBefore(bitmap, s_dir, 0u, j);
            out[j] = isRight ? rhs[onesBefore] : lhs[j - onesBefore];
        }
        return;
    }

    byteMergeThread<false, false, kMergeUnroll>(
            out, state.count, bitmap, s_dir, 0u, lhs, rhs, 0u, 0u);
}

__global__ void fastDecodeRootConstFlat1Kernel(
        const PivCoGpuTree* tree,
        uint8_t* dst,
        size_t dstSize,
        const uint8_t* bitstream,
        size_t bitstreamSize,
        const uint64_t* offsets,
        size_t blockSize,
        PivCoGpuStatus* status)
{
    __shared__ uint32_t prefix[kFastMaxRootBytes + 1];
    __shared__ uint32_t sharedNumOnes;
    __shared__ uint64_t sharedLeafBitBase;
    __shared__ int sharedValid;

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }

    const uint64_t sliceBegin = offsets[block];
    const uint64_t sliceEnd   = offsets[block + 1];
    if (sliceBegin > sliceEnd || sliceEnd > bitstreamSize) {
        setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
        return;
    }

    const size_t blockLen  = dMin(blockSize, dstSize - blockOff);
    const size_t rootBytes = dBitmapBytes(blockLen);
    if (rootBytes > kFastMaxRootBytes) {
        setStatus(status, PIVCO_GPU_STATUS_PARAMETER, blockSize);
        return;
    }

    const uint8_t* const slice = bitstream + sliceBegin;
    const uint64_t sliceBytes  = sliceEnd - sliceBegin;

    fastScanRootPrefix(prefix, slice, rootBytes, blockLen);

    if (threadIdx.x == 0) {
        const size_t countBits = dFastCountBits(blockLen);
        size_t storedNumOnes   = 0;
        sharedValid            = 1;
        if (sliceBytes < rootBytes
            || !DeviceBitReader{ slice,
                                 sliceBytes,
                                 static_cast<uint64_t>(blockLen) }
                        .readBits(countBits, &storedNumOnes)
            || storedNumOnes > blockLen || storedNumOnes != prefix[rootBytes]) {
            sharedValid = 0;
        }
        sharedNumOnes     = static_cast<uint32_t>(storedNumOnes);
        sharedLeafBitBase = dFastLeafBitBase(blockLen);
        if (sharedValid != 0
            && dFastEncodedBytes(blockLen, sharedNumOnes) != sliceBytes) {
            sharedValid = 0;
        }
    }
    __syncthreads();

    if (sharedValid == 0) {
        setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
        return;
    }

    for (size_t byteIndex = threadIdx.x; byteIndex < rootBytes;
         byteIndex += blockDim.x) {
        const uint8_t rootByte =
                slice[byteIndex] & dRootByteMask(byteIndex, blockLen);
        uint32_t rank        = prefix[byteIndex];
        const size_t baseOut = blockOff + byteIndex * 8;
        for (size_t bit = 0; bit < 8 && byteIndex * 8 + bit < blockLen; ++bit) {
            uint8_t symbol = tree->fastZeroSymbol;
            if (((rootByte >> bit) & 1u) != 0) {
                symbol = dGetBit(slice, sharedLeafBitBase + rank) == 0
                        ? tree->fastLeafZeroSymbol
                        : tree->fastLeafOneSymbol;
                ++rank;
            }
            dst[baseOut + bit] = symbol;
        }
    }
}

__global__ void fastDecodeFlatRootKernel(
        const PivCoGpuTree* tree,
        uint8_t* dst,
        size_t dstSize,
        const uint8_t* bitstream,
        size_t bitstreamSize,
        const uint64_t* offsets,
        size_t blockSize,
        PivCoGpuStatus* status)
{
    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }

    const uint64_t sliceBegin = offsets[block];
    const uint64_t sliceEnd   = offsets[block + 1];
    if (sliceBegin > sliceEnd || sliceEnd > bitstreamSize) {
        setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
        return;
    }

    const size_t blockLen = dMin(blockSize, dstSize - blockOff);
    const size_t depth    = dLeafFlatDepth(tree, 0);
    const uint64_t encoded =
            (static_cast<uint64_t>(blockLen) * depth + 7u) / 8u;
    if (sliceEnd - sliceBegin != encoded) {
        setStatus(status, PIVCO_GPU_STATUS_CORRUPTION, block);
        return;
    }

    const uint8_t* const slice = bitstream + sliceBegin;

    // Cache the flat root's symbols (numRanks == 2^depth <= 256) in shared, so
    // the per-output lookup hits shared instead of a dependent global load.
    __shared__ uint8_t s_syms[256];
    const uint32_t numSyms = tree->numRanks;
    for (uint32_t i = threadIdx.x; i < numSyms; i += blockDim.x) {
        s_syms[i] = tree->rankToSymbol[i];
    }
    __syncthreads();

    // Eight outputs per thread, mirroring scheduledFlatKernel: a group of 8
    // symbols starts at output byte g*8, i.e. bit g*8*depth = byte g*depth
    // (always byte-aligned), so load the group's packed bytes once, unpack the
    // eight depth-bit indices from the register, look each up in the shared
    // symbol table, and write one coalesced 8-byte group. This replaces the
    // per-symbol path, which was latency-bound on a dependent 1-2 byte load +
    // conflicted shared gather + byte store per output. `out` is 8-byte aligned
    // (blockOff is a multiple of blockSize) with trailing slop, matching the
    // merge kernels' store contract. The index is masked to `depth` bits and
    // depth == flat depth, so index < 2^depth == numRanks; for depth == 8 the
    // mask is 0xFF and each packed byte is one index directly.
    const uint32_t mask         = depth >= 8 ? 0xFFu : ((1u << depth) - 1u);
    uint8_t* const out          = dst + blockOff;
    const uint32_t numGroups    = static_cast<uint32_t>((blockLen + 7) / 8);
    const uint32_t encodedBytes = static_cast<uint32_t>(
            (static_cast<uint64_t>(blockLen) * depth + 7u) / 8u);
    // Depth-2 memory-level parallelism on the packed-index load: fetch two
    // groups' packed words before unpacking either, so one load hides the
    // other's latency (this path is latency bound on that dependent load).
    for (uint32_t gg = threadIdx.x; gg < numGroups; gg += blockDim.x * 2u) {
        const uint32_t g1 = gg + blockDim.x;
        const bool has1   = g1 < numGroups;
        const uint64_t p0 = dLoad8Bounded(
                slice, gg * static_cast<uint32_t>(depth), encodedBytes);
        const uint64_t p1 = has1 ? dLoad8Bounded(
                                           slice,
                                           g1 * static_cast<uint32_t>(depth),
                                           encodedBytes)
                                 : 0u;
#pragma unroll
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint32_t g      = half == 0u ? gg : g1;
            const uint64_t packed = half == 0u ? p0 : p1;
            if (half == 1u && !has1) {
                break;
            }
            uint64_t outWord = 0;
#pragma unroll
            for (uint32_t k = 0; k < 8u; ++k) {
                const uint32_t index =
                        static_cast<uint32_t>(
                                packed >> (k * static_cast<uint32_t>(depth)))
                        & mask;
                outWord |= static_cast<uint64_t>(s_syms[index]) << (k * 8u);
            }
            *reinterpret_cast<uint64_t*>(out + g * 8u) = outWord;
        }
    }
}

__global__ void fastEncodePackRootConstFlat1Kernel(
        const PivCoGpuTree* tree,
        const uint8_t* src,
        size_t srcSize,
        size_t blockSize,
        uint8_t* blockBitstreams,
        uint64_t* offsets,
        PivCoGpuStatus* status)
{
    __shared__ uint32_t chunkPrefix[kFastBlockThreads + 1];
    __shared__ uint32_t leafWords[(kFastMaxRootBytes + 7) / 4];
    __shared__ uint32_t sharedNumOnes;
    __shared__ uint64_t sharedLeafBitBase;
    __shared__ uint64_t sharedOutSize;
    __shared__ int sharedValid;

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= srcSize) {
        return;
    }

    const size_t blockLen  = dMin(blockSize, srcSize - blockOff);
    const size_t rootBytes = dBitmapBytes(blockLen);
    if (rootBytes > kFastMaxRootBytes) {
        setStatus(status, PIVCO_GPU_STATUS_PARAMETER, blockSize);
        return;
    }

    uint8_t* const out         = blockBitstreams + block * blockSize;
    const uint64_t leafBitBase = dFastLeafBitBase(blockLen);
    for (uint64_t byte = blockLen / 8u + threadIdx.x; byte < leafBitBase / 8u;
         byte += blockDim.x) {
        out[byte] = 0;
    }

    uint32_t localLeafWords[kFastThreadLeafWords] = {};
    uint32_t localCount                           = 0;
    const size_t chunkBegin = (rootBytes * threadIdx.x) / blockDim.x;
    const size_t chunkEnd   = (rootBytes * (threadIdx.x + 1)) / blockDim.x;
    for (size_t byteIndex = chunkBegin; byteIndex < chunkEnd; ++byteIndex) {
        const size_t srcBase = byteIndex * 8;
        uint8_t rootByte     = 0;
        uint8_t leafOneMask  = 0;
        if (srcBase + 8 <= blockLen) {
            const uint8_t* const srcPtr = src + blockOff + srcBase;
            const uint8_t zeroMask = dEqMask8(srcPtr, tree->fastZeroSymbol);
            const uint8_t leafZeroMask =
                    dEqMask8(srcPtr, tree->fastLeafZeroSymbol);
            leafOneMask = dEqMask8(srcPtr, tree->fastLeafOneSymbol);
            rootByte    = static_cast<uint8_t>(~zeroMask);
            const uint8_t missingMask = rootByte
                    & static_cast<uint8_t>(~(leafZeroMask | leafOneMask));
            if (missingMask != 0) {
                const uint32_t bit =
                        static_cast<uint32_t>(__ffs(missingMask) - 1);
                setStatus(status, PIVCO_GPU_STATUS_MISSING_SYMBOL, srcPtr[bit]);
            }
        } else {
            for (size_t bit = 0; srcBase + bit < blockLen; ++bit) {
                const uint8_t symbol = src[blockOff + srcBase + bit];
                if (symbol == tree->fastZeroSymbol) {
                    continue;
                }
                if (symbol != tree->fastLeafZeroSymbol
                    && symbol != tree->fastLeafOneSymbol) {
                    setStatus(status, PIVCO_GPU_STATUS_MISSING_SYMBOL, symbol);
                    continue;
                }
                rootByte |= static_cast<uint8_t>(1u << bit);
                if (symbol == tree->fastLeafOneSymbol) {
                    leafOneMask |= static_cast<uint8_t>(1u << bit);
                }
            }
        }
        uint8_t pending = rootByte;
        while (pending != 0) {
            const uint32_t bit = static_cast<uint32_t>(__ffs(pending) - 1);
            if ((leafOneMask & (1u << bit)) != 0) {
                localLeafWords[localCount / 32u] |=
                        static_cast<uint32_t>(1u << (localCount & 31u));
            }
            ++localCount;
            pending &= static_cast<uint8_t>(pending - 1);
        }
        out[byteIndex] = rootByte;
    }

    chunkPrefix[threadIdx.x + 1] = localCount;
    if (threadIdx.x == 0) {
        chunkPrefix[0] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t running = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            const uint32_t count = chunkPrefix[i + 1];
            chunkPrefix[i]       = running;
            running += count;
        }
        chunkPrefix[blockDim.x] = running;
        sharedNumOnes           = running;
        sharedLeafBitBase       = leafBitBase;
        sharedOutSize           = dFastEncodedBytes(blockLen, running);
        sharedValid             = sharedOutSize <= blockSize;
        offsets[block + 1]      = sharedOutSize;
        dOrBits(out, blockLen, running, dFastCountBits(blockLen));
    }
    __syncthreads();

    if (sharedValid == 0) {
        setStatus(status, PIVCO_GPU_STATUS_CAPACITY, block);
        return;
    }

    const uint32_t leafBytes     = (sharedNumOnes + 7u) / 8u;
    const uint32_t leafWordCount = (leafBytes + 3u) / 4u;
    for (uint32_t i = threadIdx.x; i < leafWordCount; i += blockDim.x) {
        leafWords[i] = 0;
    }
    __syncthreads();

    const uint32_t chunkBitBase = chunkPrefix[threadIdx.x];
    uint32_t remaining          = localCount;
    for (uint32_t i = 0; i < kFastThreadLeafWords && remaining != 0; ++i) {
        const uint32_t bits = remaining < 32u ? remaining : 32u;
        dAtomicOrSharedBits(
                leafWords, chunkBitBase + i * 32u, localLeafWords[i], bits);
        remaining -= bits;
    }
    __syncthreads();

    const uint64_t leafByteBase = sharedLeafBitBase / 8u;
    const uint8_t* const leafBytesPtr =
            reinterpret_cast<const uint8_t*>(leafWords);
    for (uint32_t i = threadIdx.x; i < leafBytes; i += blockDim.x) {
        out[leafByteBase + i] = leafBytesPtr[i];
    }
}

__global__ void fastEncodeCopyRootConstFlat1Kernel(
        uint8_t* dst,
        size_t dstCapacity,
        const uint8_t* blockBitstreams,
        size_t srcSize,
        size_t blockSize,
        const uint64_t* offsets,
        PivCoGpuStatus* status)
{
    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= srcSize) {
        return;
    }

    const uint64_t outBegin = offsets[block];
    const uint64_t outEnd   = offsets[block + 1];
    if (outBegin > outEnd || outEnd > dstCapacity
        || outEnd - outBegin > blockSize) {
        setStatus(status, PIVCO_GPU_STATUS_CAPACITY, block);
        return;
    }

    const uint8_t* const srcBlock = blockBitstreams + block * blockSize;
    uint8_t* const dstBlock       = dst + outBegin;
    const uint64_t outSize        = outEnd - outBegin;
    for (uint64_t i = threadIdx.x; i < outSize; i += blockDim.x) {
        dstBlock[i] = srcBlock[i];
    }
}

cudaError_t launchScheduledDecodeStage(
        uint8_t op,
        uint16_t stageOffset,
        uint16_t stageCount,
        size_t numBlocks,
        const PivCoGpuTree* tree_d,
        const PivCoGpuDecodeSchedule* schedule_d,
        uint8_t* dst,
        size_t dstSize,
        const uint8_t* bitstream,
        const uint64_t* offsets,
        size_t blockSize,
        uint8_t* workspace,
        PivCoGpuStatus* status,
        cudaStream_t stream)
{
    if (stageCount == 0) {
        return cudaSuccess;
    }

    const dim3 grid(
            static_cast<unsigned>(numBlocks),
            static_cast<unsigned>(stageCount));
    switch (op) {
        case PIVCO_GPU_SCHEDULE_OP_FLAT1:
            scheduledFlatKernel<1><<<grid, kRankSelectThreads, 0, stream>>>(
                    tree_d,
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_FLAT2:
            scheduledFlatKernel<2><<<grid, kRankSelectThreads, 0, stream>>>(
                    tree_d,
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_FLAT3:
            scheduledFlatKernel<3><<<grid, kRankSelectThreads, 0, stream>>>(
                    tree_d,
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_FLAT4:
            scheduledFlatKernel<4><<<grid, kRankSelectThreads, 0, stream>>>(
                    tree_d,
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_FLAT5:
            scheduledFlatKernel<5><<<grid, kRankSelectThreads, 0, stream>>>(
                    tree_d,
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_FLAT6:
            scheduledFlatKernel<6><<<grid, kRankSelectThreads, 0, stream>>>(
                    tree_d,
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_FLAT7:
            scheduledFlatKernel<7><<<grid, kRankSelectThreads, 0, stream>>>(
                    tree_d,
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_FLAT8:
            scheduledFlatKernel<8><<<grid, kRankSelectThreads, 0, stream>>>(
                    tree_d,
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_MERGE_VECTOR_VECTOR:
            scheduledMergeVectorVectorKernel<<<
                    grid,
                    kRankSelectThreads,
                    0,
                    stream>>>(
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_VECTOR:
            scheduledMergeConstantVectorKernel<<<
                    grid,
                    kRankSelectThreads,
                    0,
                    stream>>>(
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_MERGE_VECTOR_CONSTANT:
            scheduledMergeVectorConstantKernel<<<
                    grid,
                    kRankSelectThreads,
                    0,
                    stream>>>(
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        case PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_CONSTANT:
            scheduledMergeConstantConstantKernel<<<
                    grid,
                    kRankSelectThreads,
                    0,
                    stream>>>(
                    schedule_d,
                    dst,
                    dstSize,
                    bitstream,
                    offsets,
                    blockSize,
                    workspace,
                    status,
                    stageOffset);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

// ===========================================================================
// Chunked top-down / merge-in-shared decoder.
//
// A warp owns one contiguous CHUNK of a block's output (chunkOutputs bytes)
// and decodes the WHOLE tree for that chunk with all intermediate node streams
// resident in shared memory -- only the bitmaps/flat indices are read from
// global (L2) and only the final chunk output is written to global (once). This
// removes the per-level global round-trips and the ~30 per-level kernel
// launches of the bottom-up cascade, and turns every dependent child load from
// an L2
// (~200 cyc) access into a shared (~30 cyc) access.
//
// The monotone-rank property makes it work: a contiguous output chunk [c0,c0+S)
// descends to a CONTIGUOUS sub-range at every tree node, so all per-chunk node
// streams together fit in 2*S bytes (level-parity ping-pong). Phase 1 is a
// top-down descent (per node: its chunk sub-range [lo,lo+subLen) and its offset
// in the level buffer), using the per-block rank directories for the two
// boundary ranks per node. Phase 2 is the bottom-up merge (deepest level first)
// with the proven byte-select, reading children from shared. Phase 3 flushes
// the root buffer to global, coalesced.
// ===========================================================================
// Chunk size (outputs per warp-pass) is chosen at dispatch by input size and
// passed to the kernel at runtime: tiny inputs use the small chunk (more chunks
// -> more grid parallelism to fill the GPU), larger inputs use the big chunk
// (each per-chunk tree descent is fixed overhead, so wider chunks amortize it;
// measured +13-21% at >=8 MiB). The per-warp ping-pong buffer is sized from the
// runtime chunk (kChunkTdBufBytesFor); the shared budget is set per launch.
constexpr uint32_t kChunkTdSmallOutputs = 1024;
constexpr uint32_t kChunkTdLargeOutputs = 2048;
constexpr int kChunkTdWarps             = 8;
constexpr int kChunkTdThreads           = kChunkTdWarps * 32;
// Per-warp stream buffer size for a given chunk: the chunk's bytes plus the
// 8-byte over-read slop dLoad8Shared/dRankOnesBeforeFast need at the buffer
// end.
__host__ __device__ inline uint32_t kChunkTdBufBytesFor(uint32_t chunkOutputs)
{
    return chunkOutputs + 64u;
}
// Levels with at least this many nodes use the node-per-lane merge (32 small
// nodes decoded in parallel across the warp); shallower/wider-node levels use
// the node-per-warp merge (the whole warp cooperates on each large node).
constexpr uint32_t kChunkTdLaneThreshold = 8;

// Decode one node's whole chunk sub-range on a SINGLE lane (used when a level
// has many small nodes, so 32 nodes run in parallel across the warp). The
// running child rank accumulates in a register -- no warp scan or directory
// read.
__device__ __forceinline__ void mergeNodeSingleLane(
        const PivCoGpuScheduleNode& node,
        const ScheduledNodeState& st,
        uint8_t* out,
        uint32_t lo,
        uint32_t len,
        const uint8_t* sbuf,
        const uint16_t* streamOff,
        const uint8_t* slice,
        const PivCoGpuTree* tree)
{
    if (node.kind == PIVCO_GPU_SCHEDULE_CONSTANT) {
        for (uint32_t j = 0; j < len; ++j) {
            out[j] = node.symbol;
        }
    } else if (node.kind == PIVCO_GPU_SCHEDULE_FLAT) {
        const uint32_t depth     = node.flatDepth;
        const uint32_t firstRank = node.firstRank;
        const uint32_t msk       = (1u << depth) - 1u;
        for (uint32_t j = 0; j < len; ++j) {
            const uint64_t bp = static_cast<uint64_t>(st.leafBitBase)
                    + static_cast<uint64_t>(lo + j) * depth;
            const uint32_t bi = static_cast<uint32_t>(bp >> 3u);
            const uint32_t sh = static_cast<uint32_t>(bp & 7u);
            uint32_t win      = slice[bi];
            if (sh + depth > 8u) {
                win |= static_cast<uint32_t>(slice[bi + 1u]) << 8u;
            }
            out[j] = tree->rankToSymbol[firstRank + ((win >> sh) & msk)];
        }
    } else if (node.op == PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_CONSTANT) {
        const uint8_t ls        = node.leftSymbol;
        const uint8_t rs        = node.rightSymbol;
        const uint8_t* const bm = slice + st.bitmapByteBase;
        for (uint32_t j = 0; j < len; ++j) {
            out[j] = dGetBit(bm, lo + j) != 0 ? rs : ls;
        }
    } else {
        const bool leftConst =
                node.op == PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_VECTOR;
        const bool rightConst =
                node.op == PIVCO_GPU_SCHEDULE_OP_MERGE_VECTOR_CONSTANT;
        const uint64_t leftB =
                static_cast<uint64_t>(node.leftSymbol) * 0x0101010101010101ull;
        const uint64_t rightB =
                static_cast<uint64_t>(node.rightSymbol) * 0x0101010101010101ull;
        const uint32_t leftOff  = leftConst ? 0u : streamOff[node.leftChild];
        const uint32_t rightOff = rightConst ? 0u : streamOff[node.rightChild];
        const uint8_t* const bm = slice + st.bitmapByteBase;
        uint32_t r              = 0u;
        for (uint32_t g = 0; g < len; g += 8u) {
            const uint32_t bit = lo + g;
            const uint32_t bi  = bit >> 3u;
            const uint32_t sh  = bit & 7u;
            uint32_t win       = bm[bi];
            if (sh != 0u) {
                win |= static_cast<uint32_t>(bm[bi + 1u]) << 8u;
            }
            const uint32_t valid = (len - g) < 8u ? (len - g) : 8u;
            const uint32_t mask8 = ((win >> sh) & 0xFFu) & ((1u << valid) - 1u);
            const uint64_t lw =
                    leftConst ? leftB : dLoad8Shared(sbuf, leftOff + (g - r));
            const uint64_t rw =
                    rightConst ? rightB : dLoad8Shared(sbuf, rightOff + r);
            const uint64_t res = byteMerge8(lw, rw, mask8);
            for (uint32_t k = 0; k < valid; ++k) {
                out[g + k] = static_cast<uint8_t>(res >> (k * 8u));
            }
            r += static_cast<uint32_t>(__popc(mask8));
        }
    }
}

__global__ void __launch_bounds__(kChunkTdThreads, 6) pivcoChunkTopDownKernel(
        const PivCoGpuTree* tree,
        const PivCoGpuDecodeSchedule* schedule,
        uint8_t* dst,
        size_t dstSize,
        const uint8_t* bitstream,
        const uint64_t* offsets,
        size_t blockSize,
        uint32_t chunkOutputs,
        uint8_t* workspace,
        PivCoGpuStatus* status)
{
    if (status->code != PIVCO_GPU_STATUS_OK) {
        return;
    }
    __shared__ uint16_t s_levelBegin[PIVCO_GPU_MAX_LEVELS + 2];
    extern __shared__ uint8_t s_dyn[];

    const uint32_t S           = chunkOutputs;
    const uint32_t bufBytes    = kChunkTdBufBytesFor(chunkOutputs);
    const uint32_t nodeCount   = schedule->nodeCount;
    const int maxLevel         = static_cast<int>(schedule->maxLevel);
    const uint32_t warpInBlock = threadIdx.x >> 5u;
    const uint32_t lane        = threadIdx.x & 31u;

    // Shared layout: CTA-wide per-level node index list, then per-warp regions
    // each holding a 2*S ping-pong stream pair + node metadata (loBit u32,
    // subLen u16, streamOff u16).
    // Per-node metadata is 6 bytes: loBit/subLen/streamOff are all <= blockLen
    // <= 64 KiB, so uint16 each (loBit is a position < count <= 65536, i.e. <=
    // 65535).
    uint16_t* const s_levelNode = reinterpret_cast<uint16_t*>(s_dyn);
    const uint32_t perWarpBytes = (2u * bufBytes + 6u * nodeCount + 15u) & ~15u;
    uint8_t* const perWarpBase  = s_dyn + ((nodeCount * 2u + 15u) & ~15u);
    uint8_t* const buf0         = perWarpBase + warpInBlock * perWarpBytes;
    uint8_t* const buf1         = buf0 + bufBytes;
    uint8_t* const myMeta       = buf0 + 2u * bufBytes;
    uint16_t* const loBit       = reinterpret_cast<uint16_t*>(myMeta);
    uint16_t* const subLen =
            reinterpret_cast<uint16_t*>(myMeta + nodeCount * 2u);
    uint16_t* const streamOff =
            reinterpret_cast<uint16_t*>(myMeta + nodeCount * 4u);

    // ---- Phase 0: build per-level node index lists (CTA-wide, once). Nodes
    // are emitted in DFS pre-order, so bucket them by level for the per-level
    // passes.
    if (threadIdx.x == 0) {
        uint16_t cnt[PIVCO_GPU_MAX_LEVELS + 1];
        for (int l = 0; l <= maxLevel; ++l) {
            cnt[l] = 0;
        }
        for (uint32_t n = 0; n < nodeCount; ++n) {
            cnt[schedule->nodes[n].level]++;
        }
        uint32_t acc = 0;
        for (int l = 0; l <= maxLevel; ++l) {
            s_levelBegin[l] = static_cast<uint16_t>(acc);
            acc += cnt[l];
        }
        s_levelBegin[maxLevel + 1] = static_cast<uint16_t>(acc);
        uint16_t cur[PIVCO_GPU_MAX_LEVELS + 1];
        for (int l = 0; l <= maxLevel; ++l) {
            cur[l] = s_levelBegin[l];
        }
        for (uint32_t n = 0; n < nodeCount; ++n) {
            const uint32_t l      = schedule->nodes[n].level;
            s_levelNode[cur[l]++] = static_cast<uint16_t>(n);
        }
    }
    __syncthreads();

    const size_t block    = blockIdx.x;
    const size_t blockOff = block * blockSize;
    if (blockOff >= dstSize) {
        return;
    }
    const uint32_t blockLen =
            static_cast<uint32_t>(dMin(blockSize, dstSize - blockOff));
    const uint32_t chunkInBlock = blockIdx.y * kChunkTdWarps + warpInBlock;
    const uint32_t c0           = chunkInBlock * S;
    if (c0 >= blockLen) {
        return;
    }
    const uint32_t sLen = dMin(S, blockLen - c0);

    ScheduledNodeState* states = nullptr;
    uint16_t* directory        = nullptr;
    uint32_t dirCap            = 0;
    uint8_t* bA                = nullptr;
    uint8_t* bB                = nullptr;
    scheduledDecodeWorkspace(
            workspace,
            block,
            blockSize,
            &states,
            &directory,
            &dirCap,
            &bA,
            &bB);
    (void)dirCap;
    (void)bA;
    (void)bB;
    const uint8_t* const slice = bitstream + offsets[block];

    // ---- Phase 1: warp-parallel top-down descent. Levels are serial (parent
    // -> child), but all nodes within a level are independent, so the 32 lanes
    // cover them in parallel -- the boundary-rank L2 reads overlap instead of
    // stalling one lane. streamOff is a per-level exclusive prefix sum (warp
    // scan).
    for (uint32_t n = lane; n < nodeCount; n += 32u) {
        subLen[n] = (n == 0u) ? static_cast<uint16_t>(sLen) : uint16_t{ 0 };
    }
    if (lane == 0u) {
        loBit[0] = static_cast<uint16_t>(c0);
    }
    __syncwarp();
    for (int L = 0; L <= maxLevel; ++L) {
        const uint32_t begin = s_levelBegin[L];
        const uint32_t end   = s_levelBegin[L + 1];
        // Few-node levels use the node-per-warp merge with aligned uint64
        // stores, so 8-byte-pad their per-node offsets; many-node levels stay
        // byte-packed.
        const bool padLevel = (end - begin) < kChunkTdLaneThreshold;
        uint32_t running    = 0u;
        for (uint32_t base = begin; base < end; base += 32u) {
            const uint32_t idx = base + lane;
            const uint32_t nid = idx < end ? s_levelNode[idx] : 0u;
            uint32_t v         = idx < end ? subLen[nid] : 0u;
            if (padLevel) {
                v = (v + 7u) & ~7u;
            }
            uint32_t incl = v;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, incl, d);
                if (static_cast<int>(lane) >= d) {
                    incl += t;
                }
            }
            if (idx < end) {
                streamOff[nid] = static_cast<uint16_t>(running + incl - v);
            }
            running += __shfl_sync(0xFFFFFFFFu, incl, 31);
        }
        __syncwarp();
        if (L < maxLevel) {
            for (uint32_t idx = begin + lane; idx < end; idx += 32u) {
                const uint32_t nid              = s_levelNode[idx];
                const PivCoGpuScheduleNode node = schedule->nodes[nid];
                if (node.kind != PIVCO_GPU_SCHEDULE_INTERNAL
                    || node.op
                            == PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_CONSTANT) {
                    continue;
                }
                const uint32_t lo  = loBit[nid];
                const uint32_t len = subLen[nid];
                if (len == 0u) {
                    continue;
                }
                const uint32_t hi           = lo + len;
                const ScheduledNodeState st = states[nid];
                const uint8_t* const bm     = slice + st.bitmapByteBase;
                const uint32_t rlo =
                        dRankOnesBeforeFast(bm, directory, st.dirBase, lo);
                const uint32_t rhi =
                        dRankOnesBeforeFast(bm, directory, st.dirBase, hi);
                loBit[node.leftChild] = static_cast<uint16_t>(lo - rlo);
                subLen[node.leftChild] =
                        static_cast<uint16_t>((hi - rhi) - (lo - rlo));
                loBit[node.rightChild]  = static_cast<uint16_t>(rlo);
                subLen[node.rightChild] = static_cast<uint16_t>(rhi - rlo);
                // A constant child is emitted directly by the parent's merge
                // (leftSym/rightSym), so it never needs its own materialized
                // stream -- zero its sub-length to skip that redundant fill.
                if (node.op == PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_VECTOR) {
                    subLen[node.leftChild] = 0u;
                } else if (
                        node.op
                        == PIVCO_GPU_SCHEDULE_OP_MERGE_VECTOR_CONSTANT) {
                    subLen[node.rightChild] = 0u;
                }
            }
            __syncwarp();
        }
    }

    // ---- Phase 2: bottom-up merge, deepest level first, all in shared.
    for (int L = maxLevel; L >= 0; --L) {
        uint8_t* const dbuf       = (L & 1) ? buf1 : buf0;
        const uint8_t* const sbuf = (L & 1) ? buf0 : buf1;
        const uint32_t begin      = s_levelBegin[L];
        const uint32_t end        = s_levelBegin[L + 1];

        if (end - begin >= kChunkTdLaneThreshold) {
            // Many small nodes: one node per lane (32 nodes decoded in
            // parallel).
            for (uint32_t idx = begin + lane; idx < end; idx += 32u) {
                const uint32_t nid = s_levelNode[idx];
                const uint32_t len = subLen[nid];
                if (len == 0u) {
                    continue;
                }
                mergeNodeSingleLane(
                        schedule->nodes[nid],
                        states[nid],
                        dbuf + streamOff[nid],
                        loBit[nid],
                        len,
                        sbuf,
                        streamOff,
                        slice,
                        tree);
            }
            __syncwarp();
            continue;
        }

        for (uint32_t idx = begin; idx < end; ++idx) {
            const uint32_t nid              = s_levelNode[idx];
            const PivCoGpuScheduleNode node = schedule->nodes[nid];
            const uint32_t len              = subLen[nid];
            if (len == 0u) {
                continue;
            }
            uint8_t* const out          = dbuf + streamOff[nid];
            const uint32_t lo           = loBit[nid];
            const ScheduledNodeState st = states[nid];

            if (node.kind == PIVCO_GPU_SCHEDULE_CONSTANT) {
                for (uint32_t j = lane; j < len; j += 32u) {
                    out[j] = node.symbol;
                }
            } else if (node.kind == PIVCO_GPU_SCHEDULE_FLAT) {
                const uint32_t depth     = node.flatDepth;
                const uint32_t firstRank = node.firstRank;
                const uint32_t msk       = (1u << depth) - 1u;
                for (uint32_t j = lane; j < len; j += 32u) {
                    const uint64_t bp = static_cast<uint64_t>(st.leafBitBase)
                            + static_cast<uint64_t>(lo + j) * depth;
                    const uint32_t bi = static_cast<uint32_t>(bp >> 3u);
                    const uint32_t sh = static_cast<uint32_t>(bp & 7u);
                    uint32_t win      = slice[bi];
                    if (sh + depth > 8u) {
                        win |= static_cast<uint32_t>(slice[bi + 1u]) << 8u;
                    }
                    out[j] =
                            tree->rankToSymbol[firstRank + ((win >> sh) & msk)];
                }
            } else if (
                    node.op == PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_CONSTANT) {
                const uint8_t ls        = node.leftSymbol;
                const uint8_t rs        = node.rightSymbol;
                const uint8_t* const bm = slice + st.bitmapByteBase;
                for (uint32_t j = lane; j < len; j += 32u) {
                    out[j] = dGetBit(bm, lo + j) != 0 ? rs : ls;
                }
            } else {
                // VV / CV / VC: 8 outputs per thread via byteMerge8 (two
                // __byte_perm). The child cursors need the ones-count before
                // each group; compute it with a warp scan of per-group popc (no
                // per-group directory read), carrying the running total across
                // iterations.
                const bool leftConst =
                        node.op == PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_VECTOR;
                const bool rightConst =
                        node.op == PIVCO_GPU_SCHEDULE_OP_MERGE_VECTOR_CONSTANT;
                const uint64_t leftB = static_cast<uint64_t>(node.leftSymbol)
                        * 0x0101010101010101ull;
                const uint64_t rightB = static_cast<uint64_t>(node.rightSymbol)
                        * 0x0101010101010101ull;
                const uint32_t leftOff =
                        leftConst ? 0u : streamOff[node.leftChild];
                const uint32_t rightOff =
                        rightConst ? 0u : streamOff[node.rightChild];
                const uint8_t* const bm  = slice + st.bitmapByteBase;
                const uint32_t numGroups = (len + 7u) / 8u;
                uint32_t running         = 0u;
                for (uint32_t base = 0; base < numGroups; base += 32u) {
                    const uint32_t gb = base + lane;
                    const bool active = gb < numGroups;
                    const uint32_t g  = gb * 8u;
                    uint32_t mask8    = 0u;
                    if (active) {
                        const uint32_t bit = lo + g;
                        const uint32_t bi  = bit >> 3u;
                        const uint32_t sh  = bit & 7u;
                        uint32_t win       = bm[bi];
                        if (sh != 0u) {
                            win |= static_cast<uint32_t>(bm[bi + 1u]) << 8u;
                        }
                        const uint32_t valid = (len - g) < 8u ? (len - g) : 8u;
                        mask8 = ((win >> sh) & 0xFFu) & ((1u << valid) - 1u);
                    }
                    const uint32_t ones = static_cast<uint32_t>(__popc(mask8));
                    uint32_t incl       = ones;
#pragma unroll
                    for (int d = 1; d < 32; d <<= 1) {
                        const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, incl, d);
                        if (static_cast<int>(lane) >= d) {
                            incl += t;
                        }
                    }
                    const uint32_t rankAtGroup = running + incl - ones;
                    if (active) {
                        const uint64_t lw  = leftConst
                                 ? leftB
                                 : dLoad8Shared(
                                          sbuf, leftOff + (g - rankAtGroup));
                        const uint64_t rw  = rightConst
                                 ? rightB
                                 : dLoad8Shared(sbuf, rightOff + rankAtGroup);
                        const uint64_t res = byteMerge8(lw, rw, mask8);
                        // streamOff is 8-byte padded for these few-node levels,
                        // so out+g is aligned -> one uint64 store (over-store
                        // into the node's pad is harmless).
                        *reinterpret_cast<uint64_t*>(out + g) = res;
                    }
                    running += __shfl_sync(0xFFFFFFFFu, incl, 31);
                }
            }
        }
        __syncwarp();
    }

    // ---- Phase 3: coalesced flush of the root buffer (level 0 wrote buf0).
    uint8_t* const gout          = dst + blockOff + c0;
    const uint8_t* const rootBuf = buf0;
    for (uint32_t i = lane * 8u; i < sLen; i += 32u * 8u) {
        if (i + 8u <= sLen) {
            *reinterpret_cast<uint64_t*>(gout + i) =
                    *reinterpret_cast<const uint64_t*>(rootBuf + i);
        } else {
            for (uint32_t k = i; k < sLen; ++k) {
                gout[k] = rootBuf[k];
            }
        }
    }
}

bool chunkTopDownEnabled()
{
    static const bool enabled = [] {
        const char* v = getenv("PIVCO_CHUNK_TD");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool chunkTopDownDisabled()
{
    static const bool disabled = [] {
        const char* v = getenv("PIVCO_CHUNK_TD");
        return v != nullptr && v[0] == '0';
    }();
    return disabled;
}

// The single-kernel chunk-in-shared decoder decodes the whole tree per output
// chunk with intermediates in shared, so it pays only 3 kernel launches vs the
// bottom-up cascade's ~30. For small inputs that fixed launch overhead
// dominates and the chunk decoder is faster; past the crossover the cascade's
// higher steady-state throughput (its O(depth) ALU is hidden behind DRAM
// latency) wins. The wider (2048) chunk lifts the chunk decoder's steady-state
// enough to hold the win out to ~12 MiB (measured +19..+38% at 10 MiB) for the
// deep trees that dominate real data. Shallow trees (small tableLog, e.g. an
// incompressible .gz) have a fast baseline (few merge levels -> few launches)
// and cross over earlier (~6 MiB), so gate the wider window on tree depth: deep
// trees get 12 MiB; shallow ones only win to ~4 MiB (measured: an
// incompressible .gz ties at 4 MiB and loses ~5% by 8 MiB), so cap them there.
constexpr size_t kChunkTdMaxAutoBytesDeep    = 12u * 1024u * 1024u;
constexpr size_t kChunkTdMaxAutoBytesShallow = 4u * 1024u * 1024u;
constexpr int kChunkTdDeepTableLog           = 10;

inline size_t chunkTdMaxAutoBytesFor(int tableLog)
{
    return tableLog >= kChunkTdDeepTableLog ? kChunkTdMaxAutoBytesDeep
                                            : kChunkTdMaxAutoBytesShallow;
}
// Below this size the small (1024) chunk wins (its extra chunks give the grid
// enough parallelism to fill the GPU); at/above it the wide (2048) chunk's
// per-chunk-descent amortization wins. Between ~4 and ~8 MiB the small chunk is
// still ahead, so keep the boundary at 6 MiB.
constexpr size_t kChunkTdWideChunkBytes = 6u * 1024u * 1024u;

// Pick the chunk size for a given output size (see the two constants above).
inline uint32_t chunkTdOutputsFor(size_t dstSize)
{
    return dstSize <= kChunkTdWideChunkBytes ? kChunkTdSmallOutputs
                                             : kChunkTdLargeOutputs;
}

} // namespace

extern "C" size_t pivcoGpuEncodeWorkspaceBytes(size_t srcSize, size_t blockSize)
{
    return workspaceBytesFor(srcSize, blockSize);
}

extern "C" size_t pivcoGpuDecodeWorkspaceBytes(size_t dstSize, size_t blockSize)
{
    return decodeWorkspaceBytesFor(dstSize, blockSize);
}

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
        cudaStream_t stream)
{
    if (context == nullptr || offsets_d == nullptr || status_d == nullptr
        || totalSize_d == nullptr || blockSize == 0
        || blockSize > PIVCO_GPU_MAX_BLOCK_SIZE
        || addOverflows(srcSize, blockSize - 1)) {
        return cudaErrorInvalidValue;
    }

    const size_t numBlocks = numBlocksFor(srcSize, blockSize);
    if (offsetsCapacity < numBlocks + 1 || numBlocks > kMaxGridX) {
        return cudaErrorInvalidValue;
    }
    if (srcSize == 0) {
        return cudaMemsetAsync(offsets_d, 0, sizeof(uint64_t), stream);
    }
    if (src_d == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (context->hostTree.tableLog == 0) {
        cudaError_t err =
                cudaMemsetAsync(totalSize_d, 0, sizeof(uint64_t), stream);
        if (err != cudaSuccess) {
            return err;
        }
        return cudaMemsetAsync(
                offsets_d, 0, sizeof(uint64_t) * (numBlocks + 1), stream);
    }

    if (dst_d == nullptr || workspace_d == nullptr
        || workspaceBytes < workspaceBytesFor(srcSize, blockSize)) {
        return cudaErrorInvalidValue;
    }

    auto* const tree_d = static_cast<PivCoGpuTree*>(workspace_d);
    auto* const blockWorkspace =
            static_cast<uint8_t*>(workspace_d) + kTreeWorkspaceBytes;
    cudaError_t err = cudaMemcpyAsync(
            tree_d,
            &context->hostTree,
            sizeof(context->hostTree),
            cudaMemcpyHostToDevice,
            stream);
    if (err != cudaSuccess) {
        return err;
    }

    if (canUseFastRootConstFlat1Encode(context->hostTree, blockSize)) {
        auto* const blockBitstreams      = blockWorkspace;
        const size_t blockBitstreamBytes = numBlocks * blockSize;
        const size_t numScanChunks =
                (numBlocks + kScanItemsPerBlock - 1) / kScanItemsPerBlock;
        uint64_t* const scanChunkSums = reinterpret_cast<uint64_t*>(
                blockWorkspace
                + alignUpSize(blockBitstreamBytes, alignof(uint64_t)));
        uint64_t* const scanChunkOffsets = scanChunkSums + numScanChunks;
        fastEncodePackRootConstFlat1Kernel<<<
                numBlocks,
                kFastBlockThreads,
                0,
                stream>>>(
                tree_d,
                static_cast<const uint8_t*>(src_d),
                srcSize,
                blockSize,
                blockBitstreams,
                offsets_d,
                status_d);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        if (numScanChunks <= kScanItemsPerBlock) {
            scanBlockSizesKernel<<<
                    numScanChunks,
                    kScanItemsPerBlock,
                    0,
                    stream>>>(offsets_d, numBlocks, scanChunkSums);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return err;
            }
            scanChunkSumsKernel<<<1, kScanItemsPerBlock, 0, stream>>>(
                    scanChunkSums,
                    scanChunkOffsets,
                    numScanChunks,
                    totalSize_d);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return err;
            }
            addChunkOffsetsKernel<<<
                    numScanChunks,
                    kScanItemsPerBlock,
                    0,
                    stream>>>(
                    offsets_d, numBlocks, scanChunkOffsets, totalSize_d);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return err;
            }
        } else {
            scanOffsetsKernel<<<1, 1, 0, stream>>>(
                    offsets_d, numBlocks, totalSize_d, status_d);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return err;
            }
        }

        fastEncodeCopyRootConstFlat1Kernel<<<
                numBlocks,
                kFastBlockThreads,
                0,
                stream>>>(
                static_cast<uint8_t*>(dst_d),
                dstCapacity,
                blockBitstreams,
                srcSize,
                blockSize,
                offsets_d,
                status_d);
        return cudaGetLastError();
    }

    encodeLayoutKernel<<<numBlocks, 1, 0, stream>>>(
            tree_d,
            static_cast<const uint8_t*>(src_d),
            srcSize,
            blockSize,
            blockWorkspace,
            offsets_d,
            status_d);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    scanOffsetsKernel<<<1, 1, 0, stream>>>(
            offsets_d, numBlocks, totalSize_d, status_d);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    encodeEmitKernel<<<numBlocks, 1, 0, stream>>>(
            tree_d,
            static_cast<uint8_t*>(dst_d),
            dstCapacity,
            static_cast<const uint8_t*>(src_d),
            srcSize,
            blockSize,
            blockWorkspace,
            offsets_d,
            status_d);
    return cudaGetLastError();
}

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
        cudaStream_t stream)
{
    if (context == nullptr || offsets_d == nullptr || status_d == nullptr
        || blockSize == 0 || blockSize > PIVCO_GPU_MAX_BLOCK_SIZE
        || addOverflows(dstSize, blockSize - 1)) {
        return cudaErrorInvalidValue;
    }

    const size_t numBlocks = numBlocksFor(dstSize, blockSize);
    if (offsetsCount < numBlocks + 1 || numBlocks > kMaxGridX) {
        return cudaErrorInvalidValue;
    }
    if (dstSize == 0) {
        return bitstreamSize == 0 ? cudaSuccess : cudaErrorInvalidValue;
    }
    if (dst_d == nullptr || workspace_d == nullptr
        || workspaceBytes < decodeWorkspaceBytesFor(dstSize, blockSize)
        || (bitstreamSize != 0 && bitstream_d == nullptr)) {
        return cudaErrorInvalidValue;
    }

    if (context->hostTree.tableLog == 0 && bitstreamSize != 0) {
        return cudaErrorInvalidValue;
    }

    auto* const workspaceBytesPtr = static_cast<uint8_t*>(workspace_d);
    auto* const tree_d     = reinterpret_cast<PivCoGpuTree*>(workspaceBytesPtr);
    auto* const schedule_d = reinterpret_cast<PivCoGpuDecodeSchedule*>(
            workspaceBytesPtr + kTreeWorkspaceBytes);
    auto* const blockWorkspace =
            workspaceBytesPtr + kDecodeStaticWorkspaceBytes;
    cudaError_t err = cudaMemcpyAsync(
            tree_d,
            &context->hostTree,
            sizeof(context->hostTree),
            cudaMemcpyHostToDevice,
            stream);
    if (err != cudaSuccess) {
        return err;
    }

    if (canUseFastRootConstFlat1Decode(context->hostTree, blockSize)) {
        fastDecodeRootConstFlat1Kernel<<<
                numBlocks,
                kFastBlockThreads,
                0,
                stream>>>(
                tree_d,
                static_cast<uint8_t*>(dst_d),
                dstSize,
                static_cast<const uint8_t*>(bitstream_d),
                bitstreamSize,
                offsets_d,
                blockSize,
                status_d);
        return cudaGetLastError();
    }

    if (canUseFastFlatRoot(context->hostTree)) {
        fastDecodeFlatRootKernel<<<numBlocks, kFastBlockThreads, 0, stream>>>(
                tree_d,
                static_cast<uint8_t*>(dst_d),
                dstSize,
                static_cast<const uint8_t*>(bitstream_d),
                bitstreamSize,
                offsets_d,
                blockSize,
                status_d);
        return cudaGetLastError();
    }

    if ((chunkTopDownEnabled()
         || (dstSize <= chunkTdMaxAutoBytesFor(context->hostTree.tableLog)
             && !chunkTopDownDisabled()))
        && canUseScheduledDecode(
                context->hostTree, context->decodeSchedule, blockSize)) {
        err = cudaMemcpyAsync(
                schedule_d,
                &context->decodeSchedule,
                sizeof(context->decodeSchedule),
                cudaMemcpyHostToDevice,
                stream);
        if (err != cudaSuccess) {
            return err;
        }

        scheduledParseKernel<<<numBlocks, 1, 0, stream>>>(
                schedule_d,
                static_cast<const uint8_t*>(bitstream_d),
                bitstreamSize,
                offsets_d,
                dstSize,
                blockSize,
                blockWorkspace,
                status_d);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        // Rank directories (per internal node) enable the O(1) boundary-rank
        // lookups the per-chunk descent needs.
        if (context->decodeSchedule.internalCount != 0) {
            scheduledDirectoryKernel<<<
                    dim3(static_cast<unsigned>(numBlocks),
                         context->decodeSchedule.internalCount),
                    kRankSelectThreads,
                    0,
                    stream>>>(
                    schedule_d,
                    static_cast<const uint8_t*>(bitstream_d),
                    offsets_d,
                    dstSize,
                    blockSize,
                    blockWorkspace,
                    status_d);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return err;
            }
        }

        const uint32_t chunkOutputs   = chunkTdOutputsFor(dstSize);
        const uint32_t chunksPerBlock = static_cast<uint32_t>(
                (blockSize + chunkOutputs - 1) / chunkOutputs);
        const uint32_t gridY =
                (chunksPerBlock + kChunkTdWarps - 1) / kChunkTdWarps;
        const size_t nc             = context->decodeSchedule.nodeCount;
        const size_t levelNodeBytes = (nc * 2u + 15u) & ~size_t{ 15 };
        const size_t perWarpBytes =
                (2u * kChunkTdBufBytesFor(chunkOutputs) + 6u * nc + 15u)
                & ~size_t{ 15 };
        const size_t sharedBytes = levelNodeBytes
                + static_cast<size_t>(kChunkTdWarps) * perWarpBytes;
        cudaFuncSetAttribute(
                pivcoChunkTopDownKernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(sharedBytes));
        pivcoChunkTopDownKernel<<<
                dim3(static_cast<unsigned>(numBlocks), gridY),
                kChunkTdThreads,
                sharedBytes,
                stream>>>(
                tree_d,
                schedule_d,
                static_cast<uint8_t*>(dst_d),
                dstSize,
                static_cast<const uint8_t*>(bitstream_d),
                offsets_d,
                blockSize,
                chunkOutputs,
                blockWorkspace,
                status_d);
        return cudaGetLastError();
    }

    if (canUseScheduledDecode(
                context->hostTree, context->decodeSchedule, blockSize)) {
        err = cudaMemcpyAsync(
                schedule_d,
                &context->decodeSchedule,
                sizeof(context->decodeSchedule),
                cudaMemcpyHostToDevice,
                stream);
        if (err != cudaSuccess) {
            return err;
        }

        // Optional per-kernel microbenchmark: when PIVCO_KERNEL_TIMING is set,
        // each stage is timed in isolation (event record + sync per launch) so
        // its own GPU time -- the "ideal" per-kernel cost with the prior
        // stage's caches warm -- can be compared against the in-pipeline nsys
        // time. This path serializes and is diagnostic only; it is off on the
        // perf path.
        static const bool kernelTiming = [] {
            const char* v = getenv("PIVCO_KERNEL_TIMING");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        cudaEvent_t evStart                      = nullptr;
        cudaEvent_t evStop                       = nullptr;
        double parseMs                           = 0.0;
        double opMs[PIVCO_GPU_SCHEDULE_OP_COUNT] = {};
        if (kernelTiming) {
            cudaEventCreate(&evStart);
            cudaEventCreate(&evStop);
            cudaEventRecord(evStart, stream);
        }

        scheduledParseKernel<<<numBlocks, 1, 0, stream>>>(
                schedule_d,
                static_cast<const uint8_t*>(bitstream_d),
                bitstreamSize,
                offsets_d,
                dstSize,
                blockSize,
                blockWorkspace,
                status_d);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        if (kernelTiming) {
            cudaEventRecord(evStop, stream);
            cudaEventSynchronize(evStop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart, evStop);
            parseMs = ms;
        }

        // The rank directory is built in shared memory at the top of each merge
        // kernel (buildSharedDirectory), so no standalone directory kernel is
        // launched here -- this avoids the global directory write+read and the
        // redundant bitmap re-read.

        for (int level = static_cast<int>(context->decodeSchedule.maxLevel);
             level >= 0;
             --level) {
            for (uint8_t op = 0; op < PIVCO_GPU_SCHEDULE_OP_COUNT; ++op) {
                const size_t stageIndex =
                        scheduleStageIndex(static_cast<uint16_t>(level), op);
                if (kernelTiming
                    && context->decodeSchedule.stageCount[stageIndex] != 0) {
                    cudaEventRecord(evStart, stream);
                }
                err = launchScheduledDecodeStage(
                        op,
                        context->decodeSchedule.stageOffset[stageIndex],
                        context->decodeSchedule.stageCount[stageIndex],
                        numBlocks,
                        tree_d,
                        schedule_d,
                        static_cast<uint8_t*>(dst_d),
                        dstSize,
                        static_cast<const uint8_t*>(bitstream_d),
                        offsets_d,
                        blockSize,
                        blockWorkspace,
                        status_d,
                        stream);
                if (err != cudaSuccess) {
                    return err;
                }
                if (kernelTiming
                    && context->decodeSchedule.stageCount[stageIndex] != 0) {
                    cudaEventRecord(evStop, stream);
                    cudaEventSynchronize(evStop);
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, evStart, evStop);
                    opMs[op] += ms;
                }
            }
        }
        if (kernelTiming) {
            static const char* const kOpNames[PIVCO_GPU_SCHEDULE_OP_COUNT] = {
                "flat1", "flat2", "flat3",   "flat4",   "flat5",   "flat6",
                "flat7", "flat8", "mergeVV", "mergeCV", "mergeVC", "mergeCC"
            };
            fprintf(stderr, "[pivco kernel timing ms] parse=%.4f", parseMs);
            for (uint8_t op = 0; op < PIVCO_GPU_SCHEDULE_OP_COUNT; ++op) {
                if (opMs[op] > 0.0) {
                    fprintf(stderr, " %s=%.4f", kOpNames[op], opMs[op]);
                }
            }
            fprintf(stderr, "\n");
            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);
        }
        return cudaSuccess;
    }

    // No optimized decode path applies. The fast, scheduled-cascade, and
    // chunk-in-shared decoders cover fastMode trees and every schedulable
    // tree at blockSize <= kRankSelectMaxBlockSize (64 KiB); the chunk
    // decoder's per-node metadata is uint16, so larger block sizes are out
    // of range by design.
    return cudaErrorNotSupported;
}

extern "C" ZL_Report pivcoGpuEncode(
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
        ZL_GPU_Stream stream)
{
    PivCoGpuStatus* status_d = nullptr;
    uint64_t* totalSize_d    = nullptr;
    cudaError_t err          = cudaMalloc(&status_d, sizeof(PivCoGpuStatus));
    if (err != cudaSuccess) {
        return cudaReport(err);
    }
    err = cudaMalloc(&totalSize_d, sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaFree(status_d);
        return cudaReport(err);
    }
    err = cudaMemsetAsync(status_d, 0, sizeof(PivCoGpuStatus), stream);
    if (err == cudaSuccess) {
        err = cudaMemsetAsync(totalSize_d, 0, sizeof(uint64_t), stream);
    }
    if (err == cudaSuccess) {
        err = pivcoGpuEncodeAsync(
                context,
                dst_d,
                dstCapacity,
                offsets_d,
                offsetsCapacity,
                src_d,
                srcSize,
                blockSize,
                workspace_d,
                workspaceBytes,
                status_d,
                totalSize_d,
                reinterpret_cast<cudaStream_t>(stream));
    }

    PivCoGpuStatus status{};
    uint64_t totalSize = 0;
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(
                &status,
                status_d,
                sizeof(status),
                cudaMemcpyDeviceToHost,
                reinterpret_cast<cudaStream_t>(stream));
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(
                &totalSize,
                totalSize_d,
                sizeof(totalSize),
                cudaMemcpyDeviceToHost,
                reinterpret_cast<cudaStream_t>(stream));
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    }

    cudaFree(totalSize_d);
    cudaFree(status_d);

    if (err != cudaSuccess) {
        return cudaReport(err);
    }
    const ZL_Report statusResult = statusReport(status);
    if (ZL_isError(statusResult)) {
        return statusResult;
    }
    if (totalSize > SIZE_MAX) {
        return ZL_returnError(ZL_ErrorCode_integerOverflow);
    }
    return ZL_returnValue(static_cast<size_t>(totalSize));
}

extern "C" ZL_Report pivcoGpuDecode(
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
        ZL_GPU_Stream stream)
{
    PivCoGpuStatus* status_d = nullptr;
    cudaError_t err          = cudaMalloc(&status_d, sizeof(PivCoGpuStatus));
    if (err != cudaSuccess) {
        return cudaReport(err);
    }
    err = cudaMemsetAsync(status_d, 0, sizeof(PivCoGpuStatus), stream);
    if (err == cudaSuccess) {
        err = pivcoGpuDecodeAsync(
                context,
                dst_d,
                dstSize,
                bitstream_d,
                bitstreamSize,
                offsets_d,
                offsetsCount,
                blockSize,
                workspace_d,
                workspaceBytes,
                status_d,
                reinterpret_cast<cudaStream_t>(stream));
    }

    PivCoGpuStatus status{};
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(
                &status,
                status_d,
                sizeof(status),
                cudaMemcpyDeviceToHost,
                reinterpret_cast<cudaStream_t>(stream));
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    }

    cudaFree(status_d);

    if (err != cudaSuccess) {
        return cudaReport(err);
    }
    const ZL_Report statusResult = statusReport(status);
    if (ZL_isError(statusResult)) {
        return statusResult;
    }
    return ZL_returnValue(dstSize);
}
