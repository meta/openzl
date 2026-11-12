// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/pivco-huffman/gpu/pivco_gpu.cuh"
#include "contrib/pivco-huffman/gpu/pivco_gpu.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "contrib/pivco-huffman/gpu/pivco_block_index.h"
#include "contrib/pivco-huffman/gpu/pivco_gpu_tree.h"
#include "openzl/codecs/pivco_huffman/common_pivco_kernel.h"
#include "openzl/codecs/pivco_huffman/decode_pivco_kernel.h"
#include "openzl/codecs/pivco_huffman/encode_pivco_kernel.h"
#include "openzl/zl_errors.h"

namespace {

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer()                               = default;
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) {
            // Cast away the result: cudaFree is ignorable, but its HIP
            // counterpart hipFree is [[nodiscard]], so an ignored return trips
            // -Werror on the AMD build.
            (void)cudaFree(ptr_);
        }
    }

    cudaError_t reset(size_t count)
    {
        if (ptr_ != nullptr) {
            // Cast away the result: cudaFree is ignorable, but its HIP
            // counterpart hipFree is [[nodiscard]], so an ignored return trips
            // -Werror on the AMD build.
            (void)cudaFree(ptr_);
        }
        ptr_   = nullptr;
        count_ = count;
        if (count == 0) {
            return cudaSuccess;
        }
        return cudaMalloc(&ptr_, count * sizeof(T));
    }

    T* get() const
    {
        return ptr_;
    }

    size_t size() const
    {
        return count_;
    }

   private:
    T* ptr_{ nullptr };
    size_t count_{ 0 };
};

// Declare after buffers/context so queued work completes before their release.
struct StreamEventHandles {
    cudaStream_t streams[2]{};
    cudaEvent_t events[2]{};
    ~StreamEventHandles()
    {
        for (unsigned i = 0; i < 2; ++i) {
            if (streams[i]) {
                (void)cudaStreamSynchronize(streams[i]);
            }
            if (events[i]) {
                (void)cudaEventDestroy(events[i]);
            }
            if (streams[i]) {
                (void)cudaStreamDestroy(streams[i]);
            }
        }
    }
};

struct Encoded {
    std::vector<uint8_t> bytes;
    std::vector<uint64_t> offsets;
};

Encoded cpuEncode(
        const std::vector<uint8_t>& weights,
        const std::vector<uint8_t>& data,
        size_t blockSize)
{
    const int tableLog =
            ZL_PivCoHuffman_computeTableLog(weights.data(), weights.size());
    EXPECT_GE(tableLog, 0);
    const size_t bound = ZL_PivCoHuffmanEncode_bound(data.size(), blockSize);
    EXPECT_NE(bound, SIZE_MAX);

    std::vector<uint8_t> encoded(bound);
    std::vector<uint8_t> scratch(
            ZL_PivCoHuffmanEncode_scratchElements(data.size(), blockSize));
    const size_t encodedSize = ZL_PivCoHuffman_encode(
            encoded.data(),
            encoded.size(),
            scratch.data(),
            scratch.size(),
            weights.data(),
            weights.size(),
            tableLog,
            data.data(),
            data.size(),
            blockSize,
            &ZL_PivCoHuffmanEncode_generic);
    EXPECT_NE(encodedSize, SIZE_MAX);
    encoded.resize(encodedSize);

    const size_t numBlocks =
            data.empty() ? 0 : (data.size() + blockSize - 1) / blockSize;
    std::vector<uint64_t> offsets(numBlocks + 1);
    const ZL_Report indexReport = pivcoFindBlockOffsets(
            offsets.data(),
            offsets.size(),
            weights.data(),
            weights.size(),
            encoded.data(),
            encoded.size(),
            data.size(),
            blockSize);
    EXPECT_FALSE(ZL_isError(indexReport));
    return Encoded{ std::move(encoded), std::move(offsets) };
}

std::vector<uint8_t> cpuDecode(
        const std::vector<uint8_t>& weights,
        const std::vector<uint8_t>& encoded,
        size_t decodedSize,
        size_t blockSize)
{
    std::vector<uint8_t> decoded(decodedSize);
    std::vector<uint8_t> scratch(
            ZL_PivCoHuffmanDecode_scratchBytes(decodedSize, blockSize));
    EXPECT_TRUE(ZL_PivCoHuffman_decode(
            decoded.data(),
            decoded.size(),
            scratch.data(),
            scratch.size(),
            weights.data(),
            weights.size(),
            encoded.data(),
            encoded.size(),
            blockSize,
            &ZL_PivCoHuffmanDecode_generic));
    return decoded;
}

PivCoGpuContext* createContext(const std::vector<uint8_t>& weights)
{
    PivCoGpuContext* context = nullptr;
    const int tableLog =
            ZL_PivCoHuffman_computeTableLog(weights.data(), weights.size());
    const ZL_Report report = pivcoGpuContextCreate(
            &context, weights.data(), weights.size(), tableLog);
    EXPECT_FALSE(ZL_isError(report));
    return context;
}

void copyToDevice(DeviceBuffer<uint8_t>& dst, const std::vector<uint8_t>& src);
void copyToDevice(
        DeviceBuffer<uint64_t>& dst,
        const std::vector<uint64_t>& src);
std::vector<uint8_t> copyBytesFromDevice(DeviceBuffer<uint8_t>& src, size_t n);

std::vector<uint8_t> makeThreeSymbolData(size_t size)
{
    std::vector<uint8_t> data(size);
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (uint8_t& value : data) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const uint8_t r = static_cast<uint8_t>(state);
        value           = r < 160 ? 0 : (r < 220 ? 1 : 2);
    }
    return data;
}

std::vector<uint8_t> makeFlatRootData(size_t size)
{
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 5 + i / 7) & 7);
    }
    return data;
}

std::vector<uint8_t> makeRankSelectData(size_t size)
{
    std::vector<uint8_t> data(size);
    uint64_t state = 0xD1B54A32D192ED03ull;
    for (uint8_t& value : data) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const uint8_t sample = static_cast<uint8_t>(state);
        value = sample < 128 ? 0 : (sample < 192 ? 1 : (sample < 224 ? 2 : 3));
    }
    return data;
}

std::vector<uint8_t> makeDataForWeights(
        const std::vector<uint8_t>& weights,
        size_t size)
{
    std::vector<uint8_t> symbols;
    for (size_t symbol = 0; symbol < weights.size(); ++symbol) {
        if (weights[symbol] != 0) {
            symbols.push_back(static_cast<uint8_t>(symbol));
        }
    }
    EXPECT_FALSE(symbols.empty());

    std::vector<uint8_t> data;
    data.reserve(size);
    for (uint8_t symbol : symbols) {
        if (data.size() == size) {
            return data;
        }
        data.push_back(symbol);
    }
    uint64_t state = 0xA0761D6478BD642Full;
    while (data.size() < size) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        data.push_back(symbols[state % symbols.size()]);
    }
    return data;
}

uint32_t scheduleOpBit(uint8_t op)
{
    return uint32_t{ 1 } << op;
}

uint32_t flatOpMask()
{
    uint32_t mask = 0;
    for (uint8_t op = PIVCO_GPU_SCHEDULE_OP_FLAT1;
         op <= PIVCO_GPU_SCHEDULE_OP_FLAT8;
         ++op) {
        mask |= scheduleOpBit(op);
    }
    return mask;
}

uint32_t requiredMergeOpMask()
{
    return scheduleOpBit(PIVCO_GPU_SCHEDULE_OP_MERGE_VECTOR_VECTOR)
            | scheduleOpBit(PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_VECTOR)
            | scheduleOpBit(PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_CONSTANT);
}

bool coverageComplete(uint32_t opMask)
{
    return (opMask & flatOpMask()) != 0
            && (opMask & requiredMergeOpMask()) == requiredMergeOpMask();
}

uint32_t scheduleOpsFor(const PivCoGpuContext* context)
{
    uint32_t mask                          = 0;
    const PivCoGpuDecodeSchedule& schedule = context->decodeSchedule;
    EXPECT_NE(schedule.enabled, 0);
    for (size_t level = 0; level <= schedule.maxLevel; ++level) {
        for (uint8_t op = 0; op < PIVCO_GPU_SCHEDULE_OP_COUNT; ++op) {
            const size_t index = level * PIVCO_GPU_SCHEDULE_OP_COUNT + op;
            if (schedule.stageCount[index] != 0) {
                mask |= scheduleOpBit(op);
            }
        }
    }
    return mask;
}

void collectScheduledCoverageWeights(
        int tableLog,
        int weight,
        int remaining,
        size_t symbols,
        std::vector<int>& counts,
        uint32_t* opMask,
        std::vector<std::vector<uint8_t>>* weightCases)
{
    if (coverageComplete(*opMask) || symbols > 32) {
        return;
    }
    if (weight == 0) {
        if (remaining != 0 || symbols < 2) {
            return;
        }

        std::vector<uint8_t> weights;
        weights.reserve(symbols);
        for (int w = tableLog + 1; w >= 1; --w) {
            for (int i = 0; i < counts[w]; ++i) {
                weights.push_back(static_cast<uint8_t>(w));
            }
        }

        const int computedTableLog =
                ZL_PivCoHuffman_computeTableLog(weights.data(), weights.size());
        if (computedTableLog != tableLog) {
            return;
        }

        PivCoGpuContext* context = nullptr;
        const ZL_Report report   = pivcoGpuContextCreate(
                &context, weights.data(), weights.size(), tableLog);
        if (ZL_isError(report) || context == nullptr) {
            return;
        }
        if (context->hostTree.fastMode == PIVCO_GPU_FAST_NONE
            && context->decodeSchedule.enabled != 0) {
            const uint32_t caseMask = scheduleOpsFor(context)
                    & (flatOpMask() | requiredMergeOpMask());
            if (((*opMask) | caseMask) != *opMask) {
                *opMask |= caseMask;
                weightCases->push_back(std::move(weights));
            }
        }
        pivcoGpuContextDestroy(context);
        return;
    }

    const int value = 1 << (weight - 1);
    const int maxCount =
            std::min<int>(remaining / value, static_cast<int>(32 - symbols));
    for (int count = 0; count <= maxCount; ++count) {
        counts[weight] = count;
        collectScheduledCoverageWeights(
                tableLog,
                weight - 1,
                remaining - count * value,
                symbols + count,
                counts,
                opMask,
                weightCases);
        if (coverageComplete(*opMask)) {
            break;
        }
    }
    counts[weight] = 0;
}

std::vector<std::vector<uint8_t>> findScheduledCoverageWeights()
{
    uint32_t opMask = 0;
    std::vector<std::vector<uint8_t>> weightCases;
    for (int tableLog = 1; tableLog <= 8 && !coverageComplete(opMask);
         ++tableLog) {
        std::vector<int> counts(tableLog + 2);
        collectScheduledCoverageWeights(
                tableLog,
                tableLog + 1,
                1 << tableLog,
                0,
                counts,
                &opMask,
                &weightCases);
    }
    return weightCases;
}

void expectGpuDecodeMatchesCpuEncode(
        const std::vector<uint8_t>& weights,
        const std::vector<uint8_t>& data,
        size_t blockSize)
{
    const Encoded encoded = cpuEncode(weights, data, blockSize);

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->hostTree.fastMode, PIVCO_GPU_FAST_NONE);
    ASSERT_NE(context->decodeSchedule.enabled, 0);

    DeviceBuffer<uint8_t> bitstream;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> decoded;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(bitstream, encoded.bytes);
    copyToDevice(offsets, encoded.offsets);
    ASSERT_EQ(
            cudaSuccess,
            decoded.reset(data.size() + PIVCO_GPU_DECODE_DST_SLOP));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuDecodeWorkspaceBytes(data.size(), blockSize)));

    const ZL_Report decodeReport = pivcoGpuDecode(
            context,
            decoded.get(),
            data.size(),
            bitstream.get(),
            encoded.bytes.size(),
            offsets.get(),
            offsets.size(),
            blockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(decodeReport));
    EXPECT_EQ(ZL_validResult(decodeReport), data.size());
    EXPECT_EQ(copyBytesFromDevice(decoded, data.size()), data);

    pivcoGpuContextDestroy(context);
}

void copyToDevice(DeviceBuffer<uint8_t>& dst, const std::vector<uint8_t>& src)
{
    // Reserve PIVCO_GPU_DECODE_SRC_SLOP trailing bytes so the decoder's
    // loop-free bitmap over-read stays in bounds when this buffer holds the
    // compressed input (harmless for other byte buffers).
    ASSERT_EQ(cudaSuccess, dst.reset(src.size() + PIVCO_GPU_DECODE_SRC_SLOP));
    if (!src.empty()) {
        ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(
                        dst.get(),
                        src.data(),
                        src.size(),
                        cudaMemcpyHostToDevice));
    }
}

void copyToDevice(DeviceBuffer<uint64_t>& dst, const std::vector<uint64_t>& src)
{
    ASSERT_EQ(cudaSuccess, dst.reset(src.size()));
    if (!src.empty()) {
        ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(
                        dst.get(),
                        src.data(),
                        src.size() * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
    }
}

std::vector<uint8_t> copyBytesFromDevice(DeviceBuffer<uint8_t>& src, size_t n)
{
    std::vector<uint8_t> out(n);
    if (n != 0) {
        EXPECT_EQ(
                cudaSuccess,
                cudaMemcpy(out.data(), src.get(), n, cudaMemcpyDeviceToHost));
    }
    return out;
}

std::vector<uint64_t> copyOffsetsFromDevice(
        DeviceBuffer<uint64_t>& src,
        size_t n)
{
    std::vector<uint64_t> out(n);
    if (n != 0) {
        EXPECT_EQ(
                cudaSuccess,
                cudaMemcpy(
                        out.data(),
                        src.get(),
                        n * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
    }
    return out;
}

TEST(PivCoGpuTest, CpuEncodeGpuDecode)
{
    const std::vector<uint8_t> weights{ 2, 1, 1 };
    const std::vector<uint8_t> data{ 0, 1, 2, 0, 2, 1, 0, 0, 2,
                                     1, 1, 0, 2, 0, 1, 2, 2 };
    constexpr size_t kBlockSize = 5;
    const Encoded encoded       = cpuEncode(weights, data, kBlockSize);

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> bitstream;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> decoded;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(bitstream, encoded.bytes);
    copyToDevice(offsets, encoded.offsets);
    ASSERT_EQ(
            cudaSuccess,
            decoded.reset(data.size() + PIVCO_GPU_DECODE_DST_SLOP));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuDecodeWorkspaceBytes(data.size(), kBlockSize)));

    const ZL_Report report = pivcoGpuDecode(
            context,
            decoded.get(),
            data.size(),
            bitstream.get(),
            encoded.bytes.size(),
            offsets.get(),
            offsets.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(report));
    EXPECT_EQ(ZL_validResult(report), data.size());
    EXPECT_EQ(copyBytesFromDevice(decoded, data.size()), data);

    pivcoGpuContextDestroy(context);
}

TEST(PivCoGpuTest, GpuEncodeMatchesCpuBytesAndOffsets)
{
    const std::vector<uint8_t> weights{ 2, 1, 1 };
    const std::vector<uint8_t> data{ 0, 1, 2, 0, 2, 1, 0, 0, 2,
                                     1, 1, 0, 2, 0, 1, 2, 2 };
    constexpr size_t kBlockSize = 7;
    const Encoded expected      = cpuEncode(weights, data, kBlockSize);

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> src;
    DeviceBuffer<uint8_t> encoded;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(src, data);
    ASSERT_EQ(
            cudaSuccess,
            encoded.reset(
                    ZL_PivCoHuffmanEncode_bound(data.size(), kBlockSize)));
    ASSERT_EQ(cudaSuccess, offsets.reset(expected.offsets.size()));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuEncodeWorkspaceBytes(data.size(), kBlockSize)));

    const ZL_Report report = pivcoGpuEncode(
            context,
            encoded.get(),
            encoded.size(),
            offsets.get(),
            offsets.size(),
            src.get(),
            data.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);

    ASSERT_FALSE(ZL_isError(report));
    ASSERT_EQ(ZL_validResult(report), expected.bytes.size());
    EXPECT_EQ(
            copyBytesFromDevice(encoded, expected.bytes.size()),
            expected.bytes);
    EXPECT_EQ(
            copyOffsetsFromDevice(offsets, expected.offsets.size()),
            expected.offsets);
    EXPECT_EQ(
            cpuDecode(
                    weights,
                    copyBytesFromDevice(encoded, expected.bytes.size()),
                    data.size(),
                    kBlockSize),
            data);

    pivcoGpuContextDestroy(context);
}

TEST(PivCoGpuTest, GpuEncodeGpuDecodeRoundTrip)
{
    const std::vector<uint8_t> weights{ 2, 1, 1 };
    const std::vector<uint8_t> data{ 0, 0, 1, 2, 0, 1, 0, 2,
                                     2, 1, 0, 0, 2, 1, 2, 0 };
    constexpr size_t kBlockSize = 6;

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> src;
    DeviceBuffer<uint8_t> encoded;
    DeviceBuffer<uint8_t> decoded;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(src, data);
    ASSERT_EQ(
            cudaSuccess,
            encoded.reset(
                    ZL_PivCoHuffmanEncode_bound(data.size(), kBlockSize)));
    ASSERT_EQ(
            cudaSuccess,
            offsets.reset((data.size() + kBlockSize - 1) / kBlockSize + 1));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuEncodeWorkspaceBytes(data.size(), kBlockSize)));

    const ZL_Report encodeReport = pivcoGpuEncode(
            context,
            encoded.get(),
            encoded.size(),
            offsets.get(),
            offsets.size(),
            src.get(),
            data.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(encodeReport));

    ASSERT_EQ(
            cudaSuccess,
            decoded.reset(data.size() + PIVCO_GPU_DECODE_DST_SLOP));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuDecodeWorkspaceBytes(data.size(), kBlockSize)));
    const ZL_Report decodeReport = pivcoGpuDecode(
            context,
            decoded.get(),
            data.size(),
            encoded.get(),
            ZL_validResult(encodeReport),
            offsets.get(),
            offsets.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(decodeReport));
    EXPECT_EQ(ZL_validResult(decodeReport), data.size());
    EXPECT_EQ(copyBytesFromDevice(decoded, data.size()), data);

    pivcoGpuContextDestroy(context);
}

TEST(PivCoGpuTest, FastGpuEncodeMatchesCpuBytesAndOffsets)
{
    const std::vector<uint8_t> weights{ 2, 1, 1 };
    constexpr size_t kBlockSize     = 32 * 1024;
    const std::vector<uint8_t> data = makeThreeSymbolData(3 * kBlockSize + 17);
    const Encoded expected          = cpuEncode(weights, data, kBlockSize);

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> src;
    DeviceBuffer<uint8_t> encoded;
    DeviceBuffer<uint8_t> decoded;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(src, data);
    ASSERT_EQ(
            cudaSuccess,
            encoded.reset(
                    ZL_PivCoHuffmanEncode_bound(data.size(), kBlockSize)));
    ASSERT_EQ(cudaSuccess, offsets.reset(expected.offsets.size()));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuEncodeWorkspaceBytes(data.size(), kBlockSize)));

    const ZL_Report encodeReport = pivcoGpuEncode(
            context,
            encoded.get(),
            encoded.size(),
            offsets.get(),
            offsets.size(),
            src.get(),
            data.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);

    ASSERT_FALSE(ZL_isError(encodeReport));
    ASSERT_EQ(ZL_validResult(encodeReport), expected.bytes.size());
    EXPECT_EQ(
            copyBytesFromDevice(encoded, expected.bytes.size()),
            expected.bytes);
    EXPECT_EQ(
            copyOffsetsFromDevice(offsets, expected.offsets.size()),
            expected.offsets);

    ASSERT_EQ(
            cudaSuccess,
            decoded.reset(data.size() + PIVCO_GPU_DECODE_DST_SLOP));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuDecodeWorkspaceBytes(data.size(), kBlockSize)));
    const ZL_Report decodeReport = pivcoGpuDecode(
            context,
            decoded.get(),
            data.size(),
            encoded.get(),
            ZL_validResult(encodeReport),
            offsets.get(),
            offsets.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(decodeReport));
    EXPECT_EQ(ZL_validResult(decodeReport), data.size());
    EXPECT_EQ(copyBytesFromDevice(decoded, data.size()), data);

    pivcoGpuContextDestroy(context);
}

TEST(PivCoGpuTest, FlatRootGpuDecodeMatchesCpuEncode)
{
    const std::vector<uint8_t> weights{ 1, 1, 1, 1, 1, 1, 1, 1 };
    constexpr size_t kBlockSize     = 32 * 1024;
    const std::vector<uint8_t> data = makeFlatRootData(3 * kBlockSize + 11);
    const Encoded encoded           = cpuEncode(weights, data, kBlockSize);

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> bitstream;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> decoded;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(bitstream, encoded.bytes);
    copyToDevice(offsets, encoded.offsets);
    ASSERT_EQ(
            cudaSuccess,
            decoded.reset(data.size() + PIVCO_GPU_DECODE_DST_SLOP));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuDecodeWorkspaceBytes(data.size(), kBlockSize)));

    const ZL_Report decodeReport = pivcoGpuDecode(
            context,
            decoded.get(),
            data.size(),
            bitstream.get(),
            encoded.bytes.size(),
            offsets.get(),
            offsets.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(decodeReport));
    EXPECT_EQ(ZL_validResult(decodeReport), data.size());
    EXPECT_EQ(copyBytesFromDevice(decoded, data.size()), data);

    pivcoGpuContextDestroy(context);
}

TEST(PivCoGpuTest, RankSelectGpuDecodeMatchesCpuEncode)
{
    const std::vector<uint8_t> weights{ 3, 2, 1, 1 };
    constexpr size_t kBlockSize     = 32 * 1024;
    const std::vector<uint8_t> data = makeRankSelectData(2 * kBlockSize + 19);
    const Encoded encoded           = cpuEncode(weights, data, kBlockSize);

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> bitstream;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> decoded;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(bitstream, encoded.bytes);
    copyToDevice(offsets, encoded.offsets);
    ASSERT_EQ(
            cudaSuccess,
            decoded.reset(data.size() + PIVCO_GPU_DECODE_DST_SLOP));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuDecodeWorkspaceBytes(data.size(), kBlockSize)));

    const ZL_Report decodeReport = pivcoGpuDecode(
            context,
            decoded.get(),
            data.size(),
            bitstream.get(),
            encoded.bytes.size(),
            offsets.get(),
            offsets.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(decodeReport));
    EXPECT_EQ(ZL_validResult(decodeReport), data.size());
    EXPECT_EQ(copyBytesFromDevice(decoded, data.size()), data);

    pivcoGpuContextDestroy(context);
}

TEST(PivCoGpuTest, ScheduledGpuDecodeCoversMixedNodeKinds)
{
    const std::vector<std::vector<uint8_t>> weightCases =
            findScheduledCoverageWeights();
    constexpr size_t kBlockSize = 32 * 1024;

    uint32_t opMask = 0;
    for (const std::vector<uint8_t>& weights : weightCases) {
        PivCoGpuContext* context = createContext(weights);
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->hostTree.fastMode, PIVCO_GPU_FAST_NONE);
        opMask |= scheduleOpsFor(context);
        pivcoGpuContextDestroy(context);

        expectGpuDecodeMatchesCpuEncode(
                weights,
                makeDataForWeights(weights, 2 * kBlockSize + 29),
                kBlockSize);
    }

    EXPECT_NE(opMask & flatOpMask(), 0);
    EXPECT_NE(
            opMask & scheduleOpBit(PIVCO_GPU_SCHEDULE_OP_MERGE_VECTOR_VECTOR),
            0);
    EXPECT_NE(
            opMask & scheduleOpBit(PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_VECTOR),
            0);
    EXPECT_NE(
            opMask
                    & scheduleOpBit(
                            PIVCO_GPU_SCHEDULE_OP_MERGE_CONSTANT_CONSTANT),
            0);
}

TEST(PivCoGpuTest, ConstantEncodeWritesRepeatedZeroOffsets)
{
    std::vector<uint8_t> weights(6);
    weights[5] = 1;
    const std::vector<uint8_t> data(17, 5);
    constexpr size_t kBlockSize = 5;
    const size_t numOffsets = (data.size() + kBlockSize - 1) / kBlockSize + 1;

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> src;
    DeviceBuffer<uint8_t> encoded;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(src, data);
    ASSERT_EQ(cudaSuccess, encoded.reset(1));
    ASSERT_EQ(cudaSuccess, offsets.reset(numOffsets));

    const ZL_Report report = pivcoGpuEncode(
            context,
            encoded.get(),
            encoded.size(),
            offsets.get(),
            offsets.size(),
            src.get(),
            data.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(report));
    EXPECT_EQ(ZL_validResult(report), 0);
    EXPECT_EQ(
            copyOffsetsFromDevice(offsets, numOffsets),
            std::vector<uint64_t>(numOffsets, 0));

    pivcoGpuContextDestroy(context);
}

TEST(PivCoGpuTest, GpuEncodeRejectsMissingSymbol)
{
    const std::vector<uint8_t> weights{ 1, 1 };
    const std::vector<uint8_t> data{ 0, 2 };
    constexpr size_t kBlockSize = 2;

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> src;
    DeviceBuffer<uint8_t> encoded;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(src, data);
    ASSERT_EQ(cudaSuccess, encoded.reset(16));
    ASSERT_EQ(cudaSuccess, offsets.reset(2));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuEncodeWorkspaceBytes(data.size(), kBlockSize)));

    const ZL_Report report = pivcoGpuEncode(
            context,
            encoded.get(),
            encoded.size(),
            offsets.get(),
            offsets.size(),
            src.get(),
            data.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    EXPECT_TRUE(ZL_isError(report));

    pivcoGpuContextDestroy(context);
}

TEST(PivCoGpuTest, FastGpuEncodeRejectsMissingSymbol)
{
    const std::vector<uint8_t> weights{ 2, 1, 1 };
    constexpr size_t kBlockSize = 1024;
    std::vector<uint8_t> data   = makeThreeSymbolData(2 * kBlockSize);
    data[kBlockSize + 11]       = 7;

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> src;
    DeviceBuffer<uint8_t> encoded;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(src, data);
    ASSERT_EQ(
            cudaSuccess,
            encoded.reset(
                    ZL_PivCoHuffmanEncode_bound(data.size(), kBlockSize)));
    ASSERT_EQ(
            cudaSuccess,
            offsets.reset((data.size() + kBlockSize - 1) / kBlockSize + 1));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuEncodeWorkspaceBytes(data.size(), kBlockSize)));

    const ZL_Report report = pivcoGpuEncode(
            context,
            encoded.get(),
            encoded.size(),
            offsets.get(),
            offsets.size(),
            src.get(),
            data.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    EXPECT_TRUE(ZL_isError(report));

    pivcoGpuContextDestroy(context);
}

TEST(PivCoGpuTest, GpuDecodeRejectsBadOffsets)
{
    const std::vector<uint8_t> weights{ 1, 1 };
    const std::vector<uint8_t> data{ 0, 1, 1, 0, 1 };
    constexpr size_t kBlockSize = 5;
    Encoded encoded             = cpuEncode(weights, data, kBlockSize);
    encoded.offsets[1]          = encoded.bytes.size() + 1;

    PivCoGpuContext* context = createContext(weights);
    ASSERT_NE(context, nullptr);

    DeviceBuffer<uint8_t> bitstream;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<uint8_t> decoded;
    DeviceBuffer<uint8_t> workspace;
    copyToDevice(bitstream, encoded.bytes);
    copyToDevice(offsets, encoded.offsets);
    ASSERT_EQ(
            cudaSuccess,
            decoded.reset(data.size() + PIVCO_GPU_DECODE_DST_SLOP));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuDecodeWorkspaceBytes(data.size(), kBlockSize)));

    const ZL_Report report = pivcoGpuDecode(
            context,
            decoded.get(),
            data.size(),
            bitstream.get(),
            encoded.bytes.size(),
            offsets.get(),
            offsets.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    EXPECT_TRUE(ZL_isError(report));

    pivcoGpuContextDestroy(context);
}

size_t legacyEncodeWorkspaceBytes(size_t size, size_t blockSize)
{
    const size_t blocks = (size + blockSize - 1) / blockSize;
    return ((sizeof(PivCoGpuTree) + 7) / 8) * 8 + 2 * blocks * blockSize;
}

void checkCooperativeEncoding(
        const std::vector<uint8_t>& weights,
        const std::vector<uint8_t>& data,
        bool legacyWorkspace  = false,
        bool disableSchedule  = false,
        size_t blockSize      = 64 * 1024,
        bool boundedWorkspace = false)
{
    const size_t kBlockSize = blockSize;
    std::ostringstream weightTrace;
    for (size_t symbol = 0; symbol < weights.size(); ++symbol) {
        if (weights[symbol]) {
            weightTrace << symbol << ':' << unsigned(weights[symbol]) << ',';
        }
    }
    SCOPED_TRACE(
            ::testing::Message() << "inputBytes=" << data.size()
                                 << " legacyWorkspace=" << legacyWorkspace
                                 << " blockSize=" << blockSize
                                 << " boundedWorkspace=" << boundedWorkspace
                                 << " disableSchedule=" << disableSchedule
                                 << " weights=" << weightTrace.str());
    const Encoded expected = cpuEncode(weights, data, kBlockSize);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    SCOPED_TRACE(
            ::testing::Message()
            << "tableLog=" << context->hostTree.tableLog
            << " fastMode=" << unsigned(context->hostTree.fastMode)
            << " scheduleNodes=" << context->decodeSchedule.nodeCount);
    if (disableSchedule) {
        context->decodeSchedule.enabled = 0;
    }
    DeviceBuffer<uint8_t> src, encoded, decoded, workspace;
    DeviceBuffer<uint64_t> offsets;
    copyToDevice(src, data);
    ASSERT_EQ(
            cudaSuccess,
            encoded.reset(
                    ZL_PivCoHuffmanEncode_bound(data.size(), kBlockSize)
                    + PIVCO_GPU_DECODE_SRC_SLOP));
    ASSERT_EQ(cudaSuccess, offsets.reset(expected.offsets.size()));
    const size_t workspaceBytes = legacyWorkspace
            ? legacyEncodeWorkspaceBytes(data.size(), kBlockSize)
            : pivcoGpuEncodeWorkspaceBytes(
                      boundedWorkspace && data.size() > 256 * kBlockSize
                              ? 256 * kBlockSize
                              : data.size(),
                      kBlockSize);
    ASSERT_EQ(cudaSuccess, workspace.reset(workspaceBytes));
    const ZL_Report report = pivcoGpuEncode(
            context.get(),
            encoded.get(),
            encoded.size() - PIVCO_GPU_DECODE_SRC_SLOP,
            offsets.get(),
            offsets.size(),
            src.get(),
            data.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(report));
    ASSERT_EQ(ZL_validResult(report), expected.bytes.size());
    const auto actual = copyBytesFromDevice(encoded, expected.bytes.size());
    EXPECT_EQ(actual, expected.bytes);
    EXPECT_EQ(
            copyOffsetsFromDevice(offsets, expected.offsets.size()),
            expected.offsets);
    EXPECT_EQ(cpuDecode(weights, actual, data.size(), kBlockSize), data);
    if (!disableSchedule) {
        ASSERT_EQ(
                cudaSuccess,
                decoded.reset(data.size() + PIVCO_GPU_DECODE_DST_SLOP));
        ASSERT_EQ(
                cudaSuccess,
                workspace.reset(
                        pivcoGpuDecodeWorkspaceBytes(data.size(), kBlockSize)));
        const ZL_Report decodedReport = pivcoGpuDecode(
                context.get(),
                decoded.get(),
                data.size(),
                encoded.get(),
                actual.size(),
                offsets.get(),
                offsets.size(),
                kBlockSize,
                workspace.get(),
                workspace.size(),
                nullptr);
        ASSERT_FALSE(ZL_isError(decodedReport));
        EXPECT_EQ(copyBytesFromDevice(decoded, data.size()), data);
    }
}

TEST(PivCoGpuTest, CooperativeEncodeMatchesCpuWireAndGpuDecode)
{
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    for (size_t bytes : { size_t{ 1 },
                          size_t{ 33 },
                          size_t{ 65535 },
                          size_t{ 65536 },
                          size_t{ 65537 },
                          size_t{ 262161 } }) {
        SCOPED_TRACE(bytes);
        checkCooperativeEncoding(weights, makeDataForWeights(weights, bytes));
    }
    // Flat roots and sparse high symbol values also use native rank order.
    const std::vector<uint8_t> flat(256, 1);
    checkCooperativeEncoding(flat, makeDataForWeights(flat, 65536 + 17));
    std::vector<uint8_t> sparse(256);
    sparse[5]   = 4;
    sparse[128] = 3;
    sparse[200] = 2;
    sparse[254] = 1;
    sparse[255] = 1;
    checkCooperativeEncoding(sparse, makeDataForWeights(sparse, 65536 + 31));
    // Zero-count subtrees still have the same alignment/count fields.
    checkCooperativeEncoding(weights, std::vector<uint8_t>(65536 + 65, 4));
}

TEST(PivCoGpuTest, CooperativeEncodeMixedNodesAndDeepTree)
{
    for (const auto& weights : findScheduledCoverageWeights()) {
        checkCooperativeEncoding(
                weights, makeDataForWeights(weights, 2 * 65536 + 19));
    }
    const std::vector<uint8_t> deep{ 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1 };
    checkCooperativeEncoding(deep, makeDataForWeights(deep, 65536 + 3));
}

TEST(PivCoGpuTest, CooperativeEncodeMaximumGridAndLegacyFallback)
{
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    checkCooperativeEncoding(weights, makeDataForWeights(weights, 256 * 65536));
    checkCooperativeEncoding(
            weights, makeDataForWeights(weights, 65536 + 7), true);
    // A disabled/new schedule shape falls back to the existing tree encoder.
    checkCooperativeEncoding(
            weights, makeDataForWeights(weights, 65536 + 7), false, true);
}

#if !defined(__HIPCC__)
TEST(PivCoGpuTest, CooperativeEncodeMultipleBatchesAndBlockSizes)
{
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    for (size_t blockSize : { size_t{ 32768 }, size_t{ 65536 } }) {
        for (size_t bytes : { blockSize - 1,
                              blockSize,
                              blockSize + 1,
                              256 * blockSize + 1,
                              513 * blockSize + 17 }) {
            // A generic skewed tree crosses both the 256-block boundary and
            // two scratch-reuse boundaries, using only one batch of scratch.
            checkCooperativeEncoding(
                    weights,
                    makeDataForWeights(weights, bytes),
                    false,
                    false,
                    blockSize,
                    true);
        }
    }
}

TEST(PivCoGpuTest, CooperativeEncodeLateBatchFailureAndRecovery)
{
    constexpr size_t kGuardBytes = 64;
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    for (size_t blockSize : { size_t{ 32768 }, size_t{ 65536 } }) {
        SCOPED_TRACE(blockSize);
        auto data = makeDataForWeights(weights, 256 * blockSize + 17);
        const Encoded expected = cpuEncode(weights, data, blockSize);
        std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)>
                context(createContext(weights), pivcoGpuContextDestroy);
        ASSERT_NE(context.get(), nullptr);
        DeviceBuffer<uint8_t> src, encoded, workspace;
        DeviceBuffer<uint64_t> offsets;
        copyToDevice(src, data);
        ASSERT_EQ(
                cudaSuccess,
                encoded.reset(expected.bytes.size() + kGuardBytes));
        ASSERT_EQ(cudaSuccess, offsets.reset(expected.offsets.size()));
        ASSERT_EQ(
                cudaSuccess,
                workspace.reset(pivcoGpuEncodeWorkspaceBytes(
                        256 * blockSize, blockSize)));
        const auto encode = [&](size_t capacity) {
            return pivcoGpuEncode(
                    context.get(),
                    encoded.get(),
                    capacity,
                    offsets.get(),
                    offsets.size(),
                    src.get(),
                    data.size(),
                    blockSize,
                    workspace.get(),
                    workspace.size(),
                    nullptr);
        };
        const auto expectGuard = [&](size_t begin) {
            const auto actual = copyBytesFromDevice(encoded, encoded.size());
            EXPECT_EQ(
                    std::vector<uint8_t>(actual.begin() + begin, actual.end()),
                    std::vector<uint8_t>(actual.size() - begin, 0xa5));
        };
        ASSERT_EQ(cudaSuccess, cudaMemset(encoded.get(), 0xa5, encoded.size()));
        EXPECT_TRUE(ZL_isError(encode(expected.bytes.size() - 1)));
        expectGuard(expected.bytes.size() - 1);
        // Failure output is unspecified; the next call must reset all state.
        auto report = encode(expected.bytes.size());
        ASSERT_FALSE(ZL_isError(report));
        EXPECT_EQ(ZL_validResult(report), expected.bytes.size());
        EXPECT_EQ(
                copyBytesFromDevice(encoded, expected.bytes.size()),
                expected.bytes);
        EXPECT_EQ(
                copyOffsetsFromDevice(offsets, expected.offsets.size()),
                expected.offsets);
        expectGuard(expected.bytes.size());
        // A missing symbol in the second batch must fail without overrunning
        // the destination, and the same scratch must remain reusable.
        data.back() = 255;
        copyToDevice(src, data);
        EXPECT_TRUE(ZL_isError(encode(expected.bytes.size())));
        expectGuard(expected.bytes.size());
        data = makeDataForWeights(weights, data.size());
        copyToDevice(src, data);
        report = encode(expected.bytes.size());
        ASSERT_FALSE(ZL_isError(report));
        EXPECT_EQ(ZL_validResult(report), expected.bytes.size());
        EXPECT_EQ(
                copyBytesFromDevice(encoded, expected.bytes.size()),
                expected.bytes);
        EXPECT_EQ(
                copyOffsetsFromDevice(offsets, expected.offsets.size()),
                expected.offsets);
        expectGuard(expected.bytes.size());
    }
}

#endif

TEST(PivCoGpuTest, CooperativeEncodeRejectsMissingSymbolAndSmallCapacity)
{
    constexpr size_t kBlockSize = 64 * 1024;
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    auto data              = makeDataForWeights(weights, kBlockSize + 17);
    const Encoded expected = cpuEncode(weights, data, kBlockSize);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    DeviceBuffer<uint8_t> src, encoded, workspace;
    DeviceBuffer<uint64_t> offsets;
    copyToDevice(src, data);
    ASSERT_EQ(cudaSuccess, encoded.reset(expected.bytes.size()));
    ASSERT_EQ(cudaSuccess, offsets.reset(expected.offsets.size()));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuEncodeWorkspaceBytes(data.size(), kBlockSize)));
    auto encode =
            [&](size_t capacity, size_t offsetCount, size_t scratchBytes) {
                return pivcoGpuEncode(
                        context.get(),
                        encoded.get(),
                        capacity,
                        offsets.get(),
                        offsetCount,
                        src.get(),
                        data.size(),
                        kBlockSize,
                        workspace.get(),
                        scratchBytes,
                        nullptr);
            };
    EXPECT_TRUE(ZL_isError(encode(
            expected.bytes.size() - 1, offsets.size(), workspace.size())));
    EXPECT_TRUE(ZL_isError(
            encode(encoded.size(), offsets.size() - 1, workspace.size())));
    EXPECT_TRUE(ZL_isError(
            encode(encoded.size(),
                   offsets.size(),
                   legacyEncodeWorkspaceBytes(data.size(), kBlockSize) - 1)));
    data.back() = 255;
    copyToDevice(src, data);
    EXPECT_TRUE(ZL_isError(
            encode(encoded.size(), offsets.size(), workspace.size())));
}

#if !defined(__HIPCC__)
TEST(PivCoGpuTest, CooperativeEncodeRejectsAliasedWorkspace)
{
    constexpr size_t kBlockSize = 64 * 1024;
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    const auto data = makeDataForWeights(weights, kBlockSize + 3);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    DeviceBuffer<uint8_t> src, encoded, workspace;
    copyToDevice(src, data);
    ASSERT_EQ(
            cudaSuccess,
            encoded.reset(
                    ZL_PivCoHuffmanEncode_bound(data.size(), kBlockSize)));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuEncodeWorkspaceBytes(data.size(), kBlockSize)));
    const ZL_Report report = pivcoGpuEncode(
            context.get(),
            encoded.get(),
            encoded.size(),
            reinterpret_cast<uint64_t*>(workspace.get()),
            3,
            src.get(),
            data.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    EXPECT_TRUE(ZL_isError(report));
}
#endif

#if !defined(__HIPCC__)
TEST(PivCoGpuTest, CooperativeAsyncRejectsAliasBeforeWritingBuffers)
{
    constexpr size_t kBlockSize = 32 * 1024;
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    const auto data = makeDataForWeights(weights, 4097);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    DeviceBuffer<uint8_t> src, encoded, workspace;
    DeviceBuffer<uint64_t> total;
    DeviceBuffer<PivCoGpuStatus> status;
    StreamEventHandles handles;
    copyToDevice(src, data);
    ASSERT_EQ(
            cudaSuccess,
            cudaMemset(
                    src.get() + data.size(), 0xa5, src.size() - data.size()));
    auto expectedSource = data;
    expectedSource.resize(src.size(), 0xa5);
    ASSERT_EQ(
            cudaSuccess,
            encoded.reset(
                    ZL_PivCoHuffmanEncode_bound(data.size(), kBlockSize)));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuEncodeWorkspaceBytes(data.size(), kBlockSize)));
    ASSERT_EQ(cudaSuccess, total.reset(1));
    ASSERT_EQ(cudaSuccess, status.reset(1));
    ASSERT_EQ(cudaSuccess, cudaMemset(encoded.get(), 0xa5, encoded.size()));
    ASSERT_EQ(cudaSuccess, cudaMemset(workspace.get(), 0xa5, workspace.size()));
    ASSERT_EQ(cudaSuccess, cudaMemset(total.get(), 0xa5, sizeof(uint64_t)));
    ASSERT_EQ(
            cudaSuccess,
            cudaMemset(status.get(), 0xa5, sizeof(PivCoGpuStatus)));
    ASSERT_EQ(
            cudaSuccess,
            cudaStreamCreateWithFlags(
                    &handles.streams[0], cudaStreamNonBlocking));
    EXPECT_EQ(
            cudaErrorInvalidValue,
            pivcoGpuEncodeAsync(
                    context.get(),
                    encoded.get(),
                    encoded.size(),
                    reinterpret_cast<uint64_t*>(workspace.get()),
                    2,
                    src.get(),
                    data.size(),
                    kBlockSize,
                    workspace.get(),
                    workspace.size(),
                    status.get(),
                    total.get(),
                    handles.streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(handles.streams[0]));
    EXPECT_EQ(copyBytesFromDevice(src, src.size()), expectedSource);
    EXPECT_EQ(
            copyBytesFromDevice(encoded, encoded.size()),
            std::vector<uint8_t>(encoded.size(), 0xa5));
    EXPECT_EQ(
            copyBytesFromDevice(workspace, workspace.size()),
            std::vector<uint8_t>(workspace.size(), 0xa5));
    EXPECT_EQ(
            copyOffsetsFromDevice(total, 1),
            std::vector<uint64_t>{ 0xa5a5a5a5a5a5a5a5ULL });
    std::vector<uint8_t> actualStatus(sizeof(PivCoGpuStatus));
    ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                    actualStatus.data(),
                    status.get(),
                    actualStatus.size(),
                    cudaMemcpyDeviceToHost));
    EXPECT_EQ(actualStatus, std::vector<uint8_t>(actualStatus.size(), 0xa5));
}
#endif

TEST(PivCoGpuTest, CooperativeDispatchKeepsEmptyAndValidatesConstantInput)
{
    constexpr size_t kBlockSize = 64 * 1024;
    std::vector<uint8_t> weights(6);
    weights[5] = 1;
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    DeviceBuffer<uint8_t> src, encoded;
    DeviceBuffer<uint64_t> offsets;
    ASSERT_EQ(cudaSuccess, encoded.reset(1));
    ASSERT_EQ(cudaSuccess, offsets.reset(2));
    ASSERT_EQ(pivcoGpuEncodeWorkspaceBytes(0, kBlockSize), 0);
    const ZL_Report empty = pivcoGpuEncode(
            context.get(),
            encoded.get(),
            encoded.size(),
            offsets.get(),
            offsets.size(),
            nullptr,
            0,
            kBlockSize,
            nullptr,
            0,
            nullptr);
    ASSERT_FALSE(ZL_isError(empty));
    EXPECT_EQ(ZL_validResult(empty), 0);
    EXPECT_EQ(copyOffsetsFromDevice(offsets, 1), std::vector<uint64_t>{ 0 });
    std::vector<uint8_t> data(4097, 5);
    copyToDevice(src, data);
    const auto encode = [&] {
        return pivcoGpuEncode(
                context.get(),
                encoded.get(),
                encoded.size(),
                offsets.get(),
                offsets.size(),
                src.get(),
                data.size(),
                kBlockSize,
                nullptr,
                0,
                nullptr);
    };
    const ZL_Report constant = encode();
    ASSERT_FALSE(ZL_isError(constant));
    EXPECT_EQ(ZL_validResult(constant), 0);
    EXPECT_EQ(
            copyOffsetsFromDevice(offsets, 2), std::vector<uint64_t>({ 0, 0 }));
    data.back() = 4;
    copyToDevice(src, data);
    EXPECT_TRUE(ZL_isError(encode()));
}

TEST(PivCoGpuTest, ConstantEncodeGridStrideAndRecovery)
{
    constexpr size_t kBlockSize = 64 * 1024;
    constexpr uint64_t kGuard   = 0xa5a5a5a5a5a5a5a5ull;
    std::vector<uint8_t> weights(6);
    weights[5] = 1;
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    for (size_t size : { size_t{ 65537 }, size_t{ 262145 } }) {
        SCOPED_TRACE(size);
        const size_t numOffsets = (size + kBlockSize - 1) / kBlockSize + 1;
        const std::vector<uint8_t> data(size, 5);
        DeviceBuffer<uint8_t> src, encoded;
        DeviceBuffer<uint64_t> offsets;
        copyToDevice(src, data);
        ASSERT_EQ(cudaSuccess, encoded.reset(1));
        ASSERT_EQ(cudaSuccess, offsets.reset(numOffsets + 1));
        ASSERT_EQ(cudaSuccess, cudaMemset(encoded.get(), 0xa5, 1));
        ASSERT_EQ(
                cudaSuccess,
                cudaMemset(
                        offsets.get(),
                        0xa5,
                        offsets.size() * sizeof(uint64_t)));
        const auto encode = [&] {
            return pivcoGpuEncode(
                    context.get(),
                    encoded.get(),
                    encoded.size(),
                    offsets.get(),
                    numOffsets,
                    src.get(),
                    size,
                    kBlockSize,
                    nullptr,
                    0,
                    nullptr);
        };
        const auto expectSuccess = [&] {
            const ZL_Report report = encode();
            ASSERT_FALSE(ZL_isError(report));
            EXPECT_EQ(ZL_validResult(report), 0);
            std::vector<uint64_t> expected(numOffsets + 1, 0);
            expected.back() = kGuard;
            EXPECT_EQ(
                    copyOffsetsFromDevice(offsets, expected.size()), expected);
            EXPECT_EQ(
                    copyBytesFromDevice(encoded, 1),
                    std::vector<uint8_t>{ 0xa5 });
        };
        expectSuccess();
        for (size_t position : { size_t{ 0 }, size / 2, size - 1 }) {
            SCOPED_TRACE(position);
            const uint8_t invalid = 4;
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemcpy(
                            src.get() + position,
                            &invalid,
                            sizeof(invalid),
                            cudaMemcpyHostToDevice));
            EXPECT_TRUE(ZL_isError(encode()));
            const uint8_t valid = 5;
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemcpy(
                            src.get() + position,
                            &valid,
                            sizeof(valid),
                            cudaMemcpyHostToDevice));
            // The same buffers must recover after an error. The tail symbol
            // is visited after the capped 65536-thread grid's first stride.
            expectSuccess();
        }
    }
}

TEST(PivCoGpuTest, EncodeEmptyAsyncResetsOutputSize)
{
    constexpr size_t kBlockSize = 64 * 1024;
    const std::vector<uint8_t> weights{ 1 };
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    DeviceBuffer<uint64_t> offsets, total;
    DeviceBuffer<PivCoGpuStatus> status;
    ASSERT_EQ(cudaSuccess, offsets.reset(1));
    ASSERT_EQ(cudaSuccess, total.reset(1));
    ASSERT_EQ(cudaSuccess, status.reset(1));
    ASSERT_EQ(cudaSuccess, cudaMemset(offsets.get(), 0xa5, sizeof(uint64_t)));
    ASSERT_EQ(cudaSuccess, cudaMemset(total.get(), 0xa5, sizeof(uint64_t)));
    ASSERT_EQ(cudaSuccess, cudaMemset(status.get(), 0, sizeof(PivCoGpuStatus)));
    ASSERT_EQ(
            cudaSuccess,
            pivcoGpuEncodeAsync(
                    context.get(),
                    nullptr,
                    0,
                    offsets.get(),
                    offsets.size(),
                    nullptr,
                    0,
                    kBlockSize,
                    nullptr,
                    0,
                    status.get(),
                    total.get(),
                    nullptr));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    EXPECT_EQ(copyOffsetsFromDevice(offsets, 1), std::vector<uint64_t>{ 0 });
    EXPECT_EQ(copyOffsetsFromDevice(total, 1), std::vector<uint64_t>{ 0 });
    PivCoGpuStatus actualStatus{};
    ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                    &actualStatus,
                    status.get(),
                    sizeof(actualStatus),
                    cudaMemcpyDeviceToHost));
    EXPECT_EQ(actualStatus.code, PIVCO_GPU_STATUS_OK);
}

TEST(PivCoGpuTest, CooperativeAsyncStreamsAndPoisonedWorkspaceReuse)
{
    constexpr size_t kBlockSize = 64 * 1024;
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    std::vector<uint8_t> data[2] = {
        makeDataForWeights(weights, 3 * kBlockSize + 17),
        makeDataForWeights(weights, 2 * kBlockSize + 31)
    };
    DeviceBuffer<uint8_t> src[2], encoded[2], workspace[2];
    DeviceBuffer<uint64_t> offsets[2], total[2];
    DeviceBuffer<PivCoGpuStatus> status[2];
    StreamEventHandles handles;
    for (unsigned i = 0; i < 2; ++i) {
        copyToDevice(src[i], data[i]);
        ASSERT_EQ(
                cudaSuccess,
                encoded[i].reset(ZL_PivCoHuffmanEncode_bound(
                        data[i].size(), kBlockSize)));
        ASSERT_EQ(
                cudaSuccess,
                workspace[i].reset(pivcoGpuEncodeWorkspaceBytes(
                        data[i].size(), kBlockSize)));
        ASSERT_EQ(
                cudaSuccess,
                offsets[i].reset(
                        (data[i].size() + kBlockSize - 1) / kBlockSize + 1));
        ASSERT_EQ(cudaSuccess, total[i].reset(1));
        ASSERT_EQ(cudaSuccess, status[i].reset(1));
        ASSERT_EQ(
                cudaSuccess,
                cudaStreamCreateWithFlags(
                        &handles.streams[i], cudaStreamNonBlocking));
        ASSERT_EQ(
                cudaSuccess,
                cudaEventCreateWithFlags(
                        &handles.events[i], cudaEventDisableTiming));
    }
    for (unsigned round = 0; round < 3; ++round) {
        SCOPED_TRACE(round);
        Encoded expected[2];
        for (unsigned i = 0; i < 2; ++i) {
            if (round == 1) {
                data[i].resize(i == 0 ? 33 : kBlockSize - 1);
            } else if (round == 2) {
                data[i].clear();
            }
            expected[i]       = data[i].empty()
                          ? Encoded{ {}, { 0 } }
                          : cpuEncode(weights, data[i], kBlockSize);
            const auto stream = handles.streams[i];
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemsetAsync(
                            workspace[i].get(),
                            0xa5,
                            workspace[i].size(),
                            stream));
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemsetAsync(
                            status[i].get(),
                            0,
                            sizeof(PivCoGpuStatus),
                            stream));
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemsetAsync(
                            total[i].get(), 0xa5, sizeof(uint64_t), stream));
            ASSERT_EQ(
                    cudaSuccess,
                    pivcoGpuEncodeAsync(
                            context.get(),
                            encoded[i].get(),
                            encoded[i].size(),
                            offsets[i].get(),
                            offsets[i].size(),
                            src[i].get(),
                            data[i].size(),
                            kBlockSize,
                            workspace[i].get(),
                            workspace[i].size(),
                            status[i].get(),
                            total[i].get(),
                            stream));
            ASSERT_EQ(cudaSuccess, cudaEventRecord(handles.events[i], stream));
        }
        // Both disjoint streams have been submitted before either host wait.
        for (unsigned i = 0; i < 2; ++i) {
            ASSERT_EQ(cudaSuccess, cudaEventSynchronize(handles.events[i]));
            PivCoGpuStatus actualStatus{};
            uint64_t actualSize = 0;
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemcpy(
                            &actualStatus,
                            status[i].get(),
                            sizeof(actualStatus),
                            cudaMemcpyDeviceToHost));
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemcpy(
                            &actualSize,
                            total[i].get(),
                            sizeof(actualSize),
                            cudaMemcpyDeviceToHost));
            ASSERT_EQ(actualStatus.code, PIVCO_GPU_STATUS_OK);
            ASSERT_EQ(actualSize, expected[i].bytes.size());
            const auto actualBytes =
                    copyBytesFromDevice(encoded[i], size_t(actualSize));
            EXPECT_EQ(actualBytes, expected[i].bytes);
            EXPECT_EQ(
                    copyOffsetsFromDevice(
                            offsets[i], expected[i].offsets.size()),
                    expected[i].offsets);
            if (!data[i].empty()) {
                EXPECT_EQ(
                        cpuDecode(
                                weights,
                                actualBytes,
                                data[i].size(),
                                kBlockSize),
                        data[i]);
            }
        }
    }
}

#if !defined(__HIPCC__)
TEST(PivCoGpuTest, CooperativeAsyncEventOrdersSharedWorkspaceReuse)
{
    constexpr size_t kBlockSize  = 64 * 1024;
    constexpr size_t kGuardBytes = 64;
    constexpr size_t kGuardWords = kGuardBytes / sizeof(uint64_t);
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    const std::vector<uint8_t> data[2] = {
        makeDataForWeights(weights, 256 * kBlockSize + 17),
        makeDataForWeights(weights, 2 * kBlockSize + 31)
    };
    const Encoded expected[2] = { cpuEncode(weights, data[0], kBlockSize),
                                  cpuEncode(weights, data[1], kBlockSize) };
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    DeviceBuffer<uint8_t> src[2], encoded[2], workspace;
    DeviceBuffer<uint64_t> offsets[2], total[2];
    DeviceBuffer<PivCoGpuStatus> status[2];
    StreamEventHandles handles;
    const size_t workspaceBytes =
            pivcoGpuEncodeWorkspaceBytes(256 * kBlockSize, kBlockSize);
    ASSERT_EQ(cudaSuccess, workspace.reset(workspaceBytes + 2 * kGuardBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(workspace.get(), 0xa5, workspace.size()));
    for (unsigned i = 0; i < 2; ++i) {
        copyToDevice(src[i], data[i]);
        ASSERT_EQ(cudaSuccess, encoded[i].reset(expected[i].bytes.size()));
        ASSERT_EQ(
                cudaSuccess,
                offsets[i].reset(expected[i].offsets.size() + 2 * kGuardWords));
        ASSERT_EQ(
                cudaSuccess,
                cudaMemset(
                        offsets[i].get(),
                        0xa5,
                        offsets[i].size() * sizeof(uint64_t)));
        ASSERT_EQ(cudaSuccess, total[i].reset(1));
        ASSERT_EQ(cudaSuccess, status[i].reset(1));
        ASSERT_EQ(
                cudaSuccess,
                cudaStreamCreateWithFlags(
                        &handles.streams[i], cudaStreamNonBlocking));
        ASSERT_EQ(
                cudaSuccess,
                cudaEventCreateWithFlags(
                        &handles.events[i], cudaEventDisableTiming));
    }
    for (unsigned i = 0; i < 2; ++i) {
        const auto stream = handles.streams[i];
        if (i == 1) {
            // Transfer ownership of the exact same scratch through a device
            // dependency, without synchronizing either stream on the host.
            ASSERT_EQ(
                    cudaSuccess,
                    cudaStreamWaitEvent(stream, handles.events[0], 0));
        }
        ASSERT_EQ(
                cudaSuccess,
                cudaMemsetAsync(
                        workspace.get() + kGuardBytes,
                        0xa5,
                        workspaceBytes,
                        stream));
        ASSERT_EQ(
                cudaSuccess,
                cudaMemsetAsync(
                        status[i].get(), 0, sizeof(PivCoGpuStatus), stream));
        ASSERT_EQ(
                cudaSuccess,
                cudaMemsetAsync(
                        total[i].get(), 0xa5, sizeof(uint64_t), stream));
        ASSERT_EQ(
                cudaSuccess,
                pivcoGpuEncodeAsync(
                        context.get(),
                        encoded[i].get(),
                        encoded[i].size(),
                        offsets[i].get() + kGuardWords,
                        expected[i].offsets.size(),
                        src[i].get(),
                        data[i].size(),
                        kBlockSize,
                        workspace.get() + kGuardBytes,
                        workspaceBytes,
                        status[i].get(),
                        total[i].get(),
                        stream));
        ASSERT_EQ(cudaSuccess, cudaEventRecord(handles.events[i], stream));
    }
    // The second event transitively covers both encodes and the handoff.
    ASSERT_EQ(cudaSuccess, cudaEventSynchronize(handles.events[1]));
    for (unsigned i = 0; i < 2; ++i) {
        PivCoGpuStatus actualStatus{};
        uint64_t actualSize = 0;
        ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(
                        &actualStatus,
                        status[i].get(),
                        sizeof(actualStatus),
                        cudaMemcpyDeviceToHost));
        ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(
                        &actualSize,
                        total[i].get(),
                        sizeof(actualSize),
                        cudaMemcpyDeviceToHost));
        ASSERT_EQ(actualStatus.code, PIVCO_GPU_STATUS_OK);
        ASSERT_EQ(actualSize, expected[i].bytes.size());
        const auto bytes = copyBytesFromDevice(encoded[i], size_t(actualSize));
        EXPECT_EQ(bytes, expected[i].bytes);
        EXPECT_EQ(
                cpuDecode(weights, bytes, data[i].size(), kBlockSize), data[i]);
        const auto actualOffsets =
                copyOffsetsFromDevice(offsets[i], offsets[i].size());
        EXPECT_EQ(
                std::vector<uint64_t>(
                        actualOffsets.begin() + kGuardWords,
                        actualOffsets.end() - kGuardWords),
                expected[i].offsets);
        EXPECT_EQ(
                std::vector<uint64_t>(
                        actualOffsets.begin(),
                        actualOffsets.begin() + kGuardWords),
                std::vector<uint64_t>(kGuardWords, 0xa5a5a5a5a5a5a5a5ULL));
        EXPECT_EQ(
                std::vector<uint64_t>(
                        actualOffsets.end() - kGuardWords, actualOffsets.end()),
                std::vector<uint64_t>(kGuardWords, 0xa5a5a5a5a5a5a5a5ULL));
    }
    std::vector<uint8_t> guard(kGuardBytes);
    ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                    guard.data(),
                    workspace.get(),
                    guard.size(),
                    cudaMemcpyDeviceToHost));
    EXPECT_EQ(guard, std::vector<uint8_t>(kGuardBytes, 0xa5));
    ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                    guard.data(),
                    workspace.get() + kGuardBytes + workspaceBytes,
                    guard.size(),
                    cudaMemcpyDeviceToHost));
    EXPECT_EQ(guard, std::vector<uint8_t>(kGuardBytes, 0xa5));
}
#endif

struct TerminalRankCase {
    size_t sourceSize;
    size_t zeroPosition;
};

class PivCoGpuTerminalRankTest
        : public testing::TestWithParam<TerminalRankCase> {};

TEST_P(PivCoGpuTerminalRankTest, CpuWireBoundary)
{
    constexpr size_t kBlockSize     = 64 * 1024;
    const TerminalRankCase testCase = GetParam();
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    std::vector<uint8_t> data(testCase.sourceSize, 4);
    if (testCase.zeroPosition < data.size()) {
        data[testCase.zeroPosition] = 0;
    }

    // The rightmost symbol sets every internal bitmap bit. One leftmost
    // symbol keeps the root at 65535 ones while moving the zero across word
    // boundaries; a partial all-one block also exercises the masked last word.
    expectGpuDecodeMatchesCpuEncode(weights, data, kBlockSize);
}

INSTANTIATE_TEST_SUITE_P(
        TerminalRank,
        PivCoGpuTerminalRankTest,
        testing::Values(
                TerminalRankCase{ 65535, SIZE_MAX },
                TerminalRankCase{ 65536, SIZE_MAX },
                TerminalRankCase{ 65536, 0 },
                TerminalRankCase{ 65536, 31 },
                TerminalRankCase{ 65536, 32 },
                TerminalRankCase{ 65536, 65503 },
                TerminalRankCase{ 65536, 65504 },
                TerminalRankCase{ 65536, 65535 }));

TEST(PivCoGpuTest, CooperativeDecodeCpuWireFullBlockAllOnes)
{
    constexpr size_t kBlockSize = 64 * 1024;
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    // A constant source under a nonconstant context takes every right branch.
    // Its first block has 65536 ones in each internal-node bitmap; the tail
    // verifies that independent block offsets and partial endpoints still work.
    const std::vector<uint8_t> data(kBlockSize + 65, 4);
    const Encoded expected = cpuEncode(weights, data, kBlockSize);
    ASSERT_EQ(
            cpuDecode(weights, expected.bytes, data.size(), kBlockSize), data);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    ASSERT_EQ(context->hostTree.fastMode, PIVCO_GPU_FAST_NONE);
    ASSERT_NE(context->decodeSchedule.enabled, 0);
    DeviceBuffer<uint8_t> encoded, decoded, workspace;
    DeviceBuffer<uint64_t> offsets;
    ASSERT_EQ(
            cudaSuccess,
            encoded.reset(expected.bytes.size() + PIVCO_GPU_DECODE_SRC_SLOP));
    ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                    encoded.get(),
                    expected.bytes.data(),
                    expected.bytes.size(),
                    cudaMemcpyHostToDevice));
    copyToDevice(offsets, expected.offsets);
    ASSERT_EQ(
            cudaSuccess,
            decoded.reset(data.size() + PIVCO_GPU_DECODE_DST_SLOP));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(
                    pivcoGpuDecodeWorkspaceBytes(data.size(), kBlockSize)));
    const ZL_Report report = pivcoGpuDecode(
            context.get(),
            decoded.get(),
            data.size(),
            encoded.get(),
            expected.bytes.size(),
            offsets.get(),
            offsets.size(),
            kBlockSize,
            workspace.get(),
            workspace.size(),
            nullptr);
    ASSERT_FALSE(ZL_isError(report));
    EXPECT_EQ(copyBytesFromDevice(decoded, data.size()), data);
}

struct PinnedEncodeResult {
    PivCoGpuEncodeResult* ptr{};
    ~PinnedEncodeResult()
    {
        if (ptr) {
            (void)cudaFreeHost(ptr);
        }
    }
    cudaError_t reset()
    {
        return cudaMallocHost(&ptr, sizeof(*ptr));
    }
};

TEST(PivCoGpuTest, ContextWorkspaceBudgets)
{
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> generic(
            createContext({ 4, 3, 2, 1, 1 }), pivcoGpuContextDestroy);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> fast(
            createContext({ 2, 1, 1 }), pivcoGpuContextDestroy);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)>
            constant(createContext({ 1 }), pivcoGpuContextDestroy);
    ASSERT_NE(generic.get(), nullptr);
    ASSERT_NE(fast.get(), nullptr);
    ASSERT_NE(constant.get(), nullptr);
    EXPECT_EQ(
            pivcoGpuEncodeWorkspaceBytesForContext(nullptr, 1, 65536),
            SIZE_MAX);
    EXPECT_EQ(
            pivcoGpuEncodeWorkspaceBytesForContext(generic.get(), 1, 0),
            SIZE_MAX);
    EXPECT_EQ(
            pivcoGpuEncodeWorkspaceBytesForContext(
                    generic.get(), SIZE_MAX, 65536),
            SIZE_MAX);
    EXPECT_EQ(
            pivcoGpuEncodeWorkspaceBytesForContext(
                    generic.get(), 1, PIVCO_GPU_MAX_BLOCK_SIZE + 1),
            SIZE_MAX);
    EXPECT_EQ(
            pivcoGpuEncodeWorkspaceBytesForContext(generic.get(), 0, 65536), 0);
    EXPECT_EQ(
            pivcoGpuEncodeWorkspaceBytesForContext(
                    constant.get(), 1 << 20, 65536),
            0);
    for (size_t block : { size_t{ 32768 }, size_t{ 65536 } }) {
        const size_t batch   = 256 * block;
        const size_t bounded = pivcoGpuEncodeWorkspaceBytesForContext(
                generic.get(), batch, block);
#if !defined(__HIPCC__)
        EXPECT_EQ(
                bounded,
                pivcoGpuEncodeWorkspaceBytesForContext(
                        generic.get(), size_t{ 1 } << 30, block));
        EXPECT_LT(
                bounded,
                pivcoGpuEncodeWorkspaceBytes(size_t{ 1 } << 30, block));
#else
        EXPECT_EQ(bounded, pivcoGpuEncodeWorkspaceBytes(batch, block));
        EXPECT_EQ(
                pivcoGpuEncodeWorkspaceBytesForContext(
                        generic.get(), size_t{ 1 } << 30, block),
                pivcoGpuEncodeWorkspaceBytes(size_t{ 1 } << 30, block));
#endif
        EXPECT_LT(
                pivcoGpuEncodeWorkspaceBytesForContext(
                        fast.get(), batch, block),
                pivcoGpuEncodeWorkspaceBytes(batch, block));
    }
    EXPECT_EQ(
            pivcoGpuEncodeWorkspaceBytesForContext(generic.get(), 4097, 1024),
            pivcoGpuEncodeWorkspaceBytes(4097, 1024));
}

TEST(PivCoGpuTest, ContextWorkspaceExactAllocationAndGuards)
{
    for (bool specialized : { false, true }) {
        const std::vector<uint8_t> weights = specialized
                ? std::vector<uint8_t>{ 2, 1, 1 }
                : std::vector<uint8_t>{ 4, 3, 2, 1, 1 };
        std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)>
                context(createContext(weights), pivcoGpuContextDestroy);
        ASSERT_NE(context.get(), nullptr);
        for (size_t block : { size_t{ 32768 }, size_t{ 65536 } }) {
            SCOPED_TRACE(specialized);
            SCOPED_TRACE(block);
            const auto data = makeDataForWeights(
                    weights, (specialized ? 3 : 769) * block + 17);
            const Encoded expected = cpuEncode(weights, data, block);
            const size_t required  = pivcoGpuEncodeWorkspaceBytesForContext(
                    context.get(), data.size(), block);
            ASSERT_NE(required, SIZE_MAX);
            ASSERT_GT(required, 0u);
#if defined(__HIPCC__)
            if (!specialized) {
                EXPECT_EQ(
                        required,
                        pivcoGpuEncodeWorkspaceBytes(data.size(), block));
            } else
#endif
            {
                EXPECT_LT(
                        required,
                        pivcoGpuEncodeWorkspaceBytes(data.size(), block));
            }
            DeviceBuffer<uint8_t> src, dst, workspace;
            DeviceBuffer<uint64_t> offsets;
            copyToDevice(src, data);
            ASSERT_EQ(cudaSuccess, dst.reset(expected.bytes.size() + 64));
            ASSERT_EQ(cudaSuccess, offsets.reset(expected.offsets.size() + 1));
            ASSERT_EQ(cudaSuccess, workspace.reset(required + 64));
            ASSERT_EQ(cudaSuccess, cudaMemset(dst.get(), 0xa5, dst.size()));
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemset(
                            offsets.get(),
                            0xa5,
                            offsets.size() * sizeof(uint64_t)));
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemset(workspace.get(), 0xa5, workspace.size()));
            const auto encode = [&](size_t bytes) {
                return pivcoGpuEncode(
                        context.get(),
                        dst.get(),
                        expected.bytes.size(),
                        offsets.get(),
                        expected.offsets.size(),
                        src.get(),
                        data.size(),
                        block,
                        workspace.get(),
                        bytes,
                        nullptr);
            };
            auto report = encode(required);
            ASSERT_FALSE(ZL_isError(report));
            EXPECT_EQ(ZL_validResult(report), expected.bytes.size());
            EXPECT_EQ(
                    copyBytesFromDevice(dst, expected.bytes.size()),
                    expected.bytes);
            EXPECT_EQ(
                    copyOffsetsFromDevice(offsets, expected.offsets.size()),
                    expected.offsets);
            EXPECT_TRUE(ZL_isError(encode(required - 1)));
            report = encode(required);
            ASSERT_FALSE(ZL_isError(report));
            std::vector<uint8_t> guard(64);
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemcpy(
                            guard.data(),
                            workspace.get() + required,
                            64,
                            cudaMemcpyDeviceToHost));
            EXPECT_EQ(guard, std::vector<uint8_t>(64, 0xa5));
            ASSERT_EQ(
                    cudaSuccess,
                    cudaMemcpy(
                            guard.data(),
                            dst.get() + expected.bytes.size(),
                            64,
                            cudaMemcpyDeviceToHost));
            EXPECT_EQ(guard, std::vector<uint8_t>(64, 0xa5));
            const auto allOffsets =
                    copyOffsetsFromDevice(offsets, offsets.size());
            EXPECT_EQ(allOffsets.back(), 0xa5a5a5a5a5a5a5a5ull);
        }
    }
}

TEST(PivCoGpuTest, CombinedResultReuseRecoveryAndEmpty)
{
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    auto data              = makeDataForWeights(weights, 65537);
    const Encoded expected = cpuEncode(weights, data, 65536);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    DeviceBuffer<uint8_t> src, dst, workspace;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<PivCoGpuEncodeResult> result;
    PinnedEncodeResult host;
    copyToDevice(src, data);
    ASSERT_EQ(cudaSuccess, dst.reset(expected.bytes.size()));
    ASSERT_EQ(cudaSuccess, offsets.reset(expected.offsets.size()));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(pivcoGpuEncodeWorkspaceBytesForContext(
                    context.get(), data.size(), 65536)));
    ASSERT_EQ(cudaSuccess, result.reset(2));
    ASSERT_EQ(cudaSuccess, host.reset());
    ASSERT_EQ(
            cudaSuccess,
            cudaMemset(result.get(), 0xa5, 2 * sizeof(PivCoGpuEncodeResult)));
    StreamEventHandles handles;
    for (unsigned i = 0; i < 2; ++i) {
        ASSERT_EQ(
                cudaSuccess,
                cudaStreamCreateWithFlags(
                        &handles.streams[i], cudaStreamNonBlocking));
        ASSERT_EQ(
                cudaSuccess,
                cudaEventCreateWithFlags(
                        &handles.events[i], cudaEventDisableTiming));
    }
    const auto enqueue = [&](unsigned stream) {
        return pivcoGpuEncodeWithResultAsync(
                context.get(),
                dst.get(),
                dst.size(),
                offsets.get(),
                offsets.size(),
                src.get(),
                data.size(),
                65536,
                workspace.get(),
                workspace.size(),
                result.get(),
                handles.streams[stream]);
    };
    const auto complete = [&](unsigned stream) {
        EXPECT_EQ(
                cudaSuccess,
                cudaMemcpyAsync(
                        host.ptr,
                        result.get(),
                        sizeof(*host.ptr),
                        cudaMemcpyDeviceToHost,
                        handles.streams[stream]));
        EXPECT_EQ(
                cudaSuccess,
                cudaEventRecord(
                        handles.events[stream], handles.streams[stream]));
        EXPECT_EQ(cudaSuccess, cudaEventSynchronize(handles.events[stream]));
    };
    ASSERT_EQ(cudaSuccess, enqueue(0));
    // Reuse through a device event, with no host fence between submissions.
    ASSERT_EQ(
            cudaSuccess,
            cudaEventRecord(handles.events[0], handles.streams[0]));
    ASSERT_EQ(
            cudaSuccess,
            cudaStreamWaitEvent(handles.streams[1], handles.events[0], 0));
    ASSERT_EQ(cudaSuccess, enqueue(1));
    complete(1);
    auto report = pivcoGpuEncodeResultReport(host.ptr);
    ASSERT_FALSE(ZL_isError(report));
    EXPECT_EQ(ZL_validResult(report), expected.bytes.size());
    EXPECT_EQ(copyBytesFromDevice(dst, dst.size()), expected.bytes);
    EXPECT_EQ(copyOffsetsFromDevice(offsets, offsets.size()), expected.offsets);
    data.back() = 255;
    ASSERT_EQ(
            cudaSuccess,
            cudaMemcpyAsync(
                    src.get(),
                    data.data(),
                    data.size(),
                    cudaMemcpyHostToDevice,
                    handles.streams[1]));
    ASSERT_EQ(cudaSuccess, enqueue(1));
    complete(1);
    EXPECT_TRUE(ZL_isError(pivcoGpuEncodeResultReport(host.ptr)));
    data = makeDataForWeights(weights, data.size());
    ASSERT_EQ(
            cudaSuccess,
            cudaMemcpyAsync(
                    src.get(),
                    data.data(),
                    data.size(),
                    cudaMemcpyHostToDevice,
                    handles.streams[1]));
    ASSERT_EQ(cudaSuccess, enqueue(1));
    complete(1);
    ASSERT_FALSE(ZL_isError(pivcoGpuEncodeResultReport(host.ptr)));
    EXPECT_EQ(copyBytesFromDevice(dst, dst.size()), expected.bytes);
    ASSERT_EQ(
            cudaSuccess,
            pivcoGpuEncodeWithResultAsync(
                    context.get(),
                    nullptr,
                    0,
                    offsets.get(),
                    offsets.size(),
                    nullptr,
                    0,
                    65536,
                    nullptr,
                    0,
                    result.get(),
                    handles.streams[1]));
    complete(1);
    report = pivcoGpuEncodeResultReport(host.ptr);
    ASSERT_FALSE(ZL_isError(report));
    EXPECT_EQ(ZL_validResult(report), 0u);
    EXPECT_EQ(copyOffsetsFromDevice(offsets, 1), std::vector<uint64_t>{ 0 });
    std::vector<uint8_t> guard(sizeof(PivCoGpuEncodeResult));
    ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                    guard.data(),
                    result.get() + 1,
                    guard.size(),
                    cudaMemcpyDeviceToHost));
    EXPECT_EQ(guard, std::vector<uint8_t>(guard.size(), 0xa5));
}

TEST(PivCoGpuTest, CombinedResultAliasPreflight)
{
    const auto data = makeDataForWeights({ 4, 3, 2, 1, 1 }, 65537);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext({ 4, 3, 2, 1, 1 }), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    DeviceBuffer<uint8_t> src, dst, workspace;
    DeviceBuffer<uint64_t> offsets;
    DeviceBuffer<PivCoGpuEncodeResult> result;
    copyToDevice(src, data);
    ASSERT_EQ(cudaSuccess, dst.reset(data.size() * 2));
    ASSERT_EQ(cudaSuccess, offsets.reset(3));
    ASSERT_EQ(
            cudaSuccess,
            workspace.reset(pivcoGpuEncodeWorkspaceBytesForContext(
                    context.get(), data.size(), 65536)));
    ASSERT_EQ(cudaSuccess, result.reset(1));
    ASSERT_EQ(cudaSuccess, cudaMemset(dst.get(), 0xa5, dst.size()));
    ASSERT_EQ(cudaSuccess, cudaMemset(workspace.get(), 0xa5, workspace.size()));
    ASSERT_EQ(
            cudaSuccess,
            cudaMemset(offsets.get(), 0xa5, offsets.size() * sizeof(uint64_t)));
    const auto enqueue = [&](PivCoGpuEncodeResult* completion) {
        return pivcoGpuEncodeWithResultAsync(
                context.get(),
                dst.get(),
                dst.size(),
                offsets.get(),
                offsets.size(),
                src.get(),
                data.size(),
                65536,
                workspace.get(),
                workspace.size(),
                completion,
                nullptr);
    };
    for (void* pointer : { static_cast<void*>(src.get()),
                           static_cast<void*>(dst.get()),
                           static_cast<void*>(offsets.get()),
                           static_cast<void*>(workspace.get()) }) {
        EXPECT_EQ(
                cudaErrorInvalidValue,
                enqueue(static_cast<PivCoGpuEncodeResult*>(pointer)));
    }
    EXPECT_EQ(
            cudaErrorInvalidValue,
            enqueue(reinterpret_cast<PivCoGpuEncodeResult*>(
                    reinterpret_cast<uint8_t*>(result.get()) + 1)));
    EXPECT_EQ(cudaErrorInvalidValue, enqueue(nullptr));
    EXPECT_EQ(copyBytesFromDevice(src, data.size()), data);
    std::vector<uint8_t> guard(sizeof(PivCoGpuEncodeResult));
    for (const void* pointer : { static_cast<const void*>(dst.get()),
                                 static_cast<const void*>(workspace.get()),
                                 static_cast<const void*>(offsets.get()) }) {
        ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(
                        guard.data(),
                        pointer,
                        guard.size(),
                        cudaMemcpyDeviceToHost));
        EXPECT_EQ(guard, std::vector<uint8_t>(guard.size(), 0xa5));
    }
    PivCoGpuEncodeResult error{};
    error.status.code = PIVCO_GPU_STATUS_CAPACITY;
    error.totalSize   = UINT64_MAX;
    EXPECT_TRUE(ZL_isError(pivcoGpuEncodeResultReport(&error)));
    EXPECT_TRUE(ZL_isError(pivcoGpuEncodeResultReport(nullptr)));
}

TEST(PivCoGpuTest, CombinedResultConcurrentStreams)
{
    const std::vector<uint8_t> weights{ 4, 3, 2, 1, 1 };
    const auto data        = makeDataForWeights(weights, 65537);
    const Encoded expected = cpuEncode(weights, data, 65536);
    std::unique_ptr<PivCoGpuContext, decltype(&pivcoGpuContextDestroy)> context(
            createContext(weights), pivcoGpuContextDestroy);
    ASSERT_NE(context.get(), nullptr);
    DeviceBuffer<uint8_t> src, dst[2], workspace[2];
    DeviceBuffer<uint64_t> offsets[2];
    DeviceBuffer<PivCoGpuEncodeResult> result[2];
    PinnedEncodeResult host[2];
    copyToDevice(src, data);
    StreamEventHandles handles;
    for (unsigned i = 0; i < 2; ++i) {
        ASSERT_EQ(cudaSuccess, dst[i].reset(expected.bytes.size()));
        ASSERT_EQ(cudaSuccess, offsets[i].reset(expected.offsets.size()));
        ASSERT_EQ(
                cudaSuccess,
                workspace[i].reset(pivcoGpuEncodeWorkspaceBytesForContext(
                        context.get(), data.size(), 65536)));
        ASSERT_EQ(cudaSuccess, result[i].reset(1));
        ASSERT_EQ(cudaSuccess, host[i].reset());
        ASSERT_EQ(
                cudaSuccess,
                cudaStreamCreateWithFlags(
                        &handles.streams[i], cudaStreamNonBlocking));
        ASSERT_EQ(
                cudaSuccess,
                cudaEventCreateWithFlags(
                        &handles.events[i], cudaEventDisableTiming));
        ASSERT_EQ(
                cudaSuccess,
                pivcoGpuEncodeWithResultAsync(
                        context.get(),
                        dst[i].get(),
                        dst[i].size(),
                        offsets[i].get(),
                        offsets[i].size(),
                        src.get(),
                        data.size(),
                        65536,
                        workspace[i].get(),
                        workspace[i].size(),
                        result[i].get(),
                        handles.streams[i]));
        ASSERT_EQ(
                cudaSuccess,
                cudaMemcpyAsync(
                        host[i].ptr,
                        result[i].get(),
                        sizeof(*host[i].ptr),
                        cudaMemcpyDeviceToHost,
                        handles.streams[i]));
        ASSERT_EQ(
                cudaSuccess,
                cudaEventRecord(handles.events[i], handles.streams[i]));
    }
    for (unsigned i = 0; i < 2; ++i) {
        ASSERT_EQ(cudaSuccess, cudaEventSynchronize(handles.events[i]));
        auto report = pivcoGpuEncodeResultReport(host[i].ptr);
        ASSERT_FALSE(ZL_isError(report));
        EXPECT_EQ(ZL_validResult(report), expected.bytes.size());
        EXPECT_EQ(copyBytesFromDevice(dst[i], dst[i].size()), expected.bytes);
        EXPECT_EQ(
                copyOffsetsFromDevice(offsets[i], offsets[i].size()),
                expected.offsets);
    }
}

} // namespace
