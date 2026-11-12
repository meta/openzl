// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/pivco-huffman/gpu/pivco_gpu.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
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

} // namespace
