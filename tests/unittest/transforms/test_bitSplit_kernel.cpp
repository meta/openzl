// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

extern "C" {
#include "openzl/codecs/bitSplit/common_bitSplit_kernel.h"
#include "openzl/codecs/bitSplit/decode_bitSplit_kernel.h"
#include "openzl/codecs/bitSplit/encode_bitSplit_kernel.h"
}
#include "openzl/shared/mem.h"

namespace openzl {
namespace tests {

// =============================================================================
// Kernel Unit Tests
// =============================================================================

class BitSplitKernelTest : public ::testing::Test {};

TEST_F(BitSplitKernelTest, OutputEltWidth)
{
    // 1-8 bits -> u8
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(1), 1);
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(7), 1);
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(8), 1);

    // 9-16 bits -> u16
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(9), 2);
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(15), 2);
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(16), 2);

    // 17-32 bits -> u32
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(17), 4);
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(31), 4);
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(32), 4);

    // 33-64 bits -> u64
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(33), 8);
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(63), 8);
    EXPECT_EQ(ZL_bitSplit_outputEltWidth(64), 8);
}

TEST_F(BitSplitKernelTest, ParamsAreValid_Valid)
{
    uint8_t widths1[] = { 8 };
    size_t sum        = 0;
    EXPECT_TRUE(ZL_bitSplit_paramsAreValid(widths1, 1, 8, &sum));
    EXPECT_EQ(sum, 8);

    uint8_t widths2[] = { 4, 8, 12, 8 };
    EXPECT_TRUE(ZL_bitSplit_paramsAreValid(widths2, 4, 32, &sum));
    EXPECT_EQ(sum, 32);

    // Partial coverage is valid
    uint8_t widths3[] = { 3, 4, 5 };
    EXPECT_TRUE(ZL_bitSplit_paramsAreValid(widths3, 3, 32, &sum));
    EXPECT_EQ(sum, 12);
}

TEST_F(BitSplitKernelTest, ParamsAreValid_Invalid)
{
    size_t sum = 0;

    // Zero width is invalid
    uint8_t widths1[] = { 0, 8 };
    EXPECT_FALSE(ZL_bitSplit_paramsAreValid(widths1, 2, 16, &sum));

    // Sum exceeds element width
    uint8_t widths2[] = { 16, 16, 16 };
    EXPECT_FALSE(ZL_bitSplit_paramsAreValid(widths2, 3, 32, &sum));
}

TEST_F(BitSplitKernelTest, TopBitsAreZero)
{
    // Full coverage - always true
    uint8_t val8 = 0xFF;
    EXPECT_TRUE(ZL_bitSplit_topBitsAreZero(&val8, 1, 1, 8));

    uint32_t val32 = 0xFFFFFFFF;
    EXPECT_TRUE(ZL_bitSplit_topBitsAreZero(&val32, 4, 1, 32));

    // Partial coverage with top bits zero
    uint16_t val16a = 0x0FFF;
    EXPECT_TRUE(ZL_bitSplit_topBitsAreZero(&val16a, 2, 1, 12));

    uint32_t val32a = 0x00000FFF;
    EXPECT_TRUE(ZL_bitSplit_topBitsAreZero(&val32a, 4, 1, 12));

    // Partial coverage with top bits non-zero
    uint16_t val16b = 0x1FFF;
    EXPECT_FALSE(ZL_bitSplit_topBitsAreZero(&val16b, 2, 1, 12));

    uint16_t val16c = 0xFFFF;
    EXPECT_FALSE(ZL_bitSplit_topBitsAreZero(&val16c, 2, 1, 12));
}

TEST_F(BitSplitKernelTest, TopBitsAreZeroChecksEveryElement)
{
    std::vector<uint32_t> values(5000, 0x00000FFF);
    EXPECT_TRUE(
            ZL_bitSplit_topBitsAreZero(values.data(), 4, values.size(), 12));
    values.back() = 0x00001000;
    EXPECT_FALSE(
            ZL_bitSplit_topBitsAreZero(values.data(), 4, values.size(), 12));
}

TEST_F(BitSplitKernelTest, EncodeDecodeSingleValue_8bit)
{
    // Split 8-bit value 0xA5 into [4, 4]
    uint8_t bitWidths[]   = { 4, 4 };
    size_t outputWidths[] = { 1, 1 };

    uint8_t input = 0xA5;
    uint8_t out0, out1;
    void* outputs[] = { &out0, &out1 };

    ZL_bitSplitEncode(outputs, outputWidths, 1, &input, 1, bitWidths, 2);

    EXPECT_EQ(out0, 0x5); // bits 0-3
    EXPECT_EQ(out1, 0xA); // bits 4-7

    // Decode back
    const void* inputs[] = { &out0, &out1 };
    size_t inputWidths[] = { 1, 1 };
    uint8_t result;
    ZL_bitSplitDecode(&result, 1, 1, inputs, inputWidths, bitWidths, 2);
    EXPECT_EQ(result, 0xA5);
}

TEST_F(BitSplitKernelTest, EncodeDecodeSingleValue_32bit)
{
    // Split 32-bit value 0xDEADBEEF into [4, 8, 12, 8]
    uint8_t bitWidths[]   = { 4, 8, 12, 8 };
    size_t outputWidths[] = { 1, 1, 2, 1 }; // u8, u8, u16, u8

    uint32_t input = 0xDEADBEEF;
    uint8_t out0, out1, out3;
    uint16_t out2;
    void* outputs[] = { &out0, &out1, &out2, &out3 };

    ZL_bitSplitEncode(outputs, outputWidths, 1, &input, 4, bitWidths, 4);

    EXPECT_EQ(out0, 0xF);   // bits 0-3
    EXPECT_EQ(out1, 0xEE);  // bits 4-11
    EXPECT_EQ(out2, 0xADB); // bits 12-23
    EXPECT_EQ(out3, 0xDE);  // bits 24-31

    // Decode back
    const void* inputs[] = { &out0, &out1, &out2, &out3 };
    size_t inputWidths[] = { 1, 1, 2, 1 };
    uint32_t result;
    ZL_bitSplitDecode(&result, 4, 1, inputs, inputWidths, bitWidths, 4);
    EXPECT_EQ(result, 0xDEADBEEF);
}

TEST_F(BitSplitKernelTest, EncodeDecodePartialCoverage)
{
    // Split 32-bit value with only 12 bits used (top 20 bits are zero)
    // Value: 0x00000ABC (bits 0-11 used)
    uint8_t bitWidths[]   = { 3, 4, 5 };
    size_t outputWidths[] = { 1, 1, 1 };

    uint32_t input = 0x00000ABC; // = 0b101010111100
    uint8_t out0, out1, out2;
    void* outputs[] = { &out0, &out1, &out2 };

    ZL_bitSplitEncode(outputs, outputWidths, 1, &input, 4, bitWidths, 3);

    // bits 0-2: 100 = 4
    // bits 3-6: 0111 = 7
    // bits 7-11: 10101 = 21
    EXPECT_EQ(out0, 0x4);
    EXPECT_EQ(out1, 0x7);
    EXPECT_EQ(out2, 0x15);

    // Decode back
    const void* inputs[] = { &out0, &out1, &out2 };
    size_t inputWidths[] = { 1, 1, 1 };
    uint32_t result;
    ZL_bitSplitDecode(&result, 4, 1, inputs, inputWidths, bitWidths, 3);
    EXPECT_EQ(result, input);
}

TEST_F(BitSplitKernelTest, EncodeDecodeSingleStream)
{
    // Single stream: bitSplit(32) on 32-bit value (identity-like)
    uint8_t bitWidths[]   = { 32 };
    size_t outputWidths[] = { 4 }; // u32

    uint32_t input = 0x12345678;
    uint32_t out0;
    void* outputs[] = { &out0 };

    ZL_bitSplitEncode(outputs, outputWidths, 1, &input, 4, bitWidths, 1);
    EXPECT_EQ(out0, 0x12345678);

    const void* inputs[] = { &out0 };
    size_t inputWidths[] = { 4 };
    uint32_t result;
    ZL_bitSplitDecode(&result, 4, 1, inputs, inputWidths, bitWidths, 1);
    EXPECT_EQ(result, 0x12345678);
}

TEST_F(BitSplitKernelTest, EncodeDecodeMultipleElements)
{
    // Test encoding/decoding array of values - kernel owns the loop
    uint8_t bitWidths[]   = { 4, 4 };
    size_t outputWidths[] = { 1, 1 };

    std::vector<uint8_t> input = { 0x12, 0x34, 0x56, 0x78, 0x9A };
    std::vector<uint8_t> stream0(input.size());
    std::vector<uint8_t> stream1(input.size());

    void* outputs[] = { stream0.data(), stream1.data() };

    // Encode all elements in one call
    ZL_bitSplitEncode(
            outputs, outputWidths, input.size(), input.data(), 1, bitWidths, 2);

    // Verify encoded values
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_EQ(stream0[i], input[i] & 0x0F) << "stream0 mismatch at " << i;
        EXPECT_EQ(stream1[i], input[i] >> 4) << "stream1 mismatch at " << i;
    }

    // Decode all elements in one call
    const void* inputs[] = { stream0.data(), stream1.data() };
    size_t inputWidths[] = { 1, 1 };
    std::vector<uint8_t> decoded(input.size());
    ZL_bitSplitDecode(
            decoded.data(), 1, input.size(), inputs, inputWidths, bitWidths, 2);

    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_EQ(decoded[i], input[i]) << "Mismatch at index " << i;
    }
}

TEST_F(BitSplitKernelTest, EncodeDecodeWideValues)
{
    // 64-bit value split into [16, 16, 16, 16]
    uint8_t bitWidths[]   = { 16, 16, 16, 16 };
    size_t outputWidths[] = { 2, 2, 2, 2 }; // all u16

    uint64_t input = 0x123456789ABCDEF0ULL;
    uint16_t out0, out1, out2, out3;
    void* outputs[] = { &out0, &out1, &out2, &out3 };

    ZL_bitSplitEncode(outputs, outputWidths, 1, &input, 8, bitWidths, 4);

    EXPECT_EQ(out0, 0xDEF0);
    EXPECT_EQ(out1, 0x9ABC);
    EXPECT_EQ(out2, 0x5678);
    EXPECT_EQ(out3, 0x1234);

    const void* inputs[] = { &out0, &out1, &out2, &out3 };
    size_t inputWidths[] = { 2, 2, 2, 2 };
    uint64_t result;
    ZL_bitSplitDecode(&result, 8, 1, inputs, inputWidths, bitWidths, 4);
    EXPECT_EQ(result, input);
}

namespace {
uint64_t readElt(const void* src, size_t eltWidth, size_t e)
{
    return ZL_readN((const uint8_t*)src + e * eltWidth, eltWidth);
}

void writeElt(void* dst, size_t eltWidth, size_t e, uint64_t value)
{
    ZL_writeN((uint8_t*)dst + e * eltWidth, value, eltWidth);
}
} // namespace

TEST_F(BitSplitKernelTest, EncodeDecodeGenericLayouts)
{
    // Layouts outside the specialized float patterns, covering every pair of
    // element and field widths, at sizes that cross the kernels' blocks.
    struct Layout {
        size_t eltWidth;
        std::vector<uint8_t> bitWidths;
    };
    const std::vector<Layout> layouts = {
        { 1, { 3, 5 } },  { 1, { 5 } },
        { 2, { 5, 11 } }, { 2, { 8, 8 } },
        { 2, { 7, 8 } },  { 2, { 16 } },
        { 4, { 24, 8 } }, { 4, { 23, 8 } },
        { 4, { 22, 8 } }, { 4, { 1, 15, 16 } },
        { 4, { 12 } },    { 8, { 56, 8 } },
        { 8, { 20, 8 } }, { 8, { 1, 9, 17, 33 } },
        { 8, { 64 } },
    };
    std::mt19937_64 rng(42);
    for (const auto& layout : layouts) {
        size_t const nbWidths  = layout.bitWidths.size();
        size_t const sumWidths = std::accumulate(
                layout.bitWidths.begin(), layout.bitWidths.end(), size_t(0));
        uint64_t const valueMask =
                sumWidths == 64 ? ~0ULL : (1ULL << sumWidths) - 1;
        for (size_t nbElts : { 1, 7, 2048, 5000 }) {
            std::vector<uint8_t> input(nbElts * layout.eltWidth);
            for (size_t e = 0; e < nbElts; e++) {
                writeElt(input.data(), layout.eltWidth, e, rng() & valueMask);
            }

            std::vector<size_t> fieldWidths(nbWidths);
            std::vector<std::vector<uint64_t>> fields(nbWidths);
            std::vector<void*> outputs(nbWidths);
            for (size_t i = 0; i < nbWidths; i++) {
                fieldWidths[i] =
                        ZL_bitSplit_outputEltWidth(layout.bitWidths[i]);
                fields[i].resize(nbElts);
                outputs[i] = fields[i].data();
            }
            ZL_bitSplitEncode(
                    outputs.data(),
                    fieldWidths.data(),
                    nbElts,
                    input.data(),
                    layout.eltWidth,
                    layout.bitWidths.data(),
                    nbWidths);

            for (size_t e = 0; e < nbElts; e++) {
                uint64_t const value =
                        readElt(input.data(), layout.eltWidth, e);
                size_t bitPos = 0;
                for (size_t i = 0; i < nbWidths; i++) {
                    uint64_t const fieldMask =
                            ~0ULL >> (64 - layout.bitWidths[i]);
                    ASSERT_EQ(
                            readElt(fields[i].data(), fieldWidths[i], e),
                            (value >> bitPos) & fieldMask)
                            << "field " << i << " of element " << e
                            << ", layout of width " << layout.eltWidth;
                    bitPos += layout.bitWidths[i];
                }
            }

            std::vector<const void*> inputs(outputs.begin(), outputs.end());
            std::vector<uint8_t> decoded(input.size());
            ZL_bitSplitDecode(
                    decoded.data(),
                    layout.eltWidth,
                    nbElts,
                    inputs.data(),
                    fieldWidths.data(),
                    layout.bitWidths.data(),
                    nbWidths);
            ASSERT_EQ(decoded, input)
                    << nbElts << " elements of width " << layout.eltWidth;
        }
    }
}

} // namespace tests
} // namespace openzl
