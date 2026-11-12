// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/openzl.hpp"
#include "tests/codecs/test_codec.h"

namespace openzl {
namespace tests {

class BitsplitBF16Test : public CodecTest {
   public:
    /**
     * Tests round-trip compression/decompression using bitsplit_bf16.
     *
     * @param input Numeric input data (2-byte bfloat16 elements as uint16_t)
     */
    void testBitsplitBF16RoundTrip(const std::vector<uint16_t>& input)
    {
        compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);

        auto graph = compressor_.buildStaticGraph(
                ZL_NODE_BITSPLIT_BF16, { graphs::Compress{}() });
        compressor_.selectStartingGraph(graph);

        Input numericInput =
                Input::refNumeric(poly::span<const uint16_t>{ input });
        testRoundTrip(numericInput);
    }
};

// =============================================================================
// BF16 Round-Trip Tests
// =============================================================================
// bfloat16 layout: [sign:1][exponent:8][mantissa:7] = 16 bits
// No native C++ type, so we use uint16_t with manually constructed bit
// patterns.

TEST_F(BitsplitBF16Test, BF16_RoundTrip)
{
    // Sweep through a range of normal bf16 values
    std::vector<uint16_t> input(1000);
    for (size_t i = 0; i < input.size(); i++) {
        // Range from small positive normals to larger ones
        // bf16 exponent field starts at bit 7, normal range [0x0080..0x7F7F]
        input[i] = static_cast<uint16_t>(0x0080 + (i % 0x7F00));
    }
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_NegativeRoundTrip)
{
    // Sweep through a range of negative normal bf16 values
    std::vector<uint16_t> input(1000);
    for (size_t i = 0; i < input.size(); i++) {
        // Range from small negative normals to larger ones
        // bf16 exponent field starts at bit 7, normal range [0x0080..0x7F7F]
        input[i] = static_cast<uint16_t>(0x8000 | (0x0080 + (i % 0x7F00)));
    }
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_AllZero)
{
    std::vector<uint16_t> input(1000, 0x0000);
    testBitsplitBF16RoundTrip(input);
}

// =============================================================================
// BF16 Edge Cases
// =============================================================================

TEST_F(BitsplitBF16Test, BF16_EmptyInput)
{
    std::vector<uint16_t> input;
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_SingleElement)
{
    // 0x4049 ~= 3.14 in bfloat16 (truncated from fp32 0x4048F5C3)
    std::vector<uint16_t> input{ 0x4049 };
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_NaN)
{
    // bf16 quiet NaN: exponent all 1s (0xFF at bits 7-14), mantissa nonzero
    std::vector<uint16_t> input{ 0x7FC0 };
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_NaNVariants)
{
    // Signaling NaN: exponent all 1s, mantissa nonzero with MSB=0
    // Negative NaN: sign=1, exponent all 1s, mantissa nonzero
    std::vector<uint16_t> input{
        0x7F01, // positive signaling NaN
        0xFF01, // negative signaling NaN
        0xFFC0, // negative quiet NaN
    };
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_Infinity)
{
    std::vector<uint16_t> input{
        0x7F80, // +Inf (exponent all 1s, mantissa zero)
        0xFF80, // -Inf
    };
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_NegativeZero)
{
    std::vector<uint16_t> input{ 0x8000 };
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_MinMax)
{
    std::vector<uint16_t> input{
        0x0080, // smallest normal (2^-126)
        0x7F7F, // largest finite
    };
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_Subnormals)
{
    // bf16 subnormals: exponent = 0, mantissa nonzero (bits 0-6)
    std::vector<uint16_t> input(127);
    for (size_t i = 0; i < input.size(); i++) {
        // Subnormal mantissa range: 0x0001..0x007F
        input[i] = static_cast<uint16_t>((i % 0x7F) + 1);
    }
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_NegativeSubnormals)
{
    // bf16 negative subnormals: sign=1, exponent=0, mantissa nonzero (bits 0-6)
    std::vector<uint16_t> input(127);
    for (size_t i = 0; i < input.size(); i++) {
        // Negative subnormal range: 0x8001..0x807F
        input[i] = static_cast<uint16_t>(0x8000 | ((i % 0x7F) + 1));
    }
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_AllBitsSet)
{
    // 0xFFFF = negative NaN with all mantissa bits set
    std::vector<uint16_t> input{ 0xFFFF };
    testBitsplitBF16RoundTrip(input);
}

TEST_F(BitsplitBF16Test, BF16_MixedValues)
{
    std::vector<uint16_t> input{
        0x0000, // +0
        0x8000, // -0
        0x3F80, // 1.0
        0xBF80, // -1.0
        0x4049, // ~3.14
        0x7F80, // +Inf
        0xFF80, // -Inf
        0x7FC0, // NaN
        0x0001, // smallest subnormal
        0x7F7F, // largest finite
    };
    testBitsplitBF16RoundTrip(input);
}

// =============================================================================
// Split Order Case
// =============================================================================

TEST_F(BitsplitBF16Test, BF16_OutputOrder)
{
    // 0xC049 = negative ~3.14 in bfloat16
    std::vector<uint16_t> input{ 0xC049 };

    uint16_t rawBits;
    std::memcpy(&rawBits, input.data(), sizeof(rawBits));
    uint8_t mantissaVal = rawBits & 0x7F;        // bottom 7 bits
    uint8_t exponentVal = (rawBits >> 7) & 0xFF; // next 8 bits
    uint8_t signVal     = (rawBits >> 15) & 0x1; // top 1 bit

    std::vector<uint8_t> expectedMantissa{ mantissaVal };
    std::vector<uint8_t> expectedExponent{ exponentVal };
    std::vector<uint8_t> expectedSign{ signVal };

    Input inMantissa =
            Input::refNumeric(poly::span<const uint8_t>{ expectedMantissa });
    Input inExponent =
            Input::refNumeric(poly::span<const uint8_t>{ expectedExponent });
    Input inSign = Input::refNumeric(poly::span<const uint8_t>{ expectedSign });

    Input numericInput = Input::refNumeric(poly::span<const uint16_t>{ input });
    testCodecVO(
            ZL_NODE_BITSPLIT_BF16,
            numericInput,
            { &inMantissa, &inExponent, &inSign },
            24); // 24 is version where bitsplit codecs were added
}
} // namespace tests
} // namespace openzl
