// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/openzl.hpp"
#include "tests/codecs/test_codec.h"

namespace openzl {
namespace tests {

class BitsplitFPTest : public CodecTest {
   public:
    /**
     * Tests round-trip compression/decompression using bitsplit_fp.
     *
     * @param input Numeric input data (2, 4, or 8-byte IEEE 754 elements)
     */
    template <typename T>
    void testBitsplitFPRoundTrip(const std::vector<T>& input)
    {
        compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);

        auto graph = compressor_.buildStaticGraph(
                ZL_NODE_BITSPLIT_FP, { graphs::Compress{}() });
        compressor_.selectStartingGraph(graph);

        Input numericInput = Input::refNumeric(poly::span<const T>{ input });
        testRoundTrip(numericInput);
    }
};

// =============================================================================
// FP32 Round-Trip Tests
// =============================================================================

TEST_F(BitsplitFPTest, FP32_RoundTrip)
{
    std::vector<float> input(1000);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<float>(i) * 0.1f;
    }
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP32_NegativeRoundTrip)
{
    std::vector<float> input(1000);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<float>(i) * -0.1f;
    }
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP32_AllZero)
{
    std::vector<float> input(1000, 0.0f);
    testBitsplitFPRoundTrip(input);
}

// =============================================================================
// FP32 Edge Cases
// =============================================================================

TEST_F(BitsplitFPTest, FP32_EmptyInput)
{
    std::vector<float> input;
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP32_SingleElement)
{
    std::vector<float> input{ 999.999f };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP32_NaN)
{
    std::vector<float> input{ std::numeric_limits<float>::quiet_NaN() };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP32_Infinity)
{
    std::vector<float> input{
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP32_NegativeZero)
{
    std::vector<float> input{ -0.0f };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP32_MinMax)
{
    std::vector<float> input{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max(),
    };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP32_Subnormals)
{
    std::vector<float> input(1000);
    const float subnormal = std::numeric_limits<float>::denorm_min();
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = subnormal * static_cast<float>(i + 1);
    }
    testBitsplitFPRoundTrip(input);
}

// =============================================================================
// FP64 Round-Trip Tests
// =============================================================================

TEST_F(BitsplitFPTest, FP64_RoundTrip)
{
    std::vector<double> input(1000);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<double>(i) * 0.1;
    }
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP64_NegativeRoundTrip)
{
    std::vector<double> input(1000);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<double>(i) * -0.1;
    }
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP64_AllZero)
{
    std::vector<double> input(1000, 0.0);
    testBitsplitFPRoundTrip(input);
}

// =============================================================================
// FP64 Edge Cases
// =============================================================================

TEST_F(BitsplitFPTest, FP64_EmptyInput)
{
    std::vector<double> input;
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP64_SingleElement)
{
    std::vector<double> input{ 123456.789012 };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP64_NaN)
{
    std::vector<double> input{ std::numeric_limits<double>::quiet_NaN() };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP64_Infinity)
{
    std::vector<double> input{
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP64_NegativeZero)
{
    std::vector<double> input{ -0.0 };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP64_MinMax)
{
    std::vector<double> input{
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::max(),
    };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP64_Subnormals)
{
    std::vector<double> input(1000);
    const double subnormal = std::numeric_limits<double>::denorm_min();
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = subnormal * static_cast<double>(i + 1);
    }
    testBitsplitFPRoundTrip(input);
}

// =============================================================================
// FP16 Round-Trip Tests
// =============================================================================
// IEEE 754 half-precision: 1 sign + 5 exponent + 10 mantissa
// No native C++ type, so we use uint16_t with manually constructed bit
// patterns.

TEST_F(BitsplitFPTest, FP16_RoundTrip)
{
    // Sweep through a range of normal fp16 values
    std::vector<uint16_t> input(1000);
    for (size_t i = 0; i < input.size(); i++) {
        // Range from small positive normals to larger ones
        // Exponent field [0x0400..0x7BFF] covers all normal fp16 values
        input[i] = static_cast<uint16_t>(0x0400 + (i % 0x7800));
    }
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP16_NegativeRoundTrip)
{
    // Sweep through a range of negative normal fp16 values
    std::vector<uint16_t> input(1000);
    for (size_t i = 0; i < input.size(); i++) {
        // Range from small negative normals to larger ones
        // Exponent field [0x0400..0x7BFF] covers all normal fp16 values
        input[i] = static_cast<uint16_t>(0x8000 | (0x0400 + (i % 0x7800)));
    }
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP16_AllZero)
{
    std::vector<uint16_t> input(1000, 0x0000);
    testBitsplitFPRoundTrip(input);
}

// =============================================================================
// FP16 Edge Cases
// =============================================================================

TEST_F(BitsplitFPTest, FP16_EmptyInput)
{
    std::vector<uint16_t> input;
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP16_SingleElement)
{
    // 0x3C00 = 1.0 in fp16
    std::vector<uint16_t> input{ 0x3C00 };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP16_NaN)
{
    // fp16 quiet NaN: exponent all 1s, mantissa nonzero
    std::vector<uint16_t> input{ 0x7E00 };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP16_Infinity)
{
    std::vector<uint16_t> input{
        0x7C00, // +Inf
        0xFC00, // -Inf
    };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP16_NegativeZero)
{
    std::vector<uint16_t> input{ 0x8000 };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP16_MinMax)
{
    std::vector<uint16_t> input{
        0x0400, // smallest normal (2^-14)
        0x7BFF, // largest finite (65504.0)
    };
    testBitsplitFPRoundTrip(input);
}

TEST_F(BitsplitFPTest, FP16_Subnormals)
{
    // fp16 subnormals: exponent = 0, mantissa nonzero
    std::vector<uint16_t> input(1000);
    for (size_t i = 0; i < input.size(); i++) {
        // Subnormal mantissa range: 0x0001..0x03FF
        input[i] = static_cast<uint16_t>((i % 0x03FF) + 1);
    }
    testBitsplitFPRoundTrip(input);
}

// =============================================================================
// Split Order Cases
// =============================================================================

TEST_F(BitsplitFPTest, FP32_OutputOrder)
{
    std::vector<float> input{ -3.14f };

    uint32_t rawBits;
    // write the float to 4 bytes of raw bits

    std::memcpy(&rawBits, input.data(), sizeof(rawBits));
    uint32_t mantissaVal = rawBits & 0x7FFFFF;     // bottom 23 bits
    uint8_t exponentVal  = (rawBits >> 23) & 0xFF; // next 8 bits
    uint8_t signVal      = (rawBits >> 31) & 0x1;  // top 1 bit

    std::vector<uint32_t> expectedMantissa{ mantissaVal };
    std::vector<uint8_t> expectedExponent{ exponentVal };
    std::vector<uint8_t> expectedSign{ signVal };

    Input inMantissa =
            Input::refNumeric(poly::span<const uint32_t>{ expectedMantissa });
    Input inExponent =
            Input::refNumeric(poly::span<const uint8_t>{ expectedExponent });
    Input inSign = Input::refNumeric(poly::span<const uint8_t>{ expectedSign });

    Input numericInput = Input::refNumeric(poly::span<const float>{ input });
    testCodecVO(
            ZL_NODE_BITSPLIT_FP,
            numericInput,
            { &inMantissa, &inExponent, &inSign },
            24); // 24 is version where we added this codec
}

TEST_F(BitsplitFPTest, FP64_OutputOrder)
{
    std::vector<double> input{ -3.14 };

    uint64_t rawBits;
    std::memcpy(&rawBits, input.data(), sizeof(rawBits));
    uint64_t mantissaVal = rawBits & 0xFFFFFFFFFFFFF; // bottom 52 bits
    uint16_t exponentVal = (rawBits >> 52) & 0x7FF;   // next 11 bits
    uint8_t signVal      = (rawBits >> 63) & 0x1;     // top 1 bit

    std::vector<uint64_t> expectedMantissa{ mantissaVal };
    std::vector<uint16_t> expectedExponent{ exponentVal };
    std::vector<uint8_t> expectedSign{ signVal };

    Input inMantissa =
            Input::refNumeric(poly::span<const uint64_t>{ expectedMantissa });
    Input inExponent =
            Input::refNumeric(poly::span<const uint16_t>{ expectedExponent });
    Input inSign = Input::refNumeric(poly::span<const uint8_t>{ expectedSign });

    Input numericInput = Input::refNumeric(poly::span<const double>{ input });
    testCodecVO(
            ZL_NODE_BITSPLIT_FP,
            numericInput,
            { &inMantissa, &inExponent, &inSign },
            24); // 24 is version where we added this codec
}

TEST_F(BitsplitFPTest, FP16_OutputOrder)
{
    // -3.14 in fp16 = 0xC248
    std::vector<uint16_t> input{ 0xC248 };

    uint16_t rawBits;
    std::memcpy(&rawBits, input.data(), sizeof(rawBits));
    uint16_t mantissaVal = rawBits & 0x3FF;        // bottom 10 bits
    uint8_t exponentVal  = (rawBits >> 10) & 0x1F; // next 5 bits
    uint8_t signVal      = (rawBits >> 15) & 0x1;  // top 1 bit

    std::vector<uint16_t> expectedMantissa{ mantissaVal };
    std::vector<uint8_t> expectedExponent{ exponentVal };
    std::vector<uint8_t> expectedSign{ signVal };

    Input inMantissa =
            Input::refNumeric(poly::span<const uint16_t>{ expectedMantissa });
    Input inExponent =
            Input::refNumeric(poly::span<const uint8_t>{ expectedExponent });
    Input inSign = Input::refNumeric(poly::span<const uint8_t>{ expectedSign });

    Input numericInput = Input::refNumeric(poly::span<const uint16_t>{ input });
    testCodecVO(
            ZL_NODE_BITSPLIT_FP,
            numericInput,
            { &inMantissa, &inExponent, &inSign },
            24); // 24 is version where we added this codec
}
} // namespace tests
} // namespace openzl
