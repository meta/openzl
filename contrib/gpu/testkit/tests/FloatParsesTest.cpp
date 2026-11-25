// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>

#include "contrib/gpu/testkit/float_parses.h"
#include "contrib/gpu/testkit/nimble_workloads.h"
#include "contrib/gpu/testkit/parse_plan.h"

// F32LadderTest runs every pattern through compressStep(). These tests cover
// the parameter checks, which the ladder specs never reach.

namespace openzl::gpu::testkit {
namespace {

void make(const WorkloadPattern& pattern, size_t numElts)
{
    generateFrame(
            { .type = NimbleNumericType::F32, .pattern = pattern },
            numElts * sizeof(uint32_t));
}

// A WideAlphabetFloats spec that fits a frame of 4096 values.
WideAlphabetFloats wideAlphabet(uint32_t cardinality)
{
    return { .cardinality      = cardinality,
             .highMove         = 0.25,
             .longLiteralEvery = 100,
             .maxGapLog2       = 19,
             .gapGrow          = 0.35,
             .bigGapEvery      = 0,
             .gapPairs         = 8,
             .lowByteStep      = 16,
             .maxLiteralRun    = 4,
             .minCopy          = 3,
             .maxCopy          = 5,
             .longCopyLength   = 40,
             .reuseShare       = 0.35,
             .distanceOctaves  = 6 };
}

TEST(FloatParsesTest, rejectParametersThatDoNotFit)
{
    // An empty octave range for the long copies.
    EXPECT_THROW(
            make(CopyStreamFloats{ .literals             = RandomFloatBits{},
                                   .longLiteralEvery     = 10,
                                   .longLiteralMinOctave = 4,
                                   .longLiteralMaxOctave = 5,
                                   .shortLiteralEvery    = 3,
                                   .shortLiteralContinue = 0,
                                   .copyContinue         = 0,
                                   .longCopyEvery        = 50,
                                   .longCopyMinOctave    = 4,
                                   .longCopyMaxOctave    = 4,
                                   .minDistanceOctave    = 0,
                                   .maxDistanceOctave    = 10 },
                 65536),
            std::invalid_argument);
    WideAlphabetFloats shortCopies = wideAlphabet(1000);
    shortCopies.minCopy            = 4;
    shortCopies.maxCopy            = 3;
    EXPECT_THROW(make(shortCopies, 4096), std::invalid_argument);
    EXPECT_THROW(
            make(GapAlphabetFloats{ .cardinality = 300 }, 256),
            std::invalid_argument);
    EXPECT_THROW(
            make(PlannedFloats{ .literals        = RandomFloatBits{},
                                .copies          = 1,
                                .copyLength      = 0,
                                .copyOffsetCodes = 4 },
                 1024),
            std::invalid_argument);
    // Copies need exactly one distance option.
    EXPECT_THROW(
            make(PlannedFloats{ .literals = RandomFloatBits{}, .copies = 1 },
                 1024),
            std::invalid_argument);
    EXPECT_THROW(
            make(PlannedFloats{ .literals        = RandomFloatBits{},
                                .copies          = 1,
                                .copyOffsetCodes = 4,
                                .minCopyDistance = 16 },
                 1024),
            std::invalid_argument);
    // Copy distances of 2^64 or more, and runs longer than any frame.
    EXPECT_THROW(
            make(PlannedFloats{ .literals        = RandomFloatBits{},
                                .copies          = 1,
                                .copyOffsetCodes = 64 },
                 1024),
            std::invalid_argument);
    EXPECT_THROW(
            make(PlannedFloats{ .literals  = RandomFloatBits{},
                                .runs      = 1,
                                .runLength = 0xFFFFFFFE },
                 1024),
            std::invalid_argument);
}

TEST(FloatParsesTest, uniformBelowRejectsAnEmptyRange)
{
    std::mt19937_64 rng;
    EXPECT_THROW(uniformBelow(rng, 0), std::invalid_argument);
}

TEST(FloatParsesTest, rejectSpecsThatWouldOverrunTheirBuffers)
{
    // Runs of length 0.
    EXPECT_THROW(
            make(PlannedFloats{ .literals = RandomFloatBits{}, .runs = 3 },
                 512),
            std::invalid_argument);
    // Tail exponents below the normal range.
    EXPECT_THROW(
            make(PlannedFloats{ .literals =
                                        PositiveFloats{ .tailShare   = 0.5,
                                                        .tailOctaves = 200 } },
                 1024),
            std::invalid_argument);
    // Exponents below the normal range.
    EXPECT_THROW(
            make(PlannedFloats{ .literals =
                                        SignedOctaveFloats{ .octaves = 127 } },
                 1024),
            std::invalid_argument);
    // A byte 3 jitter of 8 can reach the infinity and NaN exponents.
    EXPECT_THROW(
            make(PlannedFloats{ .literals =
                                        WindowFloats{
                                                .patternShare = { 0, 0, 0 },
                                                .commonShare  = { 0, 0, 0 },
                                                .byte3Move    = 0,
                                                .byte3Grow    = 0,
                                                .byte3Jitter  = 8 } },
                 1024),
            std::invalid_argument);
    // Gaps of up to 2^20 reach the Inf patterns within 10000 values.
    EXPECT_THROW(make(wideAlphabet(10000), 16384), std::invalid_argument);
}

} // namespace
} // namespace openzl::gpu::testkit
