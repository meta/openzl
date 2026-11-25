// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "contrib/gpu/testkit/frame_verifier.h"
#include "contrib/gpu/testkit/mixed_workload.h"
#include "openzl/openzl.hpp"

namespace openzl::gpu::testkit {
namespace {

constexpr size_t kMiB = size_t{ 1 } << 20;

// The mixed workload at a smaller size, so the tests stay fast. Every part
// still gets at least one frame.
MixedWorkload smallMix(size_t decodedBytes)
{
    MixedWorkload mix = f32U64ZstdMixedWorkload();
    mix.decodedBytes  = decodedBytes;
    return mix;
}

TEST(MixedWorkloadTest, sharesSumToOne)
{
    double sum = 0;
    for (const MixedWorkloadPart& part : f32U64ZstdMixedWorkload().parts) {
        ASSERT_NE(part.step, nullptr);
        EXPECT_GT(part.share, 0);
        sum += part.share;
    }
    EXPECT_NEAR(sum, 1, 1e-9);
}

TEST(MixedWorkloadTest, eachPartGetsItsShareOfTheBytes)
{
    const MixedWorkload mix = smallMix(32 * kMiB);

    const MixedFrames frames = makeMixedFrames(mix, 1, 8);

    std::vector<size_t> bytes(mix.parts.size());
    for (size_t i = 0; i < frames.raw.size(); ++i) {
        bytes[frames.part[i]] += frames.raw[i].size();
        EXPECT_EQ(
                frames.raw[i].size(),
                mix.parts[frames.part[i]].step->frameBytes);
    }
    for (size_t p = 0; p < mix.parts.size(); ++p) {
        const MixedWorkloadPart& part = mix.parts[p];
        // Rounding to whole frames moves at most half a frame.
        const double slack = static_cast<double>(part.step->frameBytes) / 2;
        EXPECT_NEAR(
                static_cast<double>(bytes[p]),
                part.share * static_cast<double>(mix.decodedBytes),
                slack)
                << part.step->name;
    }
}

TEST(MixedWorkloadTest, framesKeepTheirGraphsAndRoundTrip)
{
    const MixedWorkload mix = smallMix(8 * kMiB);

    const MixedFrames frames = makeMixedFrames(mix, 3, 8);

    openzl::DCtx dctx;
    for (size_t i = 0; i < frames.raw.size(); ++i) {
        const LadderStep& step = *mix.parts[frames.part[i]].step;
        EXPECT_EQ(decodeGraphShape(frames.compressed[i]), step.graph)
                << step.name << " frame " << i;
        EXPECT_EQ(dctx.decompressSerial(frames.compressed[i]), frames.raw[i])
                << step.name << " frame " << i;
    }
}

// A bench splits the frames into runs, one for each thread, so every run of
// frames must mix the parts.
TEST(MixedWorkloadTest, framesMixTheParts)
{
    const MixedWorkload mix = smallMix(4 * kMiB);

    const MixedFrames frames = makeMixedFrames(mix, 1, 8);

    const std::set<size_t> firstEighth(
            frames.part.begin(), frames.part.begin() + frames.part.size() / 8);
    EXPECT_GT(firstEighth.size(), 1U);
}

TEST(MixedWorkloadTest, framesDoNotDependOnThreads)
{
    const MixedWorkload mix = smallMix(4 * kMiB);

    const MixedFrames one  = makeMixedFrames(mix, 5, 1);
    const MixedFrames four = makeMixedFrames(mix, 5, 4);

    EXPECT_EQ(one.part, four.part);
    EXPECT_EQ(one.raw, four.raw);
    EXPECT_EQ(one.compressed, four.compressed);
}

} // namespace
} // namespace openzl::gpu::testkit
