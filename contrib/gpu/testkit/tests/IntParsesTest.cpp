// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>

#include "contrib/gpu/testkit/int_parses.h"
#include "contrib/gpu/testkit/nimble_workloads.h"

// U64WorkloadsTest runs the workload specs through compressStep(), but it does
// not check the bytes that each lane model writes, and the specs pass no
// invalid parameters. These tests cover both.

namespace openzl::gpu::testkit {
namespace {

constexpr size_t kValues = 4096;

std::array<IntLane, 8> randomLanes()
{
    std::array<IntLane, 8> lanes;
    lanes.fill(RandomLane{});
    return lanes;
}

std::string make(const PlannedInts& p, NimbleNumericType type)
{
    return generateFrame(
            { .type = type, .pattern = p }, kValues * sizeof(uint64_t));
}

// The bytes of lane `b` of every value in `frame`.
std::string lane(const std::string& frame, size_t b)
{
    std::string bytes;
    for (size_t i = b; i < frame.size(); i += 8) {
        bytes.push_back(frame[i]);
    }
    return bytes;
}

TEST(IntParsesTest, lanesFollowTheirModels)
{
    PlannedInts p{ .lanes = randomLanes() };
    p.lanes[1] = ConstantLane{ 0x5A };
    p.lanes[2] = SkewedLane{ 5 };
    p.lanes[3] = WalkLane{ .move = 0.1, .grow = 0 };

    const std::string frame = make(p, NimbleNumericType::U64);

    ASSERT_EQ(frame.size(), kValues * 8);
    EXPECT_EQ(lane(frame, 1), std::string(kValues, '\x5A'));
    const std::string skewed = lane(frame, 2);
    EXPECT_EQ(std::set<char>(skewed.begin(), skewed.end()).size(), 5U);
    // A walk with grow 0 stays put or steps by exactly one.
    const std::string walk = lane(frame, 3);
    for (size_t i = 1; i < walk.size(); ++i) {
        const auto step = static_cast<uint8_t>(walk[i] - walk[i - 1]);
        EXPECT_TRUE(step == 0 || step == 1 || step == 255) << i;
    }
}

TEST(IntParsesTest, copiesRepeatEarlierValues)
{
    const PlannedInts p{ .lanes = randomLanes(),
                         .parse = { .copies          = 64,
                                    .copyLength      = 4,
                                    .copyOffsetCodes = 8,
                                    .maxCopyLength   = 16 } };

    const std::string frame = make(p, NimbleNumericType::U64);

    std::set<std::string> values;
    for (size_t i = 0; i < frame.size(); i += 8) {
        values.insert(frame.substr(i, 8));
    }
    EXPECT_LT(values.size(), kValues - 64 * 3);
}

TEST(IntParsesTest, checksItsParameters)
{
    PlannedInts p{ .lanes = randomLanes() };
    EXPECT_THROW(make(p, NimbleNumericType::F32), std::invalid_argument);

    p.lanes[0] = SkewedLane{ 1 };
    EXPECT_THROW(make(p, NimbleNumericType::U64), std::invalid_argument);
    p.lanes[0] = PatternLane{ .patterns = 0 };
    EXPECT_THROW(make(p, NimbleNumericType::U64), std::invalid_argument);
    p.lanes[0] = WalkLane{ .move = 0.5, .grow = 1 };
    EXPECT_THROW(make(p, NimbleNumericType::U64), std::invalid_argument);

    p.lanes[0] = RandomLane{};
    p.parse    = { .copies = 1, .copyLength = 2 };
    EXPECT_THROW(make(p, NimbleNumericType::U64), std::invalid_argument);
    p.parse = { .copies          = 1,
                .copyLength      = 2,
                .minCopyDistance = 1000,
                .maxCopyLength   = 4000 };
    EXPECT_THROW(make(p, NimbleNumericType::U64), std::invalid_argument);
}

} // namespace
} // namespace openzl::gpu::testkit
