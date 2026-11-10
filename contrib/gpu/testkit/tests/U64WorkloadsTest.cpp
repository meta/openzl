// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <set>
#include <string>

#include "contrib/gpu/testkit/nimble_workloads.h"
#include "contrib/gpu/testkit/tests/f32_ladder_test_util.h"
#include "contrib/gpu/testkit/u64_workloads.h"
#include "openzl/openzl.hpp"

namespace openzl::gpu::testkit {
namespace {

TEST(U64WorkloadsTest, namesTypesAndFrameSizes)
{
    std::set<std::string> names;
    for (const LadderStep& s : u64Workloads()) {
        names.insert(s.name);
        EXPECT_EQ(s.name.rfind("u64/", 0), 0U) << s.name;
        EXPECT_EQ(s.spec.type, NimbleNumericType::U64) << s.name;
        EXPECT_EQ(s.frameBytes & (s.frameBytes - 1), 0U) << s.name;
    }
    EXPECT_EQ(names.size(), u64Workloads().size());
}

class U64WorkloadTest : public ::testing::TestWithParam<const LadderStep*> {};

TEST_P(U64WorkloadTest, framesKeepTheirGraph)
{
    const LadderStep& s = *GetParam();
    expectGraphOnEveryFrame(
            s, [&](const std::string& raw) { return compressStep(s, raw); });
}

INSTANTIATE_TEST_SUITE_P(
        Workloads,
        U64WorkloadTest,
        ::testing::ValuesIn(stepsOf(u64Workloads())),
        stepTestName);

TEST(U64WorkloadsTest, framesRoundTripAndCompress)
{
    for (const LadderStep& s : u64Workloads()) {
        const std::string raw        = generateFrame(s.spec, s.frameBytes);
        const std::string compressed = compressStep(s, raw);
        EXPECT_EQ(openzl::DCtx{}.decompressSerial(compressed), raw) << s.name;
        EXPECT_LT(compressed.size(), raw.size()) << s.name;
    }
}

} // namespace
} // namespace openzl::gpu::testkit
