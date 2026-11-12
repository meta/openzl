// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <string>

#include "contrib/gpu/testkit/nimble_graph.h"
#include "contrib/gpu/testkit/tests/f32_ladder_test_util.h"
#include "contrib/gpu/testkit/u64_workloads.h"

namespace openzl::gpu::testkit {
namespace {

class U64WorkloadsNimbleTest
        : public ::testing::TestWithParam<const LadderStep*> {};

// Checks that compressLikeNimble() still writes each 64-bit workload's graph,
// so the builders, the expected text, and Nimble agree. A failure here means
// that OpenZL's selectors or the workload changed, not that compressStep()
// broke.
TEST_P(U64WorkloadsNimbleTest, nimbleWritesTheSameGraph)
{
    const LadderStep& s = *GetParam();
    expectGraphOnEveryFrame(s, [&](const std::string& raw) {
        return compressLikeNimble(raw, s.spec.type);
    });
}

INSTANTIATE_TEST_SUITE_P(
        Workloads,
        U64WorkloadsNimbleTest,
        ::testing::ValuesIn(stepsOf(u64Workloads())),
        stepTestName);

} // namespace
} // namespace openzl::gpu::testkit
