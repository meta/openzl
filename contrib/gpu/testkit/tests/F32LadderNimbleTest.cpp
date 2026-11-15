// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <string>

#include "contrib/gpu/testkit/f32_ladder.h"
#include "contrib/gpu/testkit/nimble_graph.h"
#include "contrib/gpu/testkit/tests/f32_ladder_test_util.h"

namespace openzl::gpu::testkit {
namespace {

class F32LadderNimbleTest : public ::testing::TestWithParam<const LadderStep*> {
};

// Each ladder step's graph is one that Nimble's numeric graph writes. This
// test checks that compressLikeNimble() still writes each
// step's graph for the step's workload, so the builders, the expected text,
// and Nimble agree. A failure here means that OpenZL's selectors or the
// workload changed, not that compressStep() broke. Nimble's entropy graphs
// cut their own blocks, so the test does not compare block counts with
// compressStep().
TEST_P(F32LadderNimbleTest, nimbleWritesTheSameGraph)
{
    const LadderStep& s = *GetParam();
    expectGraphOnEveryFrame(s, [&](const std::string& raw) {
        return compressLikeNimble(raw, s.spec.type);
    });
}

INSTANTIATE_TEST_SUITE_P(
        Steps,
        F32LadderNimbleTest,
        ::testing::ValuesIn(stepsOf(f32Ladder())),
        stepTestName);

INSTANTIATE_TEST_SUITE_P(
        ExtraWorkloads,
        F32LadderNimbleTest,
        ::testing::ValuesIn(stepsOf(f32ExtraWorkloads())),
        stepTestName);

} // namespace
} // namespace openzl::gpu::testkit
