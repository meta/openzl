// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "contrib/gpu/testkit/f32_ladder.h"

// Workloads of plain zstd frames. Their graph is one zstd codec, "22:b(b)",
// and zstd has no selector, so compressStep() writes the same graph for every
// seed.

namespace openzl::gpu::testkit {

// "zstd/uN-<size>" holds frames of <size> bytes, each a column of N-bit values
// from PinnedValues.
const std::vector<LadderStep>& zstdWorkloads();

} // namespace openzl::gpu::testkit
