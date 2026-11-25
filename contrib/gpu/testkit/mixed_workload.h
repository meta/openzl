// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "contrib/gpu/testkit/f32_ladder.h"

// Mixed workloads: frames of several workloads, with different graphs and
// sizes, that a benchmark decodes together. A mixed workload lists the
// workloads that its frames come from and each one's share of its decoded
// bytes.

namespace openzl::gpu::testkit {

// The ladder steps, then the float32 extra workloads, the zstd workloads, and
// the 64-bit workloads.
const std::vector<LadderStep>& allWorkloads();

struct MixedWorkloadPart {
    const LadderStep* step;
    // Share of the mixed workload's decoded bytes. The shares of a mixed
    // workload sum to 1. Each frame holds the step's frame size, at which
    // compressStep() writes the step's graph.
    double share;
};

struct MixedWorkload {
    std::string name;
    size_t decodedBytes;
    std::vector<MixedWorkloadPart> parts;
};

// Plain zstd, 64-bit, and float32 frames from 128 bytes to 2 MiB: every zstd
// workload, every 64-bit workload, and eight float32 workloads from the ladder
// and the extra workloads.
const MixedWorkload& f32U64ZstdMixedWorkload();

struct MixedFrames {
    // The part of each frame.
    std::vector<size_t> part;
    std::vector<std::string> raw;
    std::vector<std::string> compressed;
};

// Writes the frames of a mixed workload with compressStep(), in an order that
// `seed` shuffles, so that any run of frames mixes the parts. Part i gets
// round(share * decodedBytes / step frameBytes) frames, at least one. Frame j
// of a part is frame j of its step's workload with seed `seed`. The frames do
// not depend on `threads`. Throws what generateFrame() or compressStep()
// throws.
MixedFrames
makeMixedFrames(const MixedWorkload& mix, uint64_t seed, size_t threads = 1);

} // namespace openzl::gpu::testkit
