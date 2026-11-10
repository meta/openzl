// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/zstd_workloads.h"

#include <string>
#include <utility>

#include "openzl/compress/private_nodes.h"

namespace openzl::gpu::testkit {

namespace {

constexpr size_t kKiB = 1024;

// 22:b(b)
openzl::GraphID zstd(openzl::Compressor& c)
{
    return c.buildStaticGraph(ZL_NODE_ZSTD, { ZL_GRAPH_STORE });
}

// Frames of `frameBytes` that hold a PinnedValues column of `type` with
// `cardinality` distinct values from 0 through `range`.
LadderStep zstdStep(
        std::string name,
        NimbleNumericType type,
        uint64_t range,
        uint64_t cardinality,
        size_t frameBytes)
{
    return { .name            = std::move(name),
             .spec            = { .type    = type,
                                  .pattern = PinnedValues{ .base        = 0,
                                                           .range       = range,
                                                           .cardinality = cardinality } },
             .frameBytes      = frameBytes,
             .buildGraph      = zstd,
             .graph           = "22:b(b)",
             .trivialInNimble = false };
}

// Nimble writes a numeric column with its numeric graph, not with plain zstd,
// so an end-to-end Nimble benchmark cannot build these frames. A small frame
// holds few values, so it draws from fewer distinct values.
std::vector<LadderStep> buildZstdWorkloads()
{
    constexpr auto kU8          = NimbleNumericType::U8;
    constexpr auto kU32         = NimbleNumericType::U32;
    constexpr auto kU64         = NimbleNumericType::U64;
    constexpr uint64_t kRange32 = uint64_t{ 1 } << 16;
    constexpr uint64_t kRange64 = uint64_t{ 1 } << 32;
    return {
        zstdStep("zstd/u8-4k", kU8, 255, 64, 4 * kKiB),
        zstdStep("zstd/u32-128b", kU32, kRange32, 8, 128),
        zstdStep("zstd/u32-256b", kU32, kRange32, 16, 256),
        zstdStep("zstd/u32-512b", kU32, kRange32, 64, 512),
        zstdStep("zstd/u32-1k", kU32, kRange32, 256, 1 * kKiB),
        zstdStep("zstd/u32-16k", kU32, kRange32, 2048, 16 * kKiB),
        zstdStep("zstd/u32-32k", kU32, kRange32, 2048, 32 * kKiB),
        zstdStep("zstd/u32-64k", kU32, kRange32, 2048, 64 * kKiB),
        zstdStep("zstd/u64-64k", kU64, kRange64, 4096, 64 * kKiB),
        zstdStep("zstd/u64-512k", kU64, kRange64, 4096, 512 * kKiB),
        zstdStep("zstd/u64-2m", kU64, kRange64, 4096, 2048 * kKiB),
    };
}

} // namespace

const std::vector<LadderStep>& zstdWorkloads()
{
    static const std::vector<LadderStep> workloads = buildZstdWorkloads();
    return workloads;
}

} // namespace openzl::gpu::testkit
