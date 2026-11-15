// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <variant>

#include "contrib/gpu/testkit/parse_plan.h"

// 64-bit integer streams with a planned field_lz parse. field_lz cuts the
// literals of 8-byte values into 8 byte lanes, and the literal graph chooses
// a codec for each lane, so each lane of a literal comes from its own model.
// Each frame draws all its values and tables from its own seed.

namespace openzl::gpu::testkit {

// Uniform bytes. The literal graph stores them.
struct RandomLane {};

// The same byte in every value. The literal graph writes it with
// constant_serial.
struct ConstantLane {
    uint8_t value;
};

// Bytes from `symbols` distinct byte values, drawn with weight 1/k for the
// k-th value. The literal graph codes them with Huffman. `symbols` must be in
// [2, 256].
struct SkewedLane {
    uint32_t symbols;
};

// A byte that takes one step per value. A step is 0 with probability
// 1 - move. Otherwise it is 1 + min(geometric(grow), 64), with a random sign,
// and the byte wraps. The literal graph codes the steps with delta and Huffman.
struct WalkLane {
    double move;
    double grow;
};

// Bytes in 8-byte windows. A window is one of `patterns` patterns with
// probability `patternShare`. Otherwise each byte is one of 16 common values
// with probability `commonShare`, and uniform if not. The patterns draw their
// bytes the same way. zstd finds the repeated windows, so the literal graph
// writes the lane with zstd. `patterns` must be in [1, 256].
struct PatternLane {
    uint32_t patterns;
    double patternShare;
    double commonShare;
};

using IntLane = std::
        variant<RandomLane, ConstantLane, SkewedLane, WalkLane, PatternLane>;

// Literals whose byte b comes from lanes[b], with byte 0 the least
// significant, plus the runs and copies of `parse`.
struct PlannedInts {
    std::array<IntLane, 8> lanes;
    ParsePlan parse;
};

// Returns `numElts` little-endian 64-bit values. Throws std::invalid_argument
// if a lane's parameters are out of range, or if `p.parse` is invalid or
// cannot fit `numElts` values.
std::string
writeInts(const PlannedInts& p, size_t numElts, std::mt19937_64& rng);

} // namespace openzl::gpu::testkit
