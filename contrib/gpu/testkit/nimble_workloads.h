// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "contrib/gpu/testkit/float_parses.h"
#include "contrib/gpu/testkit/int_parses.h"
#include "contrib/gpu/testkit/nimble_graph.h"

// Seeded synthetic leaf streams for GPU decoder tests and benchmarks that need
// Nimble's frames. The tests and benchmarks ship these generators, not data.
// The generators draw through std:: distributions, so a spec gives the same
// bytes with one standard library, and other bytes with the same shape with
// another.

namespace openzl::gpu::testkit {

// Normal float32 values.
struct FloatNormal {
    double mean;
    double stddev;
};

// Values that hold the same range and distinct count in every frame, so that
// Nimble's numeric graph takes the same path for every seed. With F32, the
// values are float bit patterns. Every frame holds `base` and `base + range`.
// - With `cardinality` > 0, every frame holds exactly that many distinct
//   values, drawn with weight 1/k for the k-th value.
// - With `cardinality` == 0, every value in a frame is distinct. If the frame
//   holds more values than the range, the frame holds every value in the
//   range.
struct PinnedValues {
    uint64_t base;
    uint64_t range;
    uint64_t cardinality;
};

using WorkloadPattern = std::variant<
        FloatNormal,
        PinnedValues,
        PlannedInts,
        PlannedFloats,
        GapAlphabetFloats,
        WideAlphabetFloats,
        CopyStreamFloats>;

// The shape of one leaf stream, which Nimble writes with its Trivial
// encoding.
struct WorkloadSpec {
    NimbleNumericType type;
    WorkloadPattern pattern;
    uint64_t seed = 1;
};

// Returns exactly `frameBytes` bytes of frame `frameIndex` of the leaf stream
// that `spec` describes: frameBytes / width little-endian values of
// `spec.type`. `spec.seed` decides the alphabet of PinnedValues, and all
// frames of one spec share it, as the frames of one Nimble column share a
// value distribution. Each frame index draws its own values. PlannedInts
// needs U64, and the other patterns except PinnedValues need F32. Throws
// std::invalid_argument if a pattern has a type that it does not take, if the
// parameters are invalid, or if `frameBytes` is not a multiple of the element
// width.
std::string generateFrame(
        const WorkloadSpec& spec,
        size_t frameBytes,
        size_t frameIndex = 0);

struct Frame {
    std::string raw;
    std::string compressed;
};

// Returns frames 0 to count - 1 of `spec`, each `frameBytes` raw bytes, the
// same as generateFrame() gives, and each compressed with compressLikeNimble().
// makeFrames() builds the column tables once. The frames do not depend on
// `threads`. If frames throw, makeFrames() rethrows the exception of the
// lowest frame index that threw, after all threads finish.
std::vector<Frame> makeFrames(
        const WorkloadSpec& spec,
        size_t frameBytes,
        size_t count,
        const NimbleCompressParams& params = {},
        size_t threads                     = 1);

} // namespace openzl::gpu::testkit
