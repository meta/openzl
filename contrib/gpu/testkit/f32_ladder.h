// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "contrib/gpu/testkit/nimble_graph.h"
#include "contrib/gpu/testkit/nimble_workloads.h"
#include "openzl/openzl.hpp"

// A ladder of float32 decode graphs from Nimble's float graph. Each step needs
// at most two codecs that the steps before it do not need. A codec counts once
// for all of its input types and widths, because a GPU decoder implements each
// codec for every input that the codec takes. A GPU decoder that handles every
// codec up to a step can decode the frames of that step. The last step
// completes the 19 codecs that the ladder covers.
//
// Each step is a decode graph and a workload. compressStep() writes the
// workload's frames with a graph that the step builds from OpenZL nodes, so
// no selector runs and the frames take the step's graph for every seed and
// standard library. The workload sets each codec's share of the input and
// its ratio, for benchmarks. It does not choose the graph.

namespace openzl::gpu::testkit {

struct LadderStep {
    // "f32/NN-..." with the step number and what the step adds to the graphs
    // before it, or "<type>/..." for a workload outside the ladder, such as
    // "f32/indexfieldlz" or "zstd/u32-1k".
    std::string name;
    WorkloadSpec spec;
    size_t frameBytes;
    // Builds the step's graph on `compressor` and returns it.
    openzl::GraphID (*buildGraph)(openzl::Compressor& compressor);
    // decodeGraphShape() of every frame that compressStep() writes. Steps 9
    // to 11 have split codecs, which cut blocks of 32 KiB, so their block
    // counts depend on the frame size.
    std::string graph;
    // Whether Nimble's default encoding selection writes a plain column of
    // this step's values as one Trivial stream that OpenZL compresses. With
    // false, it picks another encoding (Varint, Dictionary, or Constant), so
    // an end-to-end Nimble benchmark must build this frame inside that
    // encoding or skip the step. False also marks a frame whose graph Nimble
    // never writes, such as plain zstd, which such a benchmark must skip.
    bool trivialInNimble = true;
};

// Compresses `data` with the graph of `step`, at the format version and
// compression level of NimbleCompressParams, and returns the frame. OpenZL
// keeps every codec of the graph (see makeFrameWithGraph()). Throws
// openzl::Exception if a codec rejects its input.
std::string compressStep(const LadderStep& step, std::string_view data);

// The 12 steps, in order.
const std::vector<LadderStep>& f32Ladder();

// Workloads outside the ladder. Their graphs need only the ladder's codecs, so
// they are not steps, but benchmarks run them next to the steps.
// - "f32/indexfieldlz" tokenizes the values, then runs field_lz on the
//   delta-coded alphabet and on the 2-byte indices.
// - "f32/bitpacktokens-storedcodes", "f32/splithuffman-fsecodes", and
//   "f32/splithuffman-tokenizetokens" write float32 graphs that no ladder
//   step writes.
const std::vector<LadderStep>& f32ExtraWorkloads();

} // namespace openzl::gpu::testkit
