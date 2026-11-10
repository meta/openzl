// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "contrib/gpu/testkit/f32_ladder.h"

// Workloads of 64-bit integers, which Nimble's numeric graph writes with
// field_lz on 8-byte values. Each workload is a graph and a PlannedInts spec
// whose frames take that graph, and compressStep() writes the graph without
// a selector, as for the float32 ladder.

namespace openzl::gpu::testkit {

// - "u64/tokenize": tokenize_fixed on the tokens, and FSE on the offset,
//   literal length, and match length codes. Its literal lanes are stored,
//   Huffman, delta and Huffman, and zstd.
// - "u64/hufstruct": huffman_struct_v2 on the tokens, and FSE on the codes.
// - "u64/literals": almost no matches, so field_lz stores its match streams.
// - "u64/bitpacktokens": a small frame with bitpacked tokens and stored codes.
// - "u64/splithuffman": a 2 MiB frame with tokenize_fixed on the tokens, FSE
//   on the codes, and Huffman lanes cut into blocks by splitn.
// The last four code bytes 5 and 6 with Huffman and store byte 7 as a
// constant.
const std::vector<LadderStep>& u64Workloads();

} // namespace openzl::gpu::testkit
