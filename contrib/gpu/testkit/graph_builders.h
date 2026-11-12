// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "openzl/openzl.hpp"

// Helpers that build the subtrees of the testkit workloads' decode graphs.
// Each one names the codec that regenerates the subtree, and its arguments
// build the codec's inputs in slot order. The comment above a helper with a
// fixed subtree gives the subtree in decodeGraphShape() text (see
// frame_verifier.h).

namespace openzl::gpu::testkit {

// A stored stream.
openzl::GraphID store();

// Runs node `n` and sends its output i to `successors[i]`, so the successors
// build the inputs of the codec that decodes `n`.
openzl::GraphID node(
        openzl::Compressor& c,
        openzl::NodeID n,
        std::vector<openzl::GraphID> successors);

// 9:b(next): the frame's values as 4-byte numbers.
openzl::GraphID numeric32(openzl::Compressor& c, openzl::GraphID next);

// 9:b(next): the frame's values as 8-byte numbers.
openzl::GraphID numeric64(openzl::Compressor& c, openzl::GraphID next);

// 10:nN(next)
openzl::GraphID numToSerial(openzl::Compressor& c, openzl::GraphID next);

// 10:nN(b)
openzl::GraphID storeNum(openzl::Compressor& c);

// 6:sN(b)
openzl::GraphID storeStruct(openzl::Compressor& c);

// 28:nN(b)
openzl::GraphID bitpackInt(openzl::Compressor& c);

// 27:b(b)
openzl::GraphID bitpackSerial(openzl::Compressor& c);

// 22:b(b)
openzl::GraphID zstd(openzl::Compressor& c);

// 44:b(b)
openzl::GraphID constant(openzl::Compressor& c);

// 29:b(b,b)
openzl::GraphID flatpack(openzl::Compressor& c);

// 50:b(28:n1(b),b). The private entropy nodes build their own histogram and
// table when no entropy selector passes them one.
openzl::GraphID huffman(openzl::Compressor& c);

// 49:b(52:n2(b),b)
openzl::GraphID fse(openzl::Compressor& c);

// 51:s2(10:n1(fse),b)
openzl::GraphID huffmanStruct(openzl::Compressor& c);

// 1:nN(next)
openzl::GraphID delta(openzl::Compressor& c, openzl::GraphID next);

// 9:b(1:n1(10:n1(next))): delta on the bytes of a transposed stream.
openzl::GraphID deltaBytes(openzl::Compressor& c, openzl::GraphID next);

// 37:nN(alphabet,indices). Nimble's numeric graph sorts the alphabet.
openzl::GraphID tokenize(
        openzl::Compressor& c,
        openzl::GraphID alphabet,
        openzl::GraphID indices);

// 7:s2(28:n2(b)): field_lz tokens, bitpacked.
openzl::GraphID bitpackTokens(openzl::Compressor& c);

// 36:s2(7:s2(28:n2(b)),10:n1(indices))
openzl::GraphID tokenizeTokens(openzl::Compressor& c, openzl::GraphID indices);

// 25:n4(10:n1(codes),b)
openzl::GraphID quantOffsets(openzl::Compressor& c, openzl::GraphID codes);

// 26:n4(10:n1(codes),b)
openzl::GraphID quantLengths(openzl::Compressor& c, openzl::GraphID codes);

// 4:sN(bytes...). Core's split_struct graph runs its one custom node and
// sends output i to custom graph i. A static graph cannot do this, because
// transpose_split has one variable output for all its streams.
openzl::GraphID transposeSplit(
        openzl::Compressor& c,
        std::vector<openzl::GraphID> bytes);

// splitn on a serial stream, or splitn_struct on a struct stream, written
// "*" in the shape. It cuts blocks of 32 KiB and adds the rest of the input to
// the last block, so the block count depends on the input size. splitN has one
// variable output, so a static graph sends every block to `block`.
openzl::GraphID
split(openzl::Compressor& c, ZL_Type type, openzl::GraphID block);

// split() on a serial stream.
openzl::GraphID splitSerial(openzl::Compressor& c, openzl::GraphID block);

// 8:nN(24:sN(literals,tokens,offsets,extraLiteralLengths,extraMatchLengths))
openzl::GraphID fieldLz(
        openzl::Compressor& c,
        openzl::GraphID literals,
        openzl::GraphID tokens,
        openzl::GraphID offsets,
        openzl::GraphID extraLiteralLengths,
        openzl::GraphID extraMatchLengths);

} // namespace openzl::gpu::testkit
