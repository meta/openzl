// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/f32_ladder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "contrib/gpu/testkit/frame_factory.h"
#include "openzl/codecs/zl_split.h"
#include "openzl/compress/private_nodes.h"

namespace openzl::gpu::testkit {

namespace {

constexpr size_t kKiB       = 1024;
constexpr size_t kMiB       = 1024 * kKiB;
constexpr uint64_t kOneBits = 0x3F800000;

WorkloadSpec f32(WorkloadPattern pattern)
{
    return { .type = NimbleNumericType::F32, .pattern = std::move(pattern) };
}

// The helpers below build the subtrees of the .graph texts. Each one names
// the codec that regenerates the subtree, and its arguments build the
// codec's inputs in slot order.

using openzl::Compressor;
using openzl::GraphID;

constexpr size_t kSplitBlockBytes = 32 * kKiB;

GraphID store()
{
    return ZL_GRAPH_STORE;
}

GraphID node(Compressor& c, openzl::NodeID n, std::vector<GraphID> inputs)
{
    return c.buildStaticGraph(n, inputs);
}

// 9:b(next): the frame's f32 values as 4-byte numbers.
GraphID numeric32(Compressor& c, GraphID next)
{
    return node(c, ZL_NODE_CONVERT_SERIAL_TO_NUM_LE32, { next });
}

// 10:nN(next)
GraphID numToSerial(Compressor& c, GraphID next)
{
    return node(c, ZL_NODE_CONVERT_NUM_TO_SERIAL_LE, { next });
}

// 10:nN(b)
GraphID storeNum(Compressor& c)
{
    return numToSerial(c, store());
}

// 6:sN(b)
GraphID storeStruct(Compressor& c)
{
    return node(c, ZL_NODE_CONVERT_STRUCT_TO_SERIAL, { store() });
}

// 28:nN(b)
GraphID bitpackInt(Compressor& c)
{
    return node(c, ZL_NODE_BITPACK_INT, { store() });
}

// 27:b(b)
GraphID bitpackSerial(Compressor& c)
{
    return node(c, ZL_NODE_BITPACK_SERIAL, { store() });
}

// 22:b(b)
GraphID zstd(Compressor& c)
{
    return node(c, ZL_NODE_ZSTD, { store() });
}

// 44:b(b)
GraphID constant(Compressor& c)
{
    return node(c, ZL_NODE_CONSTANT_SERIAL, { store() });
}

// 29:b(b,b)
GraphID flatpack(Compressor& c)
{
    return node(c, ZL_NODE_FLATPACK, { store(), store() });
}

// 50:b(28:n1(b),b). The private entropy nodes build their own histogram and
// table when no entropy selector passes them one.
GraphID huffman(Compressor& c)
{
    return node(
            c,
            openzl::NodeID{ ZL_PrivateStandardNodeID_huffman_v2 },
            { bitpackInt(c), store() });
}

// 49:b(52:n2(b),b)
GraphID fse(Compressor& c)
{
    const GraphID ncount =
            node(c,
                 openzl::NodeID{ ZL_PrivateStandardNodeID_fse_ncount },
                 { store() });
    return node(
            c,
            openzl::NodeID{ ZL_PrivateStandardNodeID_fse_v2 },
            { ncount, store() });
}

// 51:s2(10:n1(fse),b)
GraphID huffmanStruct(Compressor& c)
{
    return node(
            c,
            openzl::NodeID{ ZL_PrivateStandardNodeID_huffman_struct_v2 },
            { numToSerial(c, fse(c)), store() });
}

// 1:nN(next)
GraphID delta(Compressor& c, GraphID next)
{
    return node(c, ZL_NODE_DELTA_INT, { next });
}

// 9:b(1:n1(10:n1(next))): delta on the bytes of a transposed stream.
GraphID deltaBytes(Compressor& c, GraphID next)
{
    return node(
            c,
            ZL_NODE_CONVERT_SERIAL_TO_NUM8,
            { delta(c, numToSerial(c, next)) });
}

// 37:nN(alphabet,indices). Nimble's numeric graph sorts the alphabet.
GraphID tokenize(Compressor& c, GraphID alphabet, GraphID indices)
{
    openzl::LocalParams params;
    params.addIntParam(ZL_TOKENIZE_SORT_PID, 1);
    const openzl::NodeID sorted = c.parameterizeNode(
            ZL_NODE_TOKENIZE_NUMERIC, { .localParams = std::move(params) });
    return node(c, sorted, { alphabet, indices });
}

// 7:s2(28:n2(b)): field_lz tokens, bitpacked.
GraphID bitpackTokens(Compressor& c)
{
    return node(c, ZL_NODE_CONVERT_STRUCT_TO_NUM_LE, { bitpackInt(c) });
}

// 36:s2(7:s2(28:n2(b)),10:n1(indices))
GraphID tokenizeTokens(Compressor& c, GraphID indices)
{
    return node(
            c,
            ZL_NODE_TOKENIZE_STRUCT,
            { bitpackTokens(c), numToSerial(c, indices) });
}

// 25:n4(10:n1(codes),b)
GraphID quantOffsets(Compressor& c, GraphID codes)
{
    return node(
            c, ZL_NODE_QUANTIZE_OFFSETS, { numToSerial(c, codes), store() });
}

// 26:n4(10:n1(codes),b)
GraphID quantLengths(Compressor& c, GraphID codes)
{
    return node(
            c, ZL_NODE_QUANTIZE_LENGTHS, { numToSerial(c, codes), store() });
}

// 4:sN(bytes...). Core's split_struct graph runs its one custom node and
// sends output i to custom graph i. A static graph cannot do this, because
// transpose_split has one variable output for all its streams.
GraphID transposeSplit(Compressor& c, std::vector<GraphID> bytes)
{
    return c.parameterizeGraph(
            ZL_GRAPH_SPLIT_STRUCT,
            { .customGraphs = std::move(bytes),
              .customNodes =
                      std::vector<openzl::NodeID>{ ZL_NODE_TRANSPOSE_SPLIT } });
}

// Cuts the input into blocks of kSplitBlockBytes. A last size of 0 means the
// rest of the input.
ZL_SplitInstructions splitIntoBlocks(ZL_SplitState* state, const ZL_Input* in)
{
    const size_t blockElts =
            std::max<size_t>(1, kSplitBlockBytes / ZL_Input_eltWidth(in));
    const size_t numBlocks =
            std::max<size_t>(1, ZL_Input_numElts(in) / blockElts);
    auto* const sizes = static_cast<size_t*>(
            ZL_SplitState_malloc(state, numBlocks * sizeof(size_t)));
    if (sizes == nullptr) {
        return { nullptr, 0 };
    }
    std::fill(sizes, sizes + numBlocks - 1, blockElts);
    sizes[numBlocks - 1] = 0;
    return { sizes, numBlocks };
}

// splitn on a serial stream, or splitn_struct on a struct stream, written
// "*" in the shape. splitN has one variable output, so a static graph sends
// every block to `block`.
GraphID split(Compressor& c, ZL_Type type, GraphID block)
{
    return node(
            c,
            ZL_Compressor_registerSplitNode_withParser(
                    c.get(), type, splitIntoBlocks, nullptr),
            { block });
}

GraphID splitSerial(Compressor& c, GraphID block)
{
    return split(c, ZL_Type_serial, block);
}

// 8:nN(24:sN(literals,tokens,offsets,extraLiteralLengths,extraMatchLengths))
GraphID fieldLz(
        Compressor& c,
        GraphID literals,
        GraphID tokens,
        GraphID offsets,
        GraphID extraLiteralLengths,
        GraphID extraMatchLengths)
{
    const GraphID lz =
            node(c,
                 ZL_NODE_FIELD_LZ,
                 { literals,
                   tokens,
                   offsets,
                   extraLiteralLengths,
                   extraMatchLengths });
    return node(c, ZL_NODE_CONVERT_NUM_TO_STRUCT_LE, { lz });
}

// Each step's graph is one that Nimble's numeric graph writes. The workload
// parameters set each codec's share of the input and its ratio. The data must
// still suit each codec. For example, step 4 keeps byte 3 of the field_lz
// literals constant for constant_serial, and step 12 keeps the range small
// enough for 1-byte range_pack output. The ladder test checks that the frames
// keep their graphs.
//
// .graph is the decodeGraphShape() text of the frames (see frame_verifier.h).
// Each codec appears as "ID:type(inputs)". ID is the codec ID. type is the
// type of the stream that the codec regenerates: "b" is serial, "nN" is
// numeric, and "sN" is struct, with N the width in bytes. The inputs are the
// streams that the codec decodes, in slot order. A stored input appears as
// its type alone. A split codec ends in "*" and shows each of its distinct
// inputs once.
//
// The comment above each .graph draws the same graph in decode order, with
// codec names. The stored streams are on the left, and the codec that
// regenerates the frame's data is on the right. The "[type]" after a name is
// the type of the stream that the codec regenerates. "A --#i--> B" means that
// codec A regenerates input i of codec B. Inputs that join at a "+" go to the
// codec after "+-->". The drawing does not show conversion codecs and stored
// inputs, but .graph shows them.
std::vector<LadderStep> buildLadder()
{
    return {
        { .name = "f32/01-tokenize-zstd",
          .spec =
                  f32(PinnedValues{ .base        = kOneBits,
                                    .range       = 1 << 20,
                                    .cardinality = 2 }),
          .frameBytes = 64,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              tokenize(
                                      c, storeNum(c), numToSerial(c, zstd(c))));
                  },
          // zstd[b] --#1--> tokenize_numeric[n4]
          .graph           = "9:b(37:n4(10:n4(b),10:n1(22:b(b))))",
          .trivialInNimble = false },
        { .name = "f32/02-fieldlz-transpose",
          .spec =
                  f32(PlannedFloats{ .literals  = RandomFloatBits{},
                                     .runs      = 3,
                                     .runLength = 6 }),
          .frameBytes = 256,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                store() }),
                                      storeStruct(c),
                                      storeNum(c),
                                      storeNum(c),
                                      storeNum(c)));
                  },
          // transpose_split[s4] --#0--> field_lz[s4]
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,b,b),6:s2(b),10:n4(b),10:n4(b),"
                   "10:n4(b))))" },
        { .name       = "f32/03-huffman-bitpack",
          .spec       = f32(FloatNormal{ .mean = 0, .stddev = 1 }),
          .frameBytes = 4 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                huffman(c) }),
                                      storeStruct(c),
                                      storeNum(c),
                                      storeNum(c),
                                      storeNum(c)));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#3--> transpose_split[s4] --#0--> field_lz[s4]
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,b,50:b(28:n1(b),b)),6:s2(b),"
                   "10:n4(b),10:n4(b),10:n4(b))))" },
        { .name       = "f32/04-constant-delta",
          .spec       = f32(GapAlphabetFloats{ .cardinality = 64 }),
          .frameBytes = 1 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      const GraphID alphabet =
                              delta(c,
                                    fieldLz(c,
                                            transposeSplit(
                                                    c,
                                                    { store(),
                                                      store(),
                                                      store(),
                                                      constant(c) }),
                                            storeStruct(c),
                                            storeNum(c),
                                            storeNum(c),
                                            storeNum(c)));
                      return numeric32(
                              c,
                              tokenize(c, alphabet, numToSerial(c, zstd(c))));
                  },
          // clang-format off
          // constant_serial[b] --#3--> transpose_split[s4] --#0--> field_lz[s4] --#0--> delta[n4] --#0--+--> tokenize_numeric[n4]
          //                                                                                             |
          //                                                                               zstd[b] --#1--+
          // clang-format on
          .graph = "9:b(37:n4(1:n4(8:n4(24:s4(4:s4(b,b,b,44:b(b)),6:s2(b),"
                   "10:n4(b),10:n4(b),10:n4(b)))),10:n1(22:b(b))))",
          .trivialInNimble = false },
        { .name = "f32/05-tokenbitpack-quantoffsets",
          .spec =
                  f32(PlannedFloats{ .literals        = RandomFloatBits{},
                                     .runs            = 20,
                                     .runLength       = 1,
                                     .copies          = 40,
                                     .copyLength      = 2,
                                     .copyOffsetCodes = 4 }),
          .frameBytes = 1 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                store() }),
                                      bitpackTokens(c),
                                      quantOffsets(c, store()),
                                      storeNum(c),
                                      storeNum(c)));
                  },
          // transpose_split[s4] --#0--+--> field_lz[s4]
          //                           |
          //     bitpack_int[n2] --#1--+
          //                           |
          //       quant_off[n4] --#2--+
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,b,b),7:s2(28:n2(b)),"
                   "25:n4(10:n1(b),b),10:n4(b),10:n4(b))))" },
        { .name = "f32/06-bitpackserial-quantlengths",
          .spec =
                  f32(PlannedFloats{
                          .literals        = PositiveFloats{ .tailShare   = 0.1,
                                                             .tailOctaves = 32 },
                          .runs            = 16,
                          .runLength       = 2,
                          .copies          = 100,
                          .copyLength      = 2,
                          .copyOffsetCodes = 8 }),
          .frameBytes = 4 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                huffman(c) }),
                                      bitpackTokens(c),
                                      quantOffsets(c, bitpackSerial(c)),
                                      quantLengths(c, store()),
                                      storeNum(c)));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#3--> transpose_split[s4] --#0--+--> field_lz[s4]
          //                                                                         |
          //                                                   bitpack_int[n2] --#1--+
          //                                                                         |
          //                           bitpack_serial[b] --#0--> quant_off[n4] --#2--+
          //                                                                         |
          //                                                     quant_len[n4] --#3--+
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,b,50:b(28:n1(b),b)),"
                   "7:s2(28:n2(b)),25:n4(10:n1(27:b(b)),b),26:n4(10:n1(b),b),"
                   "10:n4(b))))" },
        { .name = "f32/07-bytedelta-flatpack",
          .spec =
                  f32(PlannedFloats{
                          .literals = ByteWalkFloats{ .move      = 1,
                                                      .grow      = 0.5,
                                                      .byte2Move = 0.9,
                                                      .byte2Grow = 0.85 } }),
          .frameBytes = 1 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                deltaBytes(c, store()),
                                                deltaBytes(c, flatpack(c)) }),
                                      storeStruct(c),
                                      storeNum(c),
                                      storeNum(c),
                                      storeNum(c)));
                  },
          // clang-format off
          //                     delta[n1] --#2--+--> transpose_split[s4] --#0--> field_lz[s4]
          //                                     |
          // flatpack[b] --#0--> delta[n1] --#3--+
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,9:b(1:n1(10:n1(b))),"
                   "9:b(1:n1(10:n1(29:b(b,b))))),6:s2(b),10:n4(b),10:n4(b),"
                   "10:n4(b))))" },
        { .name = "f32/08-fse",
          .spec =
                  f32(PlannedFloats{ .literals        = PositiveFloats{},
                                     .runs            = 32,
                                     .runLength       = 2,
                                     .copies          = 120,
                                     .copyLength      = 2,
                                     .minCopyDistance = 256 }),
          .frameBytes = 4 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                huffman(c) }),
                                      bitpackTokens(c),
                                      quantOffsets(c, fse(c)),
                                      quantLengths(c, store()),
                                      storeNum(c)));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#3--> transpose_split[s4] --#0--+--> field_lz[s4]
          //                                                                         |
          //                                                   bitpack_int[n2] --#1--+
          //                                                                         |
          //            fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_off[n4] --#2--+
          //                                                                         |
          //                                                     quant_len[n4] --#3--+
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,b,50:b(28:n1(b),b)),"
                   "7:s2(28:n2(b)),25:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(b),b),10:n4(b))))" },
        { .name = "f32/09-splitn-tokenizetokens",
          .spec =
                  f32(CopyStreamFloats{
                          .literals = SignedOctaveFloats{ .octaves = 10 },
                          .longLiteralEvery     = 22,
                          .longLiteralMinOctave = 4,
                          .longLiteralMaxOctave = 9,
                          .shortLiteralEvery    = 18,
                          .shortLiteralContinue = 0,
                          .copyContinue         = 0,
                          .longCopyEvery        = 122,
                          .longCopyMinOctave    = 4,
                          .longCopyMaxOctave    = 11,
                          .minDistanceOctave    = 12,
                          .maxDistanceOctave    = 20 }),
          .frameBytes = 16 * kMiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                splitSerial(c, huffman(c)) }),
                                      tokenizeTokens(
                                              c, splitSerial(c, huffman(c))),
                                      quantOffsets(c, splitSerial(c, fse(c))),
                                      quantLengths(c, fse(c)),
                                      quantLengths(c, fse(c))));
                  },
          // clang-format off
          //   bitpack_int[n1] --#0--> huffman_v2[b] --#0--> splitn[b]* --#3--> transpose_split[s4] --#0--+--> field_lz[s4]
          //                                                                                              |
          //                                          bitpack_int[n2] --#0--+--> tokenize_fixed[s2] --#1--+
          //                                                                |                             |
          // bitpack_int[n1] --#0--> huffman_v2[b] --#0--> splitn[b]* --#1--+                             |
          //                                                                                              |
          //              fse_ncount[n2] --#0--> fse_v2[b] --#0--> splitn[b]* --#0--> quant_off[n4] --#2--+
          //                                                                                              |
          //                                 fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#3--+
          //                                                                                              |
          //                                 fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#4--+
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,b,40:b(50:b(28:n1(b),b))*),"
                   "36:s2(7:s2(28:n2(b)),10:n1(40:b(50:b(28:n1(b),b))*)),"
                   "25:n4(10:n1(40:b(49:b(52:n2(b),b))*),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b))))" },
        { .name = "f32/10-huffmanstruct",
          .spec =
                  f32(CopyStreamFloats{
                          .literals =
                                  WindowFloats{ .patternShare = { 0, 0, 0.2 },
                                                .commonShare = { 0.25, 0, 0.4 },
                                                .byte3Move   = 0.5,
                                                .byte3Grow   = 0.3 },
                          .longLiteralEvery     = 333,
                          .longLiteralMinOctave = 0,
                          .longLiteralMaxOctave = 7,
                          .shortLiteralEvery    = 11,
                          .shortLiteralContinue = 0.55,
                          .copyContinue         = 0.5,
                          .longCopyEvery        = 166,
                          .longCopyMinOctave    = 0,
                          .longCopyMaxOctave    = 13,
                          .minDistanceOctave    = 0,
                          .maxDistanceOctave    = 21,
                          .rep1Every            = 8,
                          .rep0Every            = 10 }),
          .frameBytes = 8 * kMiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { huffman(c),
                                                store(),
                                                zstd(c),
                                                deltaBytes(c, huffman(c)) }),
                                      huffmanStruct(c),
                                      quantOffsets(c, splitSerial(c, fse(c))),
                                      quantLengths(c, fse(c)),
                                      quantLengths(c, fse(c))));
                  },
          // clang-format off
          //                   bitpack_int[n1] --#0--> huffman_v2[b] --#0--+--> transpose_split[s4] --#0--+--> field_lz[s4]
          //                                                               |                              |
          //                                                 zstd[b] --#2--+                              |
          //                                                               |                              |
          // bitpack_int[n1] --#0--> huffman_v2[b] --#0--> delta[n1] --#3--+                              |
          //                                                                                              |
          //                         fse_ncount[n2] --#0--> fse_v2[b] --#0--> huffman_struct_v2[s2] --#1--+
          //                                                                                              |
          //              fse_ncount[n2] --#0--> fse_v2[b] --#0--> splitn[b]* --#0--> quant_off[n4] --#2--+
          //                                                                                              |
          //                                 fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#3--+
          //                                                                                              |
          //                                 fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#4--+
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(50:b(28:n1(b),b),b,22:b(b),"
                   "9:b(1:n1(10:n1(50:b(28:n1(b),b))))),"
                   "51:s2(10:n1(49:b(52:n2(b),b)),b),"
                   "25:n4(10:n1(40:b(49:b(52:n2(b),b))*),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b))))" },
        { .name = "f32/11-splitnstruct",
          .spec =
                  f32(CopyStreamFloats{
                          .literals =
                                  WindowFloats{ .patternShare = { 0, 0, 0 },
                                                .commonShare  = { 0.3, 0, 0 },
                                                .byte3Move    = 0.05,
                                                .byte3Grow    = 0.3,
                                                .byte2Move    = 1,
                                                .byte2Grow    = 0.96,
                                                .byte3Jitter  = 7,
                                                .byte3PatternShare = 0.35 },
                          .longLiteralEvery     = 2400,
                          .longLiteralMinOctave = 0,
                          .longLiteralMaxOctave = 5,
                          .shortLiteralEvery    = 3,
                          .shortLiteralContinue = 0.4,
                          .copyContinue         = 0.38,
                          .longCopyEvery        = 16000,
                          .longCopyMinOctave    = 0,
                          .longCopyMaxOctave    = 7,
                          .minDistanceOctave    = 6,
                          .maxDistanceOctave    = 20,
                          .rep1Every            = 20,
                          .rep0Every            = 7 }),
          .frameBytes = 4 * kMiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { splitSerial(c, huffman(c)),
                                                store(),
                                                deltaBytes(
                                                        c,
                                                        splitSerial(
                                                                c, huffman(c))),
                                                deltaBytes(c, zstd(c)) }),
                                      split(c,
                                            ZL_Type_struct,
                                            tokenizeTokens(
                                                    c,
                                                    splitSerial(
                                                            c, huffman(c)))),
                                      quantOffsets(c, splitSerial(c, fse(c))),
                                      quantLengths(c, fse(c)),
                                      quantLengths(c, store())));
                  },
          // clang-format off
          //                           bitpack_int[n1] --#0--> huffman_v2[b] --#0--> splitn[b]* --#0--+--> transpose_split[s4] --#0--+--> field_lz[s4]
          //                                                                                          |                              |
          //         bitpack_int[n1] --#0--> huffman_v2[b] --#0--> splitn[b]* --#0--> delta[n1] --#2--+                              |
          //                                                                                          |                              |
          //                                                          zstd[b] --#0--> delta[n1] --#3--+                              |
          //                                                                                                                         |
          //                                          bitpack_int[n2] --#0--+--> tokenize_fixed[s2] --#0--> splitn_struct[s2]* --#1--+
          //                                                                |                                                        |
          // bitpack_int[n1] --#0--> huffman_v2[b] --#0--> splitn[b]* --#1--+                                                        |
          //                                                                                                                         |
          //                                         fse_ncount[n2] --#0--> fse_v2[b] --#0--> splitn[b]* --#0--> quant_off[n4] --#2--+
          //                                                                                                                         |
          //                                                            fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#3--+
          //                                                                                                                         |
          //                                                                                                     quant_len[n4] --#4--+
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(40:b(50:b(28:n1(b),b))*,b,"
                   "9:b(1:n1(10:n1(40:b(50:b(28:n1(b),b))*))),"
                   "9:b(1:n1(10:n1(22:b(b))))),47:s2(36:s2(7:s2(28:n2(b)),"
                   "10:n1(40:b(50:b(28:n1(b),b))*)))*,"
                   "25:n4(10:n1(40:b(49:b(52:n2(b),b))*),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b),26:n4(10:n1(b),b))))" },
        { .name = "f32/12-rangepack",
          .spec =
                  f32(PinnedValues{
                          .base = kOneBits, .range = 0, .cardinality = 0 }),
          .frameBytes = 128,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              node(c,
                                   ZL_NODE_RANGE_PACK,
                                   { tokenize(
                                           c,
                                           storeNum(c),
                                           numToSerial(c, zstd(c))) }));
                  },
          // zstd[b] --#1--> tokenize_numeric[n1] --#0--> range_pack[n4]
          .graph           = "9:b(35:n4(37:n1(10:n1(b),10:n1(22:b(b)))))",
          .trivialInNimble = false },
    };
}

std::vector<LadderStep> buildExtraWorkloads()
{
    return {
        { .name = "f32/indexfieldlz",
          .spec =
                  f32(WideAlphabetFloats{ .cardinality      = 512,
                                          .highMove         = 0.03,
                                          .longLiteralEvery = 8,
                                          .maxGapLog2       = 23,
                                          .gapGrow          = 0.45,
                                          .bigGapEvery      = 112,
                                          .gapPairs         = 16,
                                          .lowByteStep      = 16,
                                          .maxLiteralRun    = 2,
                                          .minCopy          = 2,
                                          .maxCopy          = 2,
                                          .longCopyLength   = 6,
                                          .reuseShare       = 0.65,
                                          .distanceOctaves  = 8 }),
          .frameBytes = 8 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      const GraphID alphabet =
                              delta(c,
                                    fieldLz(c,
                                            transposeSplit(
                                                    c,
                                                    { store(),
                                                      store(),
                                                      huffman(c),
                                                      zstd(c) }),
                                            bitpackTokens(c),
                                            quantOffsets(c, store()),
                                            quantLengths(c, store()),
                                            storeNum(c)));
                      const GraphID indices = fieldLz(
                              c,
                              transposeSplit(
                                      c,
                                      { deltaBytes(c, huffman(c)), zstd(c) }),
                              tokenizeTokens(c, huffman(c)),
                              quantOffsets(c, bitpackSerial(c)),
                              quantLengths(c, store()),
                              storeNum(c));
                      return numeric32(c, tokenize(c, alphabet, indices));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#2--+--> transpose_split[s4] --#0--+--> field_lz[s4] --#0--> delta[n4] --#0--+--> tokenize_numeric[n4]
          //                                             |                              |                                         |
          //                               zstd[b] --#3--+                              |                                         |
          //                                                                            |                                         |
          //                                                      bitpack_int[n2] --#1--+                                         |
          //                                                                            |                                         |
          //                                                        quant_off[n4] --#2--+                                         |
          //                                                                            |                                         |
          //                                                        quant_len[n4] --#3--+                                         |
          //                                                                                                                      |
          // bitpack_int[n1] --#0--> huffman_v2[b] --#0--> delta[n1] --#0--+--> transpose_split[s2] --#0--+--> field_lz[s2] --#1--+
          //                                                               |                              |
          //                                                 zstd[b] --#1--+                              |
          //                                                                                              |
          //                                          bitpack_int[n2] --#0--+--> tokenize_fixed[s2] --#1--+
          //                                                                |                             |
          //                    bitpack_int[n1] --#0--> huffman_v2[b] --#1--+                             |
          //                                                                                              |
          //                                                bitpack_serial[b] --#0--> quant_off[n4] --#2--+
          //                                                                                              |
          //                                                                          quant_len[n4] --#3--+
          // clang-format on
          .graph = "9:b(37:n4(1:n4(8:n4(24:s4(4:s4(b,b,50:b(28:n1(b),b),"
                   "22:b(b)),7:s2(28:n2(b)),25:n4(10:n1(b),b),26:n4(10:n1(b),"
                   "b),10:n4(b)))),"
                   "8:n2(24:s2(4:s2(9:b(1:n1(10:n1(50:b(28:n1(b),b)))),"
                   "22:b(b)),36:s2(7:s2(28:n2(b)),10:n1(50:b(28:n1(b),b))),"
                   "25:n4(10:n1(27:b(b)),b),26:n4(10:n1(b),b),10:n4(b)))))" },
        // A small frame with few matches: the offset and literal length
        // codes are too few for bitpack or FSE to gain, so field_lz stores
        // them.
        { .name = "f32/bitpacktokens-storedcodes",
          .spec =
                  f32(PlannedFloats{
                          .literals        = PositiveFloats{ .tailShare   = 0.1,
                                                             .tailOctaves = 32 },
                          .runs            = 10,
                          .runLength       = 2,
                          .copies          = 30,
                          .copyLength      = 2,
                          .copyOffsetCodes = 6 }),
          .frameBytes = 2 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                huffman(c) }),
                                      bitpackTokens(c),
                                      quantOffsets(c, store()),
                                      quantLengths(c, store()),
                                      storeNum(c)));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#3--> transpose_split[s4] --#0--+--> field_lz[s4]
          //                                                                         |
          //                                                   bitpack_int[n2] --#1--+
          //                                                                         |
          //                                                     quant_off[n4] --#2--+
          //                                                                         |
          //                                                     quant_len[n4] --#3--+
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,b,50:b(28:n1(b),b)),"
                   "7:s2(28:n2(b)),25:n4(10:n1(b),b),26:n4(10:n1(b),b),"
                   "10:n4(b))))" },
        // A large frame that is mostly literals: the exponent byte passes
        // the 100000 bytes at which the entropy graph cuts 32 KiB blocks,
        // while the tokens stay few enough to bitpack. The literals span 30
        // octaves, so that few of them repeat by chance and add matches.
        { .name = "f32/splithuffman-fsecodes",
          .spec =
                  f32(PlannedFloats{
                          .literals   = SignedOctaveFloats{ .octaves = 30 },
                          .runs       = 4,
                          .runLength  = 24,
                          .copies     = 80,
                          .copyLength = 2,
                          .minCopyDistance = 256 }),
          .frameBytes = 512 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                splitSerial(c, huffman(c)) }),
                                      bitpackTokens(c),
                                      quantOffsets(c, fse(c)),
                                      quantLengths(c, fse(c)),
                                      quantLengths(c, store())));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#0--> splitn[b]* --#3--> transpose_split[s4] --#0--+--> field_lz[s4]
          //                                                                                            |
          //                                                                      bitpack_int[n2] --#1--+
          //                                                                                            |
          //                               fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_off[n4] --#2--+
          //                                                                                            |
          //                               fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#3--+
          //                                                                                            |
          //                                                                        quant_len[n4] --#4--+
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,b,40:b(50:b(28:n1(b),b))*),"
                   "7:s2(28:n2(b)),25:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b),26:n4(10:n1(b),b))))" },
        // A large frame with more copies than f32/splithuffman-fsecodes: more
        // than 128 tokens make Nimble tokenize them, and fewer than 256
        // distinct tokens keep tokenize_fixed.
        { .name = "f32/splithuffman-tokenizetokens",
          .spec =
                  f32(PlannedFloats{
                          .literals   = SignedOctaveFloats{ .octaves = 30 },
                          .runs       = 4,
                          .runLength  = 24,
                          .copies     = 800,
                          .copyLength = 2,
                          .minCopyDistance = 256 }),
          .frameBytes = 512 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric32(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                splitSerial(c, huffman(c)) }),
                                      tokenizeTokens(c, huffman(c)),
                                      quantOffsets(c, fse(c)),
                                      quantLengths(c, fse(c)),
                                      quantLengths(c, store())));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#0--> splitn[b]* --#3--> transpose_split[s4] --#0--+--> field_lz[s4]
          //                                                                                            |
          //                                        bitpack_int[n2] --#0--+--> tokenize_fixed[s2] --#1--+
          //                                                              |                             |
          //                  bitpack_int[n1] --#0--> huffman_v2[b] --#1--+                             |
          //                                                                                            |
          //                               fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_off[n4] --#2--+
          //                                                                                            |
          //                               fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#3--+
          //                                                                                            |
          //                                                                        quant_len[n4] --#4--+
          // clang-format on
          .graph = "9:b(8:n4(24:s4(4:s4(b,b,b,40:b(50:b(28:n1(b),b))*),"
                   "36:s2(7:s2(28:n2(b)),10:n1(50:b(28:n1(b),b))),"
                   "25:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b),26:n4(10:n1(b),b))))" },
    };
}

} // namespace

std::string compressStep(const LadderStep& step, std::string_view data)
{
    const NimbleCompressParams params;
    openzl::Compressor compressor;
    compressor.setParameter(
            openzl::CParam::CompressionLevel, params.compressionLevel);
    return makeFrameWithGraph(
            compressor,
            step.buildGraph(compressor),
            openzl::Input::refSerial(data.data(), data.size()),
            params.formatVersion);
}

const std::vector<LadderStep>& f32Ladder()
{
    static const std::vector<LadderStep> ladder = buildLadder();
    return ladder;
}

const std::vector<LadderStep>& f32ExtraWorkloads()
{
    static const std::vector<LadderStep> extras = buildExtraWorkloads();
    return extras;
}

} // namespace openzl::gpu::testkit
