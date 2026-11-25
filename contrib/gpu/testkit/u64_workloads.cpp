// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/u64_workloads.h"

#include <array>
#include <utility>

#include "contrib/gpu/testkit/graph_builders.h"

namespace openzl::gpu::testkit {

namespace {

constexpr size_t kKiB = 1024;

using openzl::Compressor;
using openzl::GraphID;

WorkloadSpec u64(PlannedInts pattern)
{
    return { .type = NimbleNumericType::U64, .pattern = std::move(pattern) };
}

const IntLane kRandom = RandomLane{};

// Bytes 0 to 4 are random, so every frame spans more than 2^32 and no value
// repeats by chance: Nimble's numeric graph neither range-packs nor tokenizes
// the values, and runs field_lz on them.
std::array<IntLane, 8> wideLanes(uint32_t byte5Symbols)
{
    return { kRandom,          kRandom,          kRandom,
             kRandom,          kRandom,          SkewedLane{ byte5Symbols },
             SkewedLane{ 24 }, ConstantLane{ 1 } };
}

// .graph and the drawing above it follow the conventions of f32_ladder.cpp.
std::vector<LadderStep> buildU64Workloads()
{
    return {
        // Many short repeats and long copies of varied length. The tokens
        // pass 128 but keep fewer than 256 distinct values, so Nimble
        // tokenizes them, and the long copies give FSE varied match length
        // codes.
        { .name = "u64/tokenize",
          .spec =
                  u64({ .lanes = { kRandom,
                                   kRandom,
                                   SkewedLane{ 16 },
                                   kRandom,
                                   kRandom,
                                   kRandom,
                                   WalkLane{ .move = 0.3, .grow = 0.3 },
                                   PatternLane{ .patterns     = 32,
                                                .patternShare = 0.7,
                                                .commonShare  = 0.6 } },
                        .parse = { .runs            = 700,
                                   .runLength       = 3,
                                   .copies          = 1000,
                                   .copyLength      = 20,
                                   .copyOffsetCodes = 12,
                                   .maxCopyLength   = 80 } }),
          .frameBytes = 256 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric64(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                huffman(c),
                                                store(),
                                                store(),
                                                store(),
                                                deltaBytes(c, huffman(c)),
                                                zstd(c) }),
                                      tokenizeTokens(c, huffman(c)),
                                      quantOffsets(c, fse(c)),
                                      quantLengths(c, fse(c)),
                                      quantLengths(c, fse(c))));
                  },
          // clang-format off
          //                   bitpack_int[n1] --#0--> huffman_v2[b] --#2--+--> transpose_split[s8] --#0--+--> field_lz[s8]
          //                                                               |                              |
          // bitpack_int[n1] --#0--> huffman_v2[b] --#0--> delta[n1] --#6--+                              |
          //                                                               |                              |
          //                                                 zstd[b] --#7--+                              |
          //                                                                                              |
          //                                          bitpack_int[n2] --#0--+--> tokenize_fixed[s2] --#1--+
          //                                                                |                             |
          //                    bitpack_int[n1] --#0--> huffman_v2[b] --#1--+                             |
          //                                                                                              |
          //                                 fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_off[n4] --#2--+
          //                                                                                              |
          //                                 fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#3--+
          //                                                                                              |
          //                                 fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#4--+
          // clang-format on
          .graph = "9:b(8:n8(24:s8(4:s8(b,b,50:b(28:n1(b),b),b,b,b,"
                   "9:b(1:n1(10:n1(50:b(28:n1(b),b)))),22:b(b)),"
                   "36:s2(7:s2(28:n2(b)),10:n1(50:b(28:n1(b),b))),"
                   "25:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b))))" },
        // Copies of many lengths, between repeats: 256 or more distinct
        // tokens make Nimble write them with huffman_struct_v2. The copies
        // cover enough of the frame that each literal lane stays below the
        // 100000 bytes at which the entropy graph cuts blocks.
        { .name = "u64/hufstruct",
          .spec =
                  u64({ .lanes = wideLanes(12),
                        .parse = { .runs            = 1500,
                                   .runLength       = 4,
                                   .copies          = 3500,
                                   .copyLength      = 2,
                                   .copyOffsetCodes = 14,
                                   .maxCopyLength   = 40 } }),
          .frameBytes = 1024 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric64(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                store(),
                                                store(),
                                                huffman(c),
                                                huffman(c),
                                                constant(c) }),
                                      huffmanStruct(c),
                                      quantOffsets(c, fse(c)),
                                      quantLengths(c, fse(c)),
                                      quantLengths(c, fse(c))));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#5--+--> transpose_split[s8] --#0--+--> field_lz[s8]
          //                                             |                              |
          // bitpack_int[n1] --#0--> huffman_v2[b] --#6--+                              |
          //                                             |                              |
          //                    constant_serial[b] --#7--+                              |
          //                                                                            |
          //       fse_ncount[n2] --#0--> fse_v2[b] --#0--> huffman_struct_v2[s2] --#1--+
          //                                                                            |
          //               fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_off[n4] --#2--+
          //                                                                            |
          //               fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#3--+
          //                                                                            |
          //               fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#4--+
          // clang-format on
          .graph = "9:b(8:n8(24:s8(4:s8(b,b,b,b,b,50:b(28:n1(b),b),"
                   "50:b(28:n1(b),b),44:b(b)),"
                   "51:s2(10:n1(49:b(52:n2(b),b)),b),"
                   "25:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b))))" },
        // One repeat, so field_lz finds almost no matches and the frame is
        // almost all literals. field_lz stores its small match streams.
        { .name = "u64/literals",
          .spec =
                  u64({ .lanes = wideLanes(12),
                        .parse = { .runs = 1, .runLength = 2 } }),
          .frameBytes = 128 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric64(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                store(),
                                                store(),
                                                huffman(c),
                                                huffman(c),
                                                constant(c) }),
                                      storeStruct(c),
                                      storeNum(c),
                                      storeNum(c),
                                      storeNum(c)));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#5--+--> transpose_split[s8] --#0--+--> field_lz[s8]
          //                                             |                              |
          // bitpack_int[n1] --#0--> huffman_v2[b] --#6--+                              |
          //                                             |                              |
          //                    constant_serial[b] --#7--+                              |
          // clang-format on
          .graph = "9:b(8:n8(24:s8(4:s8(b,b,b,b,b,50:b(28:n1(b),b),"
                   "50:b(28:n1(b),b),44:b(b)),6:s2(b),10:n4(b),10:n4(b),"
                   "10:n4(b))))" },
        // A small frame with few matches: at most 128 tokens, so field_lz
        // bitpacks them, and too few offset and literal length codes for
        // bitpack or FSE to gain, so field_lz stores them. Byte 5 draws 48
        // symbols. With 12 or 32 symbols, Nimble writes byte 5 with flatpack
        // at this frame size.
        { .name = "u64/bitpacktokens",
          .spec =
                  u64({ .lanes = wideLanes(48),
                        .parse = { .runs            = 10,
                                   .runLength       = 2,
                                   .copies          = 30,
                                   .copyLength      = 2,
                                   .copyOffsetCodes = 6 } }),
          .frameBytes = 8 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric64(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                store(),
                                                store(),
                                                huffman(c),
                                                huffman(c),
                                                constant(c) }),
                                      bitpackTokens(c),
                                      quantOffsets(c, store()),
                                      quantLengths(c, store()),
                                      storeNum(c)));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#5--+--> transpose_split[s8] --#0--+--> field_lz[s8]
          //                                             |                              |
          // bitpack_int[n1] --#0--> huffman_v2[b] --#6--+                              |
          //                                             |                              |
          //                    constant_serial[b] --#7--+                              |
          //                                                                            |
          //                                                      bitpack_int[n2] --#1--+
          //                                                                            |
          //                                                        quant_off[n4] --#2--+
          //                                                                            |
          //                                                        quant_len[n4] --#3--+
          // clang-format on
          .graph = "9:b(8:n8(24:s8(4:s8(b,b,b,b,b,50:b(28:n1(b),b),"
                   "50:b(28:n1(b),b),44:b(b)),7:s2(28:n2(b)),"
                   "25:n4(10:n1(b),b),26:n4(10:n1(b),b),10:n4(b))))" },
        // The copies cover little of the 2 MiB frame, so each literal lane
        // passes the 100000 bytes at which the entropy graph cuts blocks,
        // and Nimble writes bytes 5 and 6 with splitn and Huffman. The tokens
        // keep fewer than 256 distinct values, so Nimble tokenizes them.
        { .name = "u64/splithuffman",
          .spec =
                  u64({ .lanes = wideLanes(12),
                        .parse = { .runs            = 1500,
                                   .runLength       = 3,
                                   .copies          = 3000,
                                   .copyLength      = 8,
                                   .copyOffsetCodes = 14,
                                   .maxCopyLength   = 40 } }),
          .frameBytes = 2048 * kKiB,
          .buildGraph =
                  [](Compressor& c) {
                      return numeric64(
                              c,
                              fieldLz(c,
                                      transposeSplit(
                                              c,
                                              { store(),
                                                store(),
                                                store(),
                                                store(),
                                                store(),
                                                splitSerial(c, huffman(c)),
                                                splitSerial(c, huffman(c)),
                                                constant(c) }),
                                      tokenizeTokens(c, huffman(c)),
                                      quantOffsets(c, fse(c)),
                                      quantLengths(c, fse(c)),
                                      quantLengths(c, fse(c))));
                  },
          // clang-format off
          // bitpack_int[n1] --#0--> huffman_v2[b] --#0--> splitn[b]* --#5--+--> transpose_split[s8] --#0--+--> field_lz[s8]
          //                                                                |                              |
          // bitpack_int[n1] --#0--> huffman_v2[b] --#0--> splitn[b]* --#6--+                              |
          //                                                                |                              |
          //                                       constant_serial[b] --#7--+                              |
          //                                                                                               |
          //                                           bitpack_int[n2] --#0--+--> tokenize_fixed[s2] --#1--+
          //                                                                 |                             |
          //                     bitpack_int[n1] --#0--> huffman_v2[b] --#1--+                             |
          //                                                                                               |
          //                                  fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_off[n4] --#2--+
          //                                                                                               |
          //                                  fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#3--+
          //                                                                                               |
          //                                  fse_ncount[n2] --#0--> fse_v2[b] --#0--> quant_len[n4] --#4--+
          // clang-format on
          .graph = "9:b(8:n8(24:s8(4:s8(b,b,b,b,b,"
                   "40:b(50:b(28:n1(b),b))*,40:b(50:b(28:n1(b),b))*,"
                   "44:b(b)),"
                   "36:s2(7:s2(28:n2(b)),10:n1(50:b(28:n1(b),b))),"
                   "25:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b),"
                   "26:n4(10:n1(49:b(52:n2(b),b)),b))))" },
    };
}

} // namespace

const std::vector<LadderStep>& u64Workloads()
{
    static const std::vector<LadderStep> workloads = buildU64Workloads();
    return workloads;
}

} // namespace openzl::gpu::testkit
