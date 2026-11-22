// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "openzl/zl_config.h"

#if ZL_ALLOW_INTROSPECTION

#    include <gtest/gtest.h>

#    include "cpp/tests/experimental/trace/TraceTestHelpers.hpp"
#    include "openzl/cpp/CCtx.hpp"
#    include "openzl/cpp/Compressor.hpp"
#    include "openzl/cpp/DCtx.hpp"
#    include "tests/datagen/DataGen.h"

using namespace ::testing;

namespace openzl {
namespace {

// Random words drawn from a small vocabulary, so LZ finds plenty of matches.
std::string makeWords(size_t size)
{
    tests::datagen::DataGen gen;
    std::vector<std::string> vocab;
    for (size_t i = 0; i < 256; ++i) {
        vocab.push_back(gen.randString("word", 12) + " ");
    }
    std::string data;
    while (data.size() < size) {
        data += vocab[gen.randVal<size_t>("pick", 0, vocab.size() - 1)];
    }
    return data;
}

// Every codec edge must index into the chunk's streams array, which is how the
// visualizer reads it.
TEST(DecompressTraceTest, CodecEdgesIndexIntoStreams)
{
    auto data = makeWords(64 << 10);

    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    compressor.selectStartingGraph(ZL_GRAPH_LZ);
    CCtx cctx;
    cctx.refCompressor(compressor);
    cctx.writeTraces(true);
    auto compressed = cctx.compressSerial(data);

    // A fused decoder (partition + bitpack) is never traced, which leaves gaps
    // in the decompress trace's stream IDs.
    auto ctrace = test::parseTrace(cctx.getLatestTrace().first);
    ASSERT_TRUE(ctrace.has_value());
    bool hasPartition = false;
    for (const auto& chunk : ctrace->chunks) {
        for (const auto& codec : chunk.codecs) {
            hasPartition |= codec.name == "zl.partition";
        }
    }
    ASSERT_TRUE(hasPartition);

    // Decompress into a caller-provided buffer, as the CLI and wasm do.
    DCtx dctx;
    dctx.writeTraces(true);
    std::string out(data.size(), '\0');
    ASSERT_EQ(dctx.decompressSerial(out, compressed), data.size());
    ASSERT_EQ(out, data);

    auto trace = test::parseTrace(dctx.getLatestTrace().first);
    ASSERT_TRUE(trace.has_value());
    for (const auto& chunk : trace->chunks) {
        for (const auto& codec : chunk.codecs) {
            for (const auto* edges :
                 { &codec.inputStreams, &codec.outputStreams }) {
                for (auto id : *edges) {
                    EXPECT_GE(id, 0) << codec.name;
                    EXPECT_LT(id, (int64_t)chunk.numStreams) << codec.name;
                }
            }
        }
    }
}

} // namespace
} // namespace openzl

#endif // ZL_ALLOW_INTROSPECTION
