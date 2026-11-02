// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "openzl/codecs/zl_concat.h"
#include "openzl/codecs/zl_generic.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_version.h"

#include "openzl/common/limits.h" // ZL_ENCODER_INPUT_LIMIT (internal)
#include "openzl/compress/cctx.h" // CCTX_arenaMemory (internal)

namespace {

// A ZL_CCtx is routinely pooled and reused for the lifetime of a process, and
// its arenas are released only by ZL_CCtx_free, so capacity a compression
// leaves behind is retained until the context dies. Graph-arena allocations
// scale with the input count, so one outsized multi-input compression must not
// pin that capacity for every later compression.
class CCtxMemoryRetentionTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        compressor_ = ZL_Compressor_create();
        ASSERT_NE(compressor_, nullptr);

        // CONCAT_SERIAL accepts any number of serial inputs and produces two
        // streams, so one graph serves both the wide and the narrow case.
        const ZL_GraphID successors[2] = { ZL_GRAPH_COMPRESS_GENERIC,
                                           ZL_GRAPH_COMPRESS_GENERIC };
        const ZL_GraphID gid = ZL_Compressor_registerStaticGraph_fromNode(
                compressor_, ZL_NODE_CONCAT_SERIAL, successors, 2);
        ASSERT_FALSE(ZL_isError(
                ZL_Compressor_selectStartingGraphID(compressor_, gid)));

        cctx_ = ZL_CCtx_create();
        ASSERT_NE(cctx_, nullptr);
        ASSERT_FALSE(ZL_isError(ZL_CCtx_refCompressor(cctx_, compressor_)));
        // Parameters are cleared at the end of each session unless sticky, and
        // this fixture compresses repeatedly on the one context.
        ASSERT_FALSE(ZL_isError(
                ZL_CCtx_setParameter(cctx_, ZL_CParam_stickyParameters, 1)));
        ASSERT_FALSE(ZL_isError(ZL_CCtx_setParameter(
                cctx_, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));
    }

    void TearDown() override
    {
        ZL_CCtx_free(cctx_);
        ZL_Compressor_free(compressor_);
    }

    void compress(size_t nbInputs)
    {
        static constexpr size_t kBytesPerInput = 16;

        const std::string buffer(kBytesPerInput, 'x');
        std::vector<ZL_TypedRef*> owned;
        std::vector<const ZL_TypedRef*> refs;
        owned.reserve(nbInputs);
        refs.reserve(nbInputs);
        for (size_t i = 0; i < nbInputs; i++) {
            ZL_TypedRef* const ref =
                    ZL_TypedRef_createSerial(buffer.data(), buffer.size());
            ASSERT_NE(ref, nullptr);
            owned.push_back(ref);
            refs.push_back(ref);
        }

        std::vector<char> dst(
                ZL_compressBound(nbInputs * kBytesPerInput) + nbInputs * 16
                + 4096);
        const ZL_Report report = ZL_CCtx_compressMultiTypedRef(
                cctx_, dst.data(), dst.size(), refs.data(), refs.size());
        const bool failed = ZL_isError(report);
        const std::string err =
                failed ? ZL_CCtx_getErrorContextString(cctx_, report) : "";
        for (ZL_TypedRef* ref : owned) {
            ZL_TypedRef_free(ref);
        }
        ASSERT_FALSE(failed) << "nbInputs=" << nbInputs << " err=" << err;
    }

    ZL_Compressor* compressor_{};
    ZL_CCtx* cctx_{};
};

TEST_F(CCtxMemoryRetentionTest, oneWideInputDoesNotPinArenaCapacity)
{
    constexpr size_t kWideInputs   = ZL_ENCODER_INPUT_LIMIT - 1;
    constexpr size_t kNarrowInputs = 2;
    // The arenas size down only after enough under-using sessions to trip
    // PBUFF_SIZEDOWN_THRESHOLD. The graph arena is reset per graph node, so
    // every compression contributes several sessions.
    constexpr size_t kNarrowCompressions = 50000;

    ASSERT_NO_FATAL_FAILURE(compress(kNarrowInputs));
    const size_t baseline = CCTX_arenaMemory(cctx_);

    ASSERT_NO_FATAL_FAILURE(compress(kWideInputs));
    const size_t afterWide = CCTX_arenaMemory(cctx_);
    ASSERT_GT(afterWide, baseline * 4)
            << "the wide compression did not grow the arenas, so this test "
               "would not be able to observe them failing to shrink";

    for (size_t i = 0; i < kNarrowCompressions; i++) {
        ASSERT_NO_FATAL_FAILURE(compress(kNarrowInputs));
    }

    EXPECT_LT(CCTX_arenaMemory(cctx_), afterWide / 2)
            << "arena capacity stayed pinned at the wide compression's "
               "high-water mark of "
            << afterWide << " bytes after " << kNarrowCompressions
            << " narrow compressions";
}

} // namespace
