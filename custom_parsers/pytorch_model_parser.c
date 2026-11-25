// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/pytorch_model_parser.h"

#include <string.h>

#include "custom_parsers/zip_lexer.h"
#include "openzl/common/assertion.h"
#include "openzl/shared/estimate.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_graph_api.h"
#include "openzl/zl_segmenter.h"
#include "openzl/zl_version.h"

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
static const size_t kMultiplier = 4;
#else
static const size_t kMultiplier = 8 * 1024;
#endif
static const size_t kMaxSegmentSize = 128 * kMultiplier;
static const size_t kMaxChunkSize   = 1024 * kMultiplier;

typedef enum {
    PytorchModelSuccessor_U8            = 0,
    PytorchModelSuccessor_F16           = 1,
    PytorchModelSuccessor_F32           = 2,
    PytorchModelSuccessor_F64           = 3,
    PytorchModelSuccessor_OtherFiles    = 4,
    PytorchModelSuccessor_Precompressed = 5,
    PytorchModelSuccessor_Metadata      = 6,
    PytorchModelSuccessor_NumSuccessors = 7,
} PytorchModelSuccessor;

#define PYTORCH_SEGMENT_SIZES_PID 0
#define PYTORCH_SEGMENT_TAGS_PID 1

static PytorchModelSuccessor selectSuccessor(const char* ptr, size_t size)
{
    const size_t width = ZL_guessFloatWidth(ptr, size);
    switch (width) {
        case 1:
            return PytorchModelSuccessor_U8;
        case 2:
            return PytorchModelSuccessor_F16;
        case 4:
            return PytorchModelSuccessor_F32;
        case 8:
            return PytorchModelSuccessor_F64;
        default:
            ZL_ASSERT_FAIL("unreachable");
            return PytorchModelSuccessor_U8;
    }
}

static bool
startsWithPrefix(const char* filename, size_t filenameSize, const char* prefix)
{
    return filenameSize >= strlen(prefix)
            && memcmp(filename, prefix, strlen(prefix)) == 0;
}

static bool hasDir(const char* filename, size_t filenameSize, const char* dir)
{
    for (;;) {
        if (startsWithPrefix(filename, filenameSize, dir)) {
            return true;
        }
        const char* ptr = memchr(filename, '/', filenameSize);
        if (ptr == NULL) {
            return false;
        }
        const size_t offset = (size_t)(ptr - filename) + 1;
        filename            = filename + offset;
        filenameSize -= offset;
    }
}

static bool isDataFile(const char* filename, size_t filenameSize)
{
    return hasDir(filename, filenameSize, "data/")
            || hasDir(filename, filenameSize, "xl_model_weights/");
}

static ZL_Report
pytorchModelDynGraph(ZL_Graph* gctx, ZL_Edge* sctxs[], size_t nbIns)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(gctx);

    ZL_ERR_IF(nbIns != 1, graph_invalidNumInputs);
    ZL_Edge* sctx = sctxs[0];

    const ZL_RefParam sizesParam =
            ZL_Graph_getLocalRefParam(gctx, PYTORCH_SEGMENT_SIZES_PID);
    ZL_ERR_IF_NE(sizesParam.paramId, PYTORCH_SEGMENT_SIZES_PID, graph_invalid);
    const ZL_RefParam tagsParam =
            ZL_Graph_getLocalRefParam(gctx, PYTORCH_SEGMENT_TAGS_PID);
    ZL_ERR_IF_NE(tagsParam.paramId, PYTORCH_SEGMENT_TAGS_PID, graph_invalid);

    const size_t* const inSizes  = (const size_t*)sizesParam.paramRef;
    const unsigned* const inTags = (const unsigned*)tagsParam.paramRef;
    const size_t inNbSegments    = sizesParam.paramSize / sizeof(size_t);

    if (inNbSegments == 0) {
        ZL_ERR_IF_ERR(ZL_Edge_setDestination(sctx, ZL_GRAPH_STORE));
        return ZL_returnSuccess();
    }

    size_t totalBytes = 0;
    for (size_t i = 0; i < inNbSegments; ++i) {
        totalBytes += inSizes[i];
    }
    const size_t maxNbSegments =
            inNbSegments + (totalBytes / kMaxSegmentSize) + 1;

    size_t* const segmentSizes =
            ZL_Graph_getScratchSpace(gctx, maxNbSegments * sizeof(size_t));
    unsigned* const tags =
            ZL_Graph_getScratchSpace(gctx, maxNbSegments * sizeof(unsigned));
    ZL_ERR_IF_NULL(segmentSizes, allocation);
    ZL_ERR_IF_NULL(tags, allocation);

    // Split oversized segments at kMaxSegmentSize for memory locality
    size_t nbSegments = 0;
    for (size_t i = 0; i < inNbSegments; ++i) {
        ZL_ERR_IF_GE(nbSegments, maxNbSegments, corruption);
        segmentSizes[nbSegments] = inSizes[i];
        tags[nbSegments]         = inTags[i];
        ++nbSegments;
        while (segmentSizes[nbSegments - 1] > kMaxSegmentSize) {
            ZL_ERR_IF_GE(nbSegments, maxNbSegments, corruption);
            const size_t segmentSize     = segmentSizes[nbSegments - 1];
            segmentSizes[nbSegments - 1] = kMaxSegmentSize;
            segmentSizes[nbSegments]     = segmentSize - kMaxSegmentSize;
            tags[nbSegments]             = tags[nbSegments - 1];
            ++nbSegments;
        }
    }

    // Split the input according to segmentSizes
    ZL_TRY_LET(
            ZL_EdgeList,
            streams,
            ZL_Edge_runSplitNode(sctx, segmentSizes, nbSegments));
    const ZL_GraphIDList graphs = ZL_Graph_getCustomGraphs(gctx);
    ZL_ASSERT_EQ(streams.nbStreams, nbSegments);

    // Set the destination for every segment
    for (size_t i = 0; i < streams.nbStreams; ++i) {
        ZL_ERR_IF_ERR(ZL_Edge_setDestination(
                streams.streams[i], graphs.graphids[tags[i]]));
    }

    return ZL_returnSuccess();
}

static ZL_Report pytorchModelSegmenter(ZL_Segmenter* sctx)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(sctx);
    const size_t numInputs = ZL_Segmenter_numInputs(sctx);
    ZL_ERR_IF_NE(numInputs, 1, node_invalid_input);
    const ZL_Input* const input = ZL_Segmenter_getInput(sctx, 0);
    ZL_ASSERT_NN(input);
    const size_t inputSize = ZL_Input_contentSize(input);

    const unsigned formatVersion =
            (unsigned)ZL_Segmenter_getCParam(sctx, ZL_CParam_formatVersion);
    const size_t chunkSizeLimit =
            (formatVersion < ZL_CHUNK_VERSION_MIN) ? SIZE_MAX : kMaxChunkSize;

    const ZL_GraphIDList customGraphs = ZL_Segmenter_getCustomGraphs(sctx);
    ZL_ASSERT_EQ(customGraphs.nbGraphIDs, 1);
    const ZL_GraphID functionGraph = customGraphs.graphids[0];

    ZS2_ZipLexer lexer;
    ZL_ERR_IF_ERR(ZS2_ZipLexer_init(
            &lexer, ZL_Input_ptr(input), inputSize, ZL__errorContext.opCtx));

    const size_t nbFiles   = ZS2_ZipLexer_numFiles(&lexer);
    const size_t maxNbSegs = nbFiles * 4 + 2;

    size_t* const segSizes =
            ZL_Segmenter_getScratchSpace(sctx, maxNbSegs * sizeof(size_t));
    unsigned* const segTags =
            ZL_Segmenter_getScratchSpace(sctx, maxNbSegs * sizeof(unsigned));
    ZL_ERR_IF_NULL(segSizes, allocation);
    ZL_ERR_IF_NULL(segTags, allocation);

    size_t nbChunkSegs = 0;
    size_t chunkBytes  = 0;

    while (!ZS2_ZipLexer_finished(&lexer)) {
        ZS2_ZipToken tokens[32];
        ZL_TRY_LET(size_t, nbTokens, ZS2_ZipLexer_lex(&lexer, tokens, 32));
        for (size_t i = 0; i < nbTokens; ++i) {
            const ZS2_ZipToken token = tokens[i];

            unsigned tag;
            if (token.type == ZS2_ZipTokenType_CompressedData) {
                if (token.compressionMethod != 0) {
                    tag = PytorchModelSuccessor_Precompressed;
                } else if (isDataFile(token.filename, token.filenameSize)) {
                    tag = selectSuccessor(token.ptr, token.size);
                } else {
                    tag = PytorchModelSuccessor_OtherFiles;
                }
            } else {
                tag = PytorchModelSuccessor_Metadata;
            }

            // Merge with previous segment if same tag
            if (nbChunkSegs > 0 && segTags[nbChunkSegs - 1] == tag) {
                segSizes[nbChunkSegs - 1] += token.size;
            } else {
                ZL_ERR_IF_GE(nbChunkSegs, maxNbSegs, corruption);
                segSizes[nbChunkSegs] = token.size;
                segTags[nbChunkSegs]  = tag;
                ++nbChunkSegs;
            }
            chunkBytes += token.size;

            // Flush chunks that exceed the size limit
            while (chunkBytes > chunkSizeLimit) {
                const unsigned carryTag    = segTags[nbChunkSegs - 1];
                const size_t overflowBytes = chunkBytes - chunkSizeLimit;
                ZL_ASSERT_GE(segSizes[nbChunkSegs - 1], overflowBytes);
                // Round split point down to a multiple of 8 so we don't
                // split a float or double in the middle of a field.
                size_t splitSize = segSizes[nbChunkSegs - 1] - overflowBytes;
                splitSize &= ~(size_t)7;

                const size_t excess     = segSizes[nbChunkSegs - 1] - splitSize;
                const size_t flushBytes = chunkBytes - excess;
                ZL_ASSERT_GT(excess, 0);
                ZL_ASSERT_LE(flushBytes, chunkSizeLimit);

                segSizes[nbChunkSegs - 1] = splitSize;
                if (splitSize == 0) {
                    --nbChunkSegs;
                }

                const ZL_RefParam refParams[2] = {
                    { PYTORCH_SEGMENT_SIZES_PID,
                      segSizes,
                      nbChunkSegs * sizeof(size_t) },
                    { PYTORCH_SEGMENT_TAGS_PID,
                      segTags,
                      nbChunkSegs * sizeof(unsigned) },
                };
                const ZL_LocalParams chunkParams = {
                    .refParams = { refParams, 2 },
                };
                const ZL_RuntimeGraphParameters gparams = {
                    .localParams = &chunkParams,
                };
                ZL_ERR_IF_ERR(ZL_Segmenter_processChunk(
                        sctx, &flushBytes, 1, functionGraph, &gparams));

                segSizes[0] = excess;
                segTags[0]  = carryTag;
                nbChunkSegs = 1;
                chunkBytes  = excess;
            }
        }
    }

    // Flush remaining segments
    ZL_ASSERT_LE(chunkBytes, chunkSizeLimit);
    if (nbChunkSegs > 0) {
        const ZL_RefParam refParams[2] = {
            { PYTORCH_SEGMENT_SIZES_PID,
              segSizes,
              nbChunkSegs * sizeof(size_t) },
            { PYTORCH_SEGMENT_TAGS_PID,
              segTags,
              nbChunkSegs * sizeof(unsigned) },
        };
        const ZL_LocalParams chunkParams = {
            .refParams = { refParams, 2 },
        };
        const ZL_RuntimeGraphParameters gparams = {
            .localParams = &chunkParams,
        };
        ZL_ERR_IF_ERR(ZL_Segmenter_processChunk(
                sctx, &chunkBytes, 1, functionGraph, &gparams));
    }

    return ZL_returnSuccess();
}

ZL_GraphID ZS2_createGraph_pytorchModelCompressor(ZL_Compressor* cgraph)
{
    ZL_GraphID f16Graph = ZL_Compressor_registerStaticGraph_fromNode(
            cgraph,
            ZL_NODE_BFLOAT16_DECONSTRUCT,
            ZL_GRAPHLIST(ZL_GRAPH_STORE, ZL_GRAPH_HUFFMAN));
    f16Graph = ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph, ZL_NODE_INTERPRET_AS_LE16, f16Graph);

    ZL_GraphID f32Graph = ZL_Compressor_registerStaticGraph_fromNode(
            cgraph,
            ZL_NODE_FLOAT32_DECONSTRUCT,
            ZL_GRAPHLIST(ZL_GRAPH_STORE, ZL_GRAPH_HUFFMAN));
    f32Graph = ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph, ZL_NODE_INTERPRET_AS_LE32, f32Graph);

    const ZL_GraphID f64Graph = ZL_Compressor_registerStaticGraph_fromNode1o(
            cgraph, ZL_NODE_INTERPRET_AS_LE64, ZL_GRAPH_FIELD_LZ);

    ZL_GraphID graphs[PytorchModelSuccessor_NumSuccessors];
    graphs[PytorchModelSuccessor_U8]            = ZL_GRAPH_HUFFMAN;
    graphs[PytorchModelSuccessor_F16]           = f16Graph;
    graphs[PytorchModelSuccessor_F32]           = f32Graph;
    graphs[PytorchModelSuccessor_F64]           = f64Graph;
    graphs[PytorchModelSuccessor_OtherFiles]    = ZL_GRAPH_ZSTD;
    graphs[PytorchModelSuccessor_Precompressed] = ZL_GRAPH_STORE;
    graphs[PytorchModelSuccessor_Metadata]      = ZL_GRAPH_ZSTD;

    ZL_Type const inputTypeMask       = ZL_Type_serial;
    const ZL_FunctionGraphDesc fgDesc = {
        .name                = "pytorch model compressor (inner)",
        .graph_f             = pytorchModelDynGraph,
        .inputTypeMasks      = &inputTypeMask,
        .nbInputs            = 1,
        .lastInputIsVariable = false,
        .customGraphs        = graphs,
        .nbCustomGraphs      = PytorchModelSuccessor_NumSuccessors,
    };
    const ZL_GraphID fgraphID =
            ZL_Compressor_registerFunctionGraph(cgraph, &fgDesc);

    const ZL_SegmenterDesc segDesc = {
        .name                = "pytorch model compressor",
        .segmenterFn         = pytorchModelSegmenter,
        .inputTypeMasks      = &inputTypeMask,
        .numInputs           = 1,
        .lastInputIsVariable = false,
        .customGraphs        = &fgraphID,
        .numCustomGraphs     = 1,
    };
    return ZL_Compressor_registerSegmenter(cgraph, &segDesc);
}
