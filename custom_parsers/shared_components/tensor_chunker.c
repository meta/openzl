// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/shared_components/tensor_chunker.h"

#include <limits.h>
#include <stdbool.h>

#include "custom_parsers/shared_components/tensor_graphs.h"

#include "openzl/codecs/zl_generic.h"
#include "openzl/codecs/zl_store.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/overflow.h"
#include "openzl/shared/utils.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_graph_api.h"
#include "openzl/zl_version.h"

#define TC_CHUNK_SIZE_MAX_PID 1
#define TC_SEGMENT_SIZES_PID 2
#define TC_SEGMENT_TAGS_PID 3
#define TC_SEGMENT_ROW_SIZES_PID 4

// A chunk cannot hold more than ZL_runtimeNodeLimit() nodes (20000). Measured
// costs: up to ~39 nodes per segment through the generic graph (int64, f64),
// ~15 through the Transformer, and ~16 per MiB of float data. Merging small
// neighbors and these two caps keep a chunk under ~11000 nodes, leaving room
// for trained successors.
#define TC_SMALL_SEGMENT_SIZE ((size_t)16 << 10)
#define TC_MAX_SEGMENTS_PER_CHUNK 256
#define TC_CHUNK_SIZE_LIMIT ((size_t)64 << 20)

typedef struct {
    size_t size;
    uint32_t tag;
    size_t splitAlign; // byte granularity when the segment must be split
} TC_Segment;

typedef struct {
    ZL_Segmenter* sctx;
    ZL_GraphID innerGraph;
    size_t chunkSizeMax;
    size_t* sizes;
    uint32_t* tags;
    size_t* rowSizes;
    size_t capacity;
    size_t nbSegments;
    size_t nbBytes;
} TC_Chunker;

/* ------------------------------------------------------------------------
 * Inner graph: splits a chunk into its segments and routes them by tag.
 * ------------------------------------------------------------------------ */

static ZL_Report
TC_innerGraphFn(ZL_Graph* graph, ZL_Edge* inputs[], size_t nbInputs)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(graph);
    ZL_ERR_IF_NE(nbInputs, 1, graph_invalidNumInputs);

    const ZL_RefParam sizesParam =
            ZL_Graph_getLocalRefParam(graph, TC_SEGMENT_SIZES_PID);
    const ZL_RefParam tagsParam =
            ZL_Graph_getLocalRefParam(graph, TC_SEGMENT_TAGS_PID);
    const ZL_RefParam rowSizesParam =
            ZL_Graph_getLocalRefParam(graph, TC_SEGMENT_ROW_SIZES_PID);
    ZL_ERR_IF_NE(sizesParam.paramId, TC_SEGMENT_SIZES_PID, graph_invalid);
    ZL_ERR_IF_NE(tagsParam.paramId, TC_SEGMENT_TAGS_PID, graph_invalid);
    ZL_ERR_IF_NE(
            rowSizesParam.paramId, TC_SEGMENT_ROW_SIZES_PID, graph_invalid);
    const size_t* const sizes    = (const size_t*)sizesParam.paramRef;
    const uint32_t* const tags   = (const uint32_t*)tagsParam.paramRef;
    const size_t* const rowSizes = (const size_t*)rowSizesParam.paramRef;
    const size_t nbSegments      = sizesParam.paramSize / sizeof(size_t);
    ZL_ERR_IF_NE(
            tagsParam.paramSize / sizeof(uint32_t), nbSegments, graph_invalid);
    ZL_ERR_IF_NE(
            rowSizesParam.paramSize / sizeof(size_t),
            nbSegments,
            graph_invalid);

    const ZL_GraphIDList successors = ZL_Graph_getCustomGraphs(graph);
    ZL_ERR_IF_LE(successors.nbGraphIDs, ZL_TensorClass_raw, graph_invalid);

    size_t total = 0;
    for (size_t i = 0; i < nbSegments; ++i) {
        total += sizes[i];
    }
    ZL_ERR_IF_NE(
            total,
            ZL_Input_contentSize(ZL_Edge_getData(inputs[0])),
            graph_invalid);
    if (nbSegments == 0) {
        return ZL_Edge_setDestination(inputs[0], ZL_GRAPH_STORE);
    }

    ZL_TRY_LET(
            ZL_EdgeList,
            streams,
            ZL_Edge_runSplitNode(inputs[0], sizes, nbSegments));
    ZL_ASSERT_EQ(streams.nbEdges, nbSegments);
    // Compressors serialized by an older version have fewer successors: tags
    // added since then go to the raw successor.
    for (size_t i = 0; i < nbSegments; ++i) {
        const uint32_t tag =
                tags[i] < successors.nbGraphIDs ? tags[i] : ZL_TensorClass_raw;
        // Rows are at most half of a chunk of at most 64 MiB.
        ZL_ASSERT_LE(rowSizes[i], (size_t)INT_MAX);
        ZL_ERR_IF_ERR(ZL_Edge_setIntMetadata(
                streams.edges[i],
                ZL_TENSOR_ROW_SIZE_METADATA_ID,
                (int)rowSizes[i]));
        ZL_ERR_IF_ERR(ZL_Edge_setDestination(
                streams.edges[i], successors.graphids[tag]));
    }
    return ZL_returnSuccess();
}

/* ------------------------------------------------------------------------
 * Segments
 * ------------------------------------------------------------------------ */

/// Merges consecutive small segments after the first one, so that small
/// segments are never adjacent. A merged run keeps its tag when it is shared,
/// else takes @p mixedTag, and is never split.
static size_t TC_mergeSmallSegments(
        TC_Segment* segments,
        size_t nbSegments,
        uint32_t mixedTag)
{
    size_t n = ZL_MIN(nbSegments, (size_t)1);
    for (size_t i = 1; i < nbSegments; ++i) {
        TC_Segment* const run = &segments[n - 1];
        if (n > 1 && run->size < TC_SMALL_SEGMENT_SIZE
            && segments[i].size < TC_SMALL_SEGMENT_SIZE) {
            if (run->tag != segments[i].tag) {
                run->tag = mixedTag;
            }
            run->size += segments[i].size;
            run->splitAlign = run->size;
            continue;
        }
        segments[n++] = segments[i];
    }
    return n;
}

/* ------------------------------------------------------------------------
 * Chunking
 * ------------------------------------------------------------------------ */

static ZL_Report TC_flush(TC_Chunker* ch)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(ch->sctx);
    if (ch->nbBytes == 0) {
        return ZL_returnSuccess();
    }
    const ZL_RefParam refParams[3] = {
        { TC_SEGMENT_SIZES_PID, ch->sizes, ch->nbSegments * sizeof(size_t) },
        { TC_SEGMENT_TAGS_PID, ch->tags, ch->nbSegments * sizeof(uint32_t) },
        { TC_SEGMENT_ROW_SIZES_PID,
          ch->rowSizes,
          ch->nbSegments * sizeof(size_t) },
    };
    const ZL_LocalParams localParams        = { .refParams = { refParams, 3 } };
    const ZL_RuntimeGraphParameters gparams = { .localParams = &localParams };
    ZL_ERR_IF_ERR(ZL_Segmenter_processChunk(
            ch->sctx, &ch->nbBytes, 1, ch->innerGraph, &gparams));
    ch->nbSegments = 0;
    ch->nbBytes    = 0;
    return ZL_returnSuccess();
}

/// A chunk holds at most one piece of each segment: pieces of a split segment
/// are flushed one by one, except the last.
static void TC_append(TC_Chunker* ch, size_t size, const TC_Segment* seg)
{
    ZL_ASSERT_LT(ch->nbSegments, ch->capacity);
    ch->sizes[ch->nbSegments]    = size;
    ch->tags[ch->nbSegments]     = seg->tag;
    ch->rowSizes[ch->nbSegments] = seg->splitAlign;
    ++ch->nbSegments;
    ch->nbBytes += size;
}

/// Only the last chunk may be smaller than ZL_MIN_CHUNK_SIZE, which
/// ZL_compressBound() relies on.
static ZL_Report TC_flushIfLargeEnough(TC_Chunker* ch)
{
    if (ch->nbBytes < ZL_MIN_CHUNK_SIZE) {
        return ZL_returnSuccess();
    }
    return TC_flush(ch);
}

/// Adds one segment; when it overflows the chunk, it is split on aligned
/// boundaries, its first piece filling the room left in the current chunk.
static ZL_Report TC_addSegment(TC_Chunker* ch, const TC_Segment* seg)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(ch->sctx);
    if (ch->nbSegments >= TC_MAX_SEGMENTS_PER_CHUNK) {
        ZL_ERR_IF_ERR(TC_flush(ch));
    }
    if (ch->nbBytes + seg->size > ch->chunkSizeMax) {
        ZL_ERR_IF_ERR(TC_flushIfLargeEnough(ch));
    }
    // Rows are at most half a chunk (ZL_TensorInfo), and a chunk left open by
    // TC_flushIfLargeEnough() holds less than ZL_MIN_CHUNK_SIZE, at most half a
    // chunk too: a row always fits in the room left, so a chunk never exceeds
    // chunkSizeMax.
    size_t remaining = seg->size;
    while (ch->nbBytes + remaining > ch->chunkSizeMax) {
        const size_t room =
                ch->chunkSizeMax - ZL_MIN(ch->nbBytes, ch->chunkSizeMax);
        const size_t align = seg->splitAlign;
        const size_t piece = ZL_MAX(align, room / align * align);
        if (piece >= remaining) {
            break;
        }
        TC_append(ch, piece, seg);
        ZL_ERR_IF_ERR(TC_flush(ch));
        remaining -= piece;
    }
    TC_append(ch, remaining, seg);
    return ZL_returnSuccess();
}

/// Sends @p segments, which cover the whole input, to the inner graph chunk by
/// chunk. @p sizes, @p tags and @p rowSizes are scratch buffers of
/// @p nbSegments entries.
static ZL_Report TC_process(
        ZL_Segmenter* sctx,
        size_t chunkSizeMax,
        const TC_Segment* segments,
        size_t nbSegments,
        size_t* sizes,
        uint32_t* tags,
        size_t* rowSizes)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(sctx);
    const ZL_GraphIDList customGraphs = ZL_Segmenter_getCustomGraphs(sctx);
    ZL_ERR_IF_NE(customGraphs.nbGraphIDs, 1, graph_invalid);
    TC_Chunker ch = {
        .sctx         = sctx,
        .innerGraph   = customGraphs.graphids[0],
        .chunkSizeMax = chunkSizeMax,
        .sizes        = sizes,
        .tags         = tags,
        .rowSizes     = rowSizes,
        .capacity     = nbSegments,
    };
    for (size_t i = 0; i < nbSegments; ++i) {
        ZL_ERR_IF_ERR(TC_addSegment(&ch, &segments[i]));
    }
    return TC_flush(&ch);
}

/// Adds to @p why that the input is compressed as generic data instead.
static ZL_Report
TC_warnFallback(ZL_Segmenter* sctx, const char* format, ZL_Report why)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(sctx);
    ZL_ERR_IF_ERR(
            why, "Input not parsed as %s: compressed as generic data", format);
    return why;
}

/// Records @p why the input is not parsed as a warning, retrievable via
/// ZL_CCtx_getWarnings(), then compresses it in plain chunks of @p chunkSize
/// with the generic graph.
static ZL_Report TC_fallback(
        ZL_Segmenter* sctx,
        const char* format,
        ZL_Report why,
        size_t chunkSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(sctx);
    ZL_RES_convertToWarning(sctx, TC_warnFallback(sctx, format, why));
    const size_t inputSize =
            ZL_Input_contentSize(ZL_Segmenter_getInput(sctx, 0));
    size_t pos = 0;
    while (pos < inputSize) {
        size_t size = ZL_MIN(chunkSize, inputSize - pos);
        ZL_ERR_IF_ERR(ZL_Segmenter_processChunk(
                sctx, &size, 1, ZL_GRAPH_COMPRESS_GENERIC, NULL));
        pos += size;
    }
    return ZL_returnSuccess();
}

/// The chunk size of the segmenter, or @p defaultSize if unset, raised to
/// 2 * ZL_MIN_CHUNK_SIZE and capped at 64 MiB.
static size_t TC_chunkSizeMax(ZL_Segmenter* sctx, size_t defaultSize)
{
    const ZL_IntParam param =
            ZL_Segmenter_getLocalIntParam(sctx, TC_CHUNK_SIZE_MAX_PID);
    const size_t size =
            param.paramId == TC_CHUNK_SIZE_MAX_PID && param.paramValue > 0
            ? (size_t)param.paramValue
            : defaultSize;
    // Room for a chunk to be filled to ZL_MIN_CHUNK_SIZE before row-aligned
    // splitting closes it.
    return ZL_MIN(
            ZL_MAX(size, 2 * (size_t)ZL_MIN_CHUNK_SIZE), TC_CHUNK_SIZE_LIMIT);
}

/* ------------------------------------------------------------------------
 * Parsing: tokens to segments
 * ------------------------------------------------------------------------ */

/// Converts the tokens of the file into segments.
/// @returns the number of segments written to @p segments.
static size_t TC_buildSegments(
        const ZL_TensorFormat* format,
        const void* tokens,
        size_t nbTokens,
        size_t chunkSizeMax,
        uint32_t* sample,
        TC_Segment* segments)
{
    for (size_t i = 0; i < nbTokens; ++i) {
        const ZL_TensorInfo t = format->describe(
                (const uint8_t*)tokens + i * format->tokenSize, chunkSizeMax);
        ZL_ASSERT_GE(t.splitAlign, 1);
        ZL_ASSERT_LE(t.splitAlign, chunkSizeMax / 2);
        uint32_t tag = t.tag;
        if (tag < ZL_TensorClass_count) {
            tag = ZL_TensorGraphs_refineClass(
                    (ZL_TensorClass)tag, (const uint8_t*)t.ptr, t.size, sample);
        }
        segments[i] = (TC_Segment){ t.size, tag, t.splitAlign };
    }
    return TC_mergeSmallSegments(segments, nbTokens, ZL_TensorClass_raw);
}

/// Sized by the number of tokens: each token becomes one segment, and a chunk
/// holds at most one piece of each segment.
typedef struct {
    void* tokens;
    TC_Segment* segments;
    uint32_t* sample;
    size_t* sizes;
    uint32_t* tags;
    size_t* rowSizes;
} TC_Buffers;

/// @returns false if an allocation failed or its size overflowed.
static bool TC_allocBuffers(
        ZL_Segmenter* sctx,
        size_t maxTokens,
        size_t tokenSize,
        TC_Buffers* buffers)
{
    // A segment is larger than an entry of sizes, tags or rowSizes.
    size_t tokensSize, segmentsSize;
    if (ZL_overflowMulST(maxTokens, tokenSize, &tokensSize)
        || ZL_overflowMulST(
                maxTokens, sizeof(*buffers->segments), &segmentsSize)) {
        return false;
    }
    buffers->tokens   = ZL_Segmenter_getScratchSpace(sctx, tokensSize);
    buffers->segments = ZL_Segmenter_getScratchSpace(sctx, segmentsSize);
    buffers->sample   = ZL_Segmenter_getScratchSpace(
            sctx, ZL_TENSOR_SAMPLE_SIZE * sizeof(*buffers->sample));
    buffers->sizes = ZL_Segmenter_getScratchSpace(
            sctx, maxTokens * sizeof(*buffers->sizes));
    buffers->tags = ZL_Segmenter_getScratchSpace(
            sctx, maxTokens * sizeof(*buffers->tags));
    buffers->rowSizes = ZL_Segmenter_getScratchSpace(
            sctx, maxTokens * sizeof(*buffers->rowSizes));
    return buffers->tokens && buffers->segments && buffers->sample
            && buffers->sizes && buffers->tags && buffers->rowSizes;
}

/// Lexes the input into @p buffers, allocated for the largest number of tokens
/// it can produce.
/// @returns the number of tokens, or why the input cannot be parsed.
static ZL_Report TC_lex(
        ZL_Segmenter* sctx,
        const ZL_TensorFormat* format,
        const void* src,
        size_t srcSize,
        TC_Buffers* buffers)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(sctx);
    ZL_ErrorContext* const errCtx = ZL_GET_DEFAULT_ERROR_CONTEXT(sctx);
    // Counting first bounds the allocations by the number of tensors, and lets
    // an invalid header fall back before anything is allocated.
    ZL_TRY_LET_CONST(
            size_t, maxTokens, format->lex(src, srcSize, NULL, 0, errCtx));
    ZL_ERR_IF_NOT(
            TC_allocBuffers(sctx, maxTokens, format->tokenSize, buffers),
            allocation);
    return format->lex(src, srcSize, buffers->tokens, maxTokens, errCtx);
}

ZL_Report ZL_TensorChunker_segment(
        ZL_Segmenter* sctx,
        const ZL_TensorFormat* format)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(sctx);
    ZL_ERR_IF_NE(ZL_Segmenter_numInputs(sctx), 1, node_invalid_input);
    const ZL_Input* const input = ZL_Segmenter_getInput(sctx, 0);
    const size_t inputSize      = ZL_Input_contentSize(input);
    const size_t chunkSizeMax = TC_chunkSizeMax(sctx, format->defaultChunkSize);
    if (inputSize == 0) {
        size_t size = 0;
        return ZL_Segmenter_processChunk(sctx, &size, 1, ZL_GRAPH_STORE, NULL);
    }
    const unsigned formatVersion =
            (unsigned)ZL_Segmenter_getCParam(sctx, ZL_CParam_formatVersion);
    if (formatVersion < ZL_CHUNK_VERSION_MIN) {
        // A single chunk cannot hold one stream per tensor of a large model.
        return TC_fallback(
                sctx,
                format->name,
                ZL_REPORT_ERROR(
                        graph_parser_unhandledInput,
                        "format version without chunks"),
                SIZE_MAX);
    }

    TC_Buffers buffers = { 0 };
    const ZL_Report lexed =
            TC_lex(sctx, format, ZL_Input_ptr(input), inputSize, &buffers);
    if (ZL_isError(lexed)) {
        return TC_fallback(sctx, format->name, lexed, chunkSizeMax);
    }

    const size_t nbSegments = TC_buildSegments(
            format,
            buffers.tokens,
            ZL_validResult(lexed),
            chunkSizeMax,
            buffers.sample,
            buffers.segments);
    return TC_process(
            sctx,
            chunkSizeMax,
            buffers.segments,
            nbSegments,
            buffers.sizes,
            buffers.tags,
            buffers.rowSizes);
}

/* ------------------------------------------------------------------------
 * Registration
 * ------------------------------------------------------------------------ */

/// The base graph is registered once per compressor, then parameterized with
/// the successors.
static ZL_GraphID TC_innerGraph(
        ZL_Compressor* compressor,
        const char* name,
        const ZL_GraphID* successors,
        size_t nbSuccessors)
{
    ZL_GraphID base = ZL_Compressor_getGraph(compressor, name + 1);
    if (base.gid == ZL_GRAPH_ILLEGAL.gid) {
        const ZL_Type inputType         = ZL_Type_serial;
        const ZL_FunctionGraphDesc desc = {
            .name           = name,
            .graph_f        = TC_innerGraphFn,
            .inputTypeMasks = &inputType,
            .nbInputs       = 1,
        };
        base = ZL_Compressor_registerFunctionGraph(compressor, &desc);
    }
    const ZL_ParameterizedGraphDesc desc = {
        .graph          = base,
        .customGraphs   = successors,
        .nbCustomGraphs = nbSuccessors,
    };
    return ZL_Compressor_registerParameterizedGraph(compressor, &desc);
}

ZL_RESULT_OF(ZL_GraphID)
ZL_TensorChunker_registerSegmenter(
        ZL_Compressor* compressor,
        const char* segmenterName,
        ZL_SegmenterFn segmenterFn,
        size_t chunkSizeMax,
        const char* innerName,
        const ZL_GraphID* successors,
        size_t nbSuccessors)
{
    ZL_RESULT_DECLARE_SCOPE(ZL_GraphID, compressor);
    ZL_ERR_IF_EQ(chunkSizeMax, 0, parameter_invalid);
    ZL_ERR_IF_NE(segmenterName[0], '!', parameter_invalid);
    ZL_ERR_IF_NE(innerName[0], '!', parameter_invalid);
    const ZL_GraphID inner =
            TC_innerGraph(compressor, innerName, successors, nbSuccessors);
    ZL_ERR_IF_EQ(inner.gid, ZL_GRAPH_ILLEGAL.gid, graph_invalid);

    ZL_GraphID base = ZL_Compressor_getGraph(compressor, segmenterName + 1);
    if (base.gid == ZL_GRAPH_ILLEGAL.gid) {
        const ZL_Type inputType     = ZL_Type_serial;
        const ZL_SegmenterDesc desc = {
            .name           = segmenterName,
            .segmenterFn    = segmenterFn,
            .inputTypeMasks = &inputType,
            .numInputs      = 1,
        };
        base = ZL_Compressor_registerSegmenter(compressor, &desc);
    }
    const ZL_IntParam intParam = {
        .paramId    = TC_CHUNK_SIZE_MAX_PID,
        .paramValue = (int)ZL_MIN(chunkSizeMax, (size_t)INT32_MAX),
    };
    const ZL_LocalParams localParams     = { .intParams = { &intParam, 1 } };
    const ZL_ParameterizedGraphDesc desc = {
        .graph          = base,
        .localParams    = &localParams,
        .customGraphs   = &inner,
        .nbCustomGraphs = 1,
    };
    const ZL_GraphID graph =
            ZL_Compressor_registerParameterizedGraph(compressor, &desc);
    ZL_ERR_IF_EQ(graph.gid, ZL_GRAPH_ILLEGAL.gid, graph_invalid);
    return ZL_RESULT_WRAP_VALUE(ZL_GraphID, graph);
}
