// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/numpy/numpy_parser.h"

#include "custom_parsers/numpy/numpy_lexer.h"
#include "custom_parsers/shared_components/tensor_chunker.h"
#include "custom_parsers/shared_components/tensor_graphs.h"
#include "openzl/codecs/zl_ace.h"
#include "openzl/codecs/zl_conversion.h"
#include "openzl/codecs/zl_generic.h"
#include "openzl/codecs/zl_transformer.h"
#include "openzl/codecs/zl_zstd.h"
#include "openzl/zl_errors.h"

static ZL_TensorClass NP_intClassOf(uint32_t itemSize)
{
    switch (itemSize) {
        case 1:
            return ZL_TensorClass_int8;
        case 2:
            return ZL_TensorClass_int16;
        case 4:
            return ZL_TensorClass_int32;
        case 8:
            return ZL_TensorClass_int64;
        default:
            return ZL_TensorClass_raw;
    }
}

/// Classes of the arrays of numbers, whose successors only depend on the width
/// of the numbers. Complex numbers are pairs of floats.
static ZL_TensorClass NP_classOf(const ZL_NumPyToken* t)
{
    switch (t->kind) {
        case 'b': // booleans
        case 'i':
        case 'u':
        case 'm': // timedeltas
        case 'M': // datetimes
        case 'V': // opaque elements, such as the bfloat16 of ml_dtypes
            return NP_intClassOf(t->itemSize);
        case 'f':
            switch (t->itemSize) {
                case 2:
                    return ZL_TensorClass_f16;
                case 4:
                    return ZL_TensorClass_f32;
                case 8:
                    return ZL_TensorClass_f64;
                default:
                    return ZL_TensorClass_raw;
            }
        case 'c':
            switch (t->itemSize) {
                case 8:
                    return ZL_TensorClass_f32;
                case 16:
                    return ZL_TensorClass_f64;
                default:
                    return ZL_TensorClass_raw;
            }
        default:
            return ZL_TensorClass_raw;
    }
}

/// Split granularity: whole rows when they are small enough, else elements.
/// A row holds all the dimensions but the outermost one, which is the first in
/// C order and the last in Fortran order.
static size_t
NP_splitAlign(const ZL_NumPyToken* t, ZL_TensorClass cls, size_t chunkSizeMax)
{
    if (cls == ZL_TensorClass_raw) {
        return 1;
    }
    const size_t eltSize = t->itemSize;
    if (t->ndims < 2) {
        return eltSize;
    }
    const uint32_t first = t->fortranOrder ? 0 : 1;
    const uint32_t end   = t->fortranOrder ? t->ndims - 1 : t->ndims;
    uint64_t rowBytes    = eltSize;
    for (uint32_t d = first; d < end; ++d) {
        if (t->shape[d] != 0 && rowBytes > chunkSizeMax / 2 / t->shape[d]) {
            return eltSize;
        }
        rowBytes *= t->shape[d];
    }
    return rowBytes ? (size_t)rowBytes : eltSize;
}

static ZL_Report NP_lex(
        const void* src,
        size_t srcSize,
        void* tokens,
        size_t tokensCapacity,
        ZL_ErrorContext* errCtx)
{
    return ZL_NumPy_lex(src, srcSize, tokens, tokensCapacity, errCtx);
}

static ZL_TensorInfo NP_describe(const void* token, size_t chunkSizeMax)
{
    const ZL_NumPyToken* const t = (const ZL_NumPyToken*)token;
    switch (t->type) {
        case ZL_NumPyTokenType_header:
            return (ZL_TensorInfo){ t->ptr, t->size, ZL_TensorClass_header, 1 };
        case ZL_NumPyTokenType_array: {
            const ZL_TensorClass cls = NP_classOf(t);
            return (ZL_TensorInfo){
                t->ptr, t->size, cls, NP_splitAlign(t, cls, chunkSizeMax)
            };
        }
        case ZL_NumPyTokenType_gap:
        default:
            return (ZL_TensorInfo){ t->ptr, t->size, ZL_TensorClass_raw, 1 };
    }
}

static const ZL_TensorFormat kNumPyFormat = {
    .name             = "NumPy",
    .defaultChunkSize = ZL_NUMPY_DEFAULT_CHUNK_SIZE,
    .tokenSize        = sizeof(ZL_NumPyToken),
    .lex              = NP_lex,
    .describe         = NP_describe,
};

static ZL_Report NP_segmenterFn(ZL_Segmenter* sctx)
{
    return ZL_TensorChunker_segment(sctx, &kNumPyFormat);
}

/// Reads numbers with @p convert and sends them to the Numeric Transformer.
static ZL_GraphID NP_numericSuccessor(
        ZL_Compressor* compressor,
        ZL_NodeID convert)
{
    return ZL_Compressor_buildACEGraphWithDefault(
            compressor,
            ZL_Compressor_registerStaticGraph_fromNode1o(
                    compressor, convert, ZL_GRAPH_TRANSFORMER_NUMERIC));
}

ZL_RESULT_OF(ZL_GraphID)
ZL_NumPy_registerGraph(ZL_Compressor* compressor, size_t chunkSizeMax)
{
    const ZL_GraphID raw = ZL_Compressor_buildACEGraphWithDefault(
            compressor, ZL_GRAPH_COMPRESS_GENERIC);
    const ZL_GraphID numbers8 =
            NP_numericSuccessor(compressor, ZL_NODE_CONVERT_SERIAL_TO_NUM8);
    const ZL_GraphID numbers16 =
            NP_numericSuccessor(compressor, ZL_NODE_CONVERT_SERIAL_TO_NUM_LE16);
    const ZL_GraphID numbers32 =
            NP_numericSuccessor(compressor, ZL_NODE_CONVERT_SERIAL_TO_NUM_LE32);
    const ZL_GraphID numbers64 =
            NP_numericSuccessor(compressor, ZL_NODE_CONVERT_SERIAL_TO_NUM_LE64);
    ZL_GraphID successors[ZL_TensorClass_count];
    for (size_t i = 0; i < ZL_TensorClass_count; ++i) {
        successors[i] = raw;
    }
    successors[ZL_TensorClass_header] =
            ZL_Compressor_buildACEGraphWithDefault(compressor, ZL_GRAPH_ZSTD);
    successors[ZL_TensorClass_int8]  = numbers8;
    successors[ZL_TensorClass_int16] = numbers16;
    successors[ZL_TensorClass_int32] = numbers32;
    successors[ZL_TensorClass_int64] = numbers64;
    successors[ZL_TensorClass_f16]   = numbers16;
    successors[ZL_TensorClass_f32]   = numbers32;
    successors[ZL_TensorClass_f64]   = numbers64;
    // The tensor chunker moves floats with few distinct values to these classes
    successors[ZL_TensorClass_atypical16] = numbers16;
    successors[ZL_TensorClass_atypical32] = numbers32;
    return ZL_TensorChunker_registerSegmenter(
            compressor,
            "!NumPy Segmenter",
            NP_segmenterFn,
            chunkSizeMax,
            "!NumPy Parser",
            successors,
            ZL_TensorClass_count);
}
