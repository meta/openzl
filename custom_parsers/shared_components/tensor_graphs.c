// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/shared_components/tensor_graphs.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "openzl/codecs/zl_ace.h"
#include "openzl/codecs/zl_conversion.h"
#include "openzl/codecs/zl_divide_by.h"
#include "openzl/codecs/zl_entropy.h"
#include "openzl/codecs/zl_float_deconstruct.h"
#include "openzl/codecs/zl_generic.h"
#include "openzl/codecs/zl_range_pack.h"
#include "openzl/codecs/zl_split_by_struct.h"
#include "openzl/codecs/zl_store.h"
#include "openzl/codecs/zl_transformer.h"
#include "openzl/codecs/zl_transpose.h"
#include "openzl/codecs/zl_zstd.h"
#include "openzl/common/assertion.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/data_stats.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/utils.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_selector.h"

ZL_STATIC_ASSERT(
        ZL_TensorClass_upcast32 == 14,
        "ZL_TensorClass is append-only: serialized compressors rely on it");

// A float tensor is atypical when an evenly spaced sample of its elements has
// fewer than 1/TG_DISTINCT_RATIO distinct values (masks, constants, ...).
#define TG_MIN_ELTS_FOR_SAMPLING 64
#define TG_DISTINCT_RATIO 8
// A float32 tensor is upcast when most of such a sample keeps these lowest
// bits of the fraction at zero, as float16 and bfloat16 values converted to
// float32 do.
#define TG_UPCAST_ZERO_BITS 0x1FFFu

static int TG_compareU32(const void* a, const void* b)
{
    const uint32_t x = *(const uint32_t*)a;
    const uint32_t y = *(const uint32_t*)b;
    return (x > y) - (x < y);
}

static bool TG_isAtypical(
        const uint8_t* data,
        size_t nbElts,
        size_t eltSize,
        uint32_t* sample)
{
    if (nbElts < TG_MIN_ELTS_FOR_SAMPLING) {
        return false;
    }
    const size_t nbSamples = ZL_MIN(nbElts, (size_t)ZL_TENSOR_SAMPLE_SIZE);
    const size_t stride    = nbElts / nbSamples;
    for (size_t i = 0; i < nbSamples; ++i) {
        const uint8_t* const elt = data + i * stride * eltSize;
        sample[i] = eltSize == 2 ? ZL_readLE16(elt) : ZL_readLE32(elt);
    }
    qsort(sample, nbSamples, sizeof(sample[0]), TG_compareU32);
    size_t nbDistinct = 1;
    for (size_t i = 1; i < nbSamples; ++i) {
        nbDistinct += sample[i] != sample[i - 1];
    }
    return nbDistinct * TG_DISTINCT_RATIO < nbSamples;
}

static bool TG_isUpcast(const uint8_t* data, size_t nbElts)
{
    if (nbElts < TG_MIN_ELTS_FOR_SAMPLING) {
        return false;
    }
    const size_t nbSamples = ZL_MIN(nbElts, (size_t)ZL_TENSOR_SAMPLE_SIZE);
    const size_t stride    = nbElts / nbSamples;
    size_t nbUpcast        = 0;
    for (size_t i = 0; i < nbSamples; ++i) {
        const uint32_t value = ZL_readLE32(data + i * stride * 4);
        nbUpcast += (value & TG_UPCAST_ZERO_BITS) == 0;
    }
    // Native float32 values have these bits all zero once in 8192, values
    // converted from 16 bits always: half the sample leaves a wide margin.
    return nbUpcast * 2 > nbSamples;
}

ZL_TensorClass ZL_TensorGraphs_refineClass(
        ZL_TensorClass cls,
        const uint8_t* data,
        size_t size,
        uint32_t* sample)
{
    if (cls != ZL_TensorClass_bf16 && cls != ZL_TensorClass_f16
        && cls != ZL_TensorClass_f32) {
        return cls;
    }
    const size_t eltSize = cls == ZL_TensorClass_f32 ? 4 : 2;
    if (TG_isAtypical(data, size / eltSize, eltSize, sample)) {
        return eltSize == 2 ? ZL_TensorClass_atypical16
                            : ZL_TensorClass_atypical32;
    }
    if (cls == ZL_TensorClass_f32 && TG_isUpcast(data, size / 4)) {
        return ZL_TensorClass_upcast32;
    }
    return cls;
}

enum { TG_scale16_huffman, TG_scale16_bytes, TG_scale16_zstd, TG_SCALE16_NB };

/// The entropy graphs code large inputs in independent chunks, each with its
/// own header (encode_entropy_binding.c): the estimates follow the same chunks.
#define TG_FSE_CHUNK_SIZE ((size_t)1 << 15)
#define TG_FSE_CHUNKED_SIZE 100000
#define TG_HUFFMAN_CHUNK_ELTS ((size_t)1 << 18)
#define TG_HUFFMAN_CHUNKED_ELTS 400000
/// Up to this number of symbols, Huffman code lengths are computed rather than
/// taken from the entropy, which they exceed the most on few symbols.
#define TG_EXACT_HUFFMAN_SYMBOLS 256

/// Scratch space of the estimates. TG_scale16Stats() spends on it a time
/// following the number of distinct scales rather than the 65536 possible ones.
typedef struct {
    unsigned count[65536];
    unsigned counts[65536]; // count of each distinct scale
    uint16_t scales[65536]; // distinct scales
    // internal nodes of a Huffman tree
    uint64_t nodes[TG_EXACT_HUFFMAN_SYMBOLS];
    unsigned byteCounts[2][256]; // count of each value of each byte
} TG_Scale16Workspace;

/// FSE codes each byte of the scales with a header of about 10 bits per
/// symbol.
static double TG_bytesSizeEstimate(
        const uint8_t* src,
        size_t nbElts,
        TG_Scale16Workspace* wksp)
{
    const size_t chunkSize =
            nbElts >= TG_FSE_CHUNKED_SIZE ? TG_FSE_CHUNK_SIZE : nbElts;
    double size = 0;
    for (size_t start = 0; start < nbElts; start += chunkSize) {
        const size_t n               = ZL_MIN(chunkSize, nbElts - start);
        unsigned (*const count)[256] = wksp->byteCounts;
        memset(wksp->byteCounts, 0, sizeof(wksp->byteCounts));
        for (size_t i = start; i < start + n; ++i) {
            count[0][src[2 * i]]++;
            count[1][src[2 * i + 1]]++;
        }
        for (size_t b = 0; b < 2; ++b) {
            size_t cardinality = 0;
            for (size_t v = 0; v < 256; ++v) {
                cardinality += count[b][v] != 0;
            }
            const double laneSize =
                    ZL_calculateEntropy(count[b], 255, n) * (double)n / 8
                    + 1.25 * (double)cardinality;
            size += ZL_MIN(laneSize, (double)n);
        }
    }
    return size;
}

typedef struct {
    double huffmanSize;
    size_t repeats;       // scales equal to the previous one
    double chanceRepeats; // repeats expected from the histogram alone
} TG_Scale16Stats;

static int TG_compareCounts(const void* a, const void* b)
{
    const unsigned x = *(const unsigned*)a;
    const unsigned y = *(const unsigned*)b;
    return (x > y) - (x < y);
}

/// @returns the size in bits of the Huffman code of the @p nbSymbols counts,
/// which it sorts: the sum of the weights of the internal nodes of the tree,
/// built in @p nodes in increasing weight order.
static double
TG_huffmanBits(unsigned* counts, size_t nbSymbols, uint64_t* nodes)
{
    qsort(counts, nbSymbols, sizeof(*counts), TG_compareCounts);
    size_t leaf = 0, node = 0;
    uint64_t bits = 0;
    for (size_t n = 0; n + 1 < nbSymbols; ++n) {
        uint64_t weight = 0;
        for (int child = 0; child < 2; ++child) {
            if (node < n && (leaf == nbSymbols || nodes[node] < counts[leaf])) {
                weight += nodes[node++];
            } else {
                weight += counts[leaf++];
            }
        }
        nodes[n] = weight;
        bits += weight;
    }
    return (double)bits;
}

/// The 16-bit Huffman graph bitpacks or stores a chunk whose Huffman code it
/// expects to gain too little, with a header estimate of max(100, maxScale / 4)
/// bytes, over 8 KiB on signed scales (encode_entropy_binding.c).
/// @returns the size of a chunk of @p n scales, whose Huffman code would take
/// @p huffmanSize bytes.
static double TG_huffmanChunkSize(
        size_t n,
        double entropy,
        unsigned maxScale,
        size_t minGain,
        double huffmanSize)
{
    const size_t storeSize = 2 * n;
    const size_t estimate  = (size_t)(ZL_MAX(1, entropy) * (double)n + 7) / 8
            + ZL_MAX(100, maxScale / 4);
    const size_t bitpackSize =
            (n * (size_t)ZL_nextPow2(maxScale + 1) + 7) / 8 + 2;
    if (bitpackSize <= estimate && bitpackSize + minGain <= storeSize) {
        return (double)bitpackSize;
    }
    if (entropy > 15 || estimate + minGain > storeSize) {
        return (double)storeSize;
    }
    return ZL_MIN(huffmanSize, (double)storeSize);
}

static TG_Scale16Stats
TG_scale16Stats(const uint8_t* src, size_t nbElts, TG_Scale16Workspace* wksp)
{
    TG_Scale16Stats stats = { 0 };
    const size_t chunkElts =
            nbElts >= TG_HUFFMAN_CHUNKED_ELTS ? TG_HUFFMAN_CHUNK_ELTS : nbElts;
    // The minimum gain of the entropy graphs, shared among the chunks
    const size_t minGain = ZL_MAX(32, 2 * nbElts / 100);
    memset(wksp->count, 0, sizeof(wksp->count));
    uint16_t prev = (uint16_t)~ZL_readLE16(src);
    for (size_t start = 0; start < nbElts; start += chunkElts) {
        const size_t n     = ZL_MIN(chunkElts, nbElts - start);
        size_t cardinality = 0;
        unsigned maxScale  = 0;
        for (size_t i = start; i < start + n; ++i) {
            const uint16_t v = ZL_readLE16(src + 2 * i);
            if (wksp->count[v]++ == 0) {
                wksp->scales[cardinality++] = v;
            }
            maxScale = ZL_MAX(maxScale, v);
            stats.repeats += v == prev;
            prev = v;
        }
        double sumSquares = 0;
        for (size_t k = 0; k < cardinality; ++k) {
            const unsigned c             = wksp->count[wksp->scales[k]];
            wksp->counts[k]              = c;
            wksp->count[wksp->scales[k]] = 0;
            sumSquares += (double)c * (double)c;
        }
        // 16-bit Huffman spends at least a bit per symbol, and about 1.5 byte
        // of header per symbol. Its whole code lengths cost more than the
        // entropy on few symbols, e.g. 1.67 bits instead of 1.58 on 3 equally
        // frequent ones.
        const double entropy =
                ZL_calculateEntropy(wksp->counts, cardinality - 1, n);
        const double bits = cardinality <= TG_EXACT_HUFFMAN_SYMBOLS
                ? TG_huffmanBits(wksp->counts, cardinality, wksp->nodes)
                : entropy * (double)n;
        const double huffmanSize =
                ZL_MAX(bits, (double)n) / 8 + 1.5 * (double)cardinality;
        stats.huffmanSize += TG_huffmanChunkSize(
                n,
                entropy,
                maxScale,
                (size_t)((uint64_t)minGain * n / nbElts),
                huffmanSize);
        stats.chanceRepeats += sumSquares / (double)n;
    }
    return stats;
}

static double
TG_trySize(const ZL_Selector* selector, const ZL_Input* input, ZL_GraphID graph)
{
    const ZL_GraphReport report = ZL_Selector_tryGraph(selector, input, graph);
    return ZL_isError(report.finalCompressedSize)
            ? (double)ZL_Input_contentSize(input)
            : (double)ZL_validResult(report.finalCompressedSize);
}

/// zstd is first tried on 3 slices of the scales, which show repeats within a
/// row, e.g. scales shared by a row or repeating every row.
#define TG_SAMPLE_SLICE_SIZE ((size_t)16 << 10)

static double TG_zstdSampleSize(
        const ZL_Selector* selector,
        const ZL_Input* input,
        ZL_GraphID zstd)
{
    const size_t size = ZL_Input_contentSize(input);
    if (size <= 3 * TG_SAMPLE_SLICE_SIZE) {
        return TG_trySize(selector, input, zstd);
    }
    const uint8_t* const src = (const uint8_t*)ZL_Input_ptr(input);
    uint8_t* const sample    = (uint8_t*)ZL_Selector_getScratchSpace(
            selector, 3 * TG_SAMPLE_SLICE_SIZE);
    if (sample == NULL) {
        return (double)size;
    }
    const size_t middle = (size / 2 - TG_SAMPLE_SLICE_SIZE / 2) & ~(size_t)1;
    memcpy(sample, src, TG_SAMPLE_SLICE_SIZE);
    memcpy(sample + TG_SAMPLE_SLICE_SIZE, src + middle, TG_SAMPLE_SLICE_SIZE);
    memcpy(sample + 2 * TG_SAMPLE_SLICE_SIZE,
           src + size - TG_SAMPLE_SLICE_SIZE,
           TG_SAMPLE_SLICE_SIZE);
    ZL_TypedRef* const sampleRef =
            ZL_TypedRef_createSerial(sample, 3 * TG_SAMPLE_SLICE_SIZE);
    if (sampleRef == NULL) {
        return (double)size;
    }
    const double sampleSize = TG_trySize(selector, sampleRef, zstd);
    ZL_TypedRef_free(sampleRef);
    return sampleSize * (double)size / (double)(3 * TG_SAMPLE_SLICE_SIZE);
}

/// No codec wins on all 16-bit scales: 16-bit Huffman suits most tensors,
/// coding their bytes separately suits scales taking many values for their
/// number, and zstd catches repeated scales. The entropy coders are estimated
/// from histograms, which costs a fraction of trying them.
static ZL_GraphID TG_selectScale16(
        const ZL_Selector* selector,
        const ZL_Input* input,
        const ZL_GraphID* candidates,
        size_t nbCandidates)
{
    // A serialized compressor can replace the candidates.
    if (nbCandidates != TG_SCALE16_NB) {
        return ZL_GRAPH_ILLEGAL;
    }
    // Only 16-bit scales are estimated, and zstd takes any other input.
    if (ZL_Input_eltWidth(input) != 2) {
        return candidates[TG_scale16_zstd];
    }
    const uint8_t* const src = (const uint8_t*)ZL_Input_ptr(input);
    const size_t nbElts      = ZL_Input_contentSize(input) / 2;
    TG_Scale16Workspace* const wksp =
            (TG_Scale16Workspace*)ZL_Selector_getScratchSpace(
                    selector, sizeof(TG_Scale16Workspace));
    if (wksp == NULL || nbElts < 2) {
        return candidates[TG_scale16_huffman];
    }
    const TG_Scale16Stats stats = TG_scale16Stats(src, nbElts, wksp);
    // Huffman stores constant scales as a constant.
    if (stats.repeats == nbElts - 1) {
        return candidates[TG_scale16_huffman];
    }
    double sizes[TG_SCALE16_NB];
    sizes[TG_scale16_huffman] = stats.huffmanSize;
    sizes[TG_scale16_bytes]   = TG_bytesSizeEstimate(src, nbElts, wksp);
    // zstd only wins on repeated scales. A sample shows repeats within a row,
    // but can be more repetitive than the whole scales, and misses repeats
    // paying off only over the whole scales, e.g. scales duplicated by a
    // conversion from larger groups, which repeat their predecessor more often
    // than chance: zstd is then tried on the whole scales.
    const ZL_GraphID zstd = candidates[TG_scale16_zstd];
    const bool repeated =
            (double)stats.repeats > stats.chanceRepeats + (double)nbElts / 8;
    sizes[TG_scale16_zstd] = repeated
            ? TG_trySize(selector, input, zstd)
            : TG_zstdSampleSize(selector, input, zstd);
    if (!repeated
        && sizes[TG_scale16_zstd]
                < ZL_MIN(sizes[TG_scale16_huffman], sizes[TG_scale16_bytes])
        && ZL_Input_contentSize(input) > 3 * TG_SAMPLE_SLICE_SIZE) {
        sizes[TG_scale16_zstd] = TG_trySize(selector, input, zstd);
    }
    size_t best = 0;
    for (size_t i = 1; i < TG_SCALE16_NB; ++i) {
        if (sizes[i] < sizes[best]) {
            best = i;
        }
    }
    if (sizes[best] >= 2 * (double)nbElts) {
        return ZL_GRAPH_STORE;
    }
    return candidates[best];
}

ZL_RESULT_OF(ZL_GraphID)
ZL_TensorGraphs_registerScale16Graph(ZL_Compressor* compressor)
{
    // Serialized compressors refer to selectors by name, so this one is
    // anchored, and registered once per compressor.
    const ZL_GraphID existing =
            ZL_Compressor_getGraph(compressor, "Tensor Scale16");
    if (existing.gid != ZL_GRAPH_ILLEGAL.gid) {
        return ZL_RESULT_WRAP_VALUE(ZL_GraphID, existing);
    }
    const ZL_GraphID candidates[TG_SCALE16_NB] = {
        [TG_scale16_huffman] = ZL_GRAPH_HUFFMAN,
        [TG_scale16_bytes]   = ZL_Compressor_registerTransposeSplitGraph(
                compressor, ZL_GRAPH_FSE),
        // Keeps most of the gain of higher levels on repeated scales, at a
        // fraction of their cost.
        [TG_scale16_zstd] =
                ZL_Compressor_registerZstdGraph_withLevel(compressor, 3),
    };
    const ZL_SelectorDesc desc = {
        .selector_f     = TG_selectScale16,
        .inStreamType   = ZL_Type_serial | ZL_Type_struct,
        .customGraphs   = candidates,
        .nbCustomGraphs = TG_SCALE16_NB,
        .name           = "!Tensor Scale16",
    };
    return ZL_Compressor_registerSelectorGraph2(compressor, &desc);
}

/// 8-bit floats take most of their 256 values with statistics that hold over
/// a tensor, so FSE codes them in large chunks with an accurate table.
static ZL_GraphID TG_fp8Graph(ZL_Compressor* compressor)
{
    const ZL_IntParam params[2] = {
        { ZL_ENTROPY_CHUNK_SIZE_PID, 1 << 19 },
        { ZL_ENTROPY_FSE_MAX_TABLE_LOG_PID, 12 },
    };
    const ZL_LocalParams localParams     = { .intParams = { params, 2 } };
    const ZL_ParameterizedGraphDesc desc = {
        .graph       = ZL_GRAPH_FSE,
        .localParams = &localParams,
    };
    return ZL_Compressor_registerParameterizedGraph(compressor, &desc);
}

/// Trainable successor for numeric data of width @p interpretNode.
static ZL_GraphID TG_numericSuccessor(
        ZL_Compressor* compressor,
        ZL_NodeID interpretNode,
        ZL_GraphID defaultGraph)
{
    return ZL_Compressor_registerStaticGraph_fromNode1o(
            compressor,
            interpretNode,
            ZL_Compressor_buildACEGraphWithDefault(compressor, defaultGraph));
}

/// Exponents are entropy coded, signs and fractions go to @p signFracGraph.
static ZL_GraphID TG_floatSuccessor(
        ZL_Compressor* compressor,
        ZL_NodeID interpretNode,
        ZL_NodeID deconstructNode,
        ZL_GraphID signFracGraph)
{
    const ZL_GraphID deconstruct = ZL_Compressor_registerStaticGraph_fromNode(
            compressor,
            deconstructNode,
            ZL_GRAPHLIST(signFracGraph, ZL_GRAPH_FSE));
    return TG_numericSuccessor(compressor, interpretNode, deconstruct);
}

/// float16 keeps 2 mantissa bits next to sign and exponent in its high byte,
/// so its bytes are transposed into 2 streams. The low byte is noise in native
/// float16 weights, where FSE falls back to raw storage at no cost, but weights
/// converted from bfloat16 leave its 3 lowest bits at zero.
static ZL_GraphID TG_f16Successor(ZL_Compressor* compressor)
{
    return ZL_Compressor_buildACEGraphWithDefault(
            compressor,
            ZL_Compressor_registerStaticGraph_fromNode1o(
                    compressor,
                    ZL_NODE_CONVERT_SERIAL_TO_STRUCT2,
                    ZL_Compressor_registerTransposeSplitGraph(
                            compressor, ZL_GRAPH_FSE)));
}

enum { TG_upcast32_bf16, TG_upcast32_fraction, TG_UPCAST32_NB };

/// Sends float32 values whose 16 low bits are all zero, as bfloat16 values
/// converted to float32 are, to the bfloat16 graph.
static ZL_GraphID TG_selectUpcast32(
        const ZL_Selector* selector,
        const ZL_Input* input,
        const ZL_GraphID* candidates,
        size_t nbCandidates)
{
    (void)selector;
    // A serialized compressor can replace the candidates.
    if (nbCandidates != TG_UPCAST32_NB) {
        return ZL_GRAPH_ILLEGAL;
    }
    const uint8_t* const src = (const uint8_t*)ZL_Input_ptr(input);
    const size_t size        = ZL_Input_contentSize(input);
    if (size == 0 || size % 4 != 0) {
        return candidates[TG_upcast32_fraction];
    }
    uint32_t min = UINT16_MAX;
    uint32_t max = 0;
    for (size_t pos = 0; pos < size; pos += 4) {
        const uint32_t value = ZL_readLE32(src + pos);
        if ((value & 0xFFFF) != 0) {
            return candidates[TG_upcast32_fraction];
        }
        min = ZL_MIN(min, value >> 16);
        max = ZL_MAX(max, value >> 16);
    }
    // range_pack narrows ranges below 256 to 8 bits, which the bfloat16 graph
    // does not take.
    if (max - min < 256) {
        return candidates[TG_upcast32_fraction];
    }
    return candidates[TG_upcast32_bf16];
}

ZL_RESULT_OF(ZL_GraphID)
ZL_TensorGraphs_registerUpcast32Graph(ZL_Compressor* compressor)
{
    // Serialized compressors refer to selectors by name, so this one is
    // anchored, and registered once per compressor.
    const ZL_GraphID existing =
            ZL_Compressor_getGraph(compressor, "Tensor Upcast32");
    if (existing.gid != ZL_GRAPH_ILLEGAL.gid) {
        return ZL_RESULT_WRAP_VALUE(ZL_GraphID, existing);
    }
    // Dividing by 2^16 leaves the bfloat16 values, which range_pack narrows to
    // 16 bits after subtracting the smallest one.
    const ZL_GraphID bf16 = ZL_Compressor_registerStaticGraph_fromNode1o(
            compressor,
            ZL_NODE_RANGE_PACK,
            ZL_Compressor_registerStaticGraph_fromNode(
                    compressor,
                    ZL_NODE_BFLOAT16_DECONSTRUCT,
                    ZL_GRAPHLIST(ZL_GRAPH_STORE, ZL_GRAPH_FSE)));
    const ZL_GraphID candidates[TG_UPCAST32_NB] = {
        [TG_upcast32_bf16] = ZL_Compressor_registerStaticGraph_fromNode1o(
                compressor,
                ZL_NODE_INTERPRET_AS_LE32,
                ZL_Compressor_registerStaticGraph_fromNode1o(
                        compressor,
                        ZL_Compressor_registerDivideByNode(compressor, 1 << 16),
                        bf16)),
        // The zeros of the fraction are in its low bytes, entropy coded one by
        // one, which would only slow down native float32, where these bytes
        // are noise.
        [TG_upcast32_fraction] = TG_floatSuccessor(
                compressor,
                ZL_NODE_INTERPRET_AS_LE32,
                ZL_NODE_FLOAT32_DECONSTRUCT,
                ZL_Compressor_registerTransposeSplitGraph(
                        compressor, ZL_GRAPH_FSE)),
    };
    const ZL_SelectorDesc desc = {
        .selector_f     = TG_selectUpcast32,
        .inStreamType   = ZL_Type_serial,
        .customGraphs   = candidates,
        .nbCustomGraphs = TG_UPCAST32_NB,
        .name           = "!Tensor Upcast32",
    };
    return ZL_Compressor_registerSelectorGraph2(compressor, &desc);
}

void ZL_TensorGraphs_buildSuccessors(
        ZL_Compressor* compressor,
        ZL_GraphID scale16,
        ZL_GraphID successors[ZL_TensorClass_count])
{
    successors[ZL_TensorClass_header] =
            ZL_Compressor_buildACEGraphWithDefault(compressor, ZL_GRAPH_ZSTD);
    successors[ZL_TensorClass_raw] = ZL_Compressor_buildACEGraphWithDefault(
            compressor, ZL_GRAPH_COMPRESS_GENERIC);
    // Same default as raw, but kept apart so that training specializes typed
    // byte tensors (masks, quantized weights) separately from unknown bytes.
    successors[ZL_TensorClass_int8] = ZL_Compressor_buildACEGraphWithDefault(
            compressor, ZL_GRAPH_COMPRESS_GENERIC);
    successors[ZL_TensorClass_int16] = TG_numericSuccessor(
            compressor, ZL_NODE_INTERPRET_AS_LE16, ZL_GRAPH_COMPRESS_GENERIC);
    successors[ZL_TensorClass_int32] = TG_numericSuccessor(
            compressor, ZL_NODE_INTERPRET_AS_LE32, ZL_GRAPH_COMPRESS_GENERIC);
    successors[ZL_TensorClass_int64] = TG_numericSuccessor(
            compressor, ZL_NODE_INTERPRET_AS_LE64, ZL_GRAPH_COMPRESS_GENERIC);
    successors[ZL_TensorClass_f16]  = TG_f16Successor(compressor);
    successors[ZL_TensorClass_bf16] = TG_floatSuccessor(
            compressor,
            ZL_NODE_INTERPRET_AS_LE16,
            ZL_NODE_BFLOAT16_DECONSTRUCT,
            ZL_GRAPH_STORE);
    successors[ZL_TensorClass_f32] = TG_floatSuccessor(
            compressor,
            ZL_NODE_INTERPRET_AS_LE32,
            ZL_NODE_FLOAT32_DECONSTRUCT,
            ZL_GRAPH_STORE);
    successors[ZL_TensorClass_f64] = TG_numericSuccessor(
            compressor, ZL_NODE_INTERPRET_AS_LE64, ZL_GRAPH_COMPRESS_GENERIC);
    successors[ZL_TensorClass_atypical16] = TG_numericSuccessor(
            compressor,
            ZL_NODE_INTERPRET_AS_LE16,
            ZL_GRAPH_TRANSFORMER_NUMERIC);
    successors[ZL_TensorClass_atypical32] = TG_numericSuccessor(
            compressor,
            ZL_NODE_INTERPRET_AS_LE32,
            ZL_GRAPH_TRANSFORMER_NUMERIC);
    successors[ZL_TensorClass_fp8] = ZL_Compressor_buildACEGraphWithDefault(
            compressor, TG_fp8Graph(compressor));
    successors[ZL_TensorClass_scale16] = ZL_Compressor_buildACEGraphWithDefault(
            compressor,
            ZL_Compressor_registerStaticGraph_fromNode1o(
                    compressor, ZL_NODE_CONVERT_SERIAL_TO_STRUCT2, scale16));
    // Errors leave an illegal graph, which the parser registration rejects.
    const ZL_RESULT_OF(ZL_GraphID) upcast32 =
            ZL_TensorGraphs_registerUpcast32Graph(compressor);
    successors[ZL_TensorClass_upcast32] = ZL_RES_isError(upcast32)
            ? ZL_GRAPH_ILLEGAL
            : ZL_Compressor_buildACEGraphWithDefault(
                      compressor, ZL_RES_value(upcast32));
}
