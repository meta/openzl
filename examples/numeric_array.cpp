// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <array>
#include <string>

#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_public_nodes.h"
#include "openzl/zl_selector.h"

#include "examples/example_utils.h"

namespace {
using namespace zstrong::examples;

// --8<-- [start:setup-compressor]
void parameterizeCompressor(ZL_Compressor* compressor)
{
    // Set the format version. This should be set to the maximum format version
    // that all deployed decompressors support.
    abortIfError(
            compressor,
            ZL_Compressor_setParameter(
                    compressor,
                    ZL_CParam_formatVersion,
                    kExampleFormatVersion));
    // Set the compression level.
    abortIfError(
            compressor,
            ZL_Compressor_setParameter(
                    compressor,
                    ZL_CParam_compressionLevel,
                    kExampleCompressionLevel));
}

ZL_GraphID buildCompressorStandard(ZL_Compressor* compressor)
{
    parameterizeCompressor(compressor);
    return ZL_GRAPH_COMPRESS_GENERIC;
}
// --8<-- [end:setup-compressor]

// --8<-- [start:customizing-compressor1]
ZL_GraphID buildCompressorSorted(ZL_Compressor* compressor)
{
    parameterizeCompressor(compressor);
    return ZL_Compressor_registerStaticGraph_fromNode1o(
            compressor, ZL_NODE_DELTA_INT, ZL_GRAPH_COMPRESS_GENERIC);
}
// --8<-- [end:customizing-compressor1]

// --8<-- [start:customizing-compressor2]
ZL_GraphID buildCompressorInt(ZL_Compressor* compressor)
{
    parameterizeCompressor(compressor);
    return ZL_GRAPH_FIELD_LZ;
}
// --8<-- [end:customizing-compressor2]

// --8<-- [start:customizing-compressor3]
ZL_GraphID buildCompressorBFloat16(ZL_Compressor* compressor)
{
    parameterizeCompressor(compressor);
    std::array<ZL_GraphID, 2> successors = {
        ZL_GRAPH_STORE, // sign+fraction
        ZL_GRAPH_FSE,   // exponent
    };
    // Separate the exponent from the sign+fraction bits.
    // Pass the exponent to FSE, and store the sign+fraction bits as-is.
    return ZL_Compressor_registerStaticGraph_fromNode(
            compressor,
            ZL_NODE_BFLOAT16_DECONSTRUCT,
            successors.data(),
            successors.size());
}

ZL_GraphID buildCompressorFloat16(ZL_Compressor* compressor)
{
    parameterizeCompressor(compressor);
    const ZL_GraphID bitpack = ZL_Compressor_registerStaticGraph_fromNode1o(
            compressor, ZL_NODE_INTERPRET_TOKEN_AS_LE, ZL_GRAPH_BITPACK);
    std::array<ZL_GraphID, 2> successors = {
        bitpack,      // sign+fraction
        ZL_GRAPH_FSE, // exponent
    };
    // Separate the exponent from the sign+fraction bits.
    // Pass the exponent to FSE, and bitpack the sign+fraction bits as-is.
    return ZL_Compressor_registerStaticGraph_fromNode(
            compressor,
            ZL_NODE_FLOAT16_DECONSTRUCT,
            successors.data(),
            successors.size());
}

ZL_GraphID buildCompressorFloat32(ZL_Compressor* compressor)
{
    parameterizeCompressor(compressor);
    std::array<ZL_GraphID, 2> successors = {
        ZL_GRAPH_STORE, // sign+fraction
        ZL_GRAPH_FSE,   // exponent
    };
    // Separate the exponent from the sign+fraction bits.
    // Pass the exponent to FSE, and bitpack the sign+fraction bits as-is.
    return ZL_Compressor_registerStaticGraph_fromNode(
            compressor,
            ZL_NODE_FLOAT32_DECONSTRUCT,
            successors.data(),
            successors.size());
}
// --8<-- [end:customizing-compressor3]

// --8<-- [start:customizing-compressor4]
ZL_GraphID buildCompressorBruteForce(ZL_Compressor* compressor)
{
    std::array<ZL_GraphID, 5> successors = {
        buildCompressorStandard(compressor),
        buildCompressorSorted(compressor),
        buildCompressorInt(compressor),
        buildCompressorBFloat16(compressor),
        buildCompressorFloat32(compressor),
    };
    auto selector = [](const ZL_Selector* selector,
                       const ZL_Input* input,
                       const ZL_GraphID* successors,
                       size_t numSuccessors) noexcept -> ZL_GraphID {
        size_t bestSize      = ZL_Input_contentSize(input);
        ZL_GraphID bestGraph = ZL_GRAPH_STORE;
        // Try each successor, if the compressed size is smaller, save as the
        // best graph
        for (size_t i = 0; i < numSuccessors; ++i) {
            auto report = ZL_Selector_tryGraph(selector, input, successors[i])
                                  .finalCompressedSize;
            if (!ZL_isError(report) && ZL_validResult(report) < bestSize) {
                bestSize  = ZL_validResult(report);
                bestGraph = successors[i];
            }
        }
        // Return the best graph, or store if nothing is better
        return bestGraph;
    };
    ZL_SelectorDesc desc = {
        // The function that selects which successor to pass the input to
        .selector_f = selector,
        // Type of the input data
        .inStreamType = ZL_Type_numeric,
        // Successors to select from
        .customGraphs   = successors.data(),
        .nbCustomGraphs = successors.size(),
        // Auto-detect the min & max format version that the selectors support
        // based on the `customGraphs`.
        .name = "brute_force_selector",
    };
    return ZL_Compressor_registerSelectorGraph(compressor, &desc);
}
// --8<-- [end:customizing-compressor4]

// --8<-- [start:setup-compressor-build1]
// --8<-- [start:customizing-compressor5]
ZL_GraphID buildCompressor(
        ZL_Compressor* compressor,
        const std::string& compressorName)
{
    if (compressorName == "standard") {
        return buildCompressorStandard(compressor);
        // --8<-- [end:setup-compressor-build1]
    } else if (compressorName == "int") {
        return buildCompressorInt(compressor);
    } else if (compressorName == "bfloat16") {
        return buildCompressorBFloat16(compressor);
    } else if (compressorName == "float16") {
        return buildCompressorFloat16(compressor);
    } else if (compressorName == "float32") {
        return buildCompressorFloat32(compressor);
    } else if (compressorName == "brute_force") {
        return buildCompressorBruteForce(compressor);
        // --8<-- [start:setup-compressor-build2]
    } else {
        fprintf(stderr, "Unknown compressor: %s\n", compressorName.c_str());
        abort();
    }
}
// --8<-- [end:setup-compressor-build2]
// --8<-- [end:customizing-compressor5]

// --8<-- [start:compress]
// --8<-- [start:select-graph]
std::string compress(
        const std::string& data,
        size_t width,
        const std::string& compressorName)
{
    // Set up the compressor
    ZL_Compressor* compressor = ZL_Compressor_create();
    abortIf(compressor == nullptr);
    ZL_GraphID graph = buildCompressor(compressor, compressorName);
    abortIfError(
            compressor, ZL_Compressor_selectStartingGraphID(compressor, graph));
    // --8<-- [end:select-graph]

    ZL_CCtx* cctx = ZL_CCtx_create();
    abortIf(cctx == nullptr);
    // Use the compressor for this compression
    abortIfError(cctx, ZL_CCtx_refCompressor(cctx, compressor));

    // Wrap the input data as an array of native-endian numeric data
    abortIf(data.size() % width != 0, "Input not multiple of width");
    ZL_TypedRef* input =
            ZL_TypedRef_createNumeric(data.data(), width, data.size() / width);
    abortIf(input == nullptr);

    // Compress
    std::string compressed;
    compressed.resize(ZL_compressBound(data.size()), '\0');
    const ZL_Report r = ZL_CCtx_compressTypedRef(
            cctx, compressed.data(), compressed.size(), input);
    abortIfError(cctx, r);
    compressed.resize(ZL_validResult(r));

    // Cleanup
    ZL_TypedRef_free(input);
    ZL_CCtx_free(cctx);
    ZL_Compressor_free(compressor);

    return compressed;
}
// --8<-- [end:compress]

// --8<-- [start:decompress]
std::string decompress(const std::string& data, size_t width)
{
    // Find the size of the output buffer
    const size_t outputBytes =
            abortIfError(ZL_getDecompressedSize(data.data(), data.size()));
    abortIf(outputBytes % width != 0,
            "Output size must be a multiple of width");

    std::string decompressed;
    decompressed.resize(outputBytes);

    // Set the output type as numeric
    // This could also be derived from the compressed frame header
    ZL_TypedBuffer* output = ZL_TypedBuffer_createWrapNumeric(
            decompressed.data(), width, decompressed.size());
    abortIf(output == nullptr);

    // Decompress
    ZL_DCtx* dctx = ZL_DCtx_create();
    abortIf(dctx == nullptr);
    abortIfError(
            dctx,
            ZL_DCtx_decompressTBuffer(dctx, output, data.data(), data.size()));

    // Cleanup
    ZL_TypedBuffer_free(output);
    ZL_DCtx_free(dctx);

    return decompressed;
}
// --8<-- [end:decompress]
} // namespace

int main(int argc, char** argv)
{
    if (argc != 4) {
        fprintf(stderr,
                "Usage: %s [standard|sorted|int|bfloat16|float16|float32|brute_force] <width> <input>\n",
                argv[0]);
        fprintf(stderr,
                "\tCompresses native-endian numeric data of the given width using the specified compressor");
        return 1;
    }

    const std::string compressorName = argv[1];
    const size_t width               = atoi(argv[2]);
    const std::string inputFilename  = argv[3];

    abortIf(width != 1 && width != 2 && width != 4 && width != 8,
            "Width must be 1, 2, 4, or 8");

    const std::string data = readFile(inputFilename.c_str());

    auto compressed   = compress(data, width, compressorName);
    auto decompressed = decompress(compressed, width);

    abortIf(data != decompressed, "Decompressed data does not match original");

    fprintf(stderr,
            "Compressed %zu bytes to %zu bytes\n",
            data.size(),
            compressed.size());

    return 0;
}
