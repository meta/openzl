// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/graph_builders.h"

#include <algorithm>
#include <utility>

#include "openzl/codecs/zl_split.h"
#include "openzl/compress/private_nodes.h"

namespace openzl::gpu::testkit {

using openzl::Compressor;
using openzl::GraphID;

namespace {

constexpr size_t kSplitBlockBytes = 32 * 1024;

// Cuts the input into blocks of kSplitBlockBytes and adds the rest of the
// input to the last block. A last size of 0 means the rest of the input.
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

} // namespace

GraphID store()
{
    return ZL_GRAPH_STORE;
}

GraphID node(Compressor& c, openzl::NodeID n, std::vector<GraphID> successors)
{
    return c.buildStaticGraph(n, successors);
}

GraphID numeric32(Compressor& c, GraphID next)
{
    return node(c, ZL_NODE_CONVERT_SERIAL_TO_NUM_LE32, { next });
}

GraphID numToSerial(Compressor& c, GraphID next)
{
    return node(c, ZL_NODE_CONVERT_NUM_TO_SERIAL_LE, { next });
}

GraphID storeNum(Compressor& c)
{
    return numToSerial(c, store());
}

GraphID storeStruct(Compressor& c)
{
    return node(c, ZL_NODE_CONVERT_STRUCT_TO_SERIAL, { store() });
}

GraphID bitpackInt(Compressor& c)
{
    return node(c, ZL_NODE_BITPACK_INT, { store() });
}

GraphID bitpackSerial(Compressor& c)
{
    return node(c, ZL_NODE_BITPACK_SERIAL, { store() });
}

GraphID zstd(Compressor& c)
{
    return node(c, ZL_NODE_ZSTD, { store() });
}

GraphID constant(Compressor& c)
{
    return node(c, ZL_NODE_CONSTANT_SERIAL, { store() });
}

GraphID flatpack(Compressor& c)
{
    return node(c, ZL_NODE_FLATPACK, { store(), store() });
}

GraphID huffman(Compressor& c)
{
    return node(
            c,
            openzl::NodeID{ ZL_PrivateStandardNodeID_huffman_v2 },
            { bitpackInt(c), store() });
}

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

GraphID huffmanStruct(Compressor& c)
{
    return node(
            c,
            openzl::NodeID{ ZL_PrivateStandardNodeID_huffman_struct_v2 },
            { numToSerial(c, fse(c)), store() });
}

GraphID delta(Compressor& c, GraphID next)
{
    return node(c, ZL_NODE_DELTA_INT, { next });
}

GraphID deltaBytes(Compressor& c, GraphID next)
{
    return node(
            c,
            ZL_NODE_CONVERT_SERIAL_TO_NUM8,
            { delta(c, numToSerial(c, next)) });
}

GraphID tokenize(Compressor& c, GraphID alphabet, GraphID indices)
{
    openzl::LocalParams params;
    params.addIntParam(ZL_TOKENIZE_SORT_PID, 1);
    const openzl::NodeID sorted = c.parameterizeNode(
            ZL_NODE_TOKENIZE_NUMERIC, { .localParams = std::move(params) });
    return node(c, sorted, { alphabet, indices });
}

GraphID bitpackTokens(Compressor& c)
{
    return node(c, ZL_NODE_CONVERT_STRUCT_TO_NUM_LE, { bitpackInt(c) });
}

GraphID tokenizeTokens(Compressor& c, GraphID indices)
{
    return node(
            c,
            ZL_NODE_TOKENIZE_STRUCT,
            { bitpackTokens(c), numToSerial(c, indices) });
}

GraphID quantOffsets(Compressor& c, GraphID codes)
{
    return node(
            c, ZL_NODE_QUANTIZE_OFFSETS, { numToSerial(c, codes), store() });
}

GraphID quantLengths(Compressor& c, GraphID codes)
{
    return node(
            c, ZL_NODE_QUANTIZE_LENGTHS, { numToSerial(c, codes), store() });
}

GraphID transposeSplit(Compressor& c, std::vector<GraphID> bytes)
{
    return c.parameterizeGraph(
            ZL_GRAPH_SPLIT_STRUCT,
            { .customGraphs = std::move(bytes),
              .customNodes =
                      std::vector<openzl::NodeID>{ ZL_NODE_TRANSPOSE_SPLIT } });
}

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

} // namespace openzl::gpu::testkit
