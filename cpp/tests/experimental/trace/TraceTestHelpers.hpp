// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "openzl/cpp/poly/StringView.hpp"
#include "openzl/shared/a1cbor.h"

#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

namespace openzl::test {

// Simple parsed representations of trace CBOR data for test assertions.

struct ParsedCodec {
    size_t chunkId{};
    std::string name;
    bool cType{};
    int64_t cID{};
    int64_t cHeaderSize{};
    std::string cFailureString; // empty if not present
    std::vector<int64_t> inputStreams;
    std::vector<int64_t> outputStreams;
};

struct ParsedGraph {
    size_t chunkId{};
    int64_t gType{};
    std::string gName;
    std::string gFailureString; // empty if not present
    std::vector<int64_t> codecIDs;
};

struct ParsedChunk {
    size_t chunkId{};
    size_t numStreams{};
    std::vector<ParsedCodec> codecs;
    std::vector<ParsedGraph> graphs;
};

struct ParsedTrace {
    int64_t libraryVersion{};
    int64_t frameVersion{};
    int64_t traceVersion{};
    int64_t operationType{};
    std::vector<ParsedChunk> chunks;
};

namespace detail {

// Arena allocator backed by a vector of unique_ptrs, matching the pattern
// used in test_a1cbor.cpp.
using Ptrs = std::vector<std::unique_ptr<uint8_t[]>>;

inline void* testCalloc(void* opaque, size_t bytes) noexcept
{
    if (bytes == 0) {
        return nullptr;
    }
    // Use nothrow new to avoid throwing in a noexcept function.
    // The () value-initializes (zeros) the array, matching calloc semantics.
    auto* raw = new (std::nothrow) uint8_t[bytes]();
    if (raw == nullptr) {
        return nullptr;
    }
    auto* ptrs = static_cast<Ptrs*>(opaque);
    ptrs->push_back(std::unique_ptr<uint8_t[]>(raw));
    return raw;
}

inline std::string extractString(const A1C_Item* item)
{
    if (item == nullptr || item->type != A1C_ItemType_string) {
        return "";
    }
    return std::string(item->string.data, item->string.size);
}

inline int64_t extractInt(const A1C_Item* item, int64_t defaultVal = 0)
{
    if (item == nullptr || item->type != A1C_ItemType_int64) {
        return defaultVal;
    }
    return item->int64;
}

inline bool extractBool(const A1C_Item* item, bool defaultVal = false)
{
    if (item == nullptr || item->type != A1C_ItemType_boolean) {
        return defaultVal;
    }
    return item->boolean;
}

inline std::vector<int64_t> extractIntArray(const A1C_Item* item)
{
    std::vector<int64_t> result;
    if (item == nullptr || item->type != A1C_ItemType_array) {
        return result;
    }
    for (size_t i = 0; i < item->array.size; ++i) {
        A1C_Item* elem = A1C_Array_get(&item->array, i);
        if (elem != nullptr && elem->type == A1C_ItemType_int64) {
            result.push_back(elem->int64);
        }
    }
    return result;
}

inline ParsedCodec parseCodec(const A1C_Item* item)
{
    ParsedCodec codec;
    if (item == nullptr || item->type != A1C_ItemType_map) {
        return codec;
    }
    const A1C_Map* m = &item->map;
    codec.chunkId =
            static_cast<size_t>(extractInt(A1C_Map_get_cstr(m, "chunkId")));
    codec.name           = extractString(A1C_Map_get_cstr(m, "name"));
    codec.cType          = extractBool(A1C_Map_get_cstr(m, "cType"));
    codec.cID            = extractInt(A1C_Map_get_cstr(m, "cID"));
    codec.cHeaderSize    = extractInt(A1C_Map_get_cstr(m, "cHeaderSize"));
    codec.cFailureString = extractString(A1C_Map_get_cstr(m, "cFailureString"));
    codec.inputStreams   = extractIntArray(A1C_Map_get_cstr(m, "inputStreams"));
    codec.outputStreams = extractIntArray(A1C_Map_get_cstr(m, "outputStreams"));
    return codec;
}

inline ParsedGraph parseGraph(const A1C_Item* item)
{
    ParsedGraph graph;
    if (item == nullptr || item->type != A1C_ItemType_map) {
        return graph;
    }
    const A1C_Map* m = &item->map;
    graph.chunkId =
            static_cast<size_t>(extractInt(A1C_Map_get_cstr(m, "chunkId")));
    graph.gType          = extractInt(A1C_Map_get_cstr(m, "gType"));
    graph.gName          = extractString(A1C_Map_get_cstr(m, "gName"));
    graph.gFailureString = extractString(A1C_Map_get_cstr(m, "gFailureString"));
    graph.codecIDs       = extractIntArray(A1C_Map_get_cstr(m, "codecIDs"));
    return graph;
}

inline ParsedChunk parseChunk(const A1C_Item* item)
{
    ParsedChunk chunk;
    if (item == nullptr || item->type != A1C_ItemType_map) {
        return chunk;
    }
    const A1C_Map* m = &item->map;
    chunk.chunkId =
            static_cast<size_t>(extractInt(A1C_Map_get_cstr(m, "chunkId")));

    const A1C_Item* streamsItem = A1C_Map_get_cstr(m, "streams");
    if (streamsItem != nullptr && streamsItem->type == A1C_ItemType_array) {
        chunk.numStreams = streamsItem->array.size;
    }

    const A1C_Item* codecsItem = A1C_Map_get_cstr(m, "codecs");
    if (codecsItem != nullptr && codecsItem->type == A1C_ItemType_array) {
        for (size_t i = 0; i < codecsItem->array.size; ++i) {
            chunk.codecs.push_back(
                    parseCodec(A1C_Array_get(&codecsItem->array, i)));
        }
    }

    const A1C_Item* graphsItem = A1C_Map_get_cstr(m, "graphs");
    if (graphsItem != nullptr && graphsItem->type == A1C_ItemType_array) {
        for (size_t i = 0; i < graphsItem->array.size; ++i) {
            chunk.graphs.push_back(
                    parseGraph(A1C_Array_get(&graphsItem->array, i)));
        }
    }

    return chunk;
}

} // namespace detail

/**
 * Decodes a CBOR trace (as returned by CCtx::getLatestTrace().first) into a
 * ParsedTrace struct for easy assertion in tests.
 *
 * Returns std::nullopt if decoding fails.
 */
inline std::optional<ParsedTrace> parseTrace(poly::string_view traceData)
{
    if (traceData.empty()) {
        return std::nullopt;
    }

    detail::Ptrs ptrs;
    A1C_Arena arena;
    arena.calloc = detail::testCalloc;
    arena.opaque = &ptrs;

    A1C_DecoderConfig config{};
    config.referenceSource = true;

    A1C_Decoder decoder;
    A1C_Decoder_init(&decoder, arena, config);

    const A1C_Item* root = A1C_Decoder_decode(
            &decoder,
            reinterpret_cast<const uint8_t*>(traceData.data()),
            traceData.size());
    if (root == nullptr || root->type != A1C_ItemType_map) {
        return std::nullopt;
    }

    ParsedTrace trace;
    const A1C_Map* m = &root->map;
    trace.libraryVersion =
            detail::extractInt(A1C_Map_get_cstr(m, "libraryVersion"));
    trace.frameVersion =
            detail::extractInt(A1C_Map_get_cstr(m, "frameVersion"));
    trace.traceVersion =
            detail::extractInt(A1C_Map_get_cstr(m, "traceVersion"));
    trace.operationType =
            detail::extractInt(A1C_Map_get_cstr(m, "operationType"));

    const A1C_Item* chunksItem = A1C_Map_get_cstr(m, "chunks");
    if (chunksItem != nullptr && chunksItem->type == A1C_ItemType_array) {
        for (size_t i = 0; i < chunksItem->array.size; ++i) {
            trace.chunks.push_back(
                    detail::parseChunk(A1C_Array_get(&chunksItem->array, i)));
        }
    }

    return trace;
}

} // namespace openzl::test
