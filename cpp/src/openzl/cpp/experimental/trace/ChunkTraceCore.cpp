// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "openzl/cpp/experimental/trace/ChunkTraceCore.hpp"

#include "openzl/common/a1cbor_helpers.h"
#include "openzl/cpp/experimental/trace/CborHelpers.hpp"

#include <sstream>
#include <stdexcept>

namespace openzl::visualizer {

namespace {

// Preview limits: save only enough data for the frontend to display ~4 lines.
static constexpr size_t MAX_PREVIEW_BYTES         = 32;
static constexpr size_t MAX_PREVIEW_NUMERIC_ELTS  = 100;
static constexpr size_t MAX_PREVIEW_STRINGS_LINES = 4;

static int64_t readNumericData(size_t eltWidth, const uint8_t* bytes, size_t i)
{
    int64_t val = 0;
    switch (eltWidth) {
        case 1:
            val = reinterpret_cast<const int8_t*>(bytes)[i];
            break;
        case 2:
            val = reinterpret_cast<const int16_t*>(bytes)[i];
            break;
        case 4:
            val = reinterpret_cast<const int32_t*>(bytes)[i];
            break;
        case 8:
            val = reinterpret_cast<const int64_t*>(bytes)[i];
            break;
        default:
            throw std::runtime_error("Unexpected numeric eltWidth!");
    }
    return val;
}
std::vector<int64_t>
getNumericData(const void* data, size_t eltWidth, size_t numElts)
{
    if (data == nullptr || numElts == 0) {
        return {};
    }
    const auto* bytes        = reinterpret_cast<const uint8_t*>(data);
    const size_t previewElts = std::min(numElts, MAX_PREVIEW_NUMERIC_ELTS);
    std::vector<int64_t> numericData;
    numericData.reserve(previewElts);

    for (size_t i = 0; i < previewElts; ++i) {
        numericData.push_back(readNumericData(eltWidth, bytes, i));
    }

    return numericData;
}

std::vector<std::string>
getStringData(const void* data, size_t numStrings, const uint32_t* stringLens)
{
    if (data == nullptr || stringLens == nullptr || numStrings == 0) {
        return {};
    }

    std::vector<std::string> strings;
    strings.reserve(numStrings);

    const auto* bytes = reinterpret_cast<const char*>(data);
    size_t offset     = 0;

    const size_t previewStrings =
            std::min(numStrings, MAX_PREVIEW_STRINGS_LINES);
    for (size_t i = 0; i < previewStrings; ++i) {
        uint32_t len          = stringLens[i];
        std::string rawStr    = std::string(bytes + offset, len);
        std::string parsedStr = "";
        if (rawStr.find('\n') != std::string::npos
            || rawStr.find('\t') != std::string::npos
            || rawStr.find('\r') != std::string::npos) {
            for (char c : rawStr) {
                switch (c) {
                    case '\n':
                        parsedStr += "\\n";
                        break;
                    case '\t':
                        parsedStr += "\\t";
                        break;
                    case '\r':
                        parsedStr += "\\r";
                        break;
                    default:
                        parsedStr += c;
                        break;
                }
            }
        } else {
            parsedStr = std::move(rawStr);
        }

        strings.push_back(std::move(parsedStr));
        offset += len;
    }

    return strings;
}

std::vector<uint8_t> getSerialData(const void* data, size_t numBytes)
{
    if (data == nullptr || numBytes == 0) {
        return {};
    }

    const size_t previewBytes = std::min(numBytes, MAX_PREVIEW_BYTES);
    std::vector<uint8_t> bytes;
    const auto* rawBytes = reinterpret_cast<const uint8_t*>(data);
    bytes.assign(rawBytes, rawBytes + previewBytes);

    return bytes;
}

} // namespace

void ChunkTraceCore::createSourceForStream(
        const char* sourceCodecName,
        const StreamID& streamID,
        Stream& stream,
        std::vector<Codec>& codecInfo,
        size_t& currCodecNum,
        size_t chunkId)
{
    Codec source = {
        .name         = sourceCodecName,
        .cType        = true, // standard
        .cID          = 0,
        .cHeaderSize  = 0,
        .cLocalParams = {},
        .chunkId      = chunkId,
    };
    source.codecNum = currCodecNum;
    codecInfo.push_back(std::move(source));
    codecInfo[currCodecNum].outEdges.push_back(streamID);
    stream.producerCodec = currCodecNum;
    ++currCodecNum;
}

void ChunkTraceCore::createSinkForStream(
        const char* sinkCodecName,
        const StreamID& streamID,
        Stream& stream,
        std::vector<Codec>& codecInfo,
        size_t& currCodecNum,
        size_t chunkId)
{
    Codec sink = {
        .name         = sinkCodecName,
        .cType        = true, // standard
        .cID          = 0,
        .cHeaderSize  = 0,
        .cLocalParams = {},
        .chunkId      = chunkId,
    };
    sink.codecNum = currCodecNum;
    codecInfo.push_back(std::move(sink));
    codecInfo[currCodecNum].inEdges.push_back(streamID);
    stream.consumerCodec = currCodecNum;
    ++currCodecNum;
}

void ChunkTraceCore::finalizeUnsourcedStreams(
        const char* sourceCodecName,
        std::map<ZL_DataID, Stream, ZL_DataIDCustomComparator>& streamInfo,
        std::vector<Codec>& codecInfo,
        size_t& currCodecNum,
        size_t chunkId)
{
    for (auto& [streamID, stream] : streamInfo) {
        if (!stream.producerCodec.has_value()) {
            createSourceForStream(
                    sourceCodecName,
                    streamID,
                    stream,
                    codecInfo,
                    currCodecNum,
                    chunkId);
        }
    }
}

void ChunkTraceCore::finalizeUnconsumedStreams(
        const char* terminalCodecName,
        std::map<ZL_DataID, Stream, ZL_DataIDCustomComparator>& streamInfo,
        std::vector<Codec>& codecInfo,
        size_t& currCodecNum,
        size_t chunkId)
{
    for (auto& [streamID, stream] : streamInfo) {
        if (!stream.consumerCodec.has_value()) {
            createSinkForStream(
                    terminalCodecName,
                    streamID,
                    stream,
                    codecInfo,
                    currCodecNum,
                    chunkId);
        }
    }
}

StreamPreview ChunkTraceCore::emptyPreview(ZL_Type type)
{
    switch (type) {
        case ZL_Type_string:
            return std::vector<std::string>{};
        case ZL_Type_numeric:
            return std::vector<int64_t>{};
        case ZL_Type_serial:
        case ZL_Type_struct:
        default:
            return std::vector<uint8_t>{};
    }
}

StreamPreview ChunkTraceCore::getStreamPreview(
        const void* data,
        ZL_Type type,
        size_t eltWidth,
        size_t numElts,
        const uint32_t* stringLens)
{
    switch (type) {
        case ZL_Type_numeric:
            return getNumericData(data, eltWidth, numElts);
        case ZL_Type_string:
            return getStringData(data, numElts, stringLens);
        case ZL_Type_struct:
        case ZL_Type_serial:
            return getSerialData(data, numElts);
        default:
            throw std::runtime_error("Unsupported stream preview type!");
    }
}

size_t ChunkTraceCore::fillCSize(
        const StreamID& streamID,
        std::map<ZL_DataID, Stream, ZL_DataIDCustomComparator>& streamInfo,
        const std::vector<Codec>& codecInfo,
        size_t totalSize)
{
    Stream& stream = streamInfo.at(streamID);

    // Already computed
    if (stream.cSize != 0) {
        return stream.cSize;
    }

    // Base case: stream has no successors
    if (stream.successors.empty()) {
        stream.cSize = stream.contentSize;
        stream.share = totalSize > 0 ? static_cast<double>(stream.cSize)
                        / static_cast<double>(totalSize) * 100
                                     : 0;
        return stream.cSize;
    }

    // Start with the header size from the consumer codec
    if (stream.consumerCodec.has_value()) {
        stream.cSize = codecInfo[stream.consumerCodec.value()].cHeaderSize;
    }

    // Recursively sum the cSize of successor streams
    for (const auto& successor : stream.successors) {
        stream.cSize += fillCSize(successor, streamInfo, codecInfo, totalSize);
    }

    // If the consumer codec has multiple inputs, assume each input
    // provides equal contribution
    if (stream.consumerCodec.has_value()) {
        stream.cSize /= codecInfo[stream.consumerCodec.value()].inEdges.size();
    }

    stream.share = totalSize > 0 ? static_cast<double>(stream.cSize)
                    / static_cast<double>(totalSize) * 100
                                 : 0;
    return stream.cSize;
}

ZL_Report ChunkTraceCore::serializeChunkDataToCBOR(
        A1C_Arena* a1c_arena,
        A1C_ArrayBuilder* chunkArrayBuilder,
        size_t chunkId,
        std::map<ZL_DataID, Stream, ZL_DataIDCustomComparator>& streamInfo,
        std::vector<Codec>& codecInfo,
        std::vector<Graph>& graphInfo,
        ZL_OperationContext* opCtx)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(opCtx);
    A1C_ARRAY_TRY_ADD(chunkItem, *chunkArrayBuilder);
    A1C_MapBuilder chunkBuilder = A1C_Item_map_builder(chunkItem, 4, a1c_arena);
    ZL_ERR_IF_NULL(chunkBuilder.map, allocation);

    ZL_ERR_IF_ERR(addIntValue(chunkBuilder, "chunkId", chunkId, opCtx));

    // Streams
    A1C_MAP_TRY_ADD(streamsPair, chunkBuilder);
    A1C_Item_string_refCStr(&streamsPair->key, "streams");
    A1C_ArrayBuilder streamsBuilder = A1C_Item_array_builder(
            &streamsPair->val, streamInfo.size(), a1c_arena);
    ZL_ERR_IF_NULL(streamsBuilder.array, allocation);
    // Codec edges refer to streams by their position in this array.
    std::map<ZL_DataID, ZL_IDType, ZL_DataIDCustomComparator> streamIdx;
    for (auto& stream : streamInfo) {
        A1C_ARRAY_TRY_ADD(a1c_stream, streamsBuilder);
        ZL_ERR_IF_ERR(
                stream.second.serializeStream(a1c_arena, a1c_stream, opCtx));
        streamIdx.emplace(stream.first, (ZL_IDType)streamIdx.size());
    }

    // Codecs
    A1C_MAP_TRY_ADD(codecsPair, chunkBuilder);
    A1C_Item_string_refCStr(&codecsPair->key, "codecs");
    A1C_ArrayBuilder codecsBuilder = A1C_Item_array_builder(
            &codecsPair->val, codecInfo.size(), a1c_arena);
    ZL_ERR_IF_NULL(codecsBuilder.array, allocation);
    for (size_t codecNum = 0; codecNum < codecInfo.size(); ++codecNum) {
        A1C_ARRAY_TRY_ADD(a1c_codec, codecsBuilder);
        Codec& codec  = codecInfo[codecNum];
        auto inEdges  = codec.inEdges;
        auto outEdges = codec.outEdges;
        for (auto* edges : { &codec.inEdges, &codec.outEdges }) {
            for (auto& edge : *edges) {
                edge.sid = streamIdx.at(edge);
            }
        }
        ZL_Report report = codec.serializeCodec(a1c_arena, a1c_codec, opCtx);
        codec.inEdges    = std::move(inEdges);
        codec.outEdges   = std::move(outEdges);
        ZL_ERR_IF_ERR(report);
    }

    // Graphs (empty array for decompress)
    A1C_MAP_TRY_ADD(graphsPair, chunkBuilder);
    A1C_Item_string_refCStr(&graphsPair->key, "graphs");
    A1C_ArrayBuilder graphsBuilder = A1C_Item_array_builder(
            &graphsPair->val, graphInfo.size(), a1c_arena);
    ZL_ERR_IF_NULL(graphsBuilder.array, allocation);
    for (auto& graph : graphInfo) {
        A1C_ARRAY_TRY_ADD(a1c_graph, graphsBuilder);
        ZL_ERR_IF_ERR(graph.serializeGraph(a1c_arena, a1c_graph, opCtx));
    }

    return ZL_returnSuccess();
}

ZL_Report ChunkTraceCore::writeSerializedStreamdump(
        std::vector<uint8_t>& buffer,
        std::string& outTrace)
{
    outTrace = std::string(buffer.begin(), buffer.end());
    return ZL_returnSuccess();
}

} // namespace openzl::visualizer
