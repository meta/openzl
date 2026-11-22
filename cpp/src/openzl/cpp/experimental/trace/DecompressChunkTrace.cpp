// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "openzl/cpp/experimental/trace/DecompressChunkTrace.hpp"

#include "openzl/common/a1cbor_helpers.h"
#include "openzl/cpp/Exception.hpp"
#include "openzl/cpp/experimental/trace/CborHelpers.hpp"
#include "openzl/decompress/dictx.h"
#include "openzl/zl_data.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_errors.h"

#include <variant>

namespace openzl::visualizer {

DecompressChunkTrace DecompressChunkTrace::makeSegmenterChunk(
        size_t chunkId,
        bool showStreamPreview)
{
    auto ret = DecompressChunkTrace(chunkId, showStreamPreview);
    Codec newCodec{ .name  = "segmenter", // TODO(segm): expose segmenter name
                    .cType = false,
                    .cID   = 0, // eh?
                    .cHeaderSize = 0,
                    .chunkId     = chunkId };
    newCodec.codecNum = 0;
    ret.codecInfo_.push_back(std::move(newCodec));
    return ret;
}

void DecompressChunkTrace::finalizeTrace(ZL_Report result)
{
    if (ZL_isError(result)) {
        ChunkTraceCore::finalizeUnsourcedStreams(
                "zl.#in_progress",
                streamInfo_,
                codecInfo_,
                currCodecNum_,
                chunkId_);
    } else {
        ChunkTraceCore::finalizeUnsourcedStreams(
                "zl.#regen", streamInfo_, codecInfo_, currCodecNum_, chunkId_);
    }

    // Fill cSize and share for all streams
    for (auto& [streamID, stream] : streamInfo_) {
        ChunkTraceCore::fillCSize(
                streamID, streamInfo_, codecInfo_, totalCompressedSize_);
    }
}

void DecompressChunkTrace::resolveErrorStrings(const ZL_DCtx* dctx)
{
    for (auto& codec : codecInfo_) {
        if (ZL_isError(codec.cFailure)) {
            const char* str =
                    ZL_DCtx_getErrorContextString(dctx, codec.cFailure);
            codec.cFailureString = str ? str : "";
        }
    }
}

ZL_Report DecompressChunkTrace::serializeToCBOR(
        A1C_Arena* a1c_arena,
        A1C_ArrayBuilder* chunkArrayBuilder,
        ZL_OperationContext* opCtx)
{
    std::vector<Graph> noGraphs;
    return ChunkTraceCore::serializeChunkDataToCBOR(
            a1c_arena,
            chunkArrayBuilder,
            chunkId_,
            streamInfo_,
            codecInfo_,
            noGraphs,
            opCtx);
}

void DecompressChunkTrace::on_codecDecode_start(
        ZL_Decoder* dictx,
        const ZL_Data* const* inStreams,
        size_t nbInStreams)
{
    // Discover new streams and create sink placeholders before pushing the
    // decode codec, so that currCodecNum_ remains valid for the decode codec.
    for (size_t i = 0; i < nbInStreams; ++i) {
        StreamID streamID = ZL_Data_id(inStreams[i]);
        if (streamInfo_.find(streamID) == streamInfo_.end()) {
            ZL_Type type    = ZL_Data_type(inStreams[i]);
            size_t eltWidth = ZL_Data_eltWidth(inStreams[i]);
            size_t numElts  = ZL_Data_numElts(inStreams[i]);

            StreamPreview preview = showStreamPreview_
                    ? ChunkTraceCore::getStreamPreview(
                              ZL_Data_rPtr(inStreams[i]),
                              type,
                              eltWidth,
                              numElts,
                              ZL_Data_rStringLens(inStreams[i]))
                    : ChunkTraceCore::emptyPreview(type);

            streamInfo_[streamID] = Stream{
                .id            = streamID,
                .type          = type,
                .outputIdx     = i,
                .eltWidth      = eltWidth,
                .numElts       = numElts,
                .contentSize   = ZL_Data_contentSize(inStreams[i]),
                .chunkId       = chunkId_,
                .streamPreview = std::move(preview),
            };
            ChunkTraceCore::createSinkForStream(
                    "zl.store",
                    streamID,
                    streamInfo_[streamID],
                    codecInfo_,
                    currCodecNum_,
                    chunkId_);
            // NB: this does not account for the various non-codec header costs
            totalCompressedSize_ += streamInfo_[streamID].contentSize;
        }
    }

    // Extract transform info from ZL_Decoder
    const char* transformName = DT_getTransformName(dictx->dt);

    Codec newCodec{
        .name         = transformName ? transformName : "",
        .cType        = true,
        .cID          = dictx->dt->miGraphDesc.CTid,
        .cHeaderSize  = 0,
        .cLocalParams = {},
        .chunkId      = chunkId_,
    };
    newCodec.codecNum = currCodecNum_;
    codecInfo_.push_back(std::move(newCodec));

    for (size_t i = 0; i < nbInStreams; ++i) {
        codecInfo_[currCodecNum_].outEdges.push_back(ZL_Data_id(inStreams[i]));
        streamInfo_[ZL_Data_id(inStreams[i])].producerCodec = currCodecNum_;
    }
}

void DecompressChunkTrace::on_codecDecode_end(
        ZL_Decoder* dictx,
        const ZL_Data* const* outStreams,
        size_t nbOutStreams,
        ZL_Report result)
{
    if (ZL_isError(result)) {
        codecInfo_[currCodecNum_].cFailure = result;
    }

    // Record output streams with full metadata from ZL_Data objects
    for (size_t i = 0; i < nbOutStreams; ++i) {
        StreamID streamID{ dictx->regensID[i] };
        if (streamInfo_.find(streamID) == streamInfo_.end()) {
            ZL_Type type    = ZL_Data_type(outStreams[i]);
            size_t eltWidth = ZL_Data_eltWidth(outStreams[i]);
            size_t numElts  = ZL_Data_numElts(outStreams[i]);

            StreamPreview preview = showStreamPreview_
                    ? ChunkTraceCore::getStreamPreview(
                              ZL_Data_rPtr(outStreams[i]),
                              type,
                              eltWidth,
                              numElts,
                              ZL_Data_rStringLens(outStreams[i]))
                    : ChunkTraceCore::emptyPreview(type);

            streamInfo_[streamID] = Stream{
                .id            = streamID,
                .type          = type,
                .outputIdx     = i,
                .eltWidth      = eltWidth,
                .numElts       = numElts,
                .contentSize   = ZL_Data_contentSize(outStreams[i]),
                .chunkId       = chunkId_,
                .streamPreview = std::move(preview),
            };
        }
        codecInfo_[currCodecNum_].inEdges.push_back(streamID);
        streamInfo_[streamID].consumerCodec = currCodecNum_;
    }

    // Capture streamdump for each output stream
    for (size_t i = 0; i < nbOutStreams; ++i) {
        streamdump(outStreams[i]);
    }

    // Connect stream successors for cSize calculation
    for (const auto& inStreamID : codecInfo_[currCodecNum_].inEdges) {
        streamInfo_[inStreamID].successors = codecInfo_[currCodecNum_].outEdges;
    }

    ++currCodecNum_;
}

void DecompressChunkTrace::streamdump(const ZL_Data* data)
{
    auto content = std::string(
            (const char*)ZL_Data_rPtr(data), ZL_Data_contentSize(data));
    std::string strLens = "";
    if (ZL_Data_type(data) == ZL_Type_string) {
        auto ptr = ZL_Data_rStringLens(data);
        strLens  = std::string(
                (const char*)ptr, ZL_Data_numElts(data) * sizeof(ptr[0]));
    }
    streamdump_[ZL_Data_id(data).sid] = { content, strLens };
}

std::map<size_t, std::pair<std::string, std::string>>&&
DecompressChunkTrace::getStreamdump()
{
    return std::move(streamdump_);
}

void DecompressChunkTrace::on_ZL_Decoder_getCodecHeader(
        const ZL_Decoder* /* dictx */,
        const void* /* trh */,
        size_t trhSize)
{
    // The header read happens between codecDecode_start and codecDecode_end,
    // so currCodecNum_ points to the current codec being decoded.
    codecInfo_[currCodecNum_].cHeaderSize = trhSize;
}

} // namespace openzl::visualizer
