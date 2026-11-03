// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <unordered_set>

#include "openzl/common/logging.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/poly/StringView.hpp"

#include "custom_parsers/csv/csv_profile.h"
#include "custom_parsers/dependency_registration.h"
#include "custom_parsers/gguf/gguf_parser.h"
#include "custom_parsers/onnx/onnx_parser.h"
#include "custom_parsers/parquet/parquet_graph.h"
#include "custom_parsers/safetensors/safetensors_parser.h"
#include "custom_parsers/shared_components/tensor_graphs.h"

namespace openzl::custom_parsers {

void processDependencies(Compressor& compressor, poly::string_view serialized)
{
    std::unordered_set<std::string> attemptedGraphDeps;

    while (true) {
        const auto deps   = compressor.getUnmetDependencies(serialized);
        bool madeProgress = false;

        for (const auto& graphName : deps.graphNames) {
            if (!attemptedGraphDeps.insert(graphName).second) {
                continue;
            }
            // An earlier registration in this pass may have met it, e.g. a
            // parser registers its segmenter too.
            if (compressor.getGraph(graphName)) {
                continue;
            }

            if (graphName == "Parquet Parser") {
                auto parquetResult = ZL_Parquet_registerGraph(
                        compressor.get(), ZL_GRAPH_STORE);
                if (parquetResult == ZL_GRAPH_ILLEGAL) {
                    throw std::runtime_error("Failed to create parquet graph");
                }
                madeProgress = true;
            }

            else if (graphName == "CSV Parser") {
                auto csvResult =
                        ZL_createGraph_genericCSVCompressor(compressor.get());
                if (csvResult == ZL_GRAPH_ILLEGAL) {
                    throw std::runtime_error("Failed to create CSV graph");
                }
                madeProgress = true;
            } else if (
                    graphName == "Safetensors Parser"
                    || graphName == "Safetensors Segmenter") {
                const auto result = ZL_Safetensors_registerGraph(
                        compressor.get(), ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
                if (ZL_RES_isError(result)) {
                    throw std::runtime_error(
                            std::string("Failed to create safetensors graph: ")
                            + ZL_Compressor_getErrorContextString_fromError(
                                    compressor.get(), ZL_RES_error(result)));
                }
                madeProgress = true;
            } else if (
                    graphName == "GGUF Parser"
                    || graphName == "GGUF Segmenter") {
                const auto result = ZL_GGUF_registerGraph(
                        compressor.get(), ZL_GGUF_DEFAULT_CHUNK_SIZE);
                if (ZL_RES_isError(result)) {
                    throw std::runtime_error(
                            std::string("Failed to create GGUF graph: ")
                            + ZL_Compressor_getErrorContextString_fromError(
                                    compressor.get(), ZL_RES_error(result)));
                }
                madeProgress = true;
            } else if (
                    graphName == "ONNX Parser"
                    || graphName == "ONNX Segmenter") {
                const auto result = ZL_ONNX_registerGraph(
                        compressor.get(), ZL_ONNX_DEFAULT_CHUNK_SIZE);
                if (ZL_RES_isError(result)) {
                    throw std::runtime_error(
                            std::string("Failed to create ONNX graph: ")
                            + ZL_Compressor_getErrorContextString_fromError(
                                    compressor.get(), ZL_RES_error(result)));
                }
                madeProgress = true;
            } else if (graphName == "Tensor Scale16") {
                const auto result =
                        ZL_TensorGraphs_registerScale16Graph(compressor.get());
                if (ZL_RES_isError(result)) {
                    throw std::runtime_error(
                            std::string("Failed to create the scale graph: ")
                            + ZL_Compressor_getErrorContextString_fromError(
                                    compressor.get(), ZL_RES_error(result)));
                }
                madeProgress = true;
            } else if (graphName == "Tensor Upcast32") {
                const auto result =
                        ZL_TensorGraphs_registerUpcast32Graph(compressor.get());
                if (ZL_RES_isError(result)) {
                    throw std::runtime_error(
                            std::string("Failed to create the upcast graph: ")
                            + ZL_Compressor_getErrorContextString_fromError(
                                    compressor.get(), ZL_RES_error(result)));
                }
                madeProgress = true;
            } else if (graphName == "Tensor Rows32") {
                const auto result =
                        ZL_TensorGraphs_registerRows32Graph(compressor.get());
                if (ZL_RES_isError(result)) {
                    throw std::runtime_error(
                            std::string("Failed to create the rows graph: ")
                            + ZL_Compressor_getErrorContextString_fromError(
                                    compressor.get(), ZL_RES_error(result)));
                }
                madeProgress = true;
            } else {
                ZL_LOG(WARN,
                       "processDependencies: unresolved graph dependency '%s'",
                       graphName.c_str());
            }
        }

        if (!madeProgress) {
            if (deps.nodeNames.size() > 0) {
                // TODO register any non-standard nodes that may appear in a
                // compressor
            }
            return;
        }
    }
}

std::unique_ptr<Compressor> createCompressorFromSerialized(
        poly::string_view serialized,
        poly::string_view fatBundle)
{
    auto compressor = std::make_unique<Compressor>();
    processDependencies(*compressor, serialized);
    compressor->deserialize(serialized, fatBundle);
    return compressor;
}

} // namespace openzl::custom_parsers
