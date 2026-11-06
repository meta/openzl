// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "openzl/common/logging.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/Exception.hpp"
#include "openzl/cpp/poly/StringView.hpp"

#include "custom_parsers/csv/csv_profile.h"
#include "custom_parsers/dependency_registration.h"
#include "custom_parsers/gguf/gguf_parser.h"
#include "custom_parsers/numpy/numpy_parser.h"
#include "custom_parsers/onnx/onnx_parser.h"
#include "custom_parsers/parquet/parquet_graph.h"
#include "custom_parsers/safetensors/safetensors_parser.h"
#include "custom_parsers/shared_components/tensor_graphs.h"

namespace openzl::custom_parsers {
namespace {

struct Registration {
    /// The graphs it registers: a parser registers its segmenter too.
    std::vector<std::string_view> graphNames;
    std::string_view label;
    ZL_RESULT_OF(ZL_GraphID) (*registerGraph)(ZL_Compressor*);
};

/// For the registration functions that return ZL_GRAPH_ILLEGAL on failure.
ZL_RESULT_OF(ZL_GraphID) checkGraph(ZL_GraphID graph)
{
    ZL_RESULT_DECLARE_SCOPE(ZL_GraphID, nullptr);
    if (!ZL_GraphID_isValid(graph)) {
        return ZL_RESULT_MAKE_ERROR(
                ZL_GraphID, graph_invalid, "Registration failed");
    }
    return ZL_RESULT_WRAP_VALUE(ZL_GraphID, graph);
}

const Registration* findRegistration(std::string_view graphName)
{
    static const std::vector<Registration> kRegistrations = {
        { { "Parquet Parser" },
          "Parquet",
          [](ZL_Compressor* c) {
              return checkGraph(ZL_Parquet_registerGraph(c, ZL_GRAPH_STORE));
          } },
        { { "CSV Parser" },
          "CSV",
          [](ZL_Compressor* c) {
              return checkGraph(ZL_createGraph_genericCSVCompressor(c));
          } },
        { { "Safetensors Parser", "Safetensors Segmenter" },
          "safetensors",
          [](ZL_Compressor* c) {
              return ZL_Safetensors_registerGraph(
                      c, ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
          } },
        { { "GGUF Parser", "GGUF Segmenter" },
          "GGUF",
          [](ZL_Compressor* c) {
              return ZL_GGUF_registerGraph(c, ZL_GGUF_DEFAULT_CHUNK_SIZE);
          } },
        { { "ONNX Parser", "ONNX Segmenter" },
          "ONNX",
          [](ZL_Compressor* c) {
              return ZL_ONNX_registerGraph(c, ZL_ONNX_DEFAULT_CHUNK_SIZE);
          } },
        { { "NumPy Parser", "NumPy Segmenter" },
          "NumPy",
          [](ZL_Compressor* c) {
              return ZL_NumPy_registerGraph(c, ZL_NUMPY_DEFAULT_CHUNK_SIZE);
          } },
        { { "Tensor Scale16" }, "scale", ZL_TensorGraphs_registerScale16Graph },
        { { "Tensor Upcast32" },
          "upcast",
          ZL_TensorGraphs_registerUpcast32Graph },
        { { "Tensor Rows32" }, "rows", ZL_TensorGraphs_registerRows32Graph },
    };
    for (const auto& registration : kRegistrations) {
        const auto& names = registration.graphNames;
        if (std::find(names.begin(), names.end(), graphName) != names.end()) {
            return &registration;
        }
    }
    return nullptr;
}

} // namespace

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

            const Registration* registration = findRegistration(graphName);
            if (registration == nullptr) {
                ZL_LOG(WARN,
                       "processDependencies: unresolved graph dependency '%s'",
                       graphName.c_str());
                continue;
            }
            unwrap(registration->registerGraph(compressor.get()),
                   "Failed to create the " + std::string(registration->label)
                           + " graph",
                   compressor.get());
            madeProgress = true;
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
