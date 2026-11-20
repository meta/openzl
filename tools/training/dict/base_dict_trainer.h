// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <vector>

#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/poly/Optional.hpp"
#include "openzl/zl_localParams.h"
#include "openzl/zl_opaque_types.h"
#include "tools/training/utils/utils.h"

namespace openzl::training {

/// Info about a dict-requiring node discovered by the base trainer.
struct DictNodeInfo {
    ZL_NodeID nodeID;
    ZL_IDType codecID;
    std::string nodeName;
};

/**
 * Base dict trainer that trains dictionaries for a specific codec type.
 * Subclasses provide a codec name for node filtering.
 * Node discovery via recursive graph walking and standard-node promotion
 * are handled by the base trainer.
 */
class DictTrainer {
   public:
    virtual ~DictTrainer() = default;

    /// Return a list of nodes to train dictionaries for.
    virtual std::vector<DictNodeInfo> findDictNodes(
            const Compressor& compressor) = 0;

    /// Train a dictionary from the given input samples and pack it into
    /// the codec-specific dict content format (ready for Dict_pack()).
    /// @param inputs      Samples collected via introspection hooks for the
    ///                    specific graph that feeds this codec node.
    /// @param compressor  The compressor that owns the node being trained.
    /// @param localParams The local params of the node being trained
    ///                    (e.g. compression level for zstd).
    /// @returns Packed dict content bytes ready for Dict_pack(), or nullopt if
    ///          the trainer declines to train a dictionary for these inputs.
    virtual poly::optional<std::string> trainDict(
            const std::vector<MultiInput>& inputs,
            const Compressor& compressor,
            ZL_LocalParams localParams) = 0;
};

} // namespace openzl::training
