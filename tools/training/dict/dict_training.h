// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <vector>

#include "openzl/cpp/Compressor.hpp"
#include "tools/training/dict/base_dict_trainer.h"
#include "tools/training/train_params.h"
#include "tools/training/trained_candidate.h"
#include "tools/training/utils/utils.h"

namespace openzl::training {

/// Train dictionaries for all dict-requiring nodes in @p compressor.
///
/// For each registered DictTrainer, discovers dict-requiring nodes, uses
/// collectInputStreamsForGraphs to capture the actual data flowing into
/// each node, trains dicts, packs into a fat bundle, loads into the
/// compressor, and re-serializes.
///
/// @param inputs            Raw training samples.
/// @param compressor        The compressor to train dicts for (modified
///                          in place — fat bundle is loaded).
/// @param trainParams       Training parameters.
/// @returns A TrainedCandidate with the re-serialized compressor,
///          bundleID, and per-dict entries populated.
TrainedCandidate trainDictsForCandidate(
        const std::vector<MultiInput>& inputs,
        Compressor& compressor,
        const TrainParams& trainParams);

/// For testing. Inject a custom set of dict trainers instead of using the
/// default
TrainedCandidate trainDictsForCandidate(
        const std::vector<MultiInput>& inputs,
        Compressor& compressor,
        const TrainParams& trainParams,
        const std::vector<std::unique_ptr<DictTrainer>>& trainers);

} // namespace openzl::training
