// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/training/dict/dict_training.h"

#include <cstring>
#include <set>
#include <vector>

#include "openzl/compress/cgraph.h"
#include "openzl/dict/dict.h"
#include "openzl/dict/dict_constants.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_reflection.h"
#include "openzl/zl_version.h"

#include "tools/logger/Logger.h"
#include "tools/training/dict/zstd_dict_trainer.h"
#include "tools/training/sample_collection/training_sample_collector.h"
#include "tools/training/train_exceptions.h"

using namespace openzl::tools::logger;

namespace openzl::training {

namespace {

/// Build the list of all registered DictTrainer instances.
std::vector<std::unique_ptr<DictTrainer>> createAllTrainers()
{
    std::vector<std::unique_ptr<DictTrainer>> trainers;
    trainers.push_back(std::make_unique<ZstdDictTrainer>());
    return trainers;
}

/// ZL_DictID is a plain C struct with no operator<, so std::set needs an
/// explicit comparator. The ID is a fixed 32-byte value; compare it bytewise.
struct DictIDCmp {
    bool operator()(const ZL_DictID& lhs, const ZL_DictID& rhs) const noexcept
    {
        return std::memcmp(lhs.id.bytes, rhs.id.bytes, sizeof(lhs.id.bytes))
                < 0;
    }
};

} // anonymous namespace

TrainedCandidate trainDictsForCandidate(
        const std::vector<MultiInput>& inputs,
        Compressor& compressor,
        const TrainParams& trainParams)
{
    auto trainers = createAllTrainers();
    return trainDictsForCandidate(inputs, compressor, trainParams, trainers);
}

TrainedCandidate trainDictsForCandidate(
        const std::vector<MultiInput>& inputs,
        Compressor& compressor,
        const TrainParams& trainParams,
        const std::vector<std::unique_ptr<DictTrainer>>& trainers)
{
    (void)trainParams;

    TrainedCandidate candidate;
    candidate.serializedCompressor = compressor.serialize();

    // Ask each registered trainer to find its dict-requiring nodes.
    struct TrainerWork {
        DictTrainer* trainer;
        DictNodeInfo node;
    };
    std::vector<TrainerWork> work;
    std::vector<std::string> nodeNames;

    for (auto& trainer : trainers) {
        auto nodes = trainer->findDictNodes(compressor);
        for (auto& node : nodes) {
            nodeNames.push_back(node.nodeName);
            work.push_back(
                    TrainerWork{
                            .trainer = trainer.get(),
                            .node    = std::move(node),
                    });
        }
    }

    if (work.empty()) {
        return candidate;
    }

    const auto formatVersion = compressor.getParameter(CParam::FormatVersion);
    if (formatVersion == 0) {
        throw FormatVersionUnsupportedError(
                "Compressor format version is not set.");
    }

    // Materialized dictionaries are only representable at
    // ZL_MATERIALIZED_DICT_VERSION_MIN and above. Below that the encoder cannot
    // back a node with a dict, so signal the orchestrator to use its fallback
    // instead of emitting an unusable candidate.
    if (formatVersion < ZL_MATERIALIZED_DICT_VERSION_MIN) {
        throw FormatVersionUnsupportedError(
                "dictionary training requires format version >= "
                + std::to_string(ZL_MATERIALIZED_DICT_VERSION_MIN)
                + "; target version " + std::to_string(formatVersion)
                + " is unsupported");
    }

    Logger::log_c(
            VERBOSE1,
            "Dict training: found %zu nodes requiring dictionaries",
            work.size());

    // Use introspection hooks to collect the actual data flowing into
    // each dict-requiring codec node, at the resolved target format version.
    auto cctx           = refCCtxForTraining(compressor);
    auto samplesPerNode = collectInputStreams(inputs, {}, nodeNames, cctx);

    std::set<ZL_DictID, DictIDCmp> uniqueDictIDs;
    for (auto& [trainer, node] : work) {
        auto it = samplesPerNode.find(node.nodeName);
        if (it == samplesPerNode.end() || it->second.empty()) {
            Logger::log_c(
                    VERBOSE1,
                    "No samples collected for node %s, skipping",
                    node.nodeName.c_str());
            continue;
        }

        const auto& nodeSamples = it->second;
        Logger::log_c(
                VERBOSE1,
                "Training dict for node %u (codec %u, name %s) "
                "with %zu samples",
                node.nodeID.nid,
                node.codecID,
                node.nodeName.c_str(),
                nodeSamples.size());

        // trainDict returns packed dict content ready for Dict_pack.
        ZL_LocalParams localParams = ZL_Compressor_Node_getLocalParams(
                compressor.get(), node.nodeID);
        auto dictContent =
                trainer->trainDict(nodeSamples, compressor, localParams);
        if (!dictContent.has_value()) {
            Logger::log_c(
                    VERBOSE1,
                    "Trainer declined to train dict for node %s, skipping",
                    node.nodeName.c_str());
            continue;
        }

        // Pack into the generic ZL_Dict wire format, with an auto-generated ID.
        std::string packedDict(ZL_DICT_HEADER_SIZE + dictContent->size(), '\0');
        ZL_Report report = Dict_pack(
                packedDict.data(),
                packedDict.size(),
                ZL_DICT_ID_NULL,
                node.codecID,
                false,
                dictContent->data(),
                dictContent->size());
        if (ZL_isError(report)) {
            throw Exception("Dict_pack failed");
        }
        packedDict.resize(ZL_validResult(report));

        ZL_DictID generatedDictID =
                Dict_extractID(packedDict.data(), packedDict.size());

        ZL_NodeParameters nodeParams = {
            .dictID = generatedDictID,
        };
        compressor.unwrap(ZL_Compressor_overrideNodeParams(
                compressor.get(), node.nodeID, &nodeParams));

        if (uniqueDictIDs.find(generatedDictID) != uniqueDictIDs.end()) {
            // This dict is a duplicate of a previously trained dict. No need to
            // add it to the candidate. OpenZL dict loading will materialize
            // properly.
            continue;
        }

        uniqueDictIDs.insert(generatedDictID);
        candidate.dicts.push_back(
                TrainedCandidate::DictEntry{
                        .dictID     = generatedDictID,
                        .packedDict = std::move(packedDict),
                });

        Logger::log_c(
                VERBOSE1,
                "Trained dict: %zu bytes content",
                dictContent->size());
    }

    if (candidate.dicts.empty()) {
        return candidate;
    }

    std::vector<ZL_DictID> allDictIDs;
    allDictIDs.reserve(candidate.dicts.size());
    for (auto& dict : candidate.dicts) {
        allDictIDs.push_back(dict.dictID);
    }
    // Compute bundleID
    auto bundleID =
            ZL_DictBundle_genBundleID(allDictIDs.data(), allDictIDs.size());

    candidate.serializedCompressor = compressor.serialize();

    // Set bundle ID on the candidate and patch dict_bundle_id in the CBOR.
    candidate.replaceBundleID(bundleID);

    Logger::log_c(
            VERBOSE1,
            "Dict training complete: %zu dicts, CBOR patched with bundleID",
            candidate.dicts.size());

    return candidate;
}

} // namespace openzl::training
