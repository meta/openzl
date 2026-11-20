// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openzl/codecs/zl_zstd.h"
#include "openzl/codecs/zstd/common_zstd.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/Exception.hpp"
#include "openzl/cpp/Input.hpp"
#include "openzl/dict/bundle.h"
#include "openzl/dict/dict.h"
#include "openzl/dict/dict_constants.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_localParams.h"
#include "openzl/zl_reflection.h"
#include "openzl/zl_unique_id.h"
#include "openzl/zl_version.h"

#include <zstd.h>

#include "tools/training/dict/base_dict_trainer.h"
#include "tools/training/dict/dict_training.h"
#include "tools/training/dict/zstd_dict_trainer.h"
#include "tools/training/train_exceptions.h"
#include "tools/training/trained_candidate.h"
#include "tools/training/utils/utils.h"

namespace openzl::training {
namespace {

static std::vector<std::string> generateSampleData(
        size_t numSamples,
        size_t sampleSize)
{
    std::vector<std::string> samples;
    samples.reserve(numSamples);
    const std::string prefix = "HEADER_v1: common_prefix_data=";
    const std::string suffix = " END_OF_RECORD\n";
    for (size_t i = 0; i < numSamples; ++i) {
        std::string sample;
        sample.reserve(sampleSize);
        while (sample.size() < sampleSize) {
            sample += prefix;
            for (size_t j = 0; j < 32; ++j) {
                sample += static_cast<char>('A' + ((i + j) % 26));
            }
            sample += suffix;
        }
        sample.resize(sampleSize);
        samples.push_back(std::move(sample));
    }
    return samples;
}

static std::vector<MultiInput> toMultiInputs(
        const std::vector<std::string>& data)
{
    std::vector<MultiInput> inputs;
    inputs.reserve(data.size());
    for (const auto& s : data) {
        MultiInput mi;
        mi.add(Input::refSerial(s.data(), s.size()));
        inputs.push_back(std::move(mi));
    }
    return inputs;
}

// A custom trainer that overrides the Zstd trainer by returning (incrementally)
// from a fixed set of return strings
class FakeZstdTrainer : public ZstdDictTrainer {
   public:
    explicit FakeZstdTrainer(std::vector<std::string> rets)
            : rets_(std::move(rets))
    {
    }

    poly::optional<std::string> trainDict(
            const std::vector<MultiInput>& /* inputs */,
            const Compressor& /* compressor */,
            ZL_LocalParams /* localParams */) override
    {
        if (idx_ >= rets_.size()) {
            std::string exc = "too many dicts! too many dicts!";
            throw std::runtime_error(exc);
        }
        return rets_[idx_++];
    }

   private:
    std::vector<std::string> rets_;
    size_t idx_{ 0 };
};

// -------------------------------------------------------
// Unit tests for BaseDictTrainer class
// -------------------------------------------------------
TEST(BaseDictTrainer, DuplicateDictsAreDeduped)
{
    auto sampleData = generateSampleData(
            2, 4096); // not really used, we inject a custom trainer
    auto inputs = toMultiInputs(sampleData);

    // Generate a compressor that splits an input into 3, and sends each to a
    // trainable zstd node
    Compressor compressor;
    compressor.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    {
        constexpr size_t kNumSegments                = 3;
        constexpr size_t kSegmentSizes[kNumSegments] = { 1024, 1024, 0 };
        // Build one trainable zstd graph per segment so training produces a
        // separate dictionary for each.
        std::vector<ZL_GraphID> successors;
        successors.reserve(kNumSegments);
        for (size_t i = 0; i < kNumSegments; ++i) {
            successors.push_back(compressor.unwrap(
                    ZL_Compressor_buildTrainableZstdGraph(compressor.get()),
                    "Failed to build trainable zstd graph"));
        }

        auto gid = ZL_Compressor_registerSplitGraph(
                compressor.get(),
                ZL_Type_serial,
                kSegmentSizes,
                successors.data(),
                kNumSegments);
        compressor.selectStartingGraph(gid);
    }

    // Train with a FakeZstdTrainer that returns identical dict content for the
    // first two nodes and different content for the third. The two identical
    // dicts hash to the same ID and must be deduplicated, leaving two dicts.
    std::vector<std::unique_ptr<DictTrainer>> trainers;
    trainers.push_back(
            std::make_unique<FakeZstdTrainer>(std::vector<std::string>{
                    "duplicate_dict_content_AAAAAAAAAAAAAAAA",
                    "duplicate_dict_content_AAAAAAAAAAAAAAAA",
                    "unique_dict_content_BBBBBBBBBBBBBBBB",
            }));

    TrainParams params{};
    auto candidate =
            trainDictsForCandidate(inputs, compressor, params, trainers);

    // Three nodes were trained, but two produced identical dicts, so only two
    // unique dicts should remain in the candidate.
    ASSERT_EQ(candidate.dicts.size(), 2u);
    EXPECT_FALSE(ZL_UniqueID_eq(
            &candidate.dicts[0].dictID.id, &candidate.dicts[1].dictID.id));
    EXPECT_NE(candidate.dicts[0].packedDict, candidate.dicts[1].packedDict);
}

// Below ZL_MATERIALIZED_DICT_VERSION_MIN, materialized dicts cannot be encoded,
// so dict training reports the unsupported target version by throwing
// FormatVersionUnsupportedError (letting the orchestrator fall back to zstd).
TEST(BaseDictTrainer, ThrowsWhenFormatVersionCannotEncodeDicts)
{
    auto sampleData = generateSampleData(2, 4096);
    auto inputs     = toMultiInputs(sampleData);

    // Compressor with a trainable zstd node that would otherwise be trained.
    Compressor compressor;
    ZL_GraphID g1 = openzl::unwrap(
            ZL_Compressor_buildTrainableZstdGraph(compressor.get()));
    const size_t segmentSizes[2]   = { 1024, 0 };
    const ZL_GraphID successors[2] = { g1, ZL_GRAPH_ZSTD };
    ZL_GraphID gHead               = ZL_Compressor_registerSplitGraph(
            compressor.get(), ZL_Type_serial, segmentSizes, successors, 2);
    compressor.selectStartingGraph(gHead);
    compressor.setParameter(
            CParam::FormatVersion, ZL_MATERIALIZED_DICT_VERSION_MIN - 1);

    // A trainer that would produce a dict if it were ever invoked.
    std::vector<std::unique_ptr<DictTrainer>> trainers;
    trainers.push_back(
            std::make_unique<FakeZstdTrainer>(std::vector<std::string>{
                    "dict_content_that_should_never_be_used",
            }));

    TrainParams params{};

    EXPECT_THROW(
            trainDictsForCandidate(inputs, compressor, params, trainers),
            FormatVersionUnsupportedError);
}

TEST(BaseDictTrainer, NoDictNodesRemainNoOpAtUnsupportedVersion)
{
    auto sampleData = generateSampleData(1, 256);
    auto inputs     = toMultiInputs(sampleData);

    Compressor compressor;
    compressor.selectStartingGraph(ZL_GRAPH_STORE);
    compressor.setParameter(
            CParam::FormatVersion, ZL_MATERIALIZED_DICT_VERSION_MIN - 1);

    TrainParams params{};
    const auto candidate = trainDictsForCandidate(inputs, compressor, params);
    EXPECT_TRUE(candidate.dicts.empty());
}

// -------------------------------------------------------
// Unit tests for ZstdDictTrainer class
// -------------------------------------------------------

TEST(ZstdDictTrainer, TrainDictReturnsPackedContent)
{
    auto sampleData = generateSampleData(100, 4096);
    auto inputs     = toMultiInputs(sampleData);

    ZstdDictTrainer trainer;
    Compressor compressor;
    compressor.setParameter(CParam::CompressionLevel, 3);
    ZL_LocalParams localParams{};
    auto dictOpt = trainer.trainDict(inputs, compressor, localParams);
    ASSERT_TRUE(dictOpt.has_value());
    std::string dictContent = dictOpt.value();

    ASSERT_FALSE(dictContent.empty());
    ASSERT_GE(dictContent.size(), ZL_TRAINED_ZSTD_CONTENT_HEADER_SIZE);
    ZL_TrainedZstdContentParsed parsed{};
    ASSERT_TRUE(ZL_TrainedZstdContent_parse(
            dictContent.data(), dictContent.size(), &parsed));
    EXPECT_EQ(parsed.clevel, 3);
    EXPECT_GT(parsed.rawDictSize, 0u);
}

TEST(ZstdDictTrainer, TrainDictUsesCompressionLevelFromLocalParams)
{
    auto sampleData = generateSampleData(100, 4096);
    auto inputs     = toMultiInputs(sampleData);

    ZstdDictTrainer trainer;
    Compressor compressor;
    ZL_IntParam clevelParam = { ZSTD_c_compressionLevel, 7 };
    ZL_LocalParams localParams{
        .intParams = { .intParams = &clevelParam, .nbIntParams = 1 },
    };
    auto dictOpt = trainer.trainDict(inputs, compressor, localParams);
    ASSERT_TRUE(dictOpt.has_value());
    std::string dictContent = dictOpt.value();

    ASSERT_FALSE(dictContent.empty());
    ZL_TrainedZstdContentParsed parsed{};
    ASSERT_TRUE(ZL_TrainedZstdContent_parse(
            dictContent.data(), dictContent.size(), &parsed));
    EXPECT_EQ(parsed.clevel, 7);
}

TEST(ZstdDictTrainer, FindDictNodesReturnsEmptyForDefaultCompressor)
{
    // A default compressor has no starting graph, so the recursive
    // walk finds nothing.
    Compressor compressor;
    auto sampleData = generateSampleData(10, 256);
    auto inputs     = toMultiInputs(sampleData);
    TrainParams params{};

    auto candidate = trainDictsForCandidate(inputs, compressor, params);
    EXPECT_TRUE(candidate.dicts.empty());
}

TEST(ZstdDictTrainer, FindDictNodesFindsMultipleZstdNodes)
{
    Compressor compressor;

    // Register two distinct trainable zstd graphs, and a custom level graph
    ZL_GraphID g1 = openzl::unwrap(
            ZL_Compressor_buildTrainableZstdGraph(compressor.get()));
    ZL_GraphID g2 = openzl::unwrap(
            ZL_Compressor_buildTrainableZstdGraph(compressor.get()));
    ZL_GraphID g3 =
            ZL_Compressor_registerZstdGraph_withLevel(compressor.get(), 12);
    ASSERT_TRUE(ZL_GraphID_isValid(g1));
    ASSERT_TRUE(ZL_GraphID_isValid(g2));
    ASSERT_TRUE(ZL_GraphID_isValid(g3));

    // Create a split graph that sends data to these different graphs
    const size_t segmentSizes[4]   = { 1024, 1024, 1024, 0 };
    const ZL_GraphID successors[4] = { g1, g2, g3, ZL_GRAPH_ZSTD };
    ZL_GraphID gHead               = ZL_Compressor_registerSplitGraph(
            compressor.get(), ZL_Type_serial, segmentSizes, successors, 4);
    ZL_Report r = ZL_Compressor_selectStartingGraphID(compressor.get(), gHead);
    ASSERT_FALSE(ZL_isError(r));

    ZstdDictTrainer zstdTrainer;
    const auto nodesToTrain = zstdTrainer.findDictNodes(compressor);
    ASSERT_EQ(nodesToTrain.size(), 2);
    EXPECT_EQ(nodesToTrain[0].nodeName, "zl.trainable.zstd#0");
    EXPECT_EQ(nodesToTrain[1].nodeName, "zl.trainable.zstd#1");
}

// -------------------------------------------------------
// Unit tests for TrainedCandidate integration helpers
// -------------------------------------------------------

TEST(TrainedCandidateHelpers, ReplaceBundleID)
{
    TrainedCandidate candidate;
    EXPECT_FALSE(ZL_UniqueID_isValid(&candidate.bundleID.id));

    ZL_BundleID newID;
    newID.id = ZL_UniqueID_computeSHA256("test_bundle", 11);
    candidate.replaceBundleID(newID);

    EXPECT_TRUE(ZL_UniqueID_eq(&candidate.bundleID.id, &newID.id));
}

TEST(TrainedCandidateHelpers, ReplaceDictIDUpdatesEntryAndPackedBlob)
{
    // Build a packed dict blob using Dict_pack.
    const std::string content = "fake_dict_content_for_testing";
    std::string packed(ZL_DICT_HEADER_SIZE + content.size(), '\0');
    ZL_Report report = Dict_pack(
            packed.data(),
            packed.size(),
            ZL_DICT_ID_NULL,
            42,
            false,
            content.data(),
            content.size());
    ASSERT_FALSE(ZL_isError(report));
    packed.resize(ZL_validResult(report));

    ZL_DictID originalID = Dict_extractID(packed.data(), packed.size());
    ASSERT_TRUE(ZL_UniqueID_isValid(&originalID.id));

    TrainedCandidate candidate;
    candidate.dicts.push_back(
            TrainedCandidate::DictEntry{
                    .dictID     = originalID,
                    .packedDict = packed,
            });

    // Replace the dictID (batch of 1).
    ZL_DictID newID;
    newID.id = ZL_UniqueID_computeSHA256("replacement", 11);
    ASSERT_FALSE(ZL_UniqueID_eq(&originalID.id, &newID.id));

    candidate.replaceDictID({ originalID }, { newID });

    // The struct field should be updated.
    EXPECT_TRUE(ZL_UniqueID_eq(&candidate.dicts[0].dictID.id, &newID.id));

    // The packed blob bytes 4-35 should be rewritten.
    ZL_DictID extractedID = Dict_extractID(
            candidate.dicts[0].packedDict.data(),
            candidate.dicts[0].packedDict.size());
    EXPECT_TRUE(ZL_UniqueID_eq(&extractedID.id, &newID.id));
}

TEST(TrainedCandidateHelpers, ReplaceDictIDThrowsOnMiss)
{
    TrainedCandidate candidate;
    ZL_DictID bogusID;
    bogusID.id = ZL_UniqueID_computeSHA256("bogus", 5);
    ZL_DictID anotherID;
    anotherID.id = ZL_UniqueID_computeSHA256("another", 7);

    EXPECT_THROW(
            candidate.replaceDictID({ bogusID }, { anotherID }), Exception);
}

TEST(TrainedCandidateHelpers, ReplaceDictIDBatch)
{
    auto makePacked = [](const std::string& content, ZL_IDType codec) {
        std::string packed(ZL_DICT_HEADER_SIZE + content.size(), '\0');
        ZL_Report report = Dict_pack(
                packed.data(),
                packed.size(),
                ZL_DICT_ID_NULL,
                codec,
                false,
                content.data(),
                content.size());
        EXPECT_FALSE(ZL_isError(report));
        packed.resize(ZL_validResult(report));
        return packed;
    };

    std::string packed1 = makePacked("dict_alpha_AAAA", 42);
    std::string packed2 = makePacked("dict_bravo_BBBB", 43);
    ZL_DictID id1       = Dict_extractID(packed1.data(), packed1.size());
    ZL_DictID id2       = Dict_extractID(packed2.data(), packed2.size());

    TrainedCandidate candidate;
    candidate.dicts.push_back(
            TrainedCandidate::DictEntry{
                    .dictID     = id1,
                    .packedDict = std::move(packed1),
            });
    candidate.dicts.push_back(
            TrainedCandidate::DictEntry{
                    .dictID     = id2,
                    .packedDict = std::move(packed2),
            });

    ZL_DictID new1, new2;
    new1.id = ZL_UniqueID_computeSHA256("new1", 4);
    new2.id = ZL_UniqueID_computeSHA256("new2", 4);

    candidate.replaceDictID({ id1, id2 }, { new1, new2 });

    EXPECT_TRUE(ZL_UniqueID_eq(&candidate.dicts[0].dictID.id, &new1.id));
    EXPECT_TRUE(ZL_UniqueID_eq(&candidate.dicts[1].dictID.id, &new2.id));

    ZL_DictID ext1 = Dict_extractID(
            candidate.dicts[0].packedDict.data(),
            candidate.dicts[0].packedDict.size());
    ZL_DictID ext2 = Dict_extractID(
            candidate.dicts[1].packedDict.data(),
            candidate.dicts[1].packedDict.size());
    EXPECT_TRUE(ZL_UniqueID_eq(&ext1.id, &new1.id));
    EXPECT_TRUE(ZL_UniqueID_eq(&ext2.id, &new2.id));
}

TEST(TrainedCandidateHelpers, PackFatBundleRoundTrip)
{
    // Create two packed dicts.
    const std::string content1 = "dict_content_one_AAAAAAAAA";
    const std::string content2 = "dict_content_two_BBBBBBBBB";

    auto makePacked = [](const std::string& content, ZL_IDType codec) {
        std::string packed(ZL_DICT_HEADER_SIZE + content.size(), '\0');
        ZL_Report report = Dict_pack(
                packed.data(),
                packed.size(),
                ZL_DICT_ID_NULL,
                codec,
                false,
                content.data(),
                content.size());
        EXPECT_FALSE(ZL_isError(report));
        packed.resize(ZL_validResult(report));
        return packed;
    };

    std::string packed1 = makePacked(content1, 42);
    std::string packed2 = makePacked(content2, 43);

    ZL_DictID id1 = Dict_extractID(packed1.data(), packed1.size());
    ZL_DictID id2 = Dict_extractID(packed2.data(), packed2.size());

    TrainedCandidate candidate;
    candidate.dicts.push_back(
            TrainedCandidate::DictEntry{
                    .dictID     = id1,
                    .packedDict = std::move(packed1),
            });
    candidate.dicts.push_back(
            TrainedCandidate::DictEntry{
                    .dictID     = id2,
                    .packedDict = std::move(packed2),
            });
    candidate.bundleID = ZL_DictBundle_genBundleID(
            std::vector<ZL_DictID>{ id1, id2 }.data(), 2);

    // Pack the fat bundle.
    std::string fatBundle = candidate.packFatBundle();
    ASSERT_FALSE(fatBundle.empty());

    // Parse the BundleInfo header from the fat bundle.
    auto parseResult = ZL_BundleInfo_parse(fatBundle.data(), fatBundle.size());
    ASSERT_FALSE(ZL_RES_isError(parseResult));

    ZL_BundleInfo info = ZL_RES_value(parseResult);
    EXPECT_TRUE(info.isFatBundle);
    EXPECT_EQ(info.numDicts, 2u);
    EXPECT_TRUE(ZL_UniqueID_eq(&info.bundleID.id, &candidate.bundleID.id));

    // The dict IDs in the header should match.
    EXPECT_TRUE(ZL_UniqueID_eq(&info.dictIDs[0].id, &id1.id));
    EXPECT_TRUE(ZL_UniqueID_eq(&info.dictIDs[1].id, &id2.id));
}

TEST(TrainedCandidateHelpers, PackFatBundleThrowsOnEmpty)
{
    TrainedCandidate candidate;
    EXPECT_THROW(candidate.packFatBundle(), Exception);
}

} // namespace
} // namespace openzl::training
