// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>
#include <utility>
#include <vector>
#include "custom_parsers/dependency_registration.h"
#include "openzl/codecs/zl_clustering.h"
#include "openzl/codecs/zl_quantize.h"
#include "openzl/cpp/Compressor.hpp"
#if defined(ZL_IS_FBCODE) && (ZL_IS_FBCODE == 1)
#    if ZL_FBCODE_IS_RELEASE
#        include "openzl/prod/tools/protobuf/tests/test_schema.pb.h"
#    else
#        include "openzl/dev/tools/protobuf/tests/test_schema.pb.h"
#    endif
#else
#    include "tools/protobuf/tests/test_schema.pb.h"
#endif
#include "tools/protobuf/ProtoDeserializer.h"
#include "tools/protobuf/ProtoSerializer.h"
#include "tools/training/train.h"

using namespace ::testing;
using namespace openzl::protobuf;

namespace {
openzl::Compressor makePartialDomainCompressor()
{
    openzl::Compressor compressor;
    compressor.setParameter(
            openzl::CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    compressor.setParameter(
            openzl::CParam::StoreOnExpansion, ZL_TernaryParam_disable);
    auto quantize = compressor.buildStaticGraph(
            ZL_NODE_QUANTIZE_OFFSETS, { ZL_GRAPH_STORE, ZL_GRAPH_STORE });

    // Only int32 values use quantize-offsets, whose domain excludes zero.
    // All other protobuf streams use the clustering graph's generic default.
    int tag                             = static_cast<int>(InputType::INT32);
    ZL_ClusteringConfig_Cluster cluster = {};
    cluster.typeSuccessor      = { ZL_Type_numeric, sizeof(int32_t), 0, 2 };
    cluster.memberTags         = &tag;
    cluster.nbMemberTags       = 1;
    ZL_ClusteringConfig config = {};
    config.clusters            = &cluster;
    config.nbClusters          = 1;
    compressor.selectStartingGraph(ZL_Clustering_registerGraph(
            compressor.get(), &config, &quantize, 1));
    return compressor;
}

TestSchema genTestSchema()
{
    TestSchema obj;
    obj.set_optional_int32(1);
    obj.set_optional_int64(2);
    obj.set_optional_uint32(3);
    obj.set_optional_uint64(4);
    obj.set_optional_sint32(5);
    obj.set_optional_sint64(6);
    obj.set_optional_fixed32(7);
    obj.set_optional_fixed64(8);
    obj.set_optional_sfixed32(9);
    obj.set_optional_sfixed64(10);
    obj.set_optional_float(11);
    obj.set_optional_double(12);
    obj.set_optional_bool(true);
    obj.set_optional_string("string");
    obj.set_optional_bytes("bytes");

    // Nested Message
    obj.mutable_optional_nested()->set_optional_int32(13);

    // Enum
    obj.set_optional_enum(EnumSchema::ONE);

    // Repeated Fields
    obj.add_repeated_int32(1);
    obj.add_repeated_int32(2);
    obj.add_repeated_int32(3);

    obj.add_repeated_nested()->set_optional_int32(1);
    obj.add_repeated_nested()->set_optional_int32(2);
    obj.add_repeated_nested()->set_optional_int32(3);
    obj.add_repeated_nested()->set_optional_int32(4);

    obj.add_repeated_enum(EnumSchema::ZERO);
    obj.add_repeated_enum(EnumSchema::ONE);
    return obj;
}
} // namespace

TEST(TestProtoSerializer, BasicRoundTrip)
{
    auto obj = genTestSchema();

    // Serialize
    auto serializer = ProtoSerializer();
    auto serialized = serializer.serialize(obj);

    // Deserialize
    auto deserializer = ProtoDeserializer();
    auto deserialized = deserializer.deserialize<TestSchema>(serialized);

    // Check Round Trip
    EXPECT_TRUE(
            google::protobuf::util::MessageDifferencer::Equivalent(
                    obj, deserialized));
}

TEST(TestProtoSerializer, CustomCompressor)
{
    ProtoSerializer serializer;

    openzl::Compressor compressor;
    compressor.selectStartingGraph(ZL_GRAPH_COMPRESS_GENERIC);
    // disable anti-inflation guard so compressed output differs from STORE
    compressor.setParameter(
            openzl::CParam::StoreOnExpansion, ZL_TernaryParam_disable);
    serializer.setCompressor(std::move(compressor));
    ProtoDeserializer deserializer;

    auto obj          = genTestSchema();
    auto serialized   = serializer.serialize(obj);
    auto deserialized = deserializer.deserialize<TestSchema>(serialized);

    EXPECT_TRUE(
            google::protobuf::util::MessageDifferencer::Equivalent(
                    obj, deserialized));

    openzl::Compressor store;
    store.selectStartingGraph(ZL_GRAPH_STORE);
    serializer.setCompressor(std::move(store));
    auto stored   = serializer.serialize(obj);
    auto destored = deserializer.deserialize<TestSchema>(serialized);

    EXPECT_TRUE(
            google::protobuf::util::MessageDifferencer::Equivalent(
                    obj, destored));
    EXPECT_NE(stored.size(), serialized.size());
}

TEST(TestProtoSerializer, TrainedCompressorRoundTrip)
{
    auto obj = genTestSchema();
    ProtoSerializer serializer;
    std::vector<openzl::training::MultiInput> inputs;
    inputs.emplace_back(serializer.getTrainingInputs(obj));

    openzl::training::TrainParams params;
    params.compressorGenFunc =
            openzl::custom_parsers::createCompressorFromSerialized;
    params.clusteringTrainer = openzl::training::FullSplit;
    params.threads           = 1;
    params.maxTimeSecs       = 1;

    auto trained = openzl::training::train(
            inputs, *serializer.getCompressor(), params);
    ASSERT_FALSE(trained.empty());

    auto compressor = openzl::custom_parsers::createCompressorFromSerialized(
            trained.front().serializedCompressor, "");
    ProtoSerializer trainedSerializer;
    trainedSerializer.setCompressor(std::move(*compressor));
    auto serialized   = trainedSerializer.serialize(obj);
    auto deserialized = ProtoDeserializer().deserialize<TestSchema>(serialized);

    EXPECT_TRUE(
            google::protobuf::util::MessageDifferencer::Equivalent(
                    obj, deserialized));
}

TEST(TestProtoSerializer, PermissiveCompression)
{
    for (bool reload : { false, true }) {
        SCOPED_TRACE(reload ? "reloaded compressor" : "original compressor");
        auto compressor = makePartialDomainCompressor();
        if (reload) {
            auto serialized = compressor.serialize();
            compressor      = openzl::Compressor();
            compressor.deserialize(serialized);
        }

        ProtoSerializer serializer;
        serializer.setCompressor(std::move(compressor));
        ProtoDeserializer deserializer;
        auto roundTrip = [&](const TestSchema& message) {
            auto serialized = serializer.serialize(message);
            auto decoded    = deserializer.deserialize<TestSchema>(serialized);
            EXPECT_TRUE(
                    google::protobuf::util::MessageDifferencer::Equals(
                            message, decoded));
        };

        auto positive = genTestSchema();
        auto zero     = positive;
        zero.set_optional_int32(0);

        // Default strict mode accepts the graph's domain and rejects zero.
        roundTrip(positive);
        EXPECT_THROW(serializer.serialize(zero), openzl::Exception);

        serializer.setPermissiveCompression(true);
        EXPECT_NE(
                serializer.getCompressor()->getParameter(
                        openzl::CParam::PermissiveCompression),
                ZL_TernaryParam_enable);
        // Fallback must preserve the explicitly present zero, and the setting
        // must persist across successive compression sessions.
        roundTrip(zero);
        roundTrip(positive);
        roundTrip(zero);

        // Replacing the compressor also preserves the serializer's setting.
        auto replacement = makePartialDomainCompressor();
        serializer.setCompressor(std::move(replacement));
        roundTrip(zero);

        // Explicit strict mode must also override a permissive compressor.
        serializer.getCompressor()->setParameter(
                openzl::CParam::PermissiveCompression, ZL_TernaryParam_enable);
        serializer.setPermissiveCompression(false);
        EXPECT_THROW(serializer.serialize(zero), openzl::Exception);
        roundTrip(positive);
    }
}
