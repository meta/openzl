// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "custom_parsers/dependency_registration.h"
#include "custom_parsers/safetensors/safetensors_lexer.h"
#include "custom_parsers/safetensors/safetensors_parser.h"
#include "custom_parsers/safetensors/tests/safetensors_builder.h"
#include "custom_parsers/shared_components/tensor_chunker.h"
#include "custom_parsers/shared_components/tensor_graphs.h"
#include "openzl/cpp/CompressIntrospectionHooks.hpp"
#include "openzl/openzl.hpp"
#include "openzl/zl_reflection.h"

namespace openzl::custom_parsers {
namespace {

using testing::buildSafetensors;
using testing::pseudoRandomBytes;
using testing::TensorSpec;

struct LexedToken {
    ZL_SafetensorsTokenType type;
    size_t offset;
    size_t size;
    std::string name;
    ZL_SafetensorsDtype dtype;
    std::vector<uint64_t> shape;

    bool operator==(const LexedToken&) const = default;
};

LexedToken headerToken(size_t size)
{
    return { ZL_SafetensorsTokenType_header, 0, size, "", {}, {} };
}

LexedToken gapToken(size_t offset, size_t size)
{
    return { ZL_SafetensorsTokenType_gap, offset, size, "", {}, {} };
}

LexedToken tensorToken(
        size_t offset,
        size_t size,
        std::string name,
        ZL_SafetensorsDtype dtype,
        std::vector<uint64_t> shape)
{
    return { ZL_SafetensorsTokenType_tensor,
             offset,
             size,
             std::move(name),
             dtype,
             std::move(shape) };
}

struct LexResult {
    ZL_ErrorCode error;
    std::vector<LexedToken> tokens;
};

LexResult lex(std::string_view file)
{
    const ZL_Report maxTokens =
            ZL_Safetensors_lex(file.data(), file.size(), nullptr, 0, nullptr);
    if (ZL_isError(maxTokens)) {
        return { ZL_errorCode(maxTokens), {} };
    }
    std::vector<ZL_SafetensorsToken> out(ZL_validResult(maxTokens));
    const ZL_Report nbTokens = ZL_Safetensors_lex(
            file.data(), file.size(), out.data(), out.size(), nullptr);
    if (ZL_isError(nbTokens)) {
        return { ZL_errorCode(nbTokens), {} };
    }
    LexResult result{ ZL_ErrorCode_no_error, {} };
    for (size_t i = 0; i < ZL_validResult(nbTokens); ++i) {
        const auto& t       = out[i];
        const size_t offset = size_t(t.ptr - file.data());
        if (t.type == ZL_SafetensorsTokenType_tensor) {
            result.tokens.push_back(tensorToken(
                    offset,
                    t.size,
                    std::string(t.name, t.nameSize),
                    t.dtype,
                    { t.shape, t.shape + t.ndims }));
        } else {
            result.tokens.push_back({ t.type, offset, t.size, "", {}, {} });
        }
    }
    return result;
}

std::string withHeader(std::string_view json, std::string_view data = "")
{
    std::string out(8, '\0');
    const uint64_t size = json.size();
    for (size_t i = 0; i < 8; ++i) {
        out[i] = (char)(size >> (8 * i));
    }
    return out + std::string(json) + std::string(data);
}

TEST(SafetensorsLexerTest, ReturnsTensorsInDataOrder)
{
    const auto file = withHeader(
            R"({"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]},)"
            R"("a":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]}})",
            std::string(12, 'x'));
    const size_t dataStart                 = file.size() - 12;
    const std::vector<LexedToken> expected = {
        headerToken(dataStart),
        tensorToken(dataStart, 4, "a", ZL_SafetensorsDtype_bf16, { 1, 2 }),
        tensorToken(dataStart + 4, 8, "b", ZL_SafetensorsDtype_f32, { 2 }),
    };
    const auto result = lex(file);
    ASSERT_EQ(result.error, ZL_ErrorCode_no_error);
    EXPECT_EQ(result.tokens, expected);
}

TEST(SafetensorsLexerTest, CoversDataOutsideTensorsWithGaps)
{
    const auto file = withHeader(
            R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[2,4]},)"
            R"("empty":{"dtype":"U8","shape":[0],"data_offsets":[4,4]},)"
            R"("b":{"dtype":"U8","shape":[1],"data_offsets":[6,7]}})",
            "123456789");
    const size_t dataStart                 = file.size() - 9;
    const std::vector<LexedToken> expected = {
        headerToken(dataStart),
        gapToken(dataStart, 2),
        tensorToken(dataStart + 2, 2, "a", ZL_SafetensorsDtype_u8, { 2 }),
        gapToken(dataStart + 4, 2),
        tensorToken(dataStart + 6, 1, "b", ZL_SafetensorsDtype_u8, { 1 }),
        gapToken(dataStart + 7, 2),
    };
    const auto result = lex(file);
    ASSERT_EQ(result.error, ZL_ErrorCode_no_error);
    EXPECT_EQ(result.tokens, expected);
}

TEST(SafetensorsLexerTest, SkipsMetadataAndPadding)
{
    const auto file = withHeader(
            R"({"__metadata__":{"format":"pt","k":"{\"x\":[1]}"},)"
            R"("t":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}   )",
            "abc");
    const std::vector<LexedToken> expected = {
        headerToken(file.size() - 3),
        tensorToken(file.size() - 3, 3, "t", ZL_SafetensorsDtype_u8, { 3 }),
    };
    const auto result = lex(file);
    ASSERT_EQ(result.error, ZL_ErrorCode_no_error);
    EXPECT_EQ(result.tokens, expected);
}

TEST(SafetensorsLexerTest, ReportsInconsistentOrUnknownTensorsAsUnknown)
{
    const auto file = withHeader(
            R"({"mismatch":{"dtype":"F16","shape":[3],"data_offsets":[0,4]},)"
            R"("future":{"dtype":"F4","shape":[2],"data_offsets":[4,5]}})",
            "12345");
    const size_t dataStart                 = file.size() - 5;
    const std::vector<LexedToken> expected = {
        headerToken(dataStart),
        tensorToken(
                dataStart, 4, "mismatch", ZL_SafetensorsDtype_unknown, { 3 }),
        tensorToken(
                dataStart + 4, 1, "future", ZL_SafetensorsDtype_unknown, { 2 }),
    };
    const auto result = lex(file);
    ASSERT_EQ(result.error, ZL_ErrorCode_no_error);
    EXPECT_EQ(result.tokens, expected);
}

TEST(SafetensorsLexerTest, CountsTokensWithoutCheckingLayout)
{
    // Overlapping offsets: only the full lexing checks the layout.
    const auto file = withHeader(
            R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},)"
            R"("b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}})",
            "xyz");
    const ZL_Report maxTokens =
            ZL_Safetensors_lex(file.data(), file.size(), nullptr, 0, nullptr);

    ASSERT_FALSE(ZL_isError(maxTokens));
    EXPECT_EQ(ZL_validResult(maxTokens), 6u); // header, 2 tensors and 3 gaps
}

TEST(SafetensorsLexerTest, CountingRejectsInvalidHeaderSize)
{
    std::string file(64, '\0');
    std::memset(file.data(), 0xff, 8);
    EXPECT_EQ(
            ZL_errorCode(ZL_Safetensors_lex(
                    file.data(), file.size(), nullptr, 0, nullptr)),
            ZL_ErrorCode_graph_parser_malformedInput);
}

TEST(SafetensorsLexerTest, RejectsTooManyTensors)
{
    std::string json = "{";
    for (size_t i = 0; i <= ZL_SAFETENSORS_MAX_TENSORS; ++i) {
        json += (i ? ",\"" : "\"") + std::to_string(i)
                + R"(":{"dtype":"U8","shape":[0],"data_offsets":[0,0]})";
    }
    const auto file = withHeader(json + "}");

    const ZL_Report maxTokens =
            ZL_Safetensors_lex(file.data(), file.size(), nullptr, 0, nullptr);

    EXPECT_EQ(
            ZL_errorCode(maxTokens), ZL_ErrorCode_graph_parser_unhandledInput);
}

TEST(SafetensorsLexerTest, RejectsMalformedHeaders)
{
    const std::string deep = std::string(100, '[') + std::string(100, ']');
    const std::vector<std::string> malformed = {
        std::string("\x05\0\0\0\0\0\0\0{}", 10), // header size beyond input
        withHeader("[]"),
        withHeader(R"({"t":{"dtype":"U8","shape":[1]}})"),
        withHeader(
                R"({"t":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}})",
                "x"),
        withHeader(
                R"({"t":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}})",
                "x"),
        withHeader(
                R"({"t":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}})",
                "x"),
        withHeader(
                R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},)"
                R"("b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}})",
                "xyz"),
        withHeader(
                R"({"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}} x)",
                "x"),
        withHeader(R"({"__metadata__":)" + deep + "}"),
        withHeader(
                R"({"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1])", "x"),
    };
    for (const auto& file : malformed) {
        EXPECT_EQ(lex(file).error, ZL_ErrorCode_graph_parser_malformedInput)
                << file.substr(8);
    }
}

std::vector<TensorSpec> mixedDtypeTensors()
{
    return {
        { "model.layers.0.w", "BF16", { 64, 48 }, pseudoRandomBytes(6144, 1) },
        { "model.layers.0.h", "F16", { 3000 }, pseudoRandomBytes(6000, 2) },
        { "model.layers.0.empty", "F32", { 0, 4 }, "" },
        { "model.layers.1.f", "F32", { 10, 100 }, pseudoRandomBytes(4000, 3) },
        { "model.layers.1.d", "F64", {}, pseudoRandomBytes(8, 4), 5 },
        { "model.layers.1.i", "I64", { 30 }, pseudoRandomBytes(240, 5) },
        { "mask", "BOOL", { 1, 64, 64 }, std::string(4096, '\1') },
        { "future", "F4", { 10 }, pseudoRandomBytes(5, 6) },
    };
}

class SafetensorsCompressorTest : public ::testing::Test {
   protected:
    void useChunkSize(size_t chunkSize)
    {
        const auto gid =
                ZL_Safetensors_registerGraph(compressor_.get(), chunkSize);
        ASSERT_FALSE(ZL_RES_isError(gid));
        compressor_.selectStartingGraph(ZL_RES_value(gid));
    }

    void attachHooks(CompressIntrospectionHooks& hooks)
    {
        openzl::unwrap(
                ZL_CCtx_attachIntrospectionHooks(
                        cctx_.get(), hooks.getRawHooks()),
                "Failed to attach introspection hooks",
                cctx_.get());
    }

    /// @returns the compressed size
    size_t expectRoundTrip(std::string_view input)
    {
        cctx_.setParameter(CParam::FormatVersion, formatVersion_);
        cctx_.refCompressor(compressor_);
        const auto compressed = cctx_.compressSerial(input);
        EXPECT_EQ(dctx_.decompressSerial(compressed), input);
        return compressed.size();
    }

    Compressor compressor_;
    CCtx cctx_;
    DCtx dctx_;
    int formatVersion_ = ZL_MAX_FORMAT_VERSION;
};

TEST_F(SafetensorsCompressorTest, RoundTripsMixedDtypes)
{
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    expectRoundTrip(buildSafetensors(mixedDtypeTensors(), R"({"a":"b"})"));
}

TEST_F(SafetensorsCompressorTest, RoundTripsWithChunksSmallerThanTensors)
{
    auto tensors = mixedDtypeTensors();
    tensors.push_back(
            { "model.layers.2.big",
              "F32",
              { 300, 300 },
              pseudoRandomBytes(360000, 8) });
    useChunkSize(2 * ZL_MIN_CHUNK_SIZE);
    expectRoundTrip(buildSafetensors(tensors));
}

TEST_F(SafetensorsCompressorTest, RoundTripsInvalidInputs)
{
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    const auto valid = buildSafetensors(mixedDtypeTensors());
    expectRoundTrip("");
    expectRoundTrip("tiny");
    expectRoundTrip(pseudoRandomBytes(10000, 7));
    expectRoundTrip(valid.substr(0, valid.size() / 2));
}

TEST_F(SafetensorsCompressorTest, ReportsWhyInvalidInputsAreNotParsed)
{
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    const auto valid = buildSafetensors(mixedDtypeTensors());

    expectRoundTrip(valid.substr(0, valid.size() / 2));

    const auto warnings = get_warning_strings(cctx_);
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(
            warnings[0].second.find("tensor data out of bounds"),
            std::string::npos)
            << warnings[0].second;
}

TEST_F(SafetensorsCompressorTest, FallsBackOnHugeDeclaredHeaderSize)
{
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    std::string file(1 << 20, '\0');
    std::memset(file.data(), 0xff, 8);

    expectRoundTrip(file);

    const auto warnings = get_warning_strings(cctx_);
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(warnings[0].second.find("invalid header size"), std::string::npos)
            << warnings[0].second;
}

TEST_F(SafetensorsCompressorTest, CompressesManyTinyTensorsOfMixedDtypes)
{
    // One stream per tensor would exceed the stream limit of a chunk.
    const size_t nbTensors = 60000;
    std::vector<TensorSpec> tensors;
    tensors.reserve(nbTensors);
    for (size_t i = 0; i < nbTensors; ++i) {
        const bool bf16 = i % 2 == 0;
        tensors.push_back(
                { "t" + std::to_string(i),
                  bf16 ? "BF16" : "F32",
                  { 1 },
                  pseudoRandomBytes(bf16 ? 2 : 4, (uint32_t)i) });
    }
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);

    expectRoundTrip(buildSafetensors(tensors));
}

TEST_F(SafetensorsCompressorTest, FallsBackOnFormatVersionsWithoutChunks)
{
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    formatVersion_ = ZL_CHUNK_VERSION_MIN - 1;

    expectRoundTrip(buildSafetensors(mixedDtypeTensors()));

    const auto warnings = get_warning_strings(cctx_);
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(
            warnings[0].second.find("format version without chunks"),
            std::string::npos)
            << warnings[0].second;
}

TEST_F(SafetensorsCompressorTest, ValidInputsProduceNoWarning)
{
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);

    expectRoundTrip(buildSafetensors(mixedDtypeTensors()));

    EXPECT_TRUE(get_warning_strings(cctx_).empty());
}

/// Records the size of each chunk handed to the inner parser graph.
class ChunkRecorder : public CompressIntrospectionHooks {
   public:
    void on_migraphEncode_start(
            ZL_Graph*,
            const ZL_Compressor* compressor,
            ZL_GraphID gid,
            ZL_Edge* inputs[],
            size_t nbInputs) override
    {
        const std::string_view name =
                ZL_Compressor_Graph_getName(compressor, gid);
        if (nbInputs == 1 && name.starts_with("Safetensors Parser")) {
            chunkSizes.push_back(
                    ZL_Input_contentSize(ZL_Edge_getData(inputs[0])));
        }
    }

    std::vector<size_t> chunkSizes;
};

TEST_F(SafetensorsCompressorTest, CapsTheNumberOfSegmentsPerChunk)
{
    const size_t kMaxSegmentsPerChunk = 256; // TC_MAX_SEGMENTS_PER_CHUNK
    const size_t tensorBytes          = 16 << 10;
    std::vector<TensorSpec> tensors;
    tensors.reserve(kMaxSegmentsPerChunk);
    // With the header, one more segment than fits in a chunk.
    for (size_t i = 0; i < kMaxSegmentsPerChunk; ++i) {
        tensors.push_back(
                { "t" + std::to_string(i),
                  "U8",
                  { tensorBytes },
                  std::string(tensorBytes, (char)i) });
    }
    const auto file = buildSafetensors(tensors);
    useChunkSize(size_t(1) << 30);
    ChunkRecorder recorder;
    attachHooks(recorder);

    expectRoundTrip(file);

    EXPECT_EQ(recorder.chunkSizes.size(), 2u);
}

/// Int64 values cycling through blocks of 8: a ramp, a constant, small random
/// values and large irregular steps, for which the generic graph builds ~39
/// nodes per tensor.
std::string int64Blocks(size_t nbElts, uint64_t seed)
{
    uint64_t state = seed * 0x9E3779B97F4A7C15ull + 1;
    auto next      = [&] {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    uint64_t steps = 0;
    std::string out(nbElts * sizeof(uint64_t), '\0');
    for (size_t i = 0; i < nbElts; ++i) {
        uint64_t v;
        switch (i % 32 / 8) {
            case 0:
                v = i * 3;
                break;
            case 1:
                v = 5;
                break;
            case 2:
                v = next() % 256;
                break;
            default:
                v = steps += 10000000 + next() % 30000;
                break;
        }
        std::memcpy(out.data() + i * sizeof(v), &v, sizeof(v));
    }
    return out;
}

TEST_F(SafetensorsCompressorTest, KeepsChunksUnderTheNodeLimit)
{
    // As many segments as 512 per chunk would allow, past
    // ZL_runtimeNodeLimit().
    const size_t nbTensors = 511;
    std::vector<TensorSpec> tensors;
    tensors.reserve(nbTensors);
    for (size_t i = 0; i < nbTensors; ++i) {
        tensors.push_back(
                { "model.layers.0.t" + std::to_string(i),
                  "I64",
                  { 2048 },
                  int64Blocks(2048, i + 1) });
    }
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);

    expectRoundTrip(buildSafetensors(tensors));
}

TEST_F(SafetensorsCompressorTest, SplitsLargeTensorsOnRowBoundaries)
{
    const size_t rowBytes = 500;
    const auto file       = buildSafetensors(
            { { "w", "BF16", { 1000, 250 }, pseudoRandomBytes(500000, 1) } });
    const size_t headerBytes = file.size() - 500000;
    const size_t chunkSize   = 2 * ZL_MIN_CHUNK_SIZE;
    useChunkSize(chunkSize);
    ChunkRecorder recorder;
    attachHooks(recorder);

    expectRoundTrip(file);

    ASSERT_GT(recorder.chunkSizes.size(), 2u);
    EXPECT_EQ((recorder.chunkSizes[0] - headerBytes) % rowBytes, 0u);
    size_t total = 0;
    for (size_t i = 0; i < recorder.chunkSizes.size(); ++i) {
        EXPECT_LE(recorder.chunkSizes[i], chunkSize);
        EXPECT_EQ(i == 0 ? 0 : recorder.chunkSizes[i] % rowBytes, 0u);
        total += recorder.chunkSizes[i];
    }
    EXPECT_EQ(total, file.size());
}

TEST_F(SafetensorsCompressorTest, NeverExceedsTheChunkSizeOnWideRows)
{
    const size_t chunkSize = 2 * ZL_MIN_CHUNK_SIZE;
    useChunkSize(chunkSize);
    // Rows of half a chunk are kept whole, wider ones are split by element.
    for (const size_t rowBytes : { chunkSize / 2, chunkSize / 2 + 2 }) {
        const auto build = [&](size_t firstBytes) {
            return buildSafetensors(
                    { { "a",
                        "U8",
                        { firstBytes },
                        std::string(firstBytes, 'a') },
                      { "w",
                        "BF16",
                        { 5, rowBytes / 2 },
                        pseudoRandomBytes(5 * rowBytes, 1) } });
        };
        // The header and "a" leave the chunk open, just under
        // ZL_MIN_CHUNK_SIZE, when "w" starts.
        const size_t headerBytes = build(30000).size() - 30000 - 5 * rowBytes;
        const size_t firstBytes  = ZL_MIN_CHUNK_SIZE - 1 - headerBytes;
        const auto file          = build(firstBytes);
        ASSERT_EQ(file.size(), headerBytes + firstBytes + 5 * rowBytes);
        ChunkRecorder recorder;
        attachHooks(recorder);

        expectRoundTrip(file);

        ASSERT_FALSE(recorder.chunkSizes.empty());
        EXPECT_GT(recorder.chunkSizes[0], ZL_MIN_CHUNK_SIZE) << rowBytes;
        for (const size_t size : recorder.chunkSizes) {
            EXPECT_LE(size, chunkSize) << rowBytes;
        }
    }
}

std::string causalMaskF32(size_t n)
{
    std::vector<float> mask(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            mask[i * n + j] = j <= i ? 1.0f : 0.0f;
        }
    }
    std::string bytes(mask.size() * sizeof(float), '\0');
    std::memcpy(bytes.data(), mask.data(), bytes.size());
    return bytes;
}

std::string constantBf16(size_t nbElts)
{
    std::string bytes;
    for (size_t i = 0; i < nbElts; ++i) {
        bytes += "\x80\x3f"; // 1.0
    }
    return bytes;
}

TEST_F(SafetensorsCompressorTest, CompressesDegenerateFloatTensorsStrongly)
{
    const auto file = buildSafetensors(
            {
                    { "mask", "F32", { 256, 256 }, causalMaskF32(256) },
                    { "ones", "BF16", { 128, 128 }, constantBf16(128 * 128) },
            });
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);

    EXPECT_LT(expectRoundTrip(file), file.size() / 50);
}

/// Counts the inputs handed to graphs whose name contains @p name.
class GraphCounter : public CompressIntrospectionHooks {
   public:
    explicit GraphCounter(std::string_view name) : name_(name) {}

    void on_migraphEncode_start(
            ZL_Graph*,
            const ZL_Compressor* compressor,
            ZL_GraphID gid,
            ZL_Edge*[],
            size_t) override
    {
        const std::string name = ZL_Compressor_Graph_getName(compressor, gid);
        count += name.find(name_) != std::string::npos;
        seen += name + " ";
    }

    size_t count = 0;
    std::string seen;

   private:
    std::string name_;
};

TEST_F(SafetensorsCompressorTest, SendsOnlyLowCardinalityTensorsToTransformer)
{
    const auto weights = buildSafetensors(
            { { "w", "BF16", { 64, 64 }, pseudoRandomBytes(8192, 1) } });
    const auto mask = buildSafetensors(
            { { "mask", "F32", { 64, 64 }, causalMaskF32(64) } });
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    GraphCounter onWeights("transformer_numeric");
    GraphCounter onMask("transformer_numeric");

    attachHooks(onWeights);
    expectRoundTrip(weights);
    attachHooks(onMask);
    expectRoundTrip(mask);

    EXPECT_EQ(onWeights.count, 0u);
    EXPECT_GT(onMask.count, 0u) << "graphs seen: " << onMask.seen;
}

/// bfloat16 values taking 3 values, as group scales often do.
std::string fewValuesBf16(size_t nbElts)
{
    const uint16_t values[] = { 0x3c80, 0x3d00, 0x3d40 };
    std::string bytes;
    uint32_t state = 1;
    for (size_t i = 0; i < nbElts; ++i) {
        state                = state * 1664525u + 1013904223u;
        const uint16_t value = values[(state >> 16) % 3];
        bytes += (char)(value & 0xff);
        bytes += (char)(value >> 8);
    }
    return bytes;
}

TEST_F(SafetensorsCompressorTest, KeepsGroupScalesAwayFromTransformer)
{
    const auto scales    = fewValuesBf16(32 * 512);
    const auto quantized = buildSafetensors(
            {
                    { "l.qweight",
                      "I32",
                      { 64, 512 },
                      pseudoRandomBytes(131072, 1) },
                    { "l.scales", "BF16", { 32, 512 }, scales },
                    { "m.biases", "BF16", { 32, 512 }, scales },
                    { "n.weight_scale", "BF16", { 32, 512 }, scales },
                    // Mixture-of-experts layers of MLX stack their experts.
                    { "e.switch_mlp.up_proj.scales",
                      "BF16",
                      { 2, 16, 512 },
                      scales },
            });
    const auto weights =
            buildSafetensors({ { "l.weight", "BF16", { 32, 512 }, scales } });
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    GraphCounter onQuantized("transformer_numeric");
    GraphCounter onWeights("transformer_numeric");

    attachHooks(onQuantized);
    expectRoundTrip(quantized);
    attachHooks(onWeights);
    expectRoundTrip(weights);

    EXPECT_EQ(onQuantized.count, 0u) << "graphs seen: " << onQuantized.seen;
    EXPECT_GT(onWeights.count, 0u) << "graphs seen: " << onWeights.seen;
}

/// Counts the runs of @p node.
class NodeCounter : public CompressIntrospectionHooks {
   public:
    explicit NodeCounter(ZL_NodeID node) : node_(node) {}

    void on_codecEncode_start(
            ZL_Encoder*,
            const ZL_Compressor*,
            ZL_NodeID nid,
            const ZL_Input*[],
            size_t) override
    {
        count += nid.nid == node_.nid;
    }

    size_t count = 0;

   private:
    ZL_NodeID node_;
};

TEST_F(SafetensorsCompressorTest, CodesPackedValuesAsBytes)
{
    const auto packed    = pseudoRandomBytes(131072, 1);
    const auto quantized = buildSafetensors(
            {
                    { "l.qweight", "I32", { 64, 512 }, packed },
                    { "m.weight_packed", "I32", { 64, 512 }, packed },
                    // MLX stores packed values as uint32.
                    { "n.weight", "U32", { 64, 512 }, packed },
            });
    const auto integers =
            buildSafetensors({ { "n.weight", "I32", { 64, 512 }, packed } });
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    NodeCounter onQuantized(ZL_NODE_INTERPRET_AS_LE32);
    NodeCounter onIntegers(ZL_NODE_INTERPRET_AS_LE32);

    attachHooks(onQuantized);
    expectRoundTrip(quantized);
    attachHooks(onIntegers);
    expectRoundTrip(integers);

    EXPECT_EQ(onQuantized.count, 0u);
    EXPECT_GT(onIntegers.count, 0u);
}

/// Bytes of a rounded Gaussian: most of the 256 values, with homogeneous
/// statistics, like 8-bit float weights.
std::string gaussianBytes(size_t size)
{
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0, 30);
    std::string bytes(size, '\0');
    for (auto& b : bytes) {
        const double v = std::round(dist(rng));
        b              = (char)(int8_t)std::max(-127.0, std::min(127.0, v));
    }
    return bytes;
}

TEST_F(SafetensorsCompressorTest, CodesFp8TensorsBetterThanBytes)
{
    const auto bytes = gaussianBytes(1 << 20);
    const auto fp8 =
            buildSafetensors({ { "w", "F8_E4M3", { 1024, 1024 }, bytes } });
    const auto u8 = buildSafetensors({ { "w", "U8", { 1024, 1024 }, bytes } });
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);

    EXPECT_LT(expectRoundTrip(fp8), expectRoundTrip(u8));
}

/// float32 Gaussian weights keeping @p zeroBits of their fraction at zero, as
/// weights converted from float16 (0x1fff) or bfloat16 (0xffff) do.
std::string gaussianF32(size_t nbElts, uint32_t zeroBits)
{
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 0.02f);
    std::string bytes(nbElts * 4, '\0');
    for (size_t i = 0; i < nbElts; ++i) {
        const float value = dist(rng);
        uint32_t bits;
        memcpy(&bits, &value, 4);
        bits &= ~zeroBits;
        memcpy(&bytes[4 * i], &bits, 4);
    }
    return bytes;
}

TEST_F(SafetensorsCompressorTest, CodesUpcastFloat32NearTheir16BitSize)
{
    const std::vector<uint64_t> shape = { 64, 512 };
    const auto native                 = buildSafetensors(
            { { "w", "F32", shape, gaussianF32(64 * 512, 0) } });
    const auto fromF16 = buildSafetensors(
            { { "w", "F32", shape, gaussianF32(64 * 512, 0x1fff) } });
    const auto fromBf16 = buildSafetensors(
            { { "w", "F32", shape, gaussianF32(64 * 512, 0xffff) } });
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    NodeCounter onNative(ZL_NODE_TRANSPOSE_SPLIT);
    NodeCounter onUpcast(ZL_NODE_TRANSPOSE_SPLIT);

    attachHooks(onNative);
    const size_t nativeSize = expectRoundTrip(native);
    attachHooks(onUpcast);
    const size_t f16Size  = expectRoundTrip(fromF16);
    const size_t bf16Size = expectRoundTrip(fromBf16);

    EXPECT_EQ(onNative.count, 0u);
    EXPECT_GT(onUpcast.count, 0u);
    EXPECT_LT(f16Size, nativeSize * 2 / 3);
    EXPECT_LT(bf16Size, nativeSize / 2);
}

TEST_F(SafetensorsCompressorTest, CodesFloat32FromBfloat16AsBfloat16)
{
    const std::vector<uint64_t> shape = { 64, 512 };
    const std::string f32             = gaussianF32(64 * 512, 0xffff);
    std::string bf16;
    for (size_t i = 0; i < f32.size(); i += 4) {
        bf16.append(f32, i + 2, 2);
    }
    std::string almost = f32;
    almost[4 * 1000]   = 1;
    // 40 values over a range of 117 bfloat16 values.
    std::string narrow(4 * 256, '\0');
    for (size_t i = 0; i < 256; ++i) {
        const uint32_t bits = (0x3F80u + 3 * (uint32_t)(i % 40)) << 16;
        memcpy(&narrow[4 * i], &bits, 4);
    }
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);

    const size_t bf16Size =
            expectRoundTrip(buildSafetensors({ { "w", "BF16", shape, bf16 } }));
    const size_t f32Size =
            expectRoundTrip(buildSafetensors({ { "w", "F32", shape, f32 } }));
    // A single low bit set sends the tensor to the float32 graph.
    const size_t almostSize = expectRoundTrip(
            buildSafetensors({ { "w", "F32", shape, almost } }));

    // A range below 256 values would not fit the bfloat16 graph.
    expectRoundTrip(buildSafetensors({ { "n", "F32", { 256 }, narrow } }));

    EXPECT_LE(f32Size, bf16Size + 32);
    EXPECT_GT(almostSize, f32Size);
}

/// @p f32 with the 13 low bits of the fraction cleared in rows [first, last) of
/// @p cols values, as rows of float16 weights never updated in float32 keep.
std::string
withFloat16Rows(std::string f32, size_t cols, size_t first, size_t last)
{
    for (size_t i = first * cols; i < last * cols; ++i) {
        uint32_t bits;
        memcpy(&bits, &f32[4 * i], 4);
        bits &= ~0x1fffu;
        memcpy(&f32[4 * i], &bits, 4);
    }
    return f32;
}

TEST_F(SafetensorsCompressorTest, CodesFloat16RowsOfFloat32TensorsApart)
{
    const size_t rows = 256, cols = 512;
    const std::string native = gaussianF32(rows * cols, 0);
    const std::string mixed  = withFloat16Rows(native, cols, 64, 128);
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    NodeCounter onNative(ZL_NODE_DISPATCH);
    NodeCounter onMixed(ZL_NODE_DISPATCH);
    NodeCounter onOneRow(ZL_NODE_DISPATCH);
    NodeCounter onFlat(ZL_NODE_DISPATCH);

    attachHooks(onNative);
    const size_t nativeSize = expectRoundTrip(
            buildSafetensors({ { "w", "F32", { rows, cols }, native } }));
    attachHooks(onMixed);
    const size_t mixedSize = expectRoundTrip(
            buildSafetensors({ { "w", "F32", { rows, cols }, mixed } }));
    // Too few float16 values to pay for the split.
    attachHooks(onOneRow);
    expectRoundTrip(buildSafetensors(
            { { "w",
                "F32",
                { rows, cols },
                withFloat16Rows(native, cols, 0, 1) } }));
    // Without rows from the parser, the tensor is not split.
    attachHooks(onFlat);
    expectRoundTrip(
            buildSafetensors({ { "w", "F32", { rows * cols }, mixed } }));

    EXPECT_EQ(onNative.count, 0u);
    EXPECT_GT(onMixed.count, 0u);
    EXPECT_EQ(onOneRow.count, 0u);
    EXPECT_EQ(onFlat.count, 0u);
    // Most of the 13 zero bits of each float16 value are saved.
    const size_t zeroBytes = 64 * cols * 13 / 8;
    EXPECT_LT(mixedSize, nativeSize - zeroBytes * 9 / 10);
}

TEST_F(SafetensorsCompressorTest, TransposesTheBytesOfFloat16Tensors)
{
    const auto f16 = buildSafetensors(
            { { "w", "F16", { 64, 512 }, pseudoRandomBytes(65536, 1) } });
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    NodeCounter onF16(ZL_NODE_TRANSPOSE_SPLIT);

    attachHooks(onF16);
    expectRoundTrip(f16);

    EXPECT_GT(onF16.count, 0u);
}

/// A format with an 8-byte header followed by a single tensor of 16-bit
/// integers.
struct FakeToken {
    const char* ptr;
    size_t size;
    bool isTensor;
};

ZL_Report lexFakeFormat(
        const void* src,
        size_t srcSize,
        void* tokens,
        size_t,
        ZL_ErrorContext*)
{
    if (srcSize <= 8) {
        return ZL_returnError(ZL_ErrorCode_graph_parser_malformedInput);
    }
    if (tokens != nullptr) {
        const auto* const ptr = static_cast<const char*>(src);
        auto* const out       = static_cast<FakeToken*>(tokens);
        out[0]                = { ptr, 8, false };
        out[1]                = { ptr + 8, srcSize - 8, true };
    }
    return ZL_returnValue(2);
}

ZL_TensorInfo describeFakeToken(const void* token, size_t)
{
    const auto& t = *static_cast<const FakeToken*>(token);
    if (!t.isTensor) {
        return ZL_TensorInfo{ t.ptr, t.size, ZL_TensorClass_header, 1 };
    }
    return ZL_TensorInfo{ t.ptr, t.size, ZL_TensorClass_int16, 2 };
}

const ZL_TensorFormat kFakeFormat = {
    .name             = "fake",
    .defaultChunkSize = ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE,
    .tokenSize        = sizeof(FakeToken),
    .lex              = lexFakeFormat,
    .describe         = describeFakeToken,
};

ZL_Report segmentFakeFormat(ZL_Segmenter* sctx)
{
    return ZL_TensorChunker_segment(sctx, &kFakeFormat);
}

TEST_F(SafetensorsCompressorTest, SendsTagsUnknownToTheCompressorToRaw)
{
    // Like a compressor serialized before ZL_TensorClass_int16 was added.
    const ZL_GraphID successors[2] = { ZL_GRAPH_ZSTD,
                                       ZL_GRAPH_COMPRESS_GENERIC };
    const auto gid                 = ZL_TensorChunker_registerSegmenter(
            compressor_.get(),
            "!Fake Segmenter",
            segmentFakeFormat,
            ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE,
            "!Fake Parser",
            successors,
            2);
    ASSERT_FALSE(ZL_RES_isError(gid));
    compressor_.selectStartingGraph(ZL_RES_value(gid));

    expectRoundTrip(std::string(8, 'h') + pseudoRandomBytes(10000, 3));
}

/// Counts so many tokens that the sizes of their buffers wrap around to a few
/// bytes.
ZL_Report countTooManyFakeTokens(
        const void*,
        size_t,
        void* tokens,
        size_t,
        ZL_ErrorContext*)
{
    if (tokens != nullptr) {
        ADD_FAILURE() << "tokens lexed into buffers of wrapped sizes";
        return ZL_returnValue(0);
    }
    return ZL_returnValue(SIZE_MAX / 4 + 2);
}

const ZL_TensorFormat kTooManyTokensFormat = {
    .name             = "fake",
    .defaultChunkSize = ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE,
    .tokenSize        = sizeof(FakeToken),
    .lex              = countTooManyFakeTokens,
    .describe         = describeFakeToken,
};

ZL_Report segmentTooManyTokens(ZL_Segmenter* sctx)
{
    return ZL_TensorChunker_segment(sctx, &kTooManyTokensFormat);
}

TEST_F(SafetensorsCompressorTest, FallsBackWhenTokenBuffersWouldOverflow)
{
    const ZL_GraphID successors[2] = { ZL_GRAPH_ZSTD,
                                       ZL_GRAPH_COMPRESS_GENERIC };
    const auto gid                 = ZL_TensorChunker_registerSegmenter(
            compressor_.get(),
            "!Fake Segmenter",
            segmentTooManyTokens,
            ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE,
            "!Fake Parser",
            successors,
            2);
    ASSERT_FALSE(ZL_RES_isError(gid));
    compressor_.selectStartingGraph(ZL_RES_value(gid));

    expectRoundTrip(std::string(8, 'h') + pseudoRandomBytes(10000, 3));

    const auto warnings = get_warning_strings(cctx_);
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(warnings[0].second.find("not parsed as fake"), std::string::npos)
            << warnings[0].second;
}

TEST_F(SafetensorsCompressorTest, SerializedCompressorCanBeReloaded)
{
    // Shifts the names of the graphs registered after it, unless anchored.
    ZL_Compressor_registerZstdGraph_withLevel(compressor_.get(), 5);
    useChunkSize(ZL_SAFETENSORS_DEFAULT_CHUNK_SIZE);
    // Debug builds log the dependencies left unresolved.
    ::testing::internal::CaptureStderr();
    const auto reloaded =
            createCompressorFromSerialized(compressor_.serialize(), "");
    const std::string log = ::testing::internal::GetCapturedStderr();
    CCtx cctx;
    cctx.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    cctx.refCompressor(*reloaded);
    const auto input = buildSafetensors(mixedDtypeTensors());
    EXPECT_EQ(dctx_.decompressSerial(cctx.compressSerial(input)), input);
    EXPECT_EQ(log.find("unresolved"), std::string::npos) << log;
}

TEST(TensorGraphsTest, Scale16GraphIsRegisteredOncePerCompressor)
{
    Compressor compressor;
    const auto first  = ZL_TensorGraphs_registerScale16Graph(compressor.get());
    const auto second = ZL_TensorGraphs_registerScale16Graph(compressor.get());
    ASSERT_FALSE(ZL_RES_isError(first));
    ASSERT_FALSE(ZL_RES_isError(second));
    EXPECT_EQ(ZL_RES_value(first).gid, ZL_RES_value(second).gid);
}

} // namespace
} // namespace openzl::custom_parsers
