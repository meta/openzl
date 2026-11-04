// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "custom_parsers/dependency_registration.h"
#include "custom_parsers/numpy/numpy_lexer.h"
#include "custom_parsers/numpy/numpy_parser.h"
#include "custom_parsers/numpy/tests/numpy_builder.h"
#include "custom_parsers/tests/pickle_builder.h"
#include "openzl/cpp/CompressIntrospectionHooks.hpp"
#include "openzl/openzl.hpp"

namespace openzl::custom_parsers {
namespace {

using testing::buildNpy;
using testing::buildNpyWithDict;
using testing::buildNpz;
using testing::buildZip;
using testing::NpyArraySpec;
using testing::pickleArrays;
using testing::PickledArray;
using testing::pseudoRandomBytes;
using testing::ZipMember;

struct LexedToken {
    ZL_NumPyTokenType type;
    size_t offset;
    size_t size;
    std::string name;
    char kind;
    uint32_t itemSize;
    bool fortranOrder;
    std::vector<uint64_t> shape;

    bool operator==(const LexedToken&) const = default;
};

LexedToken nonArray(ZL_NumPyTokenType type, size_t offset, size_t size)
{
    return { type, offset, size, "", 0, 0, false, {} };
}

struct LexResult {
    ZL_ErrorCode error;
    std::vector<LexedToken> tokens;
};

LexResult lex(std::string_view file)
{
    const ZL_Report count =
            ZL_NumPy_lex(file.data(), file.size(), nullptr, 0, nullptr);
    if (ZL_isError(count)) {
        return { ZL_errorCode(count), {} };
    }
    std::vector<ZL_NumPyToken> out(ZL_validResult(count));
    const ZL_Report nbTokens = ZL_NumPy_lex(
            file.data(), file.size(), out.data(), out.size(), nullptr);
    if (ZL_isError(nbTokens)) {
        return { ZL_errorCode(nbTokens), {} };
    }
    LexResult result{ ZL_ErrorCode_no_error, {} };
    for (size_t i = 0; i < ZL_validResult(nbTokens); ++i) {
        const auto& t       = out[i];
        const size_t offset = size_t(t.ptr - file.data());
        if (t.type != ZL_NumPyTokenType_array) {
            result.tokens.push_back(nonArray(t.type, offset, t.size));
            continue;
        }
        result.tokens.push_back(
                { t.type,
                  offset,
                  t.size,
                  t.name ? std::string(t.name, t.nameSize) : "",
                  t.kind,
                  t.itemSize,
                  t.fortranOrder,
                  { t.shape, t.shape + t.ndims } });
    }
    return result;
}

/// @returns the only array of the .npy file of @p array.
LexedToken lexArray(const NpyArraySpec& array)
{
    const auto result = lex(buildNpy(array));
    EXPECT_EQ(result.error, ZL_ErrorCode_no_error) << array.descr;
    for (const auto& t : result.tokens) {
        if (t.type == ZL_NumPyTokenType_array) {
            return t;
        }
    }
    ADD_FAILURE() << "no array for " << array.descr;
    return {};
}

TEST(NumPyLexerTest, ReadsTheArrayOfANpyFile)
{
    const auto data  = pseudoRandomBytes(24, 1);
    const auto file  = buildNpy({ "<f4", { 2, 3 }, data });
    const size_t hdr = file.size() - data.size();
    const std::vector<LexedToken> expected = {
        nonArray(ZL_NumPyTokenType_header, 0, hdr),
        { ZL_NumPyTokenType_array, hdr, 24, "", 'f', 4, false, { 2, 3 } },
    };

    const auto result = lex(file);

    ASSERT_EQ(result.error, ZL_ErrorCode_no_error);
    EXPECT_EQ(hdr % 64, 0u);
    EXPECT_EQ(result.tokens, expected);
}

TEST(NumPyLexerTest, LeavesTheBytesPastTheArrayInAGap)
{
    const auto file  = buildNpy({ "<i2", { 4 }, std::string(8 + 5, 'x') });
    const size_t hdr = file.size() - 13;
    const std::vector<LexedToken> expected = {
        nonArray(ZL_NumPyTokenType_header, 0, hdr),
        { ZL_NumPyTokenType_array, hdr, 8, "", 'i', 2, false, { 4 } },
        nonArray(ZL_NumPyTokenType_gap, hdr + 8, 5),
    };

    EXPECT_EQ(lex(file).tokens, expected);
}

TEST(NumPyLexerTest, ReadsHeadersOfEveryVersionAndStyle)
{
    const std::string data(24, 'x');
    const std::vector<std::string> dicts = {
        "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
        "{'shape': (2, 3), 'descr': '<f4', 'fortran_order': False}",
        "{\"descr\": \"<f4\", \"fortran_order\": False, \"shape\": (2, 3)}",
        "{'descr':'<f4','fortran_order':False,'shape':(2L, 3L),}", // Python 2
        "{ 'descr' : '<f4' , 'fortran_order' : False , 'shape' : ( 2 , 3 ) }",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), "
        "'extra': [1, (2, 'x'), {'y': None}]}",
        // a repeated key takes its last value, as in Python
        "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), "
        "'fortran_order': False}",
    };
    for (const auto& dict : dicts) {
        for (const uint8_t major : { 1, 2, 3 }) {
            const auto result = lex(buildNpyWithDict(dict, data, major));
            ASSERT_EQ(result.error, ZL_ErrorCode_no_error)
                    << dict << " v" << int(major);
            ASSERT_EQ(result.tokens.size(), 2u);
            EXPECT_EQ(result.tokens[1].kind, 'f') << dict;
            EXPECT_FALSE(result.tokens[1].fortranOrder) << dict;
            EXPECT_EQ(result.tokens[1].shape, (std::vector<uint64_t>{ 2, 3 }));
        }
    }
}

TEST(NumPyLexerTest, ReadsTheKindAndSizeOfLittleEndianDescrs)
{
    struct Case {
        std::string descr;
        char kind;
        uint32_t itemSize;
    };
    const std::vector<Case> cases = {
        { "<f4", 'f', 4 },
        { "<f2", 'f', 2 },
        { "<f8", 'f', 8 },
        { "|u1", 'u', 1 },
        { "|i1", 'i', 1 },
        { "|b1", 'b', 1 },
        { "<u2", 'u', 2 },
        { "<i8", 'i', 8 },
        { "<c8", 'c', 8 },
        { "<c16", 'c', 16 },
        { "<M8[ns]", 'M', 8 },
        { "<m8[s]", 'm', 8 },
        { "<V2", 'V', 2 },
        { "|V2", 'V', 2 },
        { "<U5", 'U', 20 },
        { "|S10", 'S', 10 },
        // not little-endian numbers
        { ">f4", 0, 0 },
        { "=f4", 0, 0 },
        { "|O", 0, 0 },
        { "|O8", 0, 0 },
        { "float32", 0, 0 },
        { "<f", 0, 0 },
        { "<M8[ns", 0, 0 },
        { "[('x', '<f4'), ('y', '<i4')]", 0, 0 },
    };
    for (const auto& c : cases) {
        const auto t = lexArray({ c.descr, { 3 }, std::string(60, 'x') });
        EXPECT_EQ(t.kind, c.kind) << c.descr;
        if (c.kind != 0) {
            EXPECT_EQ(t.itemSize, c.itemSize) << c.descr;
            EXPECT_EQ(t.size, 3 * c.itemSize) << c.descr;
        } else {
            EXPECT_EQ(t.size, 60u) << c.descr;
        }
    }
}

TEST(NumPyLexerTest, ReportsArraysItCannotDescribeWithKindZero)
{
    const std::vector<uint64_t> nineDims(ZL_NUMPY_MAX_DIMS + 1, 1);

    const auto truncated   = lexArray({ "<f4", { 100 }, std::string(40, 'x') });
    const auto tooManyDims = lexArray({ "<f4", nineDims, std::string(4, 'x') });
    const auto overflow    = lexArray(
            { "<f8", { 1ull << 40, 1ull << 40 }, std::string(8, 'x') });

    EXPECT_EQ(truncated.kind, 0);
    EXPECT_EQ(truncated.size, 40u);
    EXPECT_EQ(tooManyDims.kind, 0);
    EXPECT_EQ(overflow.kind, 0);
}

TEST(NumPyLexerTest, ReadsFortranOrderAndEmptyArrays)
{
    const auto fortran =
            lexArray({ "<f4", { 2, 3 }, std::string(24, 'x'), true });
    const auto empty = lex(buildNpy({ "<f4", { 0, 3 }, "" }));

    EXPECT_TRUE(fortran.fortranOrder);
    ASSERT_EQ(empty.error, ZL_ErrorCode_no_error);
    ASSERT_EQ(empty.tokens.size(), 1u);
    EXPECT_EQ(empty.tokens[0].type, ZL_NumPyTokenType_header);
}

TEST(NumPyLexerTest, ReadsTheStoredNpyMembersOfArchives)
{
    const auto a    = pseudoRandomBytes(16, 1);
    const auto b    = pseudoRandomBytes(6, 2);
    const auto npyA = buildNpy({ "<f4", { 4 }, a });
    const auto npyB = buildNpy({ "|u1", { 2, 3 }, b });
    // A .npy file in a deflated member is only deflate bytes that look alike
    const auto deflate = buildNpy({ "<f4", { 4 }, std::string(16, 'c') });
    const auto other   = pseudoRandomBytes(30, 4);
    const auto file    = buildZip(
            {
                    { "a.npy", npyA },
                    { "b.npy", npyB },
                    { "c.npy", deflate, 8 },
                    { "d.txt", other },
            });
    const size_t offA = file.find(npyA) + npyA.size() - a.size();
    const size_t offB = file.find(npyB) + npyB.size() - b.size();
    const size_t offC = file.find(deflate);
    const size_t offD = file.find(other);
    const size_t tail = offD + other.size();
    const std::vector<LexedToken> expected = {
        nonArray(ZL_NumPyTokenType_header, 0, offA),
        { ZL_NumPyTokenType_array, offA, 16, "a.npy", 'f', 4, false, { 4 } },
        nonArray(ZL_NumPyTokenType_header, offA + 16, offB - offA - 16),
        { ZL_NumPyTokenType_array, offB, 6, "b.npy", 'u', 1, false, { 2, 3 } },
        nonArray(ZL_NumPyTokenType_header, offB + 6, offC - offB - 6),
        nonArray(ZL_NumPyTokenType_gap, offC, deflate.size()),
        nonArray(
                ZL_NumPyTokenType_header,
                offC + deflate.size(),
                offD - offC - deflate.size()),
        nonArray(ZL_NumPyTokenType_gap, offD, 30),
        nonArray(ZL_NumPyTokenType_header, tail, file.size() - tail),
    };

    const auto result = lex(file);

    ASSERT_EQ(result.error, ZL_ErrorCode_no_error);
    EXPECT_EQ(result.tokens, expected);
}

TEST(NumPyLexerTest, CountsTokensWithoutWriting)
{
    const auto file = buildNpz(
            {
                    { "a", { "<f4", { 4 }, std::string(16, 'a') } },
                    { "b", { "<f4", { 4 }, std::string(16, 'b') } },
            });

    const ZL_Report count =
            ZL_NumPy_lex(file.data(), file.size(), nullptr, 0, nullptr);

    ASSERT_FALSE(ZL_isError(count));
    EXPECT_EQ(ZL_validResult(count), 5u);
    EXPECT_EQ(lex(file).tokens.size(), 5u);
}

TEST(NumPyLexerTest, RejectsInputsThatAreNeitherNpyNorNpz)
{
    const auto valid = buildNpy({ "<f4", { 4 }, std::string(16, 'x') });
    const std::string data(16, 'x');
    auto badVersion = valid;
    badVersion[6]   = 4;
    auto longHeader = valid;
    longHeader[9]   = (char)0x7f;
    const auto npz  = buildNpz({ { "a", { "<f4", { 4 }, data } } });
    const std::vector<std::string> malformed = {
        "",
        pseudoRandomBytes(1000, 1),
        valid.substr(0, 9),
        valid.substr(0, 40),
        badVersion,
        longHeader,
        buildNpyWithDict("{'descr': '<f4', 'shape': (4,)}", data),
        buildNpyWithDict(
                "{'descr': '<f4', 'fortran_order': 0, 'shape': (4,)}", data),
        buildNpyWithDict("{'descr': '<f4' 'fortran_order': False}", data),
        buildNpyWithDict("['<f4', False, (4,)]", data),
        buildNpyWithDict(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (4,)} x",
                data),
        npz.substr(0, npz.size() - 10),
    };
    for (const auto& file : malformed) {
        EXPECT_EQ(lex(file).error, ZL_ErrorCode_graph_parser_malformedInput)
                << "input of " << file.size() << " bytes";
    }
}

TEST(NumPyLexerTest, RejectsArchivesWithTooManyMembers)
{
    const std::vector<ZipMember> members(ZL_NUMPY_MAX_MEMBERS + 1, { "m", "" });

    EXPECT_EQ(
            lex(buildZip(members)).error,
            ZL_ErrorCode_graph_parser_unhandledInput);
}

TEST(NumPyLexerTest, ReadsTheArraysOfPickles)
{
    const PickledArray f4{ "f4", { 64, 128 }, pseudoRandomBytes(32768, 1) };
    const PickledArray u2{ "u2", { 10000 }, pseudoRandomBytes(20000, 2) };
    // Smaller than ZL_NUMPY_MIN_PICKLED_ARRAY_SIZE: stays in the opcodes
    const PickledArray small{ "f4", { 4 }, pseudoRandomBytes(16, 3) };
    const auto file =
            pickleArrays({ { "a", f4 }, { "b", u2 }, { "c", small } }, 4);
    const size_t offA                      = file.find(f4.data);
    const size_t offB                      = file.find(u2.data);
    const size_t endB                      = offB + u2.data.size();
    const std::vector<LexedToken> expected = {
        nonArray(ZL_NumPyTokenType_header, 0, offA),
        { ZL_NumPyTokenType_array,
          offA,
          32768,
          "",
          'f',
          4,
          false,
          { 64, 128 } },
        nonArray(ZL_NumPyTokenType_header, offA + 32768, offB - offA - 32768),
        { ZL_NumPyTokenType_array, offB, 20000, "", 'u', 2, false, { 10000 } },
        nonArray(ZL_NumPyTokenType_header, endB, file.size() - endB),
    };

    const auto result = lex(file);

    ASSERT_EQ(result.error, ZL_ErrorCode_no_error);
    EXPECT_EQ(result.tokens, expected);
}

TEST(NumPyLexerTest, ReportsPickledArraysItCannotDescribeWithKindZero)
{
    const auto data = pseudoRandomBytes(32768, 1);
    PickledArray records{ "V8", { 4096 }, data, '|' };
    records.fieldNames                     = { "x", "y" };
    const std::vector<PickledArray> arrays = {
        { "f4", { 8192 }, data, '>' },
        { "f8", { 8192 }, data }, // 4096 elements of 8 bytes
        records,
    };
    for (const auto& array : arrays) {
        const auto result = lex(pickleArrays({ { "a", array } }, 3));
        ASSERT_EQ(result.error, ZL_ErrorCode_no_error);
        ASSERT_EQ(result.tokens.size(), 3u) << array.typeStr;
        EXPECT_EQ(result.tokens[1].type, ZL_NumPyTokenType_array);
        EXPECT_EQ(result.tokens[1].size, data.size());
        EXPECT_EQ(result.tokens[1].kind, 0) << array.typeStr;
    }
}

TEST(NumPyLexerTest, LeavesTheBytesPastAPickleInAGap)
{
    const auto pickle =
            pickleArrays({ { "a", { "u1", { 3 }, std::string(3, 'x') } } }, 2);
    const auto file                        = pickle + "not a pickle";
    const std::vector<LexedToken> expected = {
        nonArray(ZL_NumPyTokenType_header, 0, pickle.size()),
        nonArray(ZL_NumPyTokenType_gap, pickle.size(), 12),
    };

    EXPECT_EQ(lex(file).tokens, expected);
}

/// Arrays of every class, and arrays coded as raw bytes.
std::string mixedArchive()
{
    std::vector<ZipMember> members;
    const std::vector<NpyArraySpec> arrays = {
        { "<f4", { 64, 48 }, pseudoRandomBytes(12288, 1) },
        { "<f2", { 3000 }, pseudoRandomBytes(6000, 2) },
        { "<f8", {}, pseudoRandomBytes(8, 3) },
        { "<i8", { 30 }, pseudoRandomBytes(240, 4) },
        { "<u4", { 30 }, pseudoRandomBytes(120, 5) },
        { "<i2", { 30 }, pseudoRandomBytes(60, 6) },
        { "|u1", { 64, 64 }, pseudoRandomBytes(4096, 7) },
        { "|b1", { 4096 }, std::string(4096, '\1') },
        { "<c8", { 20 }, pseudoRandomBytes(160, 8) },
        { "<M8[us]", { 20 }, pseudoRandomBytes(160, 9) },
        { "<V2", { 100 }, pseudoRandomBytes(200, 10) },
        { "<f4", { 30, 20 }, pseudoRandomBytes(2400, 11), true },
        { "<U3", { 10 }, pseudoRandomBytes(120, 12) },
        { ">f4", { 10 }, pseudoRandomBytes(40, 13) },
        { "<f4", { 100 }, pseudoRandomBytes(40, 14) },
    };
    members.reserve(arrays.size() + 1);
    for (size_t i = 0; i < arrays.size(); ++i) {
        members.push_back(
                { "arr_" + std::to_string(i) + ".npy", buildNpy(arrays[i]) });
    }
    members.push_back({ "deflated.npy", pseudoRandomBytes(500, 15), 8 });
    return buildZip(members);
}

class NumPyCompressorTest : public ::testing::Test {
   protected:
    void useChunkSize(size_t chunkSize)
    {
        const auto gid = ZL_NumPy_registerGraph(compressor_.get(), chunkSize);
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
        cctx_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        cctx_.refCompressor(compressor_);
        const auto compressed = cctx_.compressSerial(input);
        EXPECT_EQ(dctx_.decompressSerial(compressed), input);
        return compressed.size();
    }

    Compressor compressor_;
    CCtx cctx_;
    DCtx dctx_;
};

TEST_F(NumPyCompressorTest, RoundTripsNpyFilesAndArchives)
{
    useChunkSize(ZL_NUMPY_DEFAULT_CHUNK_SIZE);
    expectRoundTrip(mixedArchive());
    expectRoundTrip(
            buildNpy({ "<f4", { 64, 48 }, pseudoRandomBytes(12288, 1) }));
    expectRoundTrip(buildNpy({ "<c16", { 7 }, pseudoRandomBytes(112, 2) }, 2));
}

/// A pickle of arrays of several classes, small and large.
std::string mixedPickle(int protocol, bool python2 = false)
{
    return pickleArrays(
            {
                    { "f4",
                      { "f4", { 64, 128 }, pseudoRandomBytes(32768, 1) } },
                    { "i8", { "i8", { 3000 }, pseudoRandomBytes(24000, 2) } },
                    { "u1", { "u1", { 20000 }, pseudoRandomBytes(20000, 3) } },
                    { "small", { "f8", { 3 }, pseudoRandomBytes(24, 4) } },
                    { "be",
                      { "f4", { 5000 }, pseudoRandomBytes(20000, 5), '>' } },
                    { "fortran",
                      { "f4",
                        { 100, 50 },
                        pseudoRandomBytes(20000, 6),
                        '<',
                        true } },
            },
            protocol,
            python2);
}

TEST_F(NumPyCompressorTest, RoundTripsPickles)
{
    useChunkSize(ZL_NUMPY_DEFAULT_CHUNK_SIZE);
    for (const int protocol : { 2, 3, 4, 5 }) {
        expectRoundTrip(mixedPickle(protocol));
    }
    const auto python2 = mixedPickle(2, true);
    expectRoundTrip(python2);
    expectRoundTrip(python2.substr(0, python2.size() / 2));
    expectRoundTrip(python2 + mixedPickle(5) + "not a pickle");
}

TEST_F(NumPyCompressorTest, RoundTripsWithChunksSmallerThanArrays)
{
    useChunkSize(1);
    expectRoundTrip(
            buildNpy({ "<f4", { 1024, 1024 }, pseudoRandomBytes(1 << 22, 1) }));
    expectRoundTrip(buildNpy(
            { "<f4", { 1024, 1024 }, pseudoRandomBytes(1 << 22, 2), true }));
}

TEST_F(NumPyCompressorTest, RoundTripsInvalidInputs)
{
    useChunkSize(ZL_NUMPY_DEFAULT_CHUNK_SIZE);
    const auto valid = mixedArchive();
    expectRoundTrip("");
    expectRoundTrip(pseudoRandomBytes(10000, 7));
    expectRoundTrip(valid.substr(0, valid.size() / 2));
}

TEST_F(NumPyCompressorTest, ReportsWhyInvalidInputsAreNotParsed)
{
    useChunkSize(ZL_NUMPY_DEFAULT_CHUNK_SIZE);

    expectRoundTrip(pseudoRandomBytes(10000, 7));

    const auto warnings = get_warning_strings(cctx_);
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(
            warnings[0].second.find("Input not parsed as NumPy"),
            std::string::npos)
            << warnings[0].second;
}

TEST_F(NumPyCompressorTest, ValidInputsProduceNoWarning)
{
    useChunkSize(ZL_NUMPY_DEFAULT_CHUNK_SIZE);

    expectRoundTrip(mixedArchive());

    EXPECT_TRUE(get_warning_strings(cctx_).empty());
}

/// Counts the runs of @p graph.
class GraphCounter : public CompressIntrospectionHooks {
   public:
    explicit GraphCounter(ZL_GraphID graph) : graph_(graph) {}

    void on_migraphEncode_start(
            ZL_Graph*,
            const ZL_Compressor*,
            ZL_GraphID gid,
            ZL_Edge*[],
            size_t) override
    {
        count += gid.gid == graph_.gid;
    }

    size_t count = 0;

   private:
    ZL_GraphID graph_;
};

TEST_F(NumPyCompressorTest, SendsArraysOfNumbersToTheNumericTransformer)
{
    const auto bytes   = pseudoRandomBytes(1 << 16, 1);
    const auto numbers = buildNpy({ "<i4", { 128, 128 }, bytes });
    const auto strings = buildNpy({ "<U4", { 128, 32 }, bytes });
    useChunkSize(ZL_NUMPY_DEFAULT_CHUNK_SIZE);
    GraphCounter onNumbers(ZL_GRAPH_TRANSFORMER_NUMERIC);
    GraphCounter onStrings(ZL_GRAPH_TRANSFORMER_NUMERIC);

    attachHooks(onNumbers);
    expectRoundTrip(numbers);
    attachHooks(onStrings);
    expectRoundTrip(strings);

    EXPECT_GT(onNumbers.count, 0u);
    EXPECT_EQ(onStrings.count, 0u);
}

TEST_F(NumPyCompressorTest, SendsPickledArraysToTheNumericTransformer)
{
    const auto pickle = pickleArrays(
            { { "a", { "i4", { 128, 128 }, pseudoRandomBytes(1 << 16, 1) } } },
            4);
    useChunkSize(ZL_NUMPY_DEFAULT_CHUNK_SIZE);
    GraphCounter counter(ZL_GRAPH_TRANSFORMER_NUMERIC);

    attachHooks(counter);
    expectRoundTrip(pickle);

    EXPECT_GT(counter.count, 0u);
    EXPECT_TRUE(get_warning_strings(cctx_).empty());
}

TEST_F(NumPyCompressorTest, SerializedCompressorCanBeReloaded)
{
    // Shifts the names of the graphs registered after it, unless anchored.
    ZL_Compressor_registerZstdGraph_withLevel(compressor_.get(), 5);
    useChunkSize(ZL_NUMPY_DEFAULT_CHUNK_SIZE);
    // Debug builds log the dependencies left unresolved.
    ::testing::internal::CaptureStderr();
    const auto reloaded =
            createCompressorFromSerialized(compressor_.serialize(), "");
    const std::string log = ::testing::internal::GetCapturedStderr();
    ASSERT_NE(reloaded, nullptr);
    EXPECT_EQ(log.find("unresolved"), std::string::npos) << log;

    const auto file = mixedArchive();
    cctx_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    cctx_.refCompressor(*reloaded);
    EXPECT_EQ(dctx_.decompressSerial(cctx_.compressSerial(file)), file);
}

} // namespace
} // namespace openzl::custom_parsers
