// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "custom_parsers/pickle_lexer.h"
#include "custom_parsers/tests/pickle_builder.h"

namespace openzl::custom_parsers {
namespace {

using testing::pickleArrays;
using testing::PickledArray;
using testing::PickleWriter;

struct LexedToken {
    ZL_PickleTokenType type;
    size_t offset;
    size_t size;
    std::string typeStr;
    char byteOrder;
    bool fortranOrder;
    std::vector<uint64_t> shape;

    bool operator==(const LexedToken&) const = default;
};

LexedToken opcodes(size_t offset, size_t size)
{
    return { ZL_PickleTokenType_opcodes, offset, size, "", 0, false, {} };
}

LexedToken trailing(size_t offset, size_t size)
{
    return { ZL_PickleTokenType_trailing, offset, size, "", 0, false, {} };
}

/// The token of @p array, whose bytes are found in @p file.
LexedToken arrayIn(std::string_view file, const PickledArray& array)
{
    return { ZL_PickleTokenType_array,
             file.find(array.data),
             array.data.size(),
             array.typeStr,
             array.byteOrder,
             array.fortranOrder,
             array.shape };
}

std::vector<LexedToken> lex(std::string_view file, size_t capacity = 8)
{
    ZL_PickleLexer lexer;
    ZL_PickleLexer_init(&lexer, file.data(), file.size());
    std::vector<ZL_PickleToken> out(capacity);
    std::vector<LexedToken> tokens;
    while (!ZL_PickleLexer_finished(&lexer)) {
        const size_t n = ZL_PickleLexer_lex(&lexer, out.data(), out.size());
        for (size_t i = 0; i < n; ++i) {
            const auto& t       = out[i];
            const size_t offset = size_t(t.ptr - file.data());
            if (t.type != ZL_PickleTokenType_array) {
                tokens.push_back({ t.type, offset, t.size, "", 0, false, {} });
                continue;
            }
            tokens.push_back(
                    { t.type,
                      offset,
                      t.size,
                      std::string(t.typeStr, t.typeStrSize),
                      t.byteOrder,
                      t.fortranOrder,
                      { t.shape, t.shape + t.ndims } });
        }
    }
    return tokens;
}

/// The tokens of a pickle holding @p arrays, in order, with opcodes between.
std::vector<LexedToken> expectedTokens(
        std::string_view file,
        const std::vector<PickledArray>& arrays)
{
    std::vector<LexedToken> expected;
    size_t pos = 0;
    for (const auto& a : arrays) {
        const auto t = arrayIn(file, a);
        expected.push_back(opcodes(pos, t.offset - pos));
        expected.push_back(t);
        pos = t.offset + t.size;
    }
    expected.push_back(opcodes(pos, file.size() - pos));
    return expected;
}

std::string randomBytes(size_t size, uint32_t seed)
{
    std::string out(size, '\0');
    uint32_t state = seed * 2654435761u + 1;
    for (auto& c : out) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        c = (char)state;
    }
    return out;
}

/// Arrays of several dtypes and shapes, two of them of the same dtype, which
/// the second one gets from the memo.
std::vector<PickledArray> someArrays()
{
    return {
        { "f4", { 2, 3 }, randomBytes(24, 1) },
        { "u1", { 300 }, randomBytes(300, 2), '|' },
        { "i8", {}, randomBytes(8, 3) },
        { "f8", { 2, 2 }, randomBytes(32, 4), '<', true },
        { "f4", { 2, 1, 1, 2 }, randomBytes(16, 5) },
    };
}

std::string pickleOf(
        const std::vector<PickledArray>& arrays,
        int protocol,
        bool python2 = false)
{
    std::vector<std::pair<std::string, PickledArray>> named;
    named.reserve(arrays.size());
    for (size_t i = 0; i < arrays.size(); ++i) {
        named.emplace_back("a" + std::to_string(i), arrays[i]);
    }
    return pickleArrays(named, protocol, python2);
}

TEST(PickleLexerTest, FindsTheArraysOfProtocols3To5)
{
    const auto arrays = someArrays();
    for (const int protocol : { 3, 4, 5 }) {
        const auto file = pickleOf(arrays, protocol);
        EXPECT_EQ(lex(file), expectedTokens(file, arrays))
                << "protocol " << protocol;
    }
}

TEST(PickleLexerTest, FindsTheArraysOfPython2Pickles)
{
    const auto arrays = someArrays();
    const auto file   = pickleOf(arrays, 2, true);

    EXPECT_EQ(lex(file), expectedTokens(file, arrays));
}

/// Below protocol 3, Python 3 writes the bytes of arrays as text.
TEST(PickleLexerTest, MissesTheArraysOfPython3PicklesOfProtocol2)
{
    const auto file = pickleOf(someArrays(), 2);

    EXPECT_EQ(lex(file), expectedTokens(file, {}));
}

TEST(PickleLexerTest, LexesTheSameWithAnyCapacity)
{
    const auto file = pickleOf(someArrays(), 4);

    EXPECT_EQ(lex(file, 2), lex(file, 64));
}

TEST(PickleLexerTest, GetsDtypesFromTheMemoPastManyOtherObjects)
{
    const PickledArray a{ "f4", { 4 }, randomBytes(16, 1) };
    const PickledArray b{ "f4", { 8 }, randomBytes(32, 2) };
    PickleWriter w(4);
    w.dictBegin();
    w.str("a");
    w.array(a);
    for (int i = 0; i < 1000; ++i) {
        w.str("key" + std::to_string(i));
        w.str("value" + std::to_string(i));
    }
    w.str("b");
    w.array(b);
    w.dictEnd();
    const auto file = w.finish();

    EXPECT_EQ(lex(file), expectedTokens(file, { a, b }));
}

TEST(PickleLexerTest, KeepsDtypesInUseAmongManyOthers)
{
    // Arrays of a dtype in use, between arrays of new dtypes, more than the
    // memo of the lexer holds
    std::vector<PickledArray> arrays;
    for (int i = 1; i <= 3 * ZL_PICKLE_MEMO_DTYPES; ++i) {
        arrays.push_back({ "f4", { 4 }, randomBytes(16, 2 * i) });
        arrays.push_back({ "f8", { 2 }, randomBytes(16, 2 * i + 1) });
        arrays.back().dtypeId = i;
    }
    const auto file = pickleOf(arrays, 3);

    EXPECT_EQ(lex(file), expectedTokens(file, arrays));
}

TEST(PickleLexerTest, ReportsDtypesWithFieldsWithoutByteOrder)
{
    PickledArray records{ "V8", { 4 }, randomBytes(32, 1), '|' };
    records.fieldNames = { "x", "y" };
    const auto file    = pickleOf({ records }, 3);
    records.byteOrder  = 0;

    EXPECT_EQ(lex(file), expectedTokens(file, { records }));
}

TEST(PickleLexerTest, LexesConcatenatedPickles)
{
    const PickledArray a{ "f4", { 4 }, randomBytes(16, 1) };
    const PickledArray b{ "i2", { 3 }, randomBytes(6, 2) };
    const auto file = pickleOf({ a }, 4) + pickleOf({ b }, 2, true);

    EXPECT_EQ(lex(file), expectedTokens(file, { a, b }));
}

TEST(PickleLexerTest, LeavesWhatFollowsTheLastPickleInATrailingToken)
{
    const PickledArray a{ "f4", { 4 }, randomBytes(16, 1) };
    const auto pickle = pickleOf({ a }, 3);
    const auto file   = pickle + "not a pickle";
    auto expected     = expectedTokens(pickle, { a });
    expected.push_back(trailing(pickle.size(), 12));

    EXPECT_EQ(lex(file), expected);
}

TEST(PickleLexerTest, StopsAtTheFirstOpcodeItCannotRead)
{
    const PickledArray a{ "f4", { 4 }, randomBytes(16, 1) };
    const PickledArray b{ "f4", { 100 }, randomBytes(400, 2) };
    const auto pickle = pickleOf({ a, b }, 3);
    // The bytes of b are cut short; their BINBYTES opcode takes 5 bytes.
    const size_t cut                       = pickle.find(b.data) + 200;
    const auto truncated                   = pickle.substr(0, cut);
    const size_t binbytes                  = pickle.find(b.data) - 5;
    const auto aToken                      = arrayIn(pickle, a);
    const size_t afterA                    = aToken.offset + aToken.size;
    auto unknownOpcode                     = pickle;
    unknownOpcode[binbytes]                = '\xff';
    const std::vector<LexedToken> expected = {
        opcodes(0, aToken.offset),
        aToken,
        opcodes(afterA, binbytes - afterA),
        trailing(binbytes, cut - binbytes),
    };

    EXPECT_EQ(lex(truncated), expected);
    EXPECT_EQ(
            lex(unknownOpcode).back(),
            trailing(binbytes, pickle.size() - binbytes));
}

TEST(PickleLexerTest, CoversPicklesWithoutArraysWithOneToken)
{
    PickleWriter w(4);
    w.dictBegin();
    for (int i = 0; i < 100; ++i) {
        w.str("key" + std::to_string(i));
        w.integer(i * 1000003);
    }
    w.dictEnd();
    const auto file = w.finish();

    EXPECT_EQ(lex(file), (std::vector<LexedToken>{ opcodes(0, file.size()) }));
}

TEST(PickleLexerTest, FindsArraysAfterLongFramesAndDeepNesting)
{
    const PickledArray a{ "f4", { 4 }, randomBytes(16, 1) };
    const PickledArray b{ "f8", { 2 }, randomBytes(16, 2) };
    PickleWriter w(3);
    w.listBegin();
    for (int i = 0; i < 10000; ++i) {
        w.integer(i);
    }
    w.array(a);
    // Deeper than the marks the lexer tracks: it stops there
    for (int depth = 0; depth < ZL_PICKLE_MAX_MARKS + 10; ++depth) {
        w.listBegin();
    }
    w.array(b);
    const auto file = w.finish();

    const auto tokens = lex(file);

    ASSERT_GE(tokens.size(), 3u);
    EXPECT_EQ(tokens[1], arrayIn(file, a));
    EXPECT_EQ(tokens.back().type, ZL_PickleTokenType_trailing);
    EXPECT_EQ(tokens.back().offset + tokens.back().size, file.size());
}

TEST(PickleLexerTest, RecognizesPicklesOfProtocols2To5)
{
    for (const std::string_view src : { "\x80\x02", "\x80\x05}." }) {
        EXPECT_TRUE(ZL_PickleLexer_isPickle(src.data(), src.size())) << src;
    }
    for (const std::string_view src : { std::string_view(""),
                                        std::string_view("\x80"),
                                        std::string_view("\x80\x01", 2),
                                        std::string_view("\x80\x06", 2),
                                        std::string_view("(dp0\n."),
                                        std::string_view("PK\x03\x04") }) {
        EXPECT_FALSE(ZL_PickleLexer_isPickle(src.data(), src.size()));
    }
}

} // namespace
} // namespace openzl::custom_parsers
