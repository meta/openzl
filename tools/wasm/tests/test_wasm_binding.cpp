// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "custom_parsers/dependency_registration.h"
#include "tools/wasm/src/openzl_wasm.h"

namespace {

// C hands back a buffer the caller owns, so copy it out and release it
std::vector<uint8_t> copyAndFree(uint8_t* buf, size_t size)
{
    std::vector<uint8_t> out(buf, buf + size);
    openzl_wasm_free(buf);
    return out;
}

ZL_ErrorCode serializedCompressor(
        openzl_wasm_Profile profile,
        std::vector<uint8_t>* out)
{
    if (!out) {
        return ZL_ErrorCode_parameter_invalid;
    }
    out->clear();
    uint8_t* buf = nullptr;
    size_t size  = 0;
    ZL_ErrorCode code =
            openzl_wasm_getSerializedCompressor(profile, 0, &buf, &size);
    if (code != ZL_ErrorCode_no_error) {
        EXPECT_EQ(buf, nullptr);
        return code;
    }
    *out = copyAndFree(buf, size);
    return code;
}

// The trainers spend whatever wall-clock budget they are given, so the tests
// pass an explicit one rather than inheriting the much larger default.
constexpr size_t kTestTrainMaxTimeSecs = 5;

openzl_wasm_TrainOptions testTrainOptions(bool paretoFrontier)
{
    // Zeroed, so every option not set below stays at its default
    openzl_wasm_TrainOptions options{};
    // threads is left at its default, so the tests cover the thread count real
    // callers get rather than a single-threaded special case.
    options.maxTimeSecs    = kTestTrainMaxTimeSecs;
    options.paretoFrontier = paretoFrontier ? 1 : 0;
    return options;
}

// Collects every compressor training produced, releasing each owned buffer.
// Single-compressor callers pass paretoFrontier=false and use front().
ZL_ErrorCode train(
        const std::vector<uint8_t>& compressor,
        const std::vector<uint8_t>& src,
        const openzl_wasm_TrainOptions& options,
        std::vector<std::vector<uint8_t>>* out)
{
    if (!out) {
        return ZL_ErrorCode_parameter_invalid;
    }
    out->clear();
    uint8_t* bufs[OPENZL_WASM_TRAIN_PARETO_CANDIDATES] = {};
    size_t sizes[OPENZL_WASM_TRAIN_PARETO_CANDIDATES]  = {};
    size_t count                                       = 0;
    ZL_ErrorCode code                                  = openzl_wasm_train(
            compressor.empty() ? nullptr : compressor.data(),
            compressor.size(),
            src.empty() ? nullptr : src.data(),
            src.size(),
            &options,
            bufs,
            sizes,
            OPENZL_WASM_TRAIN_PARETO_CANDIDATES,
            &count);
    if (code != ZL_ErrorCode_no_error) {
        EXPECT_EQ(count, 0u);
        return code;
    }
    out->reserve(count);
    for (size_t i = 0; i < count; ++i) {
        out->push_back(copyAndFree(bufs[i], sizes[i]));
    }
    return code;
}

ZL_ErrorCode compress(
        const std::vector<uint8_t>& src,
        const std::vector<uint8_t>& compressor,
        std::vector<uint8_t>* out)
{
    if (!out) {
        return ZL_ErrorCode_parameter_invalid;
    }
    out->clear();
    uint8_t* buf      = nullptr;
    size_t size       = 0;
    ZL_ErrorCode code = openzl_wasm_compress(
            compressor.data(),
            compressor.size(),
            src.empty() ? nullptr : src.data(),
            src.size(),
            &buf,
            &size);
    if (code != ZL_ErrorCode_no_error) {
        EXPECT_EQ(buf, nullptr);
        return code;
    }
    *out = copyAndFree(buf, size);
    return code;
}

ZL_ErrorCode decompress(
        const std::vector<uint8_t>& frame,
        std::vector<uint8_t>* out)
{
    if (!out) {
        return ZL_ErrorCode_parameter_invalid;
    }
    out->clear();
    uint8_t* buf = nullptr;
    size_t size  = 0;
    ZL_ErrorCode code =
            openzl_wasm_decompress(frame.data(), frame.size(), &buf, &size);
    if (code != ZL_ErrorCode_no_error) {
        EXPECT_EQ(buf, nullptr);
        return code;
    }
    *out = copyAndFree(buf, size);
    return code;
}

struct BenchCompress {
    ZL_ErrorCode code;
    std::vector<uint8_t> frame;
    double ms;
};

BenchCompress benchmarkCompress(
        const std::vector<uint8_t>& src,
        const std::vector<uint8_t>& compressor,
        size_t iterations)
{
    uint8_t* buf      = nullptr;
    size_t size       = 0;
    double ms         = 0;
    ZL_ErrorCode code = openzl_wasm_benchmarkCompress(
            compressor.data(),
            compressor.size(),
            src.empty() ? nullptr : src.data(),
            src.size(),
            iterations,
            &buf,
            &size,
            &ms);
    if (code != ZL_ErrorCode_no_error) {
        EXPECT_EQ(buf, nullptr);
        return { code, {}, ms };
    }
    return { code, copyAndFree(buf, size), ms };
}

struct BenchDecompress {
    ZL_ErrorCode code;
    double ms;
};

BenchDecompress benchmarkDecompress(
        const std::vector<uint8_t>& frame,
        size_t iterations)
{
    double ms         = 0;
    ZL_ErrorCode code = openzl_wasm_benchmarkDecompress(
            frame.data(), frame.size(), iterations, &ms);
    return { code, ms };
}

std::vector<uint8_t> makeSerialData(size_t size)
{
    constexpr std::string_view kPattern = "hello openzl serial data ";
    std::vector<uint8_t> out(size);
    for (size_t i = 0; i < size; ++i) {
        out[i] = static_cast<uint8_t>(kPattern[i % kPattern.size()]);
    }
    return out;
}

std::vector<uint8_t> makeIntData(
        size_t eltWidth,
        size_t count,
        bool isSigned,
        bool isBigEndian = false)
{
    std::vector<uint8_t> out;
    out.reserve(count * eltWidth);
    for (size_t i = 0; i < count; ++i) {
        const int64_t value = isSigned ? static_cast<int64_t>(i % 200) - 100
                                       : static_cast<int64_t>(i % 200);
        for (size_t b = 0; b < eltWidth; ++b) {
            const size_t byteIndex = isBigEndian ? eltWidth - b - 1 : b;
            out.push_back(
                    static_cast<uint8_t>(
                            (static_cast<uint64_t>(value) >> (byteIndex * 8))
                            & 0xFF));
        }
    }
    return out;
}

std::vector<uint8_t> makeCsvData(size_t rows)
{
    const std::string header = "id,name,value\n";
    std::vector<uint8_t> out(header.begin(), header.end());
    for (size_t i = 0; i < rows; ++i) {
        const std::string row = std::to_string(i % 100) + ",item"
                + std::to_string(i % 10) + "," + std::to_string(i % 50) + "\n";
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

void expectRoundTrip(
        const std::vector<uint8_t>& src,
        openzl_wasm_Profile profile)
{
    SCOPED_TRACE(openzl_wasm_profileName(profile));

    std::vector<uint8_t> compressor;
    const ZL_ErrorCode serCode = serializedCompressor(profile, &compressor);
    ASSERT_EQ(serCode, ZL_ErrorCode_no_error)
            << openzl_wasm_errorString(serCode);

    std::vector<uint8_t> frame;
    const ZL_ErrorCode compCode = compress(src, compressor, &frame);
    ASSERT_EQ(compCode, ZL_ErrorCode_no_error)
            << openzl_wasm_errorString(compCode);

    if (!src.empty()) {
        EXPECT_LT(frame.size(), src.size());
    }
    std::vector<uint8_t> dec;
    const ZL_ErrorCode decCode = decompress(frame, &dec);
    ASSERT_EQ(decCode, ZL_ErrorCode_no_error)
            << openzl_wasm_errorString(decCode);
    EXPECT_EQ(dec, src);
}

// Trains from the profile's base and checks the result round-trips
void expectTrainedRoundTrip(
        const std::vector<uint8_t>& src,
        openzl_wasm_Profile profile,
        bool paretoFrontier)
{
    SCOPED_TRACE(openzl_wasm_profileName(profile));

    ASSERT_EQ(openzl_wasm_profileIsTrainable(profile), 1);

    std::vector<uint8_t> base;
    ASSERT_EQ(serializedCompressor(profile, &base), ZL_ErrorCode_no_error);

    std::vector<std::vector<uint8_t>> frontier;
    const ZL_ErrorCode trainCode =
            train(base, src, testTrainOptions(paretoFrontier), &frontier);
    ASSERT_EQ(trainCode, ZL_ErrorCode_no_error)
            << openzl_wasm_errorString(trainCode);

    // The frontier's size depends on the data, so the only guarantees are
    // that training produced something and stayed within the caller's array.
    ASSERT_FALSE(frontier.empty());
    if (paretoFrontier) {
        ASSERT_LE(frontier.size(), size_t(OPENZL_WASM_TRAIN_PARETO_CANDIDATES));
    } else {
        ASSERT_EQ(frontier.size(), 1u);
    }

    // Every compressor training produced is usable, not just the first.
    for (size_t i = 0; i < frontier.size(); ++i) {
        SCOPED_TRACE(i);
        ASSERT_FALSE(frontier[i].empty());

        std::vector<uint8_t> frame;
        const ZL_ErrorCode compCode = compress(src, frontier[i], &frame);
        ASSERT_EQ(compCode, ZL_ErrorCode_no_error)
                << openzl_wasm_errorString(compCode);

        // The frontier spans to the speed-optimal end, where a compressor is
        // barely more than store and the frame can come out larger than the
        // input, so only the best-ratio end is required to compress. It is
        // first because the frontier is ordered by compressed size.
        if (i == 0) {
            EXPECT_LT(frame.size(), src.size());
        }

        std::vector<uint8_t> dec;
        const ZL_ErrorCode decCode = decompress(frame, &dec);
        ASSERT_EQ(decCode, ZL_ErrorCode_no_error)
                << openzl_wasm_errorString(decCode);
        EXPECT_EQ(dec, src);
    }
}

struct IntProfileCase {
    openzl_wasm_Profile profile;
    size_t width;
    bool isSigned;
    bool isBigEndian;
};

constexpr IntProfileCase kIntProfiles[] = {
    { OPENZL_WASM_PROFILE_U8, 1, false, false },
    { OPENZL_WASM_PROFILE_I8, 1, true, false },
    { OPENZL_WASM_PROFILE_LE_U16, 2, false, false },
    { OPENZL_WASM_PROFILE_LE_I16, 2, true, false },
    { OPENZL_WASM_PROFILE_LE_U32, 4, false, false },
    { OPENZL_WASM_PROFILE_LE_I32, 4, true, false },
    { OPENZL_WASM_PROFILE_LE_U64, 8, false, false },
    { OPENZL_WASM_PROFILE_LE_I64, 8, true, false },
    { OPENZL_WASM_PROFILE_BE_U16, 2, false, true },
    { OPENZL_WASM_PROFILE_BE_I16, 2, true, true },
    { OPENZL_WASM_PROFILE_BE_U32, 4, false, true },
    { OPENZL_WASM_PROFILE_BE_I32, 4, true, true },
    { OPENZL_WASM_PROFILE_BE_U64, 8, false, true },
    { OPENZL_WASM_PROFILE_BE_I64, 8, true, true },
};

} // namespace

TEST(WasmBindingTest, SerialRoundTrip)
{
    expectRoundTrip(makeSerialData(4096), OPENZL_WASM_PROFILE_SERIAL);
}

TEST(WasmBindingTest, TraceOutputAddressesMustBePaired)
{
    std::vector<uint8_t> compressor;
    ASSERT_EQ(
            serializedCompressor(OPENZL_WASM_PROFILE_SERIAL, &compressor),
            ZL_ErrorCode_no_error);
    const auto src = makeSerialData(4096);
    std::vector<uint8_t> frame;
    ASSERT_EQ(compress(src, compressor, &frame), ZL_ErrorCode_no_error);

    uint8_t* buf     = nullptr;
    size_t size      = 0;
    uint8_t* trace   = nullptr;
    size_t traceSize = 0;
    for (const bool includeTraceAddress : { false, true }) {
        uint8_t** outTrace   = includeTraceAddress ? &trace : nullptr;
        size_t* outTraceSize = includeTraceAddress ? nullptr : &traceSize;
        EXPECT_EQ(
                openzl_wasm_compressWithTrace(
                        compressor.data(),
                        compressor.size(),
                        src.data(),
                        src.size(),
                        &buf,
                        &size,
                        outTrace,
                        outTraceSize),
                ZL_ErrorCode_parameter_invalid);
        EXPECT_EQ(
                openzl_wasm_decompressWithTrace(
                        frame.data(),
                        frame.size(),
                        &buf,
                        &size,
                        outTrace,
                        outTraceSize),
                ZL_ErrorCode_parameter_invalid);
    }
}

TEST(WasmBindingTest, LzRoundTrip)
{
    expectRoundTrip(makeSerialData(4096), OPENZL_WASM_PROFILE_LZ);
}

TEST(WasmBindingTest, ZstdRoundTrip)
{
    expectRoundTrip(makeSerialData(4096), OPENZL_WASM_PROFILE_ZSTD);
}

TEST(WasmBindingTest, CsvRoundTrip)
{
    expectRoundTrip(makeCsvData(1024), OPENZL_WASM_PROFILE_CSV);
}

TEST(WasmBindingTest, ParquetSerializedCompressorLoads)
{
    std::vector<uint8_t> bytes;
    ASSERT_EQ(
            serializedCompressor(OPENZL_WASM_PROFILE_PARQUET, &bytes),
            ZL_ErrorCode_no_error);
    ASSERT_FALSE(bytes.empty());
    const std::string serialized(bytes.begin(), bytes.end());
    const auto compressor =
            openzl::custom_parsers::createCompressorFromSerialized(
                    serialized, "");
    EXPECT_TRUE(ZL_GraphID_isValid(compressor->getStartingGraph()));
}

TEST(WasmBindingTest, IntProfilesRoundTrip)
{
    for (const auto& test : kIntProfiles) {
        expectRoundTrip(
                makeIntData(test.width, 1024, test.isSigned, test.isBigEndian),
                test.profile);
    }
}

TEST(WasmBindingTest, SignednessDoesNotChangeNumericGraph)
{
    constexpr openzl_wasm_Profile kUnsignedProfiles[] = {
        OPENZL_WASM_PROFILE_U8,     OPENZL_WASM_PROFILE_LE_U16,
        OPENZL_WASM_PROFILE_LE_U32, OPENZL_WASM_PROFILE_LE_U64,
        OPENZL_WASM_PROFILE_BE_U16, OPENZL_WASM_PROFILE_BE_U32,
        OPENZL_WASM_PROFILE_BE_U64,
    };
    constexpr openzl_wasm_Profile kSignedProfiles[] = {
        OPENZL_WASM_PROFILE_I8,     OPENZL_WASM_PROFILE_LE_I16,
        OPENZL_WASM_PROFILE_LE_I32, OPENZL_WASM_PROFILE_LE_I64,
        OPENZL_WASM_PROFILE_BE_I16, OPENZL_WASM_PROFILE_BE_I32,
        OPENZL_WASM_PROFILE_BE_I64,
    };

    for (size_t i = 0; i < std::size(kUnsignedProfiles); ++i) {
        std::vector<uint8_t> unsignedCompressor;
        std::vector<uint8_t> signedCompressor;
        ASSERT_EQ(
                serializedCompressor(kUnsignedProfiles[i], &unsignedCompressor),
                ZL_ErrorCode_no_error);
        ASSERT_EQ(
                serializedCompressor(kSignedProfiles[i], &signedCompressor),
                ZL_ErrorCode_no_error);
        EXPECT_EQ(unsignedCompressor, signedCompressor);
    }
}

TEST(WasmBindingTest, EmptyRoundTrip)
{
    expectRoundTrip({}, OPENZL_WASM_PROFILE_SERIAL);
}

TEST(WasmBindingTest, CompressionLevelIsSerialized)
{
    uint8_t* levelOneBuf  = nullptr;
    size_t levelOneSize   = 0;
    uint8_t* levelNineBuf = nullptr;
    size_t levelNineSize  = 0;

    ASSERT_EQ(
            openzl_wasm_getSerializedCompressor(
                    OPENZL_WASM_PROFILE_SERIAL, 1, &levelOneBuf, &levelOneSize),
            ZL_ErrorCode_no_error);
    ASSERT_EQ(
            openzl_wasm_getSerializedCompressor(
                    OPENZL_WASM_PROFILE_SERIAL,
                    9,
                    &levelNineBuf,
                    &levelNineSize),
            ZL_ErrorCode_no_error);

    EXPECT_NE(
            copyAndFree(levelOneBuf, levelOneSize),
            copyAndFree(levelNineBuf, levelNineSize));

    uint8_t* invalidBuf = nullptr;
    size_t invalidSize  = 0;
    EXPECT_EQ(
            openzl_wasm_getSerializedCompressor(
                    OPENZL_WASM_PROFILE_SERIAL, -1, &invalidBuf, &invalidSize),
            ZL_ErrorCode_parameter_invalid);
}

TEST(WasmBindingTest, TrainedCompressorRoundTrips)
{
    expectTrainedRoundTrip(
            makeSerialData(4096), OPENZL_WASM_PROFILE_SERIAL, false);
}

TEST(WasmBindingTest, TrainedLzCompressorRoundTrips)
{
    expectTrainedRoundTrip(makeSerialData(4096), OPENZL_WASM_PROFILE_LZ, false);
}

TEST(WasmBindingTest, CsvTrainingFailsWithExistingDependencyRegistration)
{
    EXPECT_EQ(openzl_wasm_profileIsTrainable(OPENZL_WASM_PROFILE_CSV), 0);
    std::vector<uint8_t> base;
    ASSERT_EQ(
            serializedCompressor(OPENZL_WASM_PROFILE_CSV, &base),
            ZL_ErrorCode_no_error);
    std::vector<std::vector<uint8_t>> trained;
    EXPECT_EQ(
            train(base, makeCsvData(1024), testTrainOptions(false), &trained),
            ZL_ErrorCode_GENERIC);
    EXPECT_TRUE(trained.empty());
}

TEST(WasmBindingTest, ZstdTrainingIsUnavailableWithoutDictionaryBundles)
{
    EXPECT_EQ(openzl_wasm_profileIsTrainable(OPENZL_WASM_PROFILE_ZSTD), 0);
    std::vector<uint8_t> base;
    ASSERT_EQ(
            serializedCompressor(OPENZL_WASM_PROFILE_ZSTD, &base),
            ZL_ErrorCode_no_error);
    std::vector<std::vector<uint8_t>> trained;
    EXPECT_EQ(
            train(base,
                  makeSerialData(4096),
                  testTrainOptions(false),
                  &trained),
            ZL_ErrorCode_GENERIC);
    EXPECT_TRUE(trained.empty());
}

TEST(WasmBindingTest, TrainedIntCompressorRoundTrips)
{
    for (const auto& test : kIntProfiles) {
        expectTrainedRoundTrip(
                makeIntData(test.width, 1024, test.isSigned, test.isBigEndian),
                test.profile,
                false);
    }
}

TEST(WasmBindingTest, ParetoTrainedCompressorsAllRoundTrip)
{
    expectTrainedRoundTrip(
            makeSerialData(4096), OPENZL_WASM_PROFILE_SERIAL, true);
}

TEST(WasmBindingTest, ParetoTrainedIntCompressorsAllRoundTrip)
{
    expectTrainedRoundTrip(
            makeIntData(4, 1024, false), OPENZL_WASM_PROFILE_LE_U32, true);
}

TEST(WasmBindingTest, TrainRejectsBadInput)
{
    const std::vector<uint8_t> src = makeSerialData(1024);
    std::vector<uint8_t> base;
    ASSERT_EQ(
            serializedCompressor(OPENZL_WASM_PROFILE_SERIAL, &base),
            ZL_ErrorCode_no_error);

    // Every rejection below is cheap: none of them reach the trainers.
    std::vector<std::vector<uint8_t>> trained;
    const openzl_wasm_TrainOptions singleOptions =
            testTrainOptions(/* paretoFrontier */ false);

    // Unlike compress, an empty sample is rejected rather than handled: there
    // is nothing to train on.
    EXPECT_EQ(
            train(base, {}, singleOptions, &trained),
            ZL_ErrorCode_parameter_invalid);
    EXPECT_TRUE(trained.empty());

    EXPECT_EQ(
            train({}, src, singleOptions, &trained),
            ZL_ErrorCode_parameter_invalid);
    EXPECT_TRUE(trained.empty());

    const std::vector<uint8_t> garbage(64, 0xAB);
    EXPECT_NE(
            train(garbage, src, singleOptions, &trained),
            ZL_ErrorCode_no_error);
    EXPECT_TRUE(trained.empty());

    // Each of the three outputs is required, so omitting any one is rejected,
    // as is an output array with no room in it.
    uint8_t* bufs[OPENZL_WASM_TRAIN_PARETO_CANDIDATES] = {};
    size_t sizes[OPENZL_WASM_TRAIN_PARETO_CANDIDATES]  = {};
    size_t count                                       = 0;
    EXPECT_EQ(
            openzl_wasm_train(
                    base.data(),
                    base.size(),
                    src.data(),
                    src.size(),
                    &singleOptions,
                    nullptr,
                    sizes,
                    OPENZL_WASM_TRAIN_PARETO_CANDIDATES,
                    &count),
            ZL_ErrorCode_parameter_invalid);
    EXPECT_EQ(
            openzl_wasm_train(
                    base.data(),
                    base.size(),
                    src.data(),
                    src.size(),
                    &singleOptions,
                    bufs,
                    nullptr,
                    OPENZL_WASM_TRAIN_PARETO_CANDIDATES,
                    &count),
            ZL_ErrorCode_parameter_invalid);
    EXPECT_EQ(
            openzl_wasm_train(
                    base.data(),
                    base.size(),
                    src.data(),
                    src.size(),
                    &singleOptions,
                    bufs,
                    sizes,
                    OPENZL_WASM_TRAIN_PARETO_CANDIDATES,
                    nullptr),
            ZL_ErrorCode_parameter_invalid);
    EXPECT_EQ(
            openzl_wasm_train(
                    base.data(),
                    base.size(),
                    src.data(),
                    src.size(),
                    &singleOptions,
                    bufs,
                    sizes,
                    0,
                    &count),
            ZL_ErrorCode_parameter_invalid);
    EXPECT_EQ(count, 0u);
}

// The frontier normally runs to several compressors, so a caller with a
// shorter array is the case that used to write past its end.
TEST(WasmBindingTest, TrainHonoursOutCapacity)
{
    const std::vector<uint8_t> src = makeSerialData(4096);
    std::vector<uint8_t> base;
    ASSERT_EQ(
            serializedCompressor(OPENZL_WASM_PROFILE_SERIAL, &base),
            ZL_ErrorCode_no_error);

    const openzl_wasm_TrainOptions options =
            testTrainOptions(/* paretoFrontier */ true);

    // Deliberately shorter than the frontier, with a guard entry after it
    // that must stay untouched.
    constexpr size_t kCapacity   = 1;
    uint8_t* bufs[kCapacity + 1] = {};
    size_t sizes[kCapacity + 1]  = {};
    size_t count                 = 0;
    const ZL_ErrorCode code      = openzl_wasm_train(
            base.data(),
            base.size(),
            src.data(),
            src.size(),
            &options,
            bufs,
            sizes,
            kCapacity,
            &count);
    ASSERT_EQ(code, ZL_ErrorCode_no_error) << openzl_wasm_errorString(code);
    EXPECT_EQ(count, kCapacity);
    EXPECT_EQ(bufs[kCapacity], nullptr);
    EXPECT_EQ(sizes[kCapacity], 0u);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_NE(bufs[i], nullptr);
        openzl_wasm_free(bufs[i]);
    }
}

TEST(WasmBindingTest, BenchmarkTimesBothDirections)
{
    const std::vector<uint8_t> src = makeSerialData(4096);
    std::vector<uint8_t> compressor;
    ASSERT_EQ(
            serializedCompressor(OPENZL_WASM_PROFILE_SERIAL, &compressor),
            ZL_ErrorCode_no_error);

    const BenchCompress bench = benchmarkCompress(src, compressor, 2);
    ASSERT_EQ(bench.code, ZL_ErrorCode_no_error)
            << openzl_wasm_errorString(bench.code);
    EXPECT_GE(bench.ms, 0.0); // timing is too machine-dependent to bound

    // The frame it hands back must be a real one, so a caller timing both
    // directions needs no extra compression to get something to decompress.
    std::vector<uint8_t> frame;
    ASSERT_EQ(compress(src, compressor, &frame), ZL_ErrorCode_no_error);
    EXPECT_EQ(bench.frame, frame);

    std::vector<uint8_t> decompressed;
    ASSERT_EQ(decompress(bench.frame, &decompressed), ZL_ErrorCode_no_error);
    EXPECT_EQ(decompressed, src);

    const BenchDecompress d = benchmarkDecompress(bench.frame, 2);
    ASSERT_EQ(d.code, ZL_ErrorCode_no_error) << openzl_wasm_errorString(d.code);
    EXPECT_GE(d.ms, 0.0);
}

TEST(WasmBindingTest, BenchmarkRejectsOutOfRangeIterations)
{
    const std::vector<uint8_t> src = makeSerialData(1024);
    std::vector<uint8_t> compressor;
    ASSERT_EQ(
            serializedCompressor(OPENZL_WASM_PROFILE_SERIAL, &compressor),
            ZL_ErrorCode_no_error);

    // Out-of-range counts are rejected rather than clamped, so the count a
    // caller passes is always the count that ran. js/wasm_api.js clamps before
    // calling, so it never trips this.
    EXPECT_EQ(
            benchmarkCompress(src, compressor, 0).code,
            ZL_ErrorCode_parameter_invalid);
    EXPECT_EQ(
            benchmarkCompress(
                    src, compressor, OPENZL_WASM_BENCHMARK_MAX_ITERATIONS + 1)
                    .code,
            ZL_ErrorCode_parameter_invalid);
    EXPECT_EQ(
            benchmarkCompress(src, compressor, 1).code, ZL_ErrorCode_no_error);
    EXPECT_EQ(
            benchmarkCompress(
                    src, compressor, OPENZL_WASM_BENCHMARK_MAX_ITERATIONS)
                    .code,
            ZL_ErrorCode_no_error);

    // benchmarkDecompress carries its own copy of the check.
    std::vector<uint8_t> frame;
    ASSERT_EQ(compress(src, compressor, &frame), ZL_ErrorCode_no_error);
    EXPECT_EQ(
            benchmarkDecompress(frame, 0).code, ZL_ErrorCode_parameter_invalid);
    EXPECT_EQ(
            benchmarkDecompress(frame, OPENZL_WASM_BENCHMARK_MAX_ITERATIONS + 1)
                    .code,
            ZL_ErrorCode_parameter_invalid);
    EXPECT_EQ(benchmarkDecompress(frame, 1).code, ZL_ErrorCode_no_error);
    EXPECT_EQ(
            benchmarkDecompress(frame, OPENZL_WASM_BENCHMARK_MAX_ITERATIONS)
                    .code,
            ZL_ErrorCode_no_error);
}

TEST(WasmBindingTest, ExposesMaxBenchmarkIterations)
{
    EXPECT_EQ(
            openzl_wasm_maxBenchmarkIterations(),
            OPENZL_WASM_BENCHMARK_MAX_ITERATIONS);
}

TEST(WasmBindingTest, ProfileEnumMatchesTable)
{
    // The enum keys the metadata language bindings enumerate.
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_SERIAL), "serial");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_U8), "u8");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_I8), "i8");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_LE_U16), "le-u16");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_LE_I16), "le-i16");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_LE_U32), "le-u32");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_LE_I32), "le-i32");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_LE_U64), "le-u64");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_LE_I64), "le-i64");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_LZ), "lz");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_ZSTD), "zstd");
    EXPECT_EQ(openzl_wasm_profileIsTrainable(OPENZL_WASM_PROFILE_ZSTD), 0);
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_CSV), "csv");
    EXPECT_STREQ(
            openzl_wasm_profileName(OPENZL_WASM_PROFILE_PARQUET), "parquet");
    EXPECT_STREQ(
            openzl_wasm_profileName(OPENZL_WASM_PROFILE_PYTORCH), "pytorch");
    EXPECT_EQ(openzl_wasm_profileIsSupported(OPENZL_WASM_PROFILE_PYTORCH), 0);
    EXPECT_EQ(openzl_wasm_profileIsTrainable(OPENZL_WASM_PROFILE_PYTORCH), 0);

    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_BE_U16), "be-u16");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_BE_I16), "be-i16");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_BE_U32), "be-u32");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_BE_I32), "be-i32");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_BE_U64), "be-u64");
    EXPECT_STREQ(openzl_wasm_profileName(OPENZL_WASM_PROFILE_BE_I64), "be-i64");

    // Every value must have a row, or the table has a value-initialized gap.
    for (int i = 0; i < OPENZL_WASM_PROFILE_COUNT; ++i) {
        SCOPED_TRACE(i);
        const auto profile = static_cast<openzl_wasm_Profile>(i);
        ASSERT_NE(openzl_wasm_profileName(profile), nullptr);
        EXPECT_NE(openzl_wasm_profileName(profile)[0], '\0');
        // buildProfileGraph() decides what builds, so this catches
        // isWasmSupported drifting from it in either direction.
        std::vector<uint8_t> compressor;
        const ZL_ErrorCode code = serializedCompressor(profile, &compressor);
        const char* reason      = openzl_wasm_profileUnsupportedReason(profile);
        if (openzl_wasm_profileIsSupported(profile)) {
            EXPECT_EQ(reason, nullptr);
            EXPECT_EQ(code, ZL_ErrorCode_no_error)
                    << openzl_wasm_errorString(code);
            EXPECT_FALSE(compressor.empty());
        } else {
            ASSERT_NE(reason, nullptr);
            EXPECT_NE(reason[0], '\0');
            EXPECT_EQ(code, ZL_ErrorCode_graph_invalid);
            EXPECT_TRUE(compressor.empty());
        }
        const char* description = openzl_wasm_profileDescription(profile);
        ASSERT_NE(description, nullptr);
        EXPECT_NE(description[0], '\0');
    }
}

TEST(WasmBindingTest, RejectsUnknownProfile)
{
    const auto bad =
            static_cast<openzl_wasm_Profile>(OPENZL_WASM_PROFILE_COUNT);
    std::vector<uint8_t> compressor;
    EXPECT_NE(serializedCompressor(bad, &compressor), ZL_ErrorCode_no_error);
    EXPECT_TRUE(compressor.empty());
    EXPECT_EQ(openzl_wasm_profileName(bad), nullptr);
    EXPECT_EQ(openzl_wasm_profileDescription(bad), nullptr);
    EXPECT_EQ(openzl_wasm_profileIsSupported(bad), 0);
    EXPECT_EQ(openzl_wasm_profileUnsupportedReason(bad), nullptr);
    EXPECT_EQ(openzl_wasm_profileIsTrainable(bad), 0);
}

TEST(WasmBindingTest, RejectsGarbage)
{
    const std::vector<uint8_t> garbage(64, 0xAB);
    const std::vector<uint8_t> src = makeSerialData(128);

    // A garbage compressor, and a garbage frame in each of the three calls
    // that read one. The helpers assert that no buffer comes back.
    std::vector<uint8_t> output;
    EXPECT_NE(compress(src, garbage, &output), ZL_ErrorCode_no_error);
    EXPECT_TRUE(output.empty());
    EXPECT_NE(decompress(garbage, &output), ZL_ErrorCode_no_error);
    EXPECT_TRUE(output.empty());
    EXPECT_NE(benchmarkCompress(src, garbage, 2).code, ZL_ErrorCode_no_error);
    EXPECT_NE(benchmarkDecompress(garbage, 2).code, ZL_ErrorCode_no_error);

    size_t size = 0;
    EXPECT_NE(
            openzl_wasm_getDecompressedSize(
                    garbage.data(), garbage.size(), &size),
            ZL_ErrorCode_no_error);
}
