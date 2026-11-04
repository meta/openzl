// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "custom_parsers/numpy/tests/numpy_builder.h"

namespace {
using namespace openzl::custom_parsers::testing;
namespace fs = std::filesystem;

/// One array of each class the parser tells apart, plus one larger than the
/// chunk size and an empty one.
std::vector<NpyArraySpec> sampleArrays()
{
    return {
        { "<f4", { 64, 48 }, pseudoRandomBytes(12288, 1) },
        { "<f4", { 24576 }, pseudoRandomBytes(98304, 2) },
        { "<f2", { 100 }, pseudoRandomBytes(200, 3) },
        { "<f8", { 10, 10 }, pseudoRandomBytes(800, 4), true },
        { "|u1", { 3, 4, 5 }, pseudoRandomBytes(60, 5) },
        { "|b1", { 50 }, std::string(50, '\1') },
        { "<i8", { 20, 30 }, pseudoRandomBytes(4800, 6), true },
        { "<u2", {}, pseudoRandomBytes(2, 7) },
        { "<M8[ns]", { 16 }, pseudoRandomBytes(128, 8) },
        { "<V2", { 32 }, pseudoRandomBytes(64, 9) },
        { "<c8", { 16 }, pseudoRandomBytes(128, 10) },
        { "<c16", { 7 }, pseudoRandomBytes(112, 11) },
        { ">f4", { 25 }, pseudoRandomBytes(100, 12) },
        { "<f16", { 4 }, pseudoRandomBytes(64, 13) },
        { "<U3", { 10 }, pseudoRandomBytes(120, 14) },
        { "|O", { 3 }, pseudoRandomBytes(24, 15) },
        { "[('x', '<f4'), ('y', '|u1')]", { 20 }, pseudoRandomBytes(100, 16) },
        { "<f4", { 0, 3 }, "" },
    };
}

std::vector<std::string> sampleFiles()
{
    const auto arrays = sampleArrays();
    std::vector<std::string> files;
    std::vector<std::pair<std::string, NpyArraySpec>> named;
    for (size_t i = 0; i < arrays.size(); ++i) {
        files.push_back(buildNpy(arrays[i]));
        named.emplace_back("arr_" + std::to_string(i), arrays[i]);
    }
    files.push_back(buildNpy(arrays[0], 2));
    files.push_back(buildNpy(arrays[0], 3));
    files.push_back(buildNpz(named));
    files.push_back(buildZip({ { "arr_0.npy", buildNpy(arrays[0]), 8 } }));
    return files;
}

/// The input of RandomInputFuzzer: its chunk size, which it reads first as a
/// native uint32_t added to its minimum of 64, then the file.
std::string randomInputSeed(const std::string& file)
{
    const uint32_t chunkSize = (uint32_t)64 << 10;
    const uint32_t raw       = chunkSize - 64;
    std::string seed(sizeof(raw), '\0');
    std::memcpy(seed.data(), &raw, sizeof(raw));
    return seed + file;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::fprintf(
                stderr,
                "USAGE: %s TEST_SUITE TEST_CASE OUTPUT_DIRECTORY\n",
                argv[0]);
        return 1;
    }

    const std::string testSuite = argv[1];
    const std::string testCase  = argv[2];
    const fs::path outDir       = argv[3];

    if (testSuite != "NumPyTest") {
        std::fprintf(stderr, "Unknown test suite: %s\n", testSuite.c_str());
        return 2;
    }

    std::vector<std::string> corpus;
    if (testCase == "RandomInputFuzzer") {
        for (const auto& file : sampleFiles()) {
            corpus.push_back(randomInputSeed(file));
        }
    } else if (testCase == "StructuredInputFuzzer") {
        // It builds its files from its input, so it needs no seed
    } else {
        std::fprintf(stderr, "Unknown test case: %s\n", testCase.c_str());
        return 3;
    }

    fs::create_directories(outDir);

    for (size_t i = 0; i < corpus.size(); ++i) {
        const auto path = outDir / ("seed_" + std::to_string(i));
        std::ofstream out(path, std::ios::binary);
        if (!out.write(corpus[i].data(), (std::streamsize)corpus[i].size())) {
            std::fprintf(stderr, "Failed to write path: %s\n", path.c_str());
            return 4;
        }
    }

    return 0;
}
