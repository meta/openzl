// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <optional>

#include "contrib/gpu/src/device/nvcomp.hpp"

namespace openzl::gpu {
namespace {

int getTempSize(size_t, size_t, NvcompZstdDecompressOpts, size_t*, size_t)
{
    return 0;
}

int decompress(
        const void* const*,
        const size_t*,
        const size_t*,
        size_t*,
        size_t,
        void*,
        size_t,
        void* const*,
        NvcompZstdDecompressOpts,
        int*,
        void*)
{
    return 0;
}

int getAlignments(NvcompZstdDecompressOpts, NvcompAlignmentRequirements*)
{
    return 0;
}

TEST(NvcompTest, LoadsNothingFromALibraryThatIsMissing)
{
    EXPECT_FALSE(loadNvcomp("libnvcomp-does-not-exist.so.5").has_value());
}

TEST(NvcompTest, GivesEveryCallerTheRegisteredFunctions)
{
    // This fails if a caller gets its own copy of nvcomp, such as from a second
    // load, instead of the process's one object.
    NvcompFunctions registered;
    registered.zstd = NvcompZstdFunctions{
        .decompressGetTempSizeAsync      = getTempSize,
        .decompressAsync                 = decompress,
        .decompressGetRequiredAlignments = getAlignments,
    };
    registerNvcomp(registered);

    const NvcompFunctions& first  = nvcompFunctions();
    const NvcompFunctions& second = nvcompFunctions();

    EXPECT_EQ(&first, &second);
    ASSERT_TRUE(first.zstd.has_value());
    EXPECT_EQ(first.zstd->decompressAsync, decompress);
}

TEST(NvcompTest, EmptiesARegisteredPartThatLacksAFunction)
{
    // This fails if a kernel can get a part with a null function.
    NvcompFunctions registered;
    registered.zstd = NvcompZstdFunctions{
        .decompressGetTempSizeAsync = getTempSize,
        .decompressAsync            = decompress,
    };
    registerNvcomp(registered);

    EXPECT_FALSE(nvcompFunctions().zstd.has_value());
}

} // namespace
} // namespace openzl::gpu
