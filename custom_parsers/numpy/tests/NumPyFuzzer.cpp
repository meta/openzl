// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "security/lionhead/utils/lib_ftest/fdp/fdp/fdp_impl.h"
#include "security/lionhead/utils/lib_ftest/ftest.h"

#include "custom_parsers/numpy/numpy_parser.h"
#include "custom_parsers/numpy/tests/numpy_builder.h"
#include "custom_parsers/tests/pickle_builder.h"
#include "openzl/openzl.hpp"
#include "tests/datagen/random_producer/LionheadFDPWrapper.h"

using namespace ::testing;
using namespace facebook::security::lionhead::fdp;
using openzl::tests::datagen::LionheadFDPWrapper;

namespace openzl::custom_parsers {

class NumPyTest : public ::testing::Test {
   protected:
    void roundtrip(std::string_view input, size_t chunkSize)
    {
        const auto gid = ZL_NumPy_registerGraph(compressor_.get(), chunkSize);
        ASSERT_FALSE(ZL_RES_isError(gid));
        compressor_.selectStartingGraph(ZL_RES_value(gid));
        cctx_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        cctx_.refCompressor(compressor_);
        const auto compressed = cctx_.compressSerial(input);
        ASSERT_EQ(dctx_.decompressSerial(compressed), input);
    }

    CCtx cctx_{};
    DCtx dctx_{};
    Compressor compressor_{};
};

template <class HarnessMode>
LionheadFDPWrapper<StructuredFDP<HarnessMode>> rwFromFDP(
        StructuredFDP<HarnessMode>& fdp)
{
    return LionheadFDPWrapper<StructuredFDP<HarnessMode>>(fdp);
}

FUZZ_F(NumPyTest, RandomInputFuzzer)
{
    auto rw              = rwFromFDP(f);
    const auto chunkSize = rw.u32_range("chunkSize", 64, 1 << 20);
    const auto input     = rw.all_remaining_bytes();
    roundtrip(
            std::string_view((const char*)input.data(), input.size()),
            chunkSize);
}

/// An array of any descr, including unhandled ones, whose data mostly matches
/// its shape.
template <class RW>
testing::NpyArraySpec randomArray(RW& rw)
{
    static const std::vector<std::pair<std::string, size_t>> kDescrs = {
        { "<f4", 4 },     { "<f2", 2 },
        { "<f8", 8 },     { "|u1", 1 },
        { "|b1", 1 },     { "<i8", 8 },
        { "<u2", 2 },     { "<c8", 8 },
        { "<c16", 16 },   { "<V2", 2 },
        { "<M8[ns]", 8 }, { "<U3", 12 },
        { ">f4", 4 },     { "|O", 8 },
        { "<f16", 16 },   { "[('x', '<f4'), ('y', '|u1')]", 5 },
    };
    const auto& [descr, itemSize] =
            kDescrs[rw.usize_range("descr", 0, kDescrs.size() - 1)];
    std::vector<uint64_t> shape(rw.usize_range("ndims", 0, 3));
    size_t nbElts = 1;
    for (auto& dim : shape) {
        dim = rw.usize_range("dim", 0, 40);
        nbElts *= dim;
    }
    size_t nbBytes = nbElts * itemSize;
    if (rw.usize_range("inconsistent", 0, 7) == 0) {
        nbBytes = rw.usize_range("nbBytes", 0, 300);
    }
    return { descr,
             std::move(shape),
             rw.usize_range("constant", 0, 3) == 0
                     ? std::string(nbBytes, (char)rw.u8("byte"))
                     : testing::pseudoRandomBytes(nbBytes, rw.u32("seed")),
             rw.usize_range("fortranOrder", 0, 3) == 0 };
}

/// The array as NumPy pickles it: structured descrs become records of 5 bytes
/// with two fields.
testing::PickledArray toPickled(testing::NpyArraySpec a)
{
    testing::PickledArray p{ a.descr.substr(1),
                             std::move(a.shape),
                             std::move(a.data),
                             a.descr[0],
                             a.fortranOrder };
    if (a.descr.starts_with("[")) {
        p.typeStr    = "V5";
        p.byteOrder  = '|';
        p.fieldNames = { "x", "y" };
    }
    return p;
}

/// A pickle of a dict of arrays, sometimes followed by another one.
template <class RW>
std::string randomPickle(RW& rw)
{
    std::string file;
    const size_t nbPickles = rw.usize_range("nbPickles", 1, 2);
    for (size_t p = 0; p < nbPickles; ++p) {
        const int protocol = (int)rw.usize_range("protocol", 2, 5);
        const bool python2 = protocol == 2 && rw.usize_range("python2", 0, 1);
        std::vector<std::pair<std::string, testing::PickledArray>> arrays;
        const size_t nbArrays = rw.usize_range("nbArrays", 0, 12);
        for (size_t i = 0; i < nbArrays; ++i) {
            arrays.emplace_back(
                    "a" + std::to_string(i), toPickled(randomArray(rw)));
            arrays.back().second.dtypeId = (int)rw.usize_range("dtypeId", 0, 2);
        }
        file += testing::pickleArrays(arrays, protocol, python2);
    }
    return file;
}

/// Mostly valid .npy files, archives and pickles, to exercise segmentation and
/// chunking.
FUZZ_F(NumPyTest, StructuredInputFuzzer)
{
    auto rw = rwFromFDP(f);
    const auto chunkSize =
            rw.u32_range("chunkSize", 2 * ZL_MIN_CHUNK_SIZE, 1 << 18);
    std::string file;
    const size_t container = rw.usize_range("container", 0, 5);
    if (container == 0) {
        file = testing::buildNpy(
                randomArray(rw), (uint8_t)rw.usize_range("major", 1, 3));
    } else if (container == 1) {
        file = randomPickle(rw);
    } else {
        std::vector<testing::ZipMember> members;
        const size_t nbMembers = rw.usize_range("nbMembers", 0, 12);
        for (size_t i = 0; i < nbMembers; ++i) {
            const bool deflated = rw.usize_range("deflated", 0, 5) == 0;
            members.push_back(
                    { "arr_" + std::to_string(i) + ".npy",
                      testing::buildNpy(randomArray(rw)),
                      (uint16_t)(deflated ? 8 : 0) });
        }
        file = testing::buildZip(members);
    }
    if (rw.usize_range("corrupt", 0, 3) == 0) {
        file[rw.usize_range("corruptPos", 0, file.size() - 1)] ^=
                (char)rw.u8_range("corruptBits", 1, 255);
    }
    roundtrip(file, chunkSize);
}

} // namespace openzl::custom_parsers
