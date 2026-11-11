// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_transforms/json_extract/decode_json_extract.h"
#include "custom_transforms/json_extract/common_json_extract.h"
#include "openzl/codecs/common/copy.h"
#include "openzl/common/assertion.h"
#include "openzl/shared/bits.h"
#include "openzl/zl_data.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>

#ifdef __AVX2__
#    include <immintrin.h>
#endif

namespace zstrong {
namespace {

/// @returns true iff the character represents a token that should be replaced.
bool isInSet(char c)
{
    return c >= char(Token::START) && c <= char(Token::END);
}

#ifdef __AVX2__
/// @returns a 32-bit bitmask for whether each byte in @p srcV is a token that
/// should be replaced.
uint32_t isInSetV(__m256i const srcV)
{
    __m256i const loV =
            _mm256_cmpgt_epi8(srcV, _mm256_set1_epi8((char)Token::START - 1));
    __m256i const hiV =
            _mm256_cmpgt_epi8(_mm256_set1_epi8((char)Token::END + 1), srcV);
    __m256i const maskV = _mm256_and_si256(loV, hiV);
    return (uint32_t)_mm256_movemask_epi8(maskV);
}
#endif

template <size_t kShortLen>
void shortcopy(char* dst, char const* src, size_t srcSize)
{
    static_assert(kShortLen <= ZS_WILDCOPY_OVERLENGTH);
    memcpy(dst, src, kShortLen);
    if (ZL_UNLIKELY(srcSize > kShortLen)) {
        memcpy(dst + kShortLen, src + kShortLen, srcSize - kShortLen);
    }
}

/// Equivalent to buildBitmaskFallback(bitmask, src, isInSet, 0)
ZL_FORCE_NOINLINE std::string_view buildBitmask(
        uint64_t* bitmask,
        std::string_view& src)
{
    std::fill(bitmask, bitmask + kBitmaskSize, 0);
    size_t blockSize = 0;
#if defined(__AVX2__)
    blockSize = alignDown(std::min(src.size(), kBlockSize), 32);

    uint32_t* bitmask32 = (uint32_t*)bitmask;

    for (size_t i = 0, o = 0; i < blockSize; i += 32, ++o) {
        __m256i const srcV =
                _mm256_loadu_si256((const __m256i_u*)(src.data() + i));
        uint32_t const mask = isInSetV(srcV);
        bitmask32[o]        = mask;
    }
#endif

    return buildBitmaskFallback(bitmask, src, isInSet, blockSize);
}

/// Small wrapper around a variable size ZL_Data.
class InStream {
   public:
    explicit InStream(ZL_Input const* stream)
            : content_((char const*)ZL_Input_ptr(stream)),
              fieldSizes_(ZL_Input_stringLens(stream)),
              fieldSizesEnd_(fieldSizes_ + ZL_Input_numElts(stream))
    {
        fieldSizesFastEnd_ = fieldSizesEnd_;
        size_t size        = 0;
        while (fieldSizesFastEnd_ > fieldSizes_
               && size < ZS_WILDCOPY_OVERLENGTH) {
            --fieldSizesFastEnd_;
            size += *fieldSizesFastEnd_;
        }
    }

    /// Copies the next field into @p dst and return the past-the-end pointer.
    char* read(char* dst)
    {
        ZL_ASSERT_LT(fieldSizes_, fieldSizesEnd_);
        if (fieldSizes_ < fieldSizesFastEnd_) {
            return readFast(dst);
        }
        auto const fieldSize = fieldSizes_[0];
        memcpy(dst, content_, fieldSize);
        content_ += fieldSize;
        ++fieldSizes_;
        return dst + fieldSize;
    }

    size_t remaining() const
    {
        ZL_ASSERT_LE(fieldSizes_, fieldSizesEnd_);
        return size_t(fieldSizesEnd_ - fieldSizes_);
    }

   private:
    char* readFast(char* dst)
    {
        ZL_ASSERT_LT(fieldSizes_, fieldSizesFastEnd_);
        auto const fieldSize = fieldSizes_[0];
        shortcopy<32>(dst, content_, fieldSize);
        content_ += fieldSize;
        ++fieldSizes_;
        return dst + fieldSize;
    }

    char const* content_;
    uint32_t const* fieldSizes_;
    uint32_t const* fieldSizesEnd_;
    uint32_t const* fieldSizesFastEnd_;
};

/// Replaces the tokens in @p src with the corresponding value, given @p bitmask
/// which is 1 for every byte that is a token.
/// NOTE: @p out must be guaranteed to be large enough to hold any result.
ZL_Report replaceTokens(
        ZL_Decoder* dictx,
        char*& out,
        uint64_t const* bitmask,
        std::string_view src,
        InStream& ints,
        InStream& floats,
        InStream& strs)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    size_t idx               = 0;
    uint64_t mask            = bitmask[idx];
    uint32_t skipped         = 0;
    char const* tokenEnd     = src.data();
    char const* const srcEnd = src.data() + src.size();
    char const* fastEnd;
    if (src.size() > 32) {
        fastEnd = srcEnd - 31;
    } else {
        fastEnd = src.data();
    }
    for (;;) {
        if (mask == 0) {
            // Mask has no remaining tokens => go to the next mask
            ++idx;
            if (idx == kBitmaskSize) {
                // Copy the JSON suffix over
                memcpy(out, tokenEnd, (srcEnd - tokenEnd));
                out += (srcEnd - tokenEnd);
                break;
            }
            mask    = bitmask[idx];
            skipped = 0;
            continue;
        }
        ZL_ASSERT_NE(mask, 0);

        // Skip to the start of the token
        auto const toSkip = countFirstZeros(mask);
        mask              = skipN(mask, toSkip);
        skipped += toSkip;
        ZL_ASSERT_LT(skipped, 64);

        char const* token = src.data() + idx * 64 + skipped;

        // Write the JSON prefix
        if (token < fastEnd) {
            shortcopy<16>(out, tokenEnd, token - tokenEnd);
        } else {
            memcpy(out, tokenEnd, (token - tokenEnd));
        }
        out += (token - tokenEnd);

        // Replace the token
        if (token[0] == char(Token::INT)) {
            ZL_ERR_IF_EQ(ints.remaining(), 0, corruption);
            out = ints.read(out);
        } else if (token[0] == char(Token::FLOAT)) {
            ZL_ERR_IF_EQ(floats.remaining(), 0, corruption);
            out = floats.read(out);
        } else if (token[0] == char(Token::STR)) {
            ZL_ERR_IF_EQ(strs.remaining(), 0, corruption);
            out = strs.read(out);
        } else if (token[0] == char(Token::TRUE)) {
            memcpy(out, "true", 4);
            out += 4;
        } else if (token[0] == char(Token::FALSE)) {
            memcpy(out, "false", 5);
            out += 5;
        } else if (token[0] == char(Token::NUL)) {
            memcpy(out, "null", 4);
            out += 4;
        } else {
            ZL_ASSERT_FAIL("Impossible");
        }

        tokenEnd = token + 1;
        mask     = skipN(mask, 1);
        ++skipped;
    }

    return ZL_returnSuccess();
}

ZL_Report jsonExtractDecode(
        ZL_Decoder* dictx,
        ZL_Input const* inputs[]) noexcept
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    ZL_Input const* jsonStream = inputs[0];
    InStream ints{ inputs[1] };
    InStream floats{ inputs[2] };
    InStream strs{ inputs[3] };

    std::string_view json{ (char const*)ZL_Input_ptr(jsonStream),
                           ZL_Input_numElts(jsonStream) };

    // Determine the maximum possible output size, plus ZS_WILDCOPY_OVERLENGTH.
    // The maximum size needs 5x jsonStream because the longest replacement
    // token without corresponding bytes in another stream is "false".
    size_t const outBound = ZL_Input_contentSize(jsonStream) * 5
            + ZL_Input_contentSize(inputs[1]) + ZL_Input_contentSize(inputs[2])
            + ZL_Input_contentSize(inputs[3]) + ZS_WILDCOPY_OVERLENGTH;
    ZL_Output* outStream = ZL_Decoder_create1OutStream(dictx, outBound, 1);
    ZL_ERR_IF_NULL(outStream, allocation);

    std::array<uint64_t, kBitmaskSize> bitmask;
    char* out = (char*)ZL_Output_ptr(outStream);
    while (!json.empty()) {
        auto const block = buildBitmask(bitmask.data(), json);
        ZL_ERR_IF_ERR(replaceTokens(
                dictx, out, bitmask.data(), block, ints, floats, strs));
    }

    size_t const size = size_t(out - (char*)ZL_Output_ptr(outStream));
    ZL_ERR_IF_ERR(ZL_Output_commit(outStream, size));

    return ZL_returnSuccess();
}

} // namespace
} // namespace zstrong

ZL_Report ZS2_DCtx_registerJsonExtract(ZL_DCtx* dctx, ZL_IDType transformID)
{
    std::array<ZL_Type, 4> const kOutStreams = {
        ZL_Type_serial, ZL_Type_string, ZL_Type_string, ZL_Type_string
    };
    ZL_TypedGraphDesc graph = {
        .CTid           = transformID,
        .inStreamType   = ZL_Type_serial,
        .outStreamTypes = kOutStreams.data(),
        .nbOutStreams   = kOutStreams.size(),
    };
    ZL_TypedDecoderDesc desc = {
        .gd          = graph,
        .transform_f = zstrong::jsonExtractDecode,
        .name        = "zl_custom.json_extract",
    };
    return ZL_DCtx_registerTypedDecoder(dctx, &desc);
}
