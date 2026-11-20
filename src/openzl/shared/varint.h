// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMMON_VARINT_H
#define ZSTRONG_COMMON_VARINT_H

#include "openzl/shared/bits.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/utils.h"
#include "openzl/zl_errors.h"

#if ZL_HAS_BMI2
#    include <immintrin.h>
#endif

ZL_BEGIN_C_DECLS

/**
 * Varint format:
 * The input is coded in little-endian format. 7-bits are encoded per byte. The
 * high-bit of each byte is 1 if there is another byte following, or 0 if it is
 * the last byte. This means a 32-bit integer can be encoded in at most 5 bytes,
 * and a 64-bit integer can be encoded in at most 10 bytes.
 *
 * Details:
 * https://developers.google.com/protocol-buffers/docs/encoding#varints
 */

#define ZL_VARINT_LENGTH_32 5
#define ZL_VARINT_LENGTH_64 10

ZL_INLINE size_t ZL_varintSize(uint64_t val)
{
    ZL_ASSERT(ZL_isLittleEndian());
    size_t const highbit = 64 - (size_t)ZL_clz64(val | 1);
    return (highbit + 6) / 7;
}

ZL_INLINE size_t ZL_varintEncode(uint64_t val, uint8_t* const dst)
{
    uint8_t* ptr = dst;
    while (val >= 128) {
        *ptr++ = 0x80 | (uint8_t)(val & 0x7f);
        val >>= 7;
    }
    *ptr++ = (uint8_t)val;
    return (size_t)(ptr - dst);
}

/**
 * Encodes a 32-bit varint, but uses a fast variant that writes up to
 * ZL_VARINT_FAST_OVERWRITE_32 bytes, even if the output isn't that large.
 * @pre dst must be at least ZL_VARINT_FAST_OVERWRITE_32 bytes large.
 * @returns The encoded size of the varint.
 */
#define ZL_VARINT_FAST_OVERWRITE_32 8
ZL_INLINE size_t ZL_varintEncode32Fast(uint32_t val, uint8_t* const dst)
{
#if ZL_HAS_BMI2
    uint64_t const MSBS32[33] = {
        0x500000080808080, 0x500000080808080, 0x500000080808080,
        0x500000080808080, 0x400000000808080, 0x400000000808080,
        0x400000000808080, 0x400000000808080, 0x400000000808080,
        0x400000000808080, 0x400000000808080, 0x300000000008080,
        0x300000000008080, 0x300000000008080, 0x300000000008080,
        0x300000000008080, 0x300000000008080, 0x300000000008080,
        0x200000000000080, 0x200000000000080, 0x200000000000080,
        0x200000000000080, 0x200000000000080, 0x200000000000080,
        0x200000000000080, 0x100000000000000, 0x100000000000000,
        0x100000000000000, 0x100000000000000, 0x100000000000000,
        0x100000000000000, 0x100000000000000, 0x100000000000000,
    };
    size_t lzcnt         = _lzcnt_u32(val);
    uint64_t p           = _pdep_u64(val, 0x0000000f7f7f7f7f);
    uint64_t msbs        = MSBS32[lzcnt];
    uint8_t bytes_needed = (uint8_t)(MSBS32[lzcnt] >> 56);
    uint64_t res         = p | msbs;
    ZL_writeLE64(dst, res);
    return bytes_needed;
#else
    return ZL_varintEncode(val, dst);
#endif
}

/**
 * Encodes a 64-bit varint, but uses a fast variant that writes up to
 * ZL_VARINT_FAST_OVERWRITE_64 bytes, even if the output isn't that large.
 * @pre dst must be at least ZL_VARINT_FAST_OVERWRITE_64 bytes large.
 * @returns The encoded size of the varint.
 */
#define ZL_VARINT_FAST_OVERWRITE_64 10
ZL_FORCE_INLINE size_t ZL_varintEncode64Fast(uint64_t value, uint8_t* const dst)
{
    // Code taken from thrift/lib/cpp/util/VarintUtils-inl.h
#if ZL_HAS_BMI2
    if (value < 0x80) {
        *dst = (uint8_t)value;
        return 1;
    }

    uint64_t const kMask           = 0x8080808080808080ULL;
    static const uint8_t kSize[64] = { 10, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8,
                                       8,  8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6,
                                       6,  6, 6, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4,
                                       4,  4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2,
                                       2,  2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1 };

    int const clzll = ZL_clz64(value);
    // Only the first 56 bits of @value will be deposited in @v.
    uint8_t const size = kSize[clzll];
    uint64_t v         = _pdep_u64(value, ~kMask) | kMask;
    v                  = _bzhi_u64(v, (unsigned)(size * 8 - 1));

    ZL_writeLE64(dst, v);
    dst[8] = (uint8_t)(value >> 56);
    dst[9] = 1;
    return size;
#else
    return ZL_varintEncode(value, dst);
#endif
}

ZL_INLINE ZL_RESULT_OF(uint64_t)
        ZL_varintDecode(const uint8_t** src, const uint8_t* end)
{
    ZL_RESULT_DECLARE_SCOPE(uint64_t, (ZL_OperationContext*)NULL);
    int8_t const* ptr  = (int8_t const*)*src;
    int8_t const* endi = (int8_t const*)end;
    uint64_t val       = 0;
    if (ZL_LIKELY((size_t)(endi - ptr) >= ZL_VARINT_LENGTH_64)) {
        int64_t b;
        do {
            b   = *ptr++;
            val = (uint64_t)(b & 0x7f);
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 7;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 14;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 21;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 28;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 35;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 42;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 49;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 56;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x01) << 63;
            if (b >= 0) {
                break;
            }
            ZL_ERR(GENERIC);
        } while (false);
    } else {
        int shift = 0;
        while (ptr != endi && *ptr < 0) {
            val |= (uint64_t)(*ptr++ & 0x7f) << shift;
            shift += 7;
        }
        if (ptr == endi) {
            ZL_ERR(GENERIC);
        }
        val |= (uint64_t)(*ptr++) << shift;
    }
    *src = (uint8_t const*)ptr;
    return ZL_WRAP_VALUE(val);
}

ZL_INLINE ZL_RESULT_OF(uint64_t) ZL_varintDecodeStrictImpl(
        const uint8_t** src,
        const uint8_t* end,
        size_t kWidth)
{
    ZL_RESULT_DECLARE_SCOPE(uint64_t, (ZL_OperationContext*)NULL);
    int8_t const* ptr  = (int8_t const*)*src;
    int8_t const* endi = (int8_t const*)end;
    uint64_t val       = 0;
    int64_t b;
    if (ZL_LIKELY(
                (size_t)(endi - ptr)
                >= (kWidth == 4 ? ZL_VARINT_LENGTH_32 : ZL_VARINT_LENGTH_64))) {
        bool zeroAllowed = true;
        do {
            b   = *ptr++;
            val = (uint64_t)(b & 0x7f);
            if (b >= 0) {
                break;
            }
            zeroAllowed = false;
            b           = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 7;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 14;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 21;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 28;
            if (b >= 0) {
                if (kWidth == 4 && (b & ~0xF) != 0) {
                    ZL_ERR(GENERIC, "Varint32 has too many bits!");
                }
                break;
            }
            if (kWidth == 4) {
                ZL_ERR(GENERIC, "Varint32 has too many bytes!");
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 35;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 42;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 49;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x7f) << 56;
            if (b >= 0) {
                break;
            }
            b = *ptr++;
            val |= (uint64_t)(b & 0x01) << 63;
            if (b >= 0) {
                if ((b & ~0x01) != 0) {
                    ZL_ERR(GENERIC, "Varint64 has too many bits!");
                }
                break;
            }
            ZL_ERR(GENERIC, "Varint64 has too many bytes!");
        } while (false);
        // The last byte was zero means that the varint was encoded
        // inefficiently
        if (!zeroAllowed && b == 0x00) {
            ZL_ERR(GENERIC, "Varint has trailing 0x00 bytes");
        }
    } else {
        int iter = 0;
        while (ptr != endi && *ptr < 0) {
            val |= (uint64_t)(*ptr++ & 0x7f) << (7 * iter);
            ++iter;
            // Impossible because either the src is too small, or we would've
            // taken the other path.
            ZL_ASSERT(!(kWidth == 4 && iter >= ZL_VARINT_LENGTH_32));
            ZL_ASSERT(!(kWidth == 8 && iter >= ZL_VARINT_LENGTH_64));
        }
        if (ptr == endi) {
            ZL_ERR(GENERIC, "Varint not finished!");
        }
        // Impossible because either the src is too small, or we would've
        // taken the other path.
        ZL_ASSERT(!(kWidth == 4 && iter >= ZL_VARINT_LENGTH_32 - 1));
        ZL_ASSERT(!(kWidth == 8 && iter >= ZL_VARINT_LENGTH_64 - 1));
        if (iter > 0 && *ptr == 0) {
            ZL_ERR(GENERIC, "Varint has trailing 0x00 bytes");
        }

        val |= (uint64_t)(*ptr++) << (iter * 7);
    }
    *src = (uint8_t const*)ptr;
    return ZL_WRAP_VALUE(val);
}

/**
 * Decodes a 32-bit varint and ensures that it was encoded in the canonical
 * format.
 * - Verifies that there aren't extra bytes. E.g. the varint 0x00FF.
 * - Verifies that there are no no more 32-bits set in the varint.
 * Any varint decoded by this function is guaranteed to be re-encoded losslessly
 * with ZS_encodeVarint() and ZS_encodeVarint32Fast().
 *
 * This function is more expensive than ZL_varintDecode() so it should only be
 * used when you need a guarantee that the varint can be re-encoded losslessly.
 *
 * @returns The decoded varint or an error.
 */
ZL_INLINE ZL_RESULT_OF(uint64_t)
        ZL_varintDecode32Strict(const uint8_t** src, const uint8_t* end)
{
    return ZL_varintDecodeStrictImpl(src, end, 4);
}

/**
 * Decodes a 64-bit varint and ensures that it was encoded in the canonical
 * format.
 * - Verifies that there aren't extra bytes. E.g. the varint 0x00FF.
 * - Verifies that there are no no more 64-bits set in the varint.
 * Any varint decoded by this function is guaranteed to be re-encoded losslessly
 * with ZS_encodeVarint() and ZS_encodeVarint64Fast().
 *
 * This function is more expensive than ZL_varintDecode() so it should only be
 * used when you need a guarantee that the varint can be re-encoded losslessly.
 *
 * @returns The decoded varint or an error.
 */
ZL_INLINE ZL_RESULT_OF(uint64_t)
        ZL_varintDecode64Strict(const uint8_t** src, const uint8_t* end)
{
    return ZL_varintDecodeStrictImpl(src, end, 8);
}

ZL_END_C_DECLS

#endif
