// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>  /* assert */
#include <stdbool.h> /* bool */

#include "openzl/codecs/bitSplit/decode_bitSplit_kernel.h"

/*
 * Specialized decoder for bf16 pattern:
 * - 3 streams with bitWidths {7, 8, 1}
 * - All srcEltWidths are 1 (uint8_t)
 * - dstEltWidth is 2 (uint16_t)
 *
 * Layout: [mantissa:7][exponent:8][sign:1] = 16 bits
 */
static inline void
decodeBf16(void* dst, size_t nbElts, const void* const srcPtrs[])
{
    uint16_t* const dst16         = (uint16_t*)dst;
    const uint8_t* const mantissa = (const uint8_t*)srcPtrs[0];
    const uint8_t* const exponent = (const uint8_t*)srcPtrs[1];
    const uint8_t* const sign     = (const uint8_t*)srcPtrs[2];

    for (size_t e = 0; e < nbElts; e++) {
        uint16_t value = (uint16_t)mantissa[e]; /* bits 0-6 */
        value |= (uint16_t)exponent[e] << 7;    /* bits 7-14 */
        value |= (uint16_t)sign[e] << 15;       /* bit 15 */
        dst16[e] = value;
    }
}

/*
 * Specialized decoder for fp16 pattern:
 * - 3 streams with bitWidths {10, 5, 1}
 * - srcEltWidths are {2, 1, 1} (uint16_t, uint8_t, uint8_t)
 * - dstEltWidth is 2 (uint16_t)
 *
 * Layout: [mantissa:10][exponent:5][sign:1] = 16 bits
 */
static inline void
decodeFp16(void* dst, size_t nbElts, const void* const srcPtrs[])
{
    uint16_t* const dst16          = (uint16_t*)dst;
    const uint16_t* const mantissa = (const uint16_t*)srcPtrs[0];
    const uint8_t* const exponent  = (const uint8_t*)srcPtrs[1];
    const uint8_t* const sign      = (const uint8_t*)srcPtrs[2];

    for (size_t e = 0; e < nbElts; e++) {
        uint16_t value = mantissa[e];         /* bits 0-9 */
        value |= (uint16_t)exponent[e] << 10; /* bits 10-14 */
        value |= (uint16_t)sign[e] << 15;     /* bit 15 */
        dst16[e] = value;
    }
}

/*
 * Specialized decoder for fp32 pattern:
 * - 3 streams with bitWidths {23, 8, 1}
 * - srcEltWidths are {4, 1, 1} (uint32_t, uint8_t, uint8_t)
 * - dstEltWidth is 4 (uint32_t)
 *
 * Layout: [mantissa:23][exponent:8][sign:1] = 32 bits
 */
static inline void
decodeFp32(void* dst, size_t nbElts, const void* const srcPtrs[])
{
    uint32_t* const dst32          = (uint32_t*)dst;
    const uint32_t* const mantissa = (const uint32_t*)srcPtrs[0];
    const uint8_t* const exponent  = (const uint8_t*)srcPtrs[1];
    const uint8_t* const sign      = (const uint8_t*)srcPtrs[2];

    for (size_t e = 0; e < nbElts; e++) {
        uint32_t value = mantissa[e];         /* bits 0-22 */
        value |= (uint32_t)exponent[e] << 23; /* bits 23-30 */
        value |= (uint32_t)sign[e] << 31;     /* bit 31 */
        dst32[e] = value;
    }
}

/*
 * Specialized decoder for fp64 pattern:
 * - 3 streams with bitWidths {52, 11, 1}
 * - srcEltWidths are {8, 2, 1} (uint64_t, uint16_t, uint8_t)
 * - dstEltWidth is 8 (uint64_t)
 *
 * Layout: [mantissa:52][exponent:11][sign:1] = 64 bits
 */
static inline void
decodeFp64(void* dst, size_t nbElts, const void* const srcPtrs[])
{
    uint64_t* const dst64          = (uint64_t*)dst;
    const uint64_t* const mantissa = (const uint64_t*)srcPtrs[0];
    const uint16_t* const exponent = (const uint16_t*)srcPtrs[1];
    const uint8_t* const sign      = (const uint8_t*)srcPtrs[2];

    for (size_t e = 0; e < nbElts; e++) {
        uint64_t value = mantissa[e];         /* bits 0-51 */
        value |= (uint64_t)exponent[e] << 52; /* bits 52-62 */
        value |= (uint64_t)sign[e] << 63;     /* bit 63 */
        dst64[e] = value;
    }
}

/*
 * Check if parameters match the bf16 pattern.
 */
static inline bool isBf16Pattern(
        size_t dstEltWidth,
        const size_t* srcEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (dstEltWidth != 2)
        return false;
    if (nbWidths != 3)
        return false;
    if (bitWidths[0] != 7 || bitWidths[1] != 8 || bitWidths[2] != 1)
        return false;
    if (srcEltWidths[0] != 1 || srcEltWidths[1] != 1 || srcEltWidths[2] != 1)
        return false;
    return true;
}

/*
 * Check if parameters match the fp16 pattern.
 */
static inline bool isFp16Pattern(
        size_t dstEltWidth,
        const size_t* srcEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (dstEltWidth != 2)
        return false;
    if (nbWidths != 3)
        return false;
    if (bitWidths[0] != 10 || bitWidths[1] != 5 || bitWidths[2] != 1)
        return false;
    if (srcEltWidths[0] != 2 || srcEltWidths[1] != 1 || srcEltWidths[2] != 1)
        return false;
    return true;
}

/*
 * Check if parameters match the fp32 pattern.
 */
static inline bool isFp32Pattern(
        size_t dstEltWidth,
        const size_t* srcEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (dstEltWidth != 4)
        return false;
    if (nbWidths != 3)
        return false;
    if (bitWidths[0] != 23 || bitWidths[1] != 8 || bitWidths[2] != 1)
        return false;
    if (srcEltWidths[0] != 4 || srcEltWidths[1] != 1 || srcEltWidths[2] != 1)
        return false;
    return true;
}

/*
 * Check if parameters match the fp64 pattern.
 */
static inline bool isFp64Pattern(
        size_t dstEltWidth,
        const size_t* srcEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (dstEltWidth != 8)
        return false;
    if (nbWidths != 3)
        return false;
    if (bitWidths[0] != 52 || bitWidths[1] != 11 || bitWidths[2] != 1)
        return false;
    if (srcEltWidths[0] != 8 || srcEltWidths[1] != 2 || srcEltWidths[2] != 1)
        return false;
    return true;
}

/*
 * Generic decoder: rebuilds the elements one field at a time, over blocks small
 * enough to stay in L1 cache, so that each loop has fixed widths and
 * vectorizes.
 */
#define BITSPLIT_DECODE_BLOCK_ELTS 2048

#define BITSPLIT_DECODE_FIELD(DstType, SrcType)                                \
    do {                                                                       \
        DstType* restrict const out      = (DstType*)dst + start;              \
        const SrcType* restrict const in = (const SrcType*)srcPtrs[i] + start; \
        if (i == 0) {                                                          \
            for (size_t e = 0; e < blockElts; e++)                             \
                out[e] = (DstType)in[e];                                       \
        } else {                                                               \
            for (size_t e = 0; e < blockElts; e++)                             \
                out[e] |= (DstType)((DstType)in[e] << bitPos);                 \
        }                                                                      \
    } while (0)

#define BITSPLIT_DECODE_FIELD_INTO(DstType)           \
    switch (srcEltWidths[i]) {                        \
        case 1:                                       \
            BITSPLIT_DECODE_FIELD(DstType, uint8_t);  \
            break;                                    \
        case 2:                                       \
            BITSPLIT_DECODE_FIELD(DstType, uint16_t); \
            break;                                    \
        case 4:                                       \
            BITSPLIT_DECODE_FIELD(DstType, uint32_t); \
            break;                                    \
        case 8:                                       \
            BITSPLIT_DECODE_FIELD(DstType, uint64_t); \
            break;                                    \
        default:                                      \
            assert(false);                            \
            break;                                    \
    }

static void decodeBlock(
        void* dst,
        size_t dstEltWidth,
        size_t start,
        size_t blockElts,
        const void* const srcPtrs[],
        const size_t* srcEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    size_t bitPos = 0;
    for (size_t i = 0; i < nbWidths; i++) {
        switch (dstEltWidth) {
            case 1:
                BITSPLIT_DECODE_FIELD_INTO(uint8_t);
                break;
            case 2:
                BITSPLIT_DECODE_FIELD_INTO(uint16_t);
                break;
            case 4:
                BITSPLIT_DECODE_FIELD_INTO(uint32_t);
                break;
            case 8:
                BITSPLIT_DECODE_FIELD_INTO(uint64_t);
                break;
            default:
                assert(false);
                break;
        }
        bitPos += bitWidths[i];
    }
}

void ZL_bitSplitDecode(
        void* dst,
        size_t dstEltWidth,
        size_t nbElts,
        const void* const srcPtrs[],
        const size_t* srcEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (nbElts == 0)
        return;

    assert(dst != NULL);
    assert(dstEltWidth == 1 || dstEltWidth == 2 || dstEltWidth == 4
           || dstEltWidth == 8);
    assert(srcPtrs != NULL);
    assert(srcEltWidths != NULL);
    assert(bitWidths != NULL);
    assert(nbWidths > 0);
    assert(nbWidths <= 64);

    /* Validate sum of bit widths fits in destination */
    {
        size_t sumBitWidths = 0;
        for (size_t i = 0; i < nbWidths; i++) {
            assert(srcPtrs[i] != NULL);
            assert(srcEltWidths[i] == 1 || srcEltWidths[i] == 2
                   || srcEltWidths[i] == 4 || srcEltWidths[i] == 8);
            assert(bitWidths[i] > 0);
            assert(bitWidths[i] <= srcEltWidths[i] * 8);
            sumBitWidths += bitWidths[i];
        }
        assert(sumBitWidths <= dstEltWidth * 8);
        (void)sumBitWidths;
    }

    /* Check for specialized patterns and dispatch */
    if (isBf16Pattern(dstEltWidth, srcEltWidths, bitWidths, nbWidths)) {
        decodeBf16(dst, nbElts, srcPtrs);
        return;
    }
    if (isFp16Pattern(dstEltWidth, srcEltWidths, bitWidths, nbWidths)) {
        decodeFp16(dst, nbElts, srcPtrs);
        return;
    }
    if (isFp32Pattern(dstEltWidth, srcEltWidths, bitWidths, nbWidths)) {
        decodeFp32(dst, nbElts, srcPtrs);
        return;
    }
    if (isFp64Pattern(dstEltWidth, srcEltWidths, bitWidths, nbWidths)) {
        decodeFp64(dst, nbElts, srcPtrs);
        return;
    }

    /* Generic path */
    for (size_t start = 0; start < nbElts;
         start += BITSPLIT_DECODE_BLOCK_ELTS) {
        size_t const blockElts = nbElts - start < BITSPLIT_DECODE_BLOCK_ELTS
                ? nbElts - start
                : BITSPLIT_DECODE_BLOCK_ELTS;
        decodeBlock(
                dst,
                dstEltWidth,
                start,
                blockElts,
                srcPtrs,
                srcEltWidths,
                bitWidths,
                nbWidths);
    }
}
