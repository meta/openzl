// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/bitSplit/encode_bitSplit_kernel.h"

#include <assert.h> /* assert */
#include <string.h>

bool ZL_bitSplit_paramsAreValid(
        const uint8_t* bitWidths,
        size_t nbWidths,
        size_t inputEltWidthBits,
        size_t* sumWidths)
{
    assert(bitWidths != NULL);
    assert(nbWidths > 0);

    size_t sum = 0;
    for (size_t i = 0; i < nbWidths; i++) {
        if (bitWidths[i] == 0) {
            return false;
        }
        sum += bitWidths[i];
    }
    if (sum > inputEltWidthBits) {
        return false;
    }
    if (sumWidths != NULL) {
        *sumWidths = sum;
    }
    return true;
}

static inline bool checkTopBitsZero(
        const void* src,
        size_t srcEltWidth,
        size_t nbElts,
        uint64_t mask)
{
    uint64_t allBits = 0;
    for (size_t e = 0; e < nbElts; e++) {
        switch (srcEltWidth) {
            case 1:
                allBits |= ((const uint8_t*)src)[e];
                break;
            case 2:
                allBits |= ((const uint16_t*)src)[e];
                break;
            case 4:
                allBits |= ((const uint32_t*)src)[e];
                break;
            case 8:
                allBits |= ((const uint64_t*)src)[e];
                break;
            default:
                assert(false);
                break;
        }
    }
    return (allBits & mask) == 0;
}

/*
 * Specialized encoder for bf16 pattern:
 * - srcEltWidth is 2 (uint16_t)
 * - 3 streams with bitWidths {7, 8, 1}
 * - All dstEltWidths are 1 (uint8_t)
 *
 * bf16 layout: [mantissa:7][exponent:8][sign:1] = 16 bits
 */
static inline void
bitSplit_bf16(void* const dstPtrs[], size_t nbElts, const void* src)
{
    uint8_t* restrict const mantissa     = (uint8_t*)dstPtrs[0];
    uint8_t* restrict const exponent     = (uint8_t*)dstPtrs[1];
    uint8_t* restrict const sign         = (uint8_t*)dstPtrs[2];
    const uint16_t* restrict const src16 = (const uint16_t*)src;

    for (size_t e = 0; e < nbElts; e++) {
        uint16_t const value = src16[e];
        mantissa[e]          = (uint8_t)(value & 0x7F);         /* bits 0-6 */
        exponent[e]          = (uint8_t)((value >> 7) & 0xFF);  /* bits 7-14 */
        sign[e]              = (uint8_t)((value >> 15) & 0x01); /* bit 15 */
    }
}

/*
 * Specialized encoder for fp16 pattern:
 * - srcEltWidth is 2 (uint16_t)
 * - 3 streams with bitWidths {10, 5, 1}
 * - dstEltWidths are {2, 1, 1} (uint16_t, uint8_t, uint8_t)
 *
 * fp16 layout: [mantissa:10][exponent:5][sign:1] = 16 bits
 */
static inline void
bitSplit_fp16(void* const dstPtrs[], size_t nbElts, const void* src)
{
    uint16_t* restrict const mantissa    = (uint16_t*)dstPtrs[0];
    uint8_t* restrict const exponent     = (uint8_t*)dstPtrs[1];
    uint8_t* restrict const sign         = (uint8_t*)dstPtrs[2];
    const uint16_t* restrict const src16 = (const uint16_t*)src;

    for (size_t e = 0; e < nbElts; e++) {
        uint16_t const value = src16[e];
        mantissa[e]          = (uint16_t)(value & 0x3FF);       /* bits 0-9 */
        exponent[e]          = (uint8_t)((value >> 10) & 0x1F); /* bits 10-14 */
        sign[e]              = (uint8_t)((value >> 15) & 0x1);  /* bit 15 */
    }
}

/*
 * Specialized encoder for fp32 pattern:
 * - srcEltWidth is 4 (uint32_t)
 * - 3 streams with bitWidths {23, 8, 1}
 * - dstEltWidths are {4, 1, 1} (uint32_t, uint8_t, uint8_t)
 *
 * fp32 layout: [mantissa:23][exponent:8][sign:1] = 32 bits
 */
static inline void
bitSplit_fp32(void* const dstPtrs[], size_t nbElts, const void* src)
{
    uint32_t* restrict const mantissa    = (uint32_t*)dstPtrs[0];
    uint8_t* restrict const exponent     = (uint8_t*)dstPtrs[1];
    uint8_t* restrict const sign         = (uint8_t*)dstPtrs[2];
    const uint32_t* restrict const src32 = (const uint32_t*)src;

    for (size_t e = 0; e < nbElts; e++) {
        uint32_t const value = src32[e];
        mantissa[e]          = (uint32_t)(value & 0x7FFFFF);    /* bits 0-22 */
        exponent[e]          = (uint8_t)((value >> 23) & 0xFF); /* bits 23-30 */
        sign[e]              = (uint8_t)((value >> 31) & 0x1);  /* bit 31 */
    }
}

/*
 * Specialized encoder for fp64 pattern:
 * - srcEltWidth is 8 (uint64_t)
 * - 3 streams with bitWidths {52, 11, 1}
 * - dstEltWidths are {8, 2, 1} (uint64_t, uint16_t, uint8_t)
 *
 * fp64 layout: [mantissa:52][exponent:11][sign:1] = 64 bits
 */
static inline void
bitSplit_fp64(void* const dstPtrs[], size_t nbElts, const void* src)
{
    uint64_t* restrict const mantissa    = (uint64_t*)dstPtrs[0];
    uint16_t* restrict const exponent    = (uint16_t*)dstPtrs[1];
    uint8_t* restrict const sign         = (uint8_t*)dstPtrs[2];
    const uint64_t* restrict const src64 = (const uint64_t*)src;

    for (size_t e = 0; e < nbElts; e++) {
        uint64_t const value = src64[e];
        mantissa[e] = (uint64_t)(value & 0xFFFFFFFFFFFFF); /* bits 0-51 */
        exponent[e] = (uint16_t)((value >> 52) & 0x7FF);   /* bits 52-62 */
        sign[e]     = (uint8_t)((value >> 63) & 0x1);      /* bit 63 */
    }
}

/*
 * Check if parameters match the bf16 encode pattern:
 * - srcEltWidth is 2 (uint16_t)
 * - 3 streams with bitWidths {7, 8, 1}
 * - All dstEltWidths are 1 (uint8_t)
 */
static inline bool isEncodeBf16Pattern(
        size_t srcEltWidth,
        const size_t* dstEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (srcEltWidth != 2)
        return false;
    if (nbWidths != 3)
        return false;
    if (bitWidths[0] != 7 || bitWidths[1] != 8 || bitWidths[2] != 1)
        return false;
    if (dstEltWidths[0] != 1 || dstEltWidths[1] != 1 || dstEltWidths[2] != 1)
        return false;
    return true;
}

/*
 * Check if parameters match the fp16 encode pattern:
 * - srcEltWidth is 2 (uint16_t)
 * - 3 streams with bitWidths {10, 5, 1}
 * - dstEltWidths are {2, 1, 1} (uint16_t, uint8_t, uint8_t)
 */
static inline bool isEncodeFp16Pattern(
        size_t srcEltWidth,
        const size_t* dstEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (srcEltWidth != 2)
        return false;
    if (nbWidths != 3)
        return false;
    if (bitWidths[0] != 10 || bitWidths[1] != 5 || bitWidths[2] != 1)
        return false;
    if (dstEltWidths[0] != 2 || dstEltWidths[1] != 1 || dstEltWidths[2] != 1)
        return false;
    return true;
}

/*
 * Check if parameters match the fp32 encode pattern:
 * - srcEltWidth is 4 (uint32_t)
 * - 3 streams with bitWidths {23, 8, 1}
 * - dstEltWidths are {4, 1, 1} (uint32_t, uint8_t, uint8_t)
 */
static inline bool isEncodeFp32Pattern(
        size_t srcEltWidth,
        const size_t* dstEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (srcEltWidth != 4)
        return false;
    if (nbWidths != 3)
        return false;
    if (bitWidths[0] != 23 || bitWidths[1] != 8 || bitWidths[2] != 1)
        return false;
    if (dstEltWidths[0] != 4 || dstEltWidths[1] != 1 || dstEltWidths[2] != 1)
        return false;
    return true;
}

/*
 * Check if parameters match the fp64 encode pattern:
 * - srcEltWidth is 8 (uint64_t)
 * - 3 streams with bitWidths {52, 11, 1}
 * - dstEltWidths are {8, 2, 1} (uint64_t, uint16_t, uint8_t)
 */
static inline bool isEncodeFp64Pattern(
        size_t srcEltWidth,
        const size_t* dstEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (srcEltWidth != 8)
        return false;
    if (nbWidths != 3)
        return false;
    if (bitWidths[0] != 52 || bitWidths[1] != 11 || bitWidths[2] != 1)
        return false;
    if (dstEltWidths[0] != 8 || dstEltWidths[1] != 2 || dstEltWidths[2] != 1)
        return false;
    return true;
}

bool ZL_bitSplit_topBitsAreZero(
        const void* src,
        size_t srcEltWidth,
        size_t nbElts,
        size_t sumWidths)
{
    if (nbElts == 0)
        return true;

    assert(src != NULL);
    assert(srcEltWidth == 1 || srcEltWidth == 2 || srcEltWidth == 4
           || srcEltWidth == 8);

    if (sumWidths >= srcEltWidth * 8) {
        return true; /* Full coverage, no top bits to check */
    }
    uint64_t const mask = ~((1ULL << sumWidths) - 1);

    switch (srcEltWidth) {
        case 1:
            return checkTopBitsZero(src, 1, nbElts, mask);
        case 2:
            return checkTopBitsZero(src, 2, nbElts, mask);
        case 4:
            return checkTopBitsZero(src, 4, nbElts, mask);
        case 8:
            return checkTopBitsZero(src, 8, nbElts, mask);
        default:
            assert(false);
            break;
    }
    return true;
}

/*
 * Generic encoder: extracts one field at a time, over blocks small enough to
 * stay in L1 cache, so that each loop has fixed widths and vectorizes.
 */
#define BITSPLIT_ENCODE_BLOCK_ELTS 2048

#define BITSPLIT_ENCODE_FIELD(SrcType, DstType)                          \
    do {                                                                 \
        const SrcType* restrict const in = (const SrcType*)src + start;  \
        DstType* restrict const out      = (DstType*)dstPtrs[i] + start; \
        SrcType const mask = (SrcType)(~0ULL >> (64 - bitWidths[i]));    \
        for (size_t e = 0; e < blockElts; e++)                           \
            out[e] = (DstType)((in[e] >> bitPos) & mask);                \
    } while (0)

#define BITSPLIT_ENCODE_FIELD_FROM(SrcType)           \
    switch (dstEltWidths[i]) {                        \
        case 1:                                       \
            BITSPLIT_ENCODE_FIELD(SrcType, uint8_t);  \
            break;                                    \
        case 2:                                       \
            BITSPLIT_ENCODE_FIELD(SrcType, uint16_t); \
            break;                                    \
        case 4:                                       \
            BITSPLIT_ENCODE_FIELD(SrcType, uint32_t); \
            break;                                    \
        case 8:                                       \
            BITSPLIT_ENCODE_FIELD(SrcType, uint64_t); \
            break;                                    \
        default:                                      \
            assert(false);                            \
            break;                                    \
    }

static void encodeBlock(
        void* const dstPtrs[],
        const size_t* dstEltWidths,
        size_t start,
        size_t blockElts,
        const void* src,
        size_t srcEltWidth,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    size_t bitPos = 0;
    for (size_t i = 0; i < nbWidths; i++) {
        switch (srcEltWidth) {
            case 1:
                BITSPLIT_ENCODE_FIELD_FROM(uint8_t);
                break;
            case 2:
                BITSPLIT_ENCODE_FIELD_FROM(uint16_t);
                break;
            case 4:
                BITSPLIT_ENCODE_FIELD_FROM(uint32_t);
                break;
            case 8:
                BITSPLIT_ENCODE_FIELD_FROM(uint64_t);
                break;
            default:
                assert(false);
                break;
        }
        bitPos += bitWidths[i];
    }
}

void ZL_bitSplitEncode(
        void* const dstPtrs[],
        const size_t* dstEltWidths,
        size_t nbElts,
        const void* src,
        size_t srcEltWidth,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    if (nbElts == 0)
        return;

    assert(dstPtrs != NULL);
    assert(dstEltWidths != NULL);
    assert(src != NULL);
    assert(srcEltWidth == 1 || srcEltWidth == 2 || srcEltWidth == 4
           || srcEltWidth == 8);
    assert(bitWidths != NULL);
    assert(nbWidths > 0);
    assert(nbWidths <= 64);

    /* Validate parameters and individual streams */
    {
        size_t sumBitWidths = 0;
        for (size_t i = 0; i < nbWidths; i++) {
            assert(dstPtrs[i] != NULL);
            assert(dstEltWidths[i] == 1 || dstEltWidths[i] == 2
                   || dstEltWidths[i] == 4 || dstEltWidths[i] == 8);
            assert(bitWidths[i] > 0);
            assert(bitWidths[i] <= dstEltWidths[i] * 8);
            sumBitWidths += bitWidths[i];
        }
        assert(sumBitWidths <= srcEltWidth * 8);
        assert(ZL_bitSplit_topBitsAreZero(
                src, srcEltWidth, nbElts, sumBitWidths));
        (void)sumBitWidths;
    }

    /* Check for specialized patterns and dispatch */
    if (isEncodeBf16Pattern(srcEltWidth, dstEltWidths, bitWidths, nbWidths)) {
        bitSplit_bf16(dstPtrs, nbElts, src);
        return;
    }
    if (isEncodeFp16Pattern(srcEltWidth, dstEltWidths, bitWidths, nbWidths)) {
        bitSplit_fp16(dstPtrs, nbElts, src);
        return;
    }
    if (isEncodeFp32Pattern(srcEltWidth, dstEltWidths, bitWidths, nbWidths)) {
        bitSplit_fp32(dstPtrs, nbElts, src);
        return;
    }
    if (isEncodeFp64Pattern(srcEltWidth, dstEltWidths, bitWidths, nbWidths)) {
        bitSplit_fp64(dstPtrs, nbElts, src);
        return;
    }

    /* Generic path */
    for (size_t start = 0; start < nbElts;
         start += BITSPLIT_ENCODE_BLOCK_ELTS) {
        size_t const blockElts = nbElts - start < BITSPLIT_ENCODE_BLOCK_ELTS
                ? nbElts - start
                : BITSPLIT_ENCODE_BLOCK_ELTS;
        encodeBlock(
                dstPtrs,
                dstEltWidths,
                start,
                blockElts,
                src,
                srcEltWidth,
                bitWidths,
                nbWidths);
    }
}
