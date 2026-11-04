// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/bitSplit/encode_bitsplit_top8_binding.h"
#include "openzl/codecs/bitSplit/encode_bitSplit_binding.h" /* EI_bitSplit_withWidths */
#include "openzl/common/assertion.h"                        /* ZL_ASSERT */
#include "openzl/zl_data.h"                                 /* ZL_Input_type */
#include "openzl/zl_input.h" /* ZL_Input_ptr, ZL_Input_eltWidth, ZL_Input_numElts */

#include <stdint.h> /* uint8_t, uint16_t, uint32_t, uint64_t */

/**
 * OR all elements of a numeric input stream together. The result has the same
 * highest set bit as the maximum value, and is 0 only if all elements are 0.
 *
 * @param src Pointer to source data
 * @param eltWidth Element width in bytes (1, 2, 4, or 8)
 * @param nbElts Number of elements
 * @return OR of all elements (0 if nbElts == 0)
 */
static uint64_t orAllValues(const void* src, size_t eltWidth, size_t nbElts)
{
    if (nbElts == 0)
        return 0;

    ZL_ASSERT(src != NULL);
    uint64_t allBits = 0;

    switch (eltWidth) {
        case 1: {
            const uint8_t* p = (const uint8_t*)src;
            uint8_t acc8     = 0;
            for (size_t i = 0; i < nbElts; i++)
                acc8 |= p[i];
            allBits = acc8;
            break;
        }
        case 2: {
            const uint16_t* p = (const uint16_t*)src;
            uint16_t acc16    = 0;
            for (size_t i = 0; i < nbElts; i++)
                acc16 |= p[i];
            allBits = acc16;
            break;
        }
        case 4: {
            const uint32_t* p = (const uint32_t*)src;
            uint32_t acc32    = 0;
            for (size_t i = 0; i < nbElts; i++)
                acc32 |= p[i];
            allBits = acc32;
            break;
        }
        case 8: {
            const uint64_t* p = (const uint64_t*)src;
            for (size_t i = 0; i < nbElts; i++)
                allBits |= p[i];
            break;
        }
        default:
            ZL_ASSERT(0); /* unreachable: eltWidth must be 1, 2, 4, or 8 */
    }

    return allBits;
}

/**
 * Compute floor(log2(v)) for v > 0.
 * Returns the position of the highest set bit (0-indexed).
 */
static unsigned floorLog2(uint64_t v)
{
    ZL_ASSERT(v > 0);
    unsigned r = 0;
    while (v >>= 1)
        r++;
    return r;
}

ZL_Report
EI_bitsplit_top8(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns)
{
    ZL_ASSERT_NN(eictx);
    ZL_ASSERT_EQ(nbIns, 1);
    ZL_ASSERT_NN(ins);
    const ZL_Input* in = ins[0];
    ZL_ASSERT_NN(in);
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_numeric);

    size_t const eltWidth = ZL_Input_eltWidth(in);
    ZL_ASSERT(eltWidth == 1 || eltWidth == 2 || eltWidth == 4 || eltWidth == 8);
    size_t const nbElts = ZL_Input_numElts(in);

    /* The highest set bit across all elements gives the effective bit width */
    uint64_t const allBits = orAllValues(ZL_Input_ptr(in), eltWidth, nbElts);

    unsigned effectiveWidth;
    if (allBits == 0) {
        effectiveWidth = 0;
    } else {
        effectiveWidth = floorLog2(allBits) + 1;
    }

    /* Determine bit widths for the split */
    uint8_t bitWidths[2];
    size_t nbWidths;

    if (effectiveWidth <= 8) {
        /* All data fits in 8 bits: single output stream */
        bitWidths[0] = (effectiveWidth > 0) ? (uint8_t)effectiveWidth : 1;
        nbWidths     = 1;
    } else {
        /* Split into remainder (lower) + top 8 bits (upper) */
        unsigned remainder = effectiveWidth - 8;
        bitWidths[0]       = (uint8_t)remainder;
        bitWidths[1]       = 8;
        nbWidths           = 2;
    }

    return EI_bitSplit_withWidths(eictx, in, bitWidths, nbWidths);
}
