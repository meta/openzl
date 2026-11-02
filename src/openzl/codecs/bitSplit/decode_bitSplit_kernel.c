// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>  /* assert */
#include <stdbool.h> /* bool */

#include "openzl/codecs/bitSplit/decode_bitSplit_kernel.h"
#include "openzl/shared/portability.h"

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

#if ZL_HAS_NEON

#    include <arm_neon.h>

/*
 * Specialized decoder for bf16 pattern:
 * - 3 streams with bitWidths {7, 8, 1}
 * - All srcEltWidths are 1 (uint8_t)
 * - dstEltWidth is 2 (uint16_t)
 *
 * Layout: [mantissa:7][exponent:8][sign:1] = 16 bits
 */
static inline void
decodeBf16_neon(void* dst, size_t nbElts, const void* const srcPtrs[])
{
    uint16_t* const dst16         = (uint16_t*)dst;
    const uint8_t* const mantissa = (const uint8_t*)srcPtrs[0];
    const uint8_t* const exponent = (const uint8_t*)srcPtrs[1];
    const uint8_t* const sign     = (const uint8_t*)srcPtrs[2];

    size_t e = 0;
    for (; e + 16 <= nbElts; e += 16) {
        uint8x16_t const vs = vld1q_u8(sign + e);
        uint8x16_t const ve = vld1q_u8(exponent + e);
        uint8x16_t const vm = vld1q_u8(mantissa + e);

        uint8x16x2_t const ves = vzipq_u8(ve, vs);
        uint8x16x2_t const vmz = vzipq_u8(vm, vdupq_n_u8(0));
        uint16x8_t ves0        = vreinterpretq_u16_u8(ves.val[0]);
        uint16x8_t ves1        = vreinterpretq_u16_u8(ves.val[1]);

        ves0 = vsliq_n_u16(vreinterpretq_u16_u8(vmz.val[0]), ves0, 7);
        ves1 = vsliq_n_u16(vreinterpretq_u16_u8(vmz.val[1]), ves1, 7);
        vst1q_u16(dst16 + e + 0, ves0);
        vst1q_u16(dst16 + e + 8, ves1);
    }

    ZL_UNROLL_LOOP(1)
    for (; e < nbElts; e++) {
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
decodeFp16_neon(void* dst, size_t nbElts, const void* const srcPtrs[])
{
    uint16_t* const dst16          = (uint16_t*)dst;
    const uint16_t* const mantissa = (const uint16_t*)srcPtrs[0];
    const uint8_t* const exponent  = (const uint8_t*)srcPtrs[1];
    const uint8_t* const sign      = (const uint8_t*)srcPtrs[2];

    size_t e = 0;
    for (; e + 16 <= nbElts; e += 16) {
        uint8x16_t const ve    = vld1q_u8(exponent + e);
        uint8x16_t const vs    = vld1q_u8(sign + e);
        uint16x8_t vm0         = vld1q_u16(mantissa + e + 0);
        uint16x8_t vm1         = vld1q_u16(mantissa + e + 8);
        uint8x16x2_t const vez = vzipq_u8(ve, vdupq_n_u8(0));
        uint8x16x2_t const vsz = vzipq_u8(vs, vdupq_n_u8(0));

        vm0 = vsliq_n_u16(vm0, vreinterpretq_u16_u8(vez.val[0]), 10);
        vm1 = vsliq_n_u16(vm1, vreinterpretq_u16_u8(vez.val[1]), 10);
        vm0 = vsliq_n_u16(vm0, vreinterpretq_u16_u8(vsz.val[0]), 15);
        vm1 = vsliq_n_u16(vm1, vreinterpretq_u16_u8(vsz.val[1]), 15);
        vst1q_u16(dst16 + e + 0, vm0);
        vst1q_u16(dst16 + e + 8, vm1);
    }

    ZL_UNROLL_LOOP(1)
    for (; e < nbElts; e++) {
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
decodeFp32_neon(void* dst, size_t nbElts, const void* const srcPtrs[])
{
    static const ZL_ALIGNED(16) int8_t shuffle[] = {
        0,  16, -1, -1, 1,  17, -1, -1, 2,  18, -1, -1, 3,  19, -1, -1,
        4,  20, -1, -1, 5,  21, -1, -1, 6,  22, -1, -1, 7,  23, -1, -1,
        8,  24, -1, -1, 9,  25, -1, -1, 10, 26, -1, -1, 11, 27, -1, -1,
        12, 28, -1, -1, 13, 29, -1, -1, 14, 30, -1, -1, 15, 31, -1, -1,
    };

    uint32_t* const dst32          = (uint32_t*)dst;
    const uint32_t* const mantissa = (const uint32_t*)srcPtrs[0];
    const uint8_t* const exponent  = (const uint8_t*)srcPtrs[1];
    const uint8_t* const sign      = (const uint8_t*)srcPtrs[2];

    size_t e = 0;
    if (nbElts >= 16) {
        uint8x16_t const shuf0 = vld1q_u8((const uint8_t*)shuffle + 0);
        uint8x16_t const shuf1 = vld1q_u8((const uint8_t*)shuffle + 16);
        uint8x16_t const shuf2 = vld1q_u8((const uint8_t*)shuffle + 32);
        uint8x16_t const shuf3 = vld1q_u8((const uint8_t*)shuffle + 48);

        for (; e + 16 <= nbElts; e += 16) {
            uint8x16x2_t ves;
            ves.val[0] = vld1q_u8(exponent + e);
            ves.val[1] = vld1q_u8(sign + e);

            uint32x4_t const vm0 = vld1q_u32(mantissa + e + 0);
            uint32x4_t const vm1 = vld1q_u32(mantissa + e + 4);
            uint32x4_t const vm2 = vld1q_u32(mantissa + e + 8);
            uint32x4_t const vm3 = vld1q_u32(mantissa + e + 12);

            uint32x4_t const ves0 =
                    vreinterpretq_u32_u8(vqtbl2q_u8(ves, shuf0));
            uint32x4_t const ves1 =
                    vreinterpretq_u32_u8(vqtbl2q_u8(ves, shuf1));
            uint32x4_t const ves2 =
                    vreinterpretq_u32_u8(vqtbl2q_u8(ves, shuf2));
            uint32x4_t const ves3 =
                    vreinterpretq_u32_u8(vqtbl2q_u8(ves, shuf3));

            vst1q_u32(dst32 + e + 0, vsliq_n_u32(vm0, ves0, 23));
            vst1q_u32(dst32 + e + 4, vsliq_n_u32(vm1, ves1, 23));
            vst1q_u32(dst32 + e + 8, vsliq_n_u32(vm2, ves2, 23));
            vst1q_u32(dst32 + e + 12, vsliq_n_u32(vm3, ves3, 23));
        }
    }

    ZL_UNROLL_LOOP(1)
    for (; e < nbElts; e++) {
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
decodeFp64_neon(void* dst, size_t nbElts, const void* const srcPtrs[])
{
    static const ZL_ALIGNED(16) int8_t shuffle[] = {
        0,  1,  -1, -1, -1, -1, -1, -1, 2,  3,  -1, -1, -1, -1, -1, -1,
        4,  5,  -1, -1, -1, -1, -1, -1, 6,  7,  -1, -1, -1, -1, -1, -1,
        8,  9,  -1, -1, -1, -1, -1, -1, 10, 11, -1, -1, -1, -1, -1, -1,
        12, 13, -1, -1, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1, -1, -1,
    };

    uint64_t* const dst64          = (uint64_t*)dst;
    const uint64_t* const mantissa = (const uint64_t*)srcPtrs[0];
    const uint16_t* const exponent = (const uint16_t*)srcPtrs[1];
    const uint8_t* const sign      = (const uint8_t*)srcPtrs[2];

    size_t e = 0;
    if (nbElts >= 8) {
        uint8x16_t const shuf0 = vld1q_u8((const uint8_t*)shuffle + 0);
        uint8x16_t const shuf1 = vld1q_u8((const uint8_t*)shuffle + 16);
        uint8x16_t const shuf2 = vld1q_u8((const uint8_t*)shuffle + 32);
        uint8x16_t const shuf3 = vld1q_u8((const uint8_t*)shuffle + 48);

        for (; e + 8 <= nbElts; e += 8) {
            uint8x16_t const vslo =
                    vcombine_u8(vld1_u8(sign + e), vdup_n_u8(0));
            uint16x8_t const vs =
                    vreinterpretq_u16_u8(vzip1q_u8(vdupq_n_u8(0), vslo));
            uint16x8_t const ve  = vld1q_u16(exponent + e);
            uint8x16_t const ves = vreinterpretq_u8_u16(vmlaq_n_u16(ve, vs, 8));

            uint64x2_t const vm0 = vld1q_u64(mantissa + e + 0);
            uint64x2_t const vm1 = vld1q_u64(mantissa + e + 2);
            uint64x2_t const vm2 = vld1q_u64(mantissa + e + 4);
            uint64x2_t const vm3 = vld1q_u64(mantissa + e + 6);

            uint64x2_t const ves0 =
                    vreinterpretq_u64_u8(vqtbl1q_u8(ves, shuf0));
            uint64x2_t const ves1 =
                    vreinterpretq_u64_u8(vqtbl1q_u8(ves, shuf1));
            uint64x2_t const ves2 =
                    vreinterpretq_u64_u8(vqtbl1q_u8(ves, shuf2));
            uint64x2_t const ves3 =
                    vreinterpretq_u64_u8(vqtbl1q_u8(ves, shuf3));

            vst1q_u64(dst64 + e + 0, vsliq_n_u64(vm0, ves0, 52));
            vst1q_u64(dst64 + e + 2, vsliq_n_u64(vm1, ves1, 52));
            vst1q_u64(dst64 + e + 4, vsliq_n_u64(vm2, ves2, 52));
            vst1q_u64(dst64 + e + 6, vsliq_n_u64(vm3, ves3, 52));
        }
    }

    ZL_UNROLL_LOOP(1)
    for (; e < nbElts; e++) {
        uint64_t value = mantissa[e];         /* bits 0-51 */
        value |= (uint64_t)exponent[e] << 52; /* bits 52-62 */
        value |= (uint64_t)sign[e] << 63;     /* bit 63 */
        dst64[e] = value;
    }
}

#endif // ZL_HAS_NEON

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

static inline void decodeElements(
        void* dst,
        const size_t dstEltWidth,
        size_t nbElts,
        const void* const srcPtrs[],
        const size_t* srcEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths)
{
    for (size_t e = 0; e < nbElts; e++) {
        uint64_t value = 0;
        size_t bitPos  = 0;

        for (size_t i = 0; i < nbWidths; i++) {
            uint64_t part = 0;
            switch (srcEltWidths[i]) {
                case 1:
                    part = ((const uint8_t*)srcPtrs[i])[e];
                    break;
                case 2:
                    part = ((const uint16_t*)srcPtrs[i])[e];
                    break;
                case 4:
                    part = ((const uint32_t*)srcPtrs[i])[e];
                    break;
                case 8:
                    part = ((const uint64_t*)srcPtrs[i])[e];
                    break;
                default:
                    assert(false);
                    break;
            }

            value |= (part << bitPos);
            bitPos += bitWidths[i];
        }

        // If `dstEltWidth` is a constant and `decodeElements()` is inlined,
        // we expect the compiler to fold this switch() statement.
        switch (dstEltWidth) {
            case 1:
                ((uint8_t*)dst)[e] = (uint8_t)value;
                break;
            case 2:
                ((uint16_t*)dst)[e] = (uint16_t)value;
                break;
            case 4:
                ((uint32_t*)dst)[e] = (uint32_t)value;
                break;
            case 8:
                ((uint64_t*)dst)[e] = (uint64_t)value;
                break;
            default:
                assert(false);
                break;
        }
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
#if ZL_HAS_NEON
        decodeBf16_neon(dst, nbElts, srcPtrs);
#else
        decodeBf16(dst, nbElts, srcPtrs);
#endif
        return;
    }
    if (isFp16Pattern(dstEltWidth, srcEltWidths, bitWidths, nbWidths)) {
#if ZL_HAS_NEON
        decodeFp16_neon(dst, nbElts, srcPtrs);
#else
        decodeFp16(dst, nbElts, srcPtrs);
#endif
        return;
    }
    if (isFp32Pattern(dstEltWidth, srcEltWidths, bitWidths, nbWidths)) {
#if ZL_HAS_NEON
        decodeFp32_neon(dst, nbElts, srcPtrs);
#else
        decodeFp32(dst, nbElts, srcPtrs);
#endif
        return;
    }
    if (isFp64Pattern(dstEltWidth, srcEltWidths, bitWidths, nbWidths)) {
#if ZL_HAS_NEON
        decodeFp64_neon(dst, nbElts, srcPtrs);
#else
        decodeFp64(dst, nbElts, srcPtrs);
#endif
        return;
    }

    /* Generic path */
    // We expect the compiler to optimize decodeElements() by propagating the
    // constant (thus resulting in several instances).
    switch (dstEltWidth) {
        case 1:
            decodeElements(
                    dst, 1, nbElts, srcPtrs, srcEltWidths, bitWidths, nbWidths);
            break;
        case 2:
            decodeElements(
                    dst, 2, nbElts, srcPtrs, srcEltWidths, bitWidths, nbWidths);
            break;
        case 4:
            decodeElements(
                    dst, 4, nbElts, srcPtrs, srcEltWidths, bitWidths, nbWidths);
            break;
        case 8:
            decodeElements(
                    dst, 8, nbElts, srcPtrs, srcEltWidths, bitWidths, nbWidths);
            break;
        default:
            assert(false);
            break;
    }
}
