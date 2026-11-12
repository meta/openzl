// Copyright (c) Meta Platforms, Inc. and affiliates.

// note : relative-path include (same directory by default)
#include "encode_delta_kernel.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h> // uint32_t, uint64_t
#include <string.h>

#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

// Note: in the best world,
// I would expect some checkable assumptions to be asserted below,
// such as ensuring ptrs are non NULL when @nelts > 0.
// However, I don't want to have `<assert.h>` in competition with zstrong's
// <assertion.h>, and on the other hand, I want to keep this file free of
// zstrong specific includes for transportability and swappability purposes.

static void ZS_deltaEncode64_scalar(
        uint64_t* first,
        uint64_t* deltas,
        uint64_t const* src,
        size_t nelts)
{
    *first = src[0];
    for (size_t n = 1; n < nelts; ++n) {
        deltas[n - 1] = src[n] - src[n - 1];
    }
}

static void ZS_deltaEncode32_scalar(
        uint32_t* first,
        uint32_t* deltas,
        uint32_t const* src,
        size_t nelts)
{
    *first = src[0];
    for (size_t n = 1; n < nelts; ++n) {
        deltas[n - 1] = src[n] - src[n - 1];
    }
}

static void ZS_deltaEncode16_scalar(
        uint16_t* first,
        uint16_t* deltas,
        uint16_t const* src,
        size_t nelts)
{
    *first = src[0];
    for (size_t n = 1; n < nelts; ++n) {
        deltas[n - 1] = (uint16_t)(src[n] - src[n - 1]);
    }
}

static void ZS_deltaEncode8_scalar(
        uint8_t* first,
        uint8_t* deltas,
        uint8_t const* src,
        size_t nelts)
{
    *first = src[0];
    for (size_t n = 1; n < nelts; ++n) {
        deltas[n - 1] = (uint8_t)(src[n] - src[n - 1]);
    }
}

#if ZL_HAS_NEON

#    include <arm_neon.h>

static void ZS_deltaEncode64_neon(
        uint64_t* first,
        uint64_t* deltas,
        uint64_t const* src,
        size_t nelts)
{
    *first = src[0];

    uint64x2_t v7 = vdupq_n_u64(src[0]);
    size_t n      = 1;
    for (; n + 8 <= nelts; n += 8) {
        uint64x2_t const v1 = vld1q_u64(src + n + 0); //  0, 1
        uint64x2_t const v3 = vld1q_u64(src + n + 2); //  2, 3
        uint64x2_t const v0 = vextq_u64(v7, v1, 1);   // -1, 0
        uint64x2_t const v2 = vextq_u64(v1, v3, 1);   //  1, 2
        uint64x2_t const v5 = vld1q_u64(src + n + 4); //  4, 5
        v7                  = vld1q_u64(src + n + 6); //  6, 7
        uint64x2_t const v4 = vextq_u64(v3, v5, 1);   //  3, 4
        uint64x2_t const v6 = vextq_u64(v5, v7, 1);   //  5, 6
        vst1q_u64(deltas + n - 1, vsubq_u64(v1, v0));
        vst1q_u64(deltas + n + 1, vsubq_u64(v3, v2));
        vst1q_u64(deltas + n + 3, vsubq_u64(v5, v4));
        vst1q_u64(deltas + n + 5, vsubq_u64(v7, v6));
    }

    ZL_UNROLL_LOOP(1)
    for (; n < nelts; ++n) {
        deltas[n - 1] = src[n] - src[n - 1];
    }
}

static void ZS_deltaEncode32_neon(
        uint32_t* first,
        uint32_t* deltas,
        uint32_t const* src,
        size_t nelts)
{
    *first = src[0];

    uint32x4_t v7 = vdupq_n_u32(src[0]);
    size_t n      = 1;
    for (; n + 16 <= nelts; n += 16) {
        uint32x4_t const v1 = vld1q_u32(src + n + 0);  //  0...3
        uint32x4_t const v3 = vld1q_u32(src + n + 4);  //  4...7
        uint32x4_t const v0 = vextq_u32(v7, v1, 3);    // -1...2
        uint32x4_t const v2 = vextq_u32(v1, v3, 3);    //  3...6
        uint32x4_t const v5 = vld1q_u32(src + n + 8);  //  8..11
        v7                  = vld1q_u32(src + n + 12); // 12..15
        uint32x4_t const v4 = vextq_u32(v3, v5, 3);    //  7..10
        uint32x4_t const v6 = vextq_u32(v5, v7, 3);    // 11..14
        vst1q_u32(deltas + n - 1, vsubq_u32(v1, v0));
        vst1q_u32(deltas + n + 3, vsubq_u32(v3, v2));
        vst1q_u32(deltas + n + 7, vsubq_u32(v5, v4));
        vst1q_u32(deltas + n + 11, vsubq_u32(v7, v6));
    }

    ZL_UNROLL_LOOP(1)
    for (; n < nelts; ++n) {
        deltas[n - 1] = src[n] - src[n - 1];
    }
}

static void ZS_deltaEncode16_neon(
        uint16_t* first,
        uint16_t* deltas,
        uint16_t const* src,
        size_t nelts)
{
    *first = src[0];

    uint16x8_t v7 = vdupq_n_u16(src[0]);
    size_t n      = 1;
    for (; n + 32 <= nelts; n += 32) {
        uint16x8_t const v1 = vld1q_u16(src + n + 0);  //  0...7
        uint16x8_t const v3 = vld1q_u16(src + n + 8);  //  8..15
        uint16x8_t const v0 = vextq_u16(v7, v1, 7);    // -1...6
        uint16x8_t const v2 = vextq_u16(v1, v3, 7);    //  7..14
        uint16x8_t const v5 = vld1q_u16(src + n + 16); // 16..23
        v7                  = vld1q_u16(src + n + 24); // 24..31
        uint16x8_t const v4 = vextq_u16(v3, v5, 7);    // 15..22
        uint16x8_t const v6 = vextq_u16(v5, v7, 7);    // 23..30
        vst1q_u16(deltas + n - 1, vsubq_u16(v1, v0));
        vst1q_u16(deltas + n + 7, vsubq_u16(v3, v2));
        vst1q_u16(deltas + n + 15, vsubq_u16(v5, v4));
        vst1q_u16(deltas + n + 23, vsubq_u16(v7, v6));
    }

    ZL_UNROLL_LOOP(1)
    for (; n < nelts; ++n) {
        deltas[n - 1] = src[n] - src[n - 1];
    }
}

static void ZS_deltaEncode8_neon(
        uint8_t* first,
        uint8_t* deltas,
        uint8_t const* src,
        size_t nelts)
{
    *first = src[0];

    uint8x16_t v7 = vdupq_n_u8(src[0]);
    size_t n      = 1;
    for (; n + 64 <= nelts; n += 64) {
        uint8x16_t const v1 = vld1q_u8(src + n + 0);  //  0..15
        uint8x16_t const v3 = vld1q_u8(src + n + 16); // 16..31
        uint8x16_t const v0 = vextq_u8(v7, v1, 15);   // -1..14
        uint8x16_t const v2 = vextq_u8(v1, v3, 15);   // 15..30
        uint8x16_t const v5 = vld1q_u8(src + n + 32); // 32..47
        v7                  = vld1q_u8(src + n + 48); // 48..63
        uint8x16_t const v4 = vextq_u8(v3, v5, 15);   // 31..46
        uint8x16_t const v6 = vextq_u8(v5, v7, 15);   // 47..62
        vst1q_u8(deltas + n - 1, vsubq_u8(v1, v0));
        vst1q_u8(deltas + n + 15, vsubq_u8(v3, v2));
        vst1q_u8(deltas + n + 31, vsubq_u8(v5, v4));
        vst1q_u8(deltas + n + 47, vsubq_u8(v7, v6));
    }

    ZL_UNROLL_LOOP(1)
    for (; n < nelts; ++n) {
        deltas[n - 1] = src[n] - src[n - 1];
    }
}

#endif // ZL_HAS_NEON

void ZS_deltaEncode64(
        uint64_t* first,
        uint64_t* deltas,
        uint64_t const* src,
        size_t nelts)
{
    if (nelts == 0) {
        return;
    }
#if ZL_HAS_NEON
    ZS_deltaEncode64_neon(first, deltas, src, nelts);
#else
    ZS_deltaEncode64_scalar(first, deltas, src, nelts);
#endif
}

void ZS_deltaEncode32(
        uint32_t* first,
        uint32_t* deltas,
        uint32_t const* src,
        size_t nelts)
{
    if (nelts == 0) {
        return;
    }
#if ZL_HAS_NEON
    ZS_deltaEncode32_neon(first, deltas, src, nelts);
#else
    ZS_deltaEncode32_scalar(first, deltas, src, nelts);
#endif
}

void ZS_deltaEncode16(
        uint16_t* first,
        uint16_t* deltas,
        uint16_t const* src,
        size_t nelts)
{
    if (nelts == 0) {
        return;
    }
#if ZL_HAS_NEON
    ZS_deltaEncode16_neon(first, deltas, src, nelts);
#else
    ZS_deltaEncode16_scalar(first, deltas, src, nelts);
#endif
}

void ZS_deltaEncode8(
        uint8_t* first,
        uint8_t* deltas,
        uint8_t const* src,
        size_t nelts)
{
    if (nelts == 0) {
        return;
    }
#if ZL_HAS_NEON
    ZS_deltaEncode8_neon(first, deltas, src, nelts);
#else
    ZS_deltaEncode8_scalar(first, deltas, src, nelts);
#endif
}

void ZS_deltaEncode(
        void* first,
        void* deltas,
        void const* src,
        size_t nbElts,
        size_t eltWidth)
{
    if (nbElts == 0) {
        return;
    }
    switch (eltWidth) {
        case 1: {
            uint8_t firstU8;
            ZS_deltaEncode8(
                    &firstU8, (uint8_t*)deltas, (uint8_t const*)src, nbElts);
            memcpy(first, &firstU8, sizeof(firstU8));
            break;
        }
        case 2: {
            uint16_t firstU16;
            ZS_deltaEncode16(
                    &firstU16, (uint16_t*)deltas, (uint16_t const*)src, nbElts);
            ZL_writeLE16(first, firstU16);
            break;
        }
        case 4: {
            uint32_t firstU32;
            ZS_deltaEncode32(
                    &firstU32, (uint32_t*)deltas, (uint32_t const*)src, nbElts);
            ZL_writeLE32(first, firstU32);
            break;
        }
        case 8: {
            uint64_t firstU64;
            ZS_deltaEncode64(
                    &firstU64, (uint64_t*)deltas, (uint64_t const*)src, nbElts);
            ZL_writeLE64(first, firstU64);
            break;
        }
        default:
            assert(false);
            break;
    }
}
