// Copyright (c) Meta Platforms, Inc. and affiliates.

// note : relative-path include (same directory by default)
#include "decode_delta_kernel.h"

#include <assert.h>
#include <stdint.h> // uint32_t

#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

// note : some template capability would be welcomed here

/* Vectorized helpers are optional accelerators for ZS_deltaDecodeXX().
 * They expect dst[0] to be already written, decode as many whole vectors as
 * possible starting from element 1, and return the index of the first element
 * not yet decoded. The scalar loop in ZS_deltaDecodeXX() finishes the job, so
 * removing a helper changes speed, not results. */

#if ZL_HAS_SSSE3

#    include <tmmintrin.h>
#elif defined(__SSE2__)
#    include <emmintrin.h>
#endif

/* Each iteration computes the prefix sums of one vector of deltas, then adds
 * them to `prev`, the last value decoded so far. `prev` is updated from the
 * local prefix sums rather than from the stored result, so that only one add
 * sits on the dependency chain between iterations, not the shuffle. */

#if ZL_HAS_SSSE3
static size_t
ZS_deltaDecode8_ssse3(uint8_t* dst, uint8_t const* deltas, size_t nbElts)
{
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(__m128i) / sizeof(*deltas);
    __m128i prev              = _mm_set1_epi8((char)dst[0]);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        __m128i values = _mm_loadu_si128((__m128i_u const*)&deltas[n - 1]);
        values         = _mm_add_epi8(values, _mm_slli_si128(values, 8));
        values         = _mm_add_epi8(values, _mm_slli_si128(values, 4));
        values         = _mm_add_epi8(values, _mm_slli_si128(values, 2));
        values         = _mm_add_epi8(values, _mm_slli_si128(values, 1));
        _mm_storeu_si128((__m128i_u*)&dst[n], _mm_add_epi8(values, prev));
        prev = _mm_add_epi8(
                prev, _mm_shuffle_epi8(values, _mm_set1_epi8(0x0f)));
    }
    return n;
}

#endif

#if ZL_HAS_SSSE3 || defined(__SSE2__)
static size_t
ZS_deltaDecode16_vec(uint16_t* dst, uint16_t const* deltas, size_t nbElts)
{
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(__m128i) / sizeof(*deltas);
    __m128i prev              = _mm_set1_epi16((int16_t)dst[0]);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        __m128i values = _mm_loadu_si128((__m128i_u const*)&deltas[n - 1]);
        values         = _mm_add_epi16(values, _mm_slli_si128(values, 8));
        values         = _mm_add_epi16(values, _mm_slli_si128(values, 4));
        values         = _mm_add_epi16(values, _mm_slli_si128(values, 2));
        _mm_storeu_si128((__m128i_u*)&dst[n], _mm_add_epi16(values, prev));
#    if ZL_HAS_SSSE3
        __m128i const carry = _mm_shuffle_epi8(values, _mm_set1_epi16(0x0f0e));
#    else
        // Broadcast word 7 to the high half, then broadcast its last dword.
        __m128i const high  = _mm_shufflehi_epi16(values, 0xff);
        __m128i const carry = _mm_shuffle_epi32(high, 0xff);
#    endif
        prev = _mm_add_epi16(prev, carry);
    }
    return n;
}

#endif

#if ZL_HAS_SSSE3
static size_t
ZS_deltaDecode32_ssse3(uint32_t* dst, uint32_t const* deltas, size_t nbElts)
{
    // TODO: Consider aligning either the source or the destination buffer,
    // or both. I've measured that this loop is 10% slower when working on
    // unaligned output buffers. Interestingly, the other vectorized delta
    // decoders aren't affected by the alignment.
    // We could either align here by decoding a few scalar elements first.
    // Or ask zstrong to give us aligned buffers when it doesn't need an
    // extra memcpy to do so.
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(__m128i) / sizeof(*deltas);
    __m128i prev              = _mm_set1_epi32((int32_t)dst[0]);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        __m128i values = _mm_loadu_si128((__m128i_u const*)&deltas[n - 1]);
        values         = _mm_add_epi32(values, _mm_slli_si128(values, 8));
        values         = _mm_add_epi32(values, _mm_slli_si128(values, 4));
        _mm_storeu_si128((__m128i_u*)&dst[n], _mm_add_epi32(values, prev));
        prev = _mm_add_epi32(prev, _mm_shuffle_epi32(values, 0xff));
    }
    return n;
}

#endif // ZL_HAS_SSSE3

void ZS_deltaDecode8(
        uint8_t* dst,
        uint8_t first,
        uint8_t const* deltas,
        size_t nbElts)
{
    if (nbElts == 0) {
        return;
    }
    dst[0]   = first;
    size_t n = 1;
#if ZL_HAS_SSSE3
    n = ZS_deltaDecode8_ssse3(dst, deltas, nbElts);
#endif
    for (; n < nbElts; ++n) {
        dst[n] = (uint8_t)(dst[n - 1] + deltas[n - 1]);
    }
}

void ZS_deltaDecode16(
        uint16_t* dst,
        uint16_t first,
        uint16_t const* deltas,
        size_t nbElts)
{
    if (nbElts == 0) {
        return;
    }
    dst[0]   = first;
    size_t n = 1;
#if ZL_HAS_SSSE3 || defined(__SSE2__)
    n = ZS_deltaDecode16_vec(dst, deltas, nbElts);
#endif
    for (; n < nbElts; ++n) {
        dst[n] = (uint16_t)(dst[n - 1] + deltas[n - 1]);
    }
}

void ZS_deltaDecode32(
        uint32_t* dst,
        uint32_t first,
        uint32_t const* deltas,
        size_t nbElts)
{
    if (nbElts == 0) {
        return;
    }
    dst[0]   = first;
    size_t n = 1;
#if ZL_HAS_SSSE3
    n = ZS_deltaDecode32_ssse3(dst, deltas, nbElts);
#endif
    for (; n < nbElts; ++n) {
        dst[n] = dst[n - 1] + deltas[n - 1];
    }
}

void ZS_deltaDecode64(
        uint64_t* dst,
        uint64_t first,
        uint64_t const* deltas,
        size_t nbElts)
{
    if (nbElts == 0) {
        return;
    }
    dst[0] = first;
    for (size_t n = 1; n < nbElts; ++n) {
        dst[n] = dst[n - 1] + deltas[n - 1];
    }
}

void ZS_deltaDecode(
        void* dst,
        void const* first,
        void const* deltas,
        size_t nbElts,
        size_t eltWidth)
{
    if (nbElts == 0) {
        return;
    }
    switch (eltWidth) {
        case 1:
            ZS_deltaDecode8(
                    (uint8_t*)dst,
                    ZL_read8(first),
                    (uint8_t const*)deltas,
                    nbElts);
            break;
        case 2:
            ZS_deltaDecode16(
                    (uint16_t*)dst,
                    ZL_readLE16(first),
                    (uint16_t const*)deltas,
                    nbElts);
            break;
        case 4:
            ZS_deltaDecode32(
                    (uint32_t*)dst,
                    ZL_readLE32(first),
                    (uint32_t const*)deltas,
                    nbElts);
            break;
        case 8:
            ZS_deltaDecode64(
                    (uint64_t*)dst,
                    ZL_readLE64(first),
                    (uint64_t const*)deltas,
                    nbElts);
            break;
        default:
            assert(false);
            break;
    }
}
