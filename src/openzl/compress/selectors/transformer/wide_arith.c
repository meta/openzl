// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "wide_arith.h"

#include "numeric_fixed_point.h"

#include "openzl/common/assertion.h"

#include <string.h>

ZL_STATIC_ASSERT(
        TRS_NUMERIC_Q32_SCALE == (UINT64_C(1) << 32),
        "wide arithmetic assumes Q32 has 32 fractional bits");

/*
 * clang-cl exposes __int128, but the MSVC runtime does not provide the
 * division helpers it requires. Use the portable implementation for that ABI.
 */
#if defined(__SIZEOF_INT128__) && !defined(_MSC_VER) \
        && !defined(TRS_WIDE_FORCE_FALLBACK)
#    define TRS_WIDE_HAVE_INT128 1
#else
#    define TRS_WIDE_HAVE_INT128 0
#endif

#if !TRS_WIDE_HAVE_INT128
static int wide_u128_cmp_impl(TRS_WideU128 lhs, TRS_WideU128 rhs)
{
    if (lhs.hi != rhs.hi) {
        return (lhs.hi > rhs.hi) ? 1 : -1;
    }
    if (lhs.lo != rhs.lo) {
        return (lhs.lo > rhs.lo) ? 1 : -1;
    }
    return 0;
}

static TRS_WideU128 wide_u128_sub_impl(TRS_WideU128 lhs, TRS_WideU128 rhs)
{
    TRS_WideU128 out;
    uint64_t borrow = (lhs.lo < rhs.lo) ? 1ULL : 0ULL;
    out.lo          = lhs.lo - rhs.lo;
    out.hi          = lhs.hi - rhs.hi - borrow;
    return out;
}

static TRS_WideU128 wide_u128_shl1_impl(TRS_WideU128 x)
{
    TRS_WideU128 out;
    out.hi = (x.hi << 1) | (x.lo >> 63);
    out.lo = x.lo << 1;
    return out;
}

static int wide_u128_get_bit_impl(TRS_WideU128 x, unsigned bit)
{
    if (bit < 64) {
        return (int)((x.lo >> bit) & 1ULL);
    }
    return (int)((x.hi >> (bit - 64)) & 1ULL);
}

static void wide_u128_set_bit_impl(TRS_WideU128* x, unsigned bit)
{
    if (bit < 64) {
        x->lo |= (1ULL << bit);
    } else {
        x->hi |= (1ULL << (bit - 64));
    }
}

static void wide_u128_divmod_impl(
        TRS_WideU128 numerator,
        TRS_WideU128 denominator,
        TRS_WideU128* quotient_out,
        TRS_WideU128* remainder_out)
{
    ZL_ASSERT(denominator.hi != 0 || denominator.lo != 0, "division by zero");

    TRS_WideU128 quotient  = { 0, 0 };
    TRS_WideU128 remainder = { 0, 0 };

    for (int bit = 127; bit >= 0; bit--) {
        /*
         * The consumed numerator prefix contains at most 127 bits, so the
         * remainder's top bit is clear and this shift cannot overflow.
         */
        ZL_ASSERT_EQ(remainder.hi >> 63, 0);
        remainder = wide_u128_shl1_impl(remainder);
        if (wide_u128_get_bit_impl(numerator, (unsigned)bit)) {
            remainder.lo |= 1ULL;
        }
        if (wide_u128_cmp_impl(remainder, denominator) >= 0) {
            remainder = wide_u128_sub_impl(remainder, denominator);
            wide_u128_set_bit_impl(&quotient, (unsigned)bit);
        }
    }

    *quotient_out  = quotient;
    *remainder_out = remainder;
}
#endif

static int wide_i128_is_negative_impl(TRS_WideI128 x)
{
    return ((int64_t)x.hi) < 0;
}

static TRS_WideI128 wide_i128_negate_impl(TRS_WideI128 x)
{
    TRS_WideI128 out;
    out.lo = ~x.lo + 1ULL;
    out.hi = ~x.hi + ((out.lo == 0) ? 1ULL : 0ULL);
    return out;
}

#if TRS_WIDE_HAVE_INT128
static __uint128_t wide_u128_to_builtin(TRS_WideU128 x)
{
    return ((__uint128_t)x.hi << 64) | (__uint128_t)x.lo;
}

static TRS_WideU128 wide_u128_from_builtin(__uint128_t x)
{
    TRS_WideU128 out;
    out.hi = (uint64_t)(x >> 64);
    out.lo = (uint64_t)x;
    return out;
}

static __int128 wide_i128_to_builtin(TRS_WideI128 x)
{
    const __uint128_t bits = ((__uint128_t)x.hi << 64) | (__uint128_t)x.lo;
    __int128 value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static TRS_WideI128 wide_i128_from_builtin(__int128 x)
{
    __uint128_t ux = (__uint128_t)x;
    TRS_WideI128 out;
    out.hi = (uint64_t)(ux >> 64);
    out.lo = (uint64_t)ux;
    return out;
}
#endif

TRS_WideU128 TRS_wide_u128_from_u64(uint64_t x)
{
    TRS_WideU128 out = { .hi = 0, .lo = x };
    return out;
}

TRS_WideU128 TRS_wide_u128_mul_u64(uint64_t lhs, uint64_t rhs)
{
#if TRS_WIDE_HAVE_INT128
    return wide_u128_from_builtin((__uint128_t)lhs * (__uint128_t)rhs);
#else
    uint64_t lhs_lo = (uint32_t)lhs;
    uint64_t lhs_hi = lhs >> 32;
    uint64_t rhs_lo = (uint32_t)rhs;
    uint64_t rhs_hi = rhs >> 32;

    uint64_t p0 = lhs_lo * rhs_lo;
    uint64_t p1 = lhs_lo * rhs_hi;
    uint64_t p2 = lhs_hi * rhs_lo;
    uint64_t p3 = lhs_hi * rhs_hi;

    uint64_t mid = (p0 >> 32) + (uint32_t)p1 + (uint32_t)p2;

    TRS_WideU128 out;
    out.lo = (p0 & 0xFFFFFFFFULL) | (mid << 32);
    out.hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    return out;
#endif
}

TRS_WideU128 TRS_wide_u128_shl32(TRS_WideU128 x)
{
    ZL_ASSERT_LE(x.hi, UINT32_MAX);
#if TRS_WIDE_HAVE_INT128
    return wide_u128_from_builtin(wide_u128_to_builtin(x) << 32);
#else
    TRS_WideU128 out;
    out.hi = (x.hi << 32) | (x.lo >> 32);
    out.lo = x.lo << 32;
    return out;
#endif
}

uint64_t TRS_wide_u128_div_u64_to_u64(TRS_WideU128 x, uint64_t divisor)
{
    int quotient_fits;

    ZL_ASSERT_NE(divisor, 0);
    if (divisor == 0) {
        return UINT64_MAX;
    }

    quotient_fits = x.hi < divisor;
    ZL_ASSERT(quotient_fits);
    if (!quotient_fits) {
        return UINT64_MAX;
    }

#if TRS_WIDE_HAVE_INT128
    return (uint64_t)(wide_u128_to_builtin(x) / (__uint128_t)divisor);
#else
    TRS_WideU128 quotient;
    TRS_WideU128 remainder;
    wide_u128_divmod_impl(
            x, TRS_wide_u128_from_u64(divisor), &quotient, &remainder);
    (void)remainder;
    return quotient.lo;
#endif
}

uint64_t TRS_wide_u128_div_u128_to_u64(
        TRS_WideU128 numerator,
        TRS_WideU128 denominator)
{
    int quotient_fits;

    ZL_ASSERT(denominator.hi != 0 || denominator.lo != 0, "division by zero");
    if (denominator.hi == 0 && denominator.lo == 0) {
        return UINT64_MAX;
    }

    quotient_fits = denominator.hi != 0 || numerator.hi < denominator.lo;
    ZL_ASSERT(quotient_fits);
    if (!quotient_fits) {
        return UINT64_MAX;
    }

#if TRS_WIDE_HAVE_INT128
    return (uint64_t)(wide_u128_to_builtin(numerator)
                      / wide_u128_to_builtin(denominator));
#else
    TRS_WideU128 quotient;
    TRS_WideU128 remainder;
    wide_u128_divmod_impl(numerator, denominator, &quotient, &remainder);
    (void)remainder;
    return quotient.lo;
#endif
}

uint64_t TRS_wide_u128_scaled_div_to_u64(TRS_WideU128 sum, size_t count)
{
    ZL_ASSERT_NE(count, 0);
    if (count == 0) {
        return UINT64_MAX;
    }

#if TRS_WIDE_HAVE_INT128
    __uint128_t total = wide_u128_to_builtin(sum);
    __uint128_t q     = total / (__uint128_t)count;
    __uint128_t r     = total % (__uint128_t)count;

    if (q > (__uint128_t)(UINT64_MAX >> 32)) {
        return UINT64_MAX;
    }

    __uint128_t scaled = q * (__uint128_t)TRS_NUMERIC_Q32_SCALE
            + (r * (__uint128_t)TRS_NUMERIC_Q32_SCALE) / (__uint128_t)count;
    return (uint64_t)scaled;
#else
    TRS_WideU128 quotient;
    TRS_WideU128 remainder;
    TRS_WideU128 shifted_remainder;
    uint64_t frac;

    wide_u128_divmod_impl(
            sum,
            TRS_wide_u128_from_u64((uint64_t)count),
            &quotient,
            &remainder);

    if (quotient.hi != 0 || quotient.lo > (UINT64_MAX >> 32)) {
        return UINT64_MAX;
    }

    shifted_remainder = TRS_wide_u128_shl32(remainder);
    frac = TRS_wide_u128_div_u64_to_u64(shifted_remainder, (uint64_t)count);
    return (quotient.lo << 32) + frac;
#endif
}

uint64_t TRS_wide_u128_saturate_to_u64(TRS_WideU128 x)
{
    if (x.hi != 0) {
        return UINT64_MAX;
    }
    return x.lo;
}

double TRS_wide_u128_to_double(TRS_WideU128 x)
{
    return (double)x.hi * 0x1p64 + (double)x.lo;
}

TRS_WideI128 TRS_wide_i128_from_i64(int64_t x)
{
    TRS_WideI128 out;
    out.hi = (x < 0) ? UINT64_MAX : 0;
    out.lo = (uint64_t)x;
    return out;
}

TRS_WideI128 TRS_wide_i128_add_i64(TRS_WideI128 acc, int64_t x)
{
#if TRS_WIDE_HAVE_INT128
    return wide_i128_from_builtin(wide_i128_to_builtin(acc) + (__int128)x);
#else
    TRS_WideI128 out;
    uint64_t add_lo = (uint64_t)x;
    uint64_t add_hi = (x < 0) ? UINT64_MAX : 0;
    uint64_t carry;

    out.lo = acc.lo + add_lo;
    carry  = (out.lo < acc.lo) ? 1ULL : 0ULL;
    out.hi = acc.hi + add_hi + carry;
    return out;
#endif
}

TRS_WideI128 TRS_wide_i128_sub_i64(TRS_WideI128 acc, int64_t x)
{
#if TRS_WIDE_HAVE_INT128
    return wide_i128_from_builtin(wide_i128_to_builtin(acc) - (__int128)x);
#else
    TRS_WideI128 out;
    uint64_t sub_lo = (uint64_t)x;
    uint64_t sub_hi = (x < 0) ? UINT64_MAX : 0;
    uint64_t borrow;

    out.lo = acc.lo - sub_lo;
    borrow = (acc.lo < sub_lo) ? 1ULL : 0ULL;
    out.hi = acc.hi - sub_hi - borrow;
    return out;
#endif
}

int TRS_wide_i128_lt(TRS_WideI128 lhs, TRS_WideI128 rhs)
{
    int64_t lhs_hi = (int64_t)lhs.hi;
    int64_t rhs_hi = (int64_t)rhs.hi;
    if (lhs_hi != rhs_hi) {
        return lhs_hi < rhs_hi;
    }
    return lhs.lo < rhs.lo;
}

int TRS_wide_i128_gt(TRS_WideI128 lhs, TRS_WideI128 rhs)
{
    int64_t lhs_hi = (int64_t)lhs.hi;
    int64_t rhs_hi = (int64_t)rhs.hi;
    if (lhs_hi != rhs_hi) {
        return lhs_hi > rhs_hi;
    }
    return lhs.lo > rhs.lo;
}

TRS_WideU128 TRS_wide_i128_abs(TRS_WideI128 x)
{
    if (wide_i128_is_negative_impl(x)) {
        TRS_WideI128 negated = wide_i128_negate_impl(x);
        TRS_WideU128 out     = { .hi = negated.hi, .lo = negated.lo };
        return out;
    }

    {
        TRS_WideU128 out = { .hi = x.hi, .lo = x.lo };
        return out;
    }
}

uint64_t TRS_wide_i128_abs_to_u64(TRS_WideI128 x)
{
    return TRS_wide_u128_saturate_to_u64(TRS_wide_i128_abs(x));
}

int64_t TRS_wide_i128_div_u64_to_i64(TRS_WideI128 x, uint64_t divisor)
{
    TRS_WideU128 magnitude;
    uint64_t quotient;

    ZL_ASSERT_NE(divisor, 0);

    magnitude = TRS_wide_i128_abs(x);
    quotient  = TRS_wide_u128_div_u64_to_u64(magnitude, divisor);

    if (wide_i128_is_negative_impl(x)) {
        uint64_t const min_magnitude = UINT64_C(1) << 63;
        if (quotient >= min_magnitude) {
            /* Equality is exactly INT64_MIN; larger magnitudes saturate. */
            ZL_ASSERT_EQ(quotient, min_magnitude);
            return INT64_MIN;
        }
        return -(int64_t)quotient;
    }

    ZL_ASSERT_LE(quotient, (uint64_t)INT64_MAX);
    if (quotient > (uint64_t)INT64_MAX) {
        return INT64_MAX;
    }
    return (int64_t)quotient;
}

int64_t TRS_wide_i128_scaled_div_to_i64(TRS_WideI128 sum, size_t count)
{
    TRS_WideU128 magnitude = TRS_wide_i128_abs(sum);
    uint64_t scaled        = TRS_wide_u128_scaled_div_to_u64(magnitude, count);

    if (wide_i128_is_negative_impl(sum)) {
        if (scaled >= (1ULL << 63)) {
            return INT64_MIN;
        }
        return -(int64_t)scaled;
    }

    if (scaled > (uint64_t)INT64_MAX) {
        return INT64_MAX;
    }
    return (int64_t)scaled;
}

int64_t TRS_wide_i128_saturate_to_i64(TRS_WideI128 x)
{
    /* A fitting i64 is exactly its low half sign-extended through `x.hi`. */
    uint64_t const sign_extension =
            (x.lo >> 63) != 0 ? UINT64_MAX : UINT64_C(0);
    if (x.hi == sign_extension)
        return (int64_t)x.lo;

    return (x.hi >> 63) != 0 ? INT64_MIN : INT64_MAX;
}

double TRS_wide_i128_to_double(TRS_WideI128 x)
{
    if (((int64_t)x.hi) < 0) {
        return -TRS_wide_u128_to_double(TRS_wide_i128_abs(x));
    }

    {
        TRS_WideU128 magnitude = { .hi = x.hi, .lo = x.lo };
        return TRS_wide_u128_to_double(magnitude);
    }
}

uint64_t TRS_wide_abs_i64_to_u64(int64_t x)
{
    if (x >= 0) {
        return (uint64_t)x;
    }
    return (~(uint64_t)x) + 1ULL;
}

uint64_t TRS_wide_i64_abs_diff_to_u64(int64_t lhs, int64_t rhs)
{
    TRS_WideI128 diff = TRS_wide_i128_from_i64(lhs);
    diff              = TRS_wide_i128_sub_i64(diff, rhs);
    return TRS_wide_i128_abs_to_u64(diff);
}
