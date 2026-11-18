// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_COMPRESS_SELECTORS_TRANSFORMER_WIDE_ARITH_H
#define OPENZL_COMPRESS_SELECTORS_TRANSFORMER_WIDE_ARITH_H

#include <stddef.h>
#include <stdint.h>

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/*
 * Internal 128-bit helper layer for numeric extraction.
 *
 * The public ABI does not expose these types. Callers can force the pure C11
 * implementation by compiling wide_arith.c with
 * TRS_WIDE_FORCE_FALLBACK defined.
 */

typedef struct {
    uint64_t hi;
    uint64_t lo;
} TRS_WideU128;

typedef struct {
    uint64_t hi;
    uint64_t lo;
} TRS_WideI128;

TRS_WideU128 TRS_wide_u128_from_u64(uint64_t x);
ZL_INLINE TRS_WideU128 TRS_wide_u128_add_u64(TRS_WideU128 acc, uint64_t x)
{
    uint64_t const low = acc.lo + x;
    acc.hi += low < acc.lo;
    acc.lo = low;
    return acc;
}
TRS_WideU128 TRS_wide_u128_mul_u64(uint64_t lhs, uint64_t rhs);
/*
 * Shift left by 32 bits. The result must fit in 128 bits, so `x.hi` must not
 * exceed UINT32_MAX.
 */
TRS_WideU128 TRS_wide_u128_shl32(TRS_WideU128 x);

/*
 * Transformer callers use these for averages or bounded normalized gap
 * statistics, so their quotients fit in the destination type. Debug builds
 * assert that invariant; release builds saturate defensively if it is
 * violated. Divisors must be nonzero.
 */
uint64_t TRS_wide_u128_div_u64_to_u64(TRS_WideU128 x, uint64_t divisor);
uint64_t TRS_wide_u128_div_u128_to_u64(
        TRS_WideU128 numerator,
        TRS_WideU128 denominator);

/*
 * Return `sum / count` encoded in Q32, preserving the fractional remainder. A
 * valid quotient can overflow after scaling, so the result saturates in every
 * build.
 * `count` must be nonzero; debug builds assert that precondition and release
 * builds return UINT64_MAX if it is violated.
 */
uint64_t TRS_wide_u128_scaled_div_to_u64(TRS_WideU128 sum, size_t count);
uint64_t TRS_wide_u128_saturate_to_u64(TRS_WideU128 x);
/*
 * Combines separately rounded 64-bit halves. The result is deterministic and
 * within one ULP of a correctly rounded 128-bit conversion. This behavior is
 * shared with the trainer and must change in lockstep with it.
 */
double TRS_wide_u128_to_double(TRS_WideU128 x);

TRS_WideI128 TRS_wide_i128_from_i64(int64_t x);
TRS_WideI128 TRS_wide_i128_add_i64(TRS_WideI128 acc, int64_t x);
TRS_WideI128 TRS_wide_i128_sub_i64(TRS_WideI128 acc, int64_t x);

int TRS_wide_i128_lt(TRS_WideI128 lhs, TRS_WideI128 rhs);
int TRS_wide_i128_gt(TRS_WideI128 lhs, TRS_WideI128 rhs);
TRS_WideU128 TRS_wide_i128_abs(TRS_WideI128 x);
uint64_t TRS_wide_i128_abs_to_u64(TRS_WideI128 x);
/*
 * The quotient is expected to fit in `int64_t`; debug builds assert this and
 * release builds saturate defensively. `divisor` must be nonzero.
 */
int64_t TRS_wide_i128_div_u64_to_i64(TRS_WideI128 x, uint64_t divisor);

/*
 * Return `sum / count` encoded in Q32, preserving the fractional remainder and
 * truncating toward zero. A valid quotient can overflow after scaling, so the
 * result saturates in every build. `count` must be nonzero; debug builds
 * assert that precondition and release builds return the corresponding
 * saturation bound if it is violated.
 */
int64_t TRS_wide_i128_scaled_div_to_i64(TRS_WideI128 sum, size_t count);
int64_t TRS_wide_i128_saturate_to_i64(TRS_WideI128 x);
/* Uses the same one-ULP conversion contract as the unsigned helper. */
double TRS_wide_i128_to_double(TRS_WideI128 x);

uint64_t TRS_wide_abs_i64_to_u64(int64_t x);
uint64_t TRS_wide_i64_abs_diff_to_u64(int64_t lhs, int64_t rhs);

ZL_END_C_DECLS

#endif /* OPENZL_COMPRESS_SELECTORS_TRANSFORMER_WIDE_ARITH_H */
