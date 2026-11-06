// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 * Numeric feature extraction for the Compression Transformer selector.
 */

#include "numeric_extract.h"
#include "cardinality.h"
#include "numeric_fixed_point.h"
#include "numeric_kmv.h"
#include "numeric_stats.h"
#include "wide_arith.h"

#include "openzl/common/assertion.h"
#include "openzl/shared/bits.h"

#include <math.h>
#include <string.h> /* memset */

/* Sign-extend a zero-extended value to its original signed interpretation. */
static int64_t to_signed(uint64_t val, size_t elt_width)
{
    switch (elt_width) {
        case 1:
            return (int64_t)(int8_t)(uint8_t)val;
        case 2:
            return (int64_t)(int16_t)(uint16_t)val;
        case 4:
            return (int64_t)(int32_t)(uint32_t)val;
        default:
            return (int64_t)val;
    }
}

static TRS_NumericFeatures NumericFeatures_extract_num64(
        const uint64_t* data,
        size_t n_elements);

static int has_match4_workspace(const TRS_NumericExtractWorkspace* workspace)
{
    return workspace != NULL && workspace->match4_table != NULL
            && workspace->match4_capacity >= TRS_NUMERIC_MATCH4_TABLE_ENTRIES;
}

static int has_kmv_workspace(const TRS_NumericExtractWorkspace* workspace)
{
    return workspace != NULL && workspace->kmv_entries != NULL
            && workspace->kmv_capacity >= TRS_NUMERIC_KMV_K;
}

static int has_cardinality_workspace(
        const TRS_NumericExtractWorkspace* workspace,
        size_t required_words)
{
    return workspace != NULL && workspace->cardinality_bitmap != NULL
            && workspace->cardinality_bitmap_capacity >= required_words;
}

static int has_sorted_gap_workspace(
        const TRS_NumericExtractWorkspace* workspace)
{
    return workspace != NULL && workspace->sorted_gap_buffer != NULL
            && workspace->sorted_gap_capacity
            >= TRS_NUMERIC_SORTED_GAP_BUFFER_ENTRIES;
}

ZL_FORCE_INLINE void
seen_bitmap_set(uint64_t* seen, size_t word_count, uint32_t value)
{
    size_t const word = value >> 6;
    ZL_ASSERT_LT(word, word_count);
    seen[word] |= UINT64_C(1) << (value & 63);
}

static TRS_NumericFeatures init_numeric_features(
        size_t n_elements,
        size_t elt_width)
{
    TRS_NumericFeatures f;
    memset(&f, 0, sizeof(f));
    f.elt_width = elt_width;
    f.count     = n_elements;
    return f;
}

static TRS_NumericFeaturesV2 init_numeric_features_v2(
        size_t n_elements,
        size_t elt_width)
{
    TRS_NumericFeaturesV2 f;
    memset(&f, 0, sizeof(f));
    f.elt_width = elt_width;
    f.count     = (uint64_t)n_elements;
    return f;
}

static double decode_u64_fp(uint64_t x)
{
    return (double)x / (double)TRS_NUMERIC_Q32_SCALE;
}

static double decode_i64_fp(int64_t x)
{
    return (double)x / (double)TRS_NUMERIC_Q32_SCALE;
}

static void numeric_features_to_v2_exact_fields(
        const TRS_NumericFeatures* src,
        TRS_NumericFeaturesV2* dst)
{
    dst->max_run_length   = (uint64_t)src->max_run_length;
    dst->zero_count       = (uint64_t)src->zero_count;
    dst->delta_up_count   = (uint64_t)src->delta_up_count;
    dst->delta_down_count = (uint64_t)src->delta_down_count;

    dst->min_u = src->min_u;
    dst->max_u = src->max_u;
    dst->min_s = src->min_s;
    dst->max_s = src->max_s;

    dst->max_abs_delta_u = src->max_abs_delta_u;
    dst->max_abs_delta_s = src->max_abs_delta_s;
    dst->range_u         = src->range_u;
    dst->range_s         = src->range_s;

    dst->cardinality_est      = src->cardinality_est;
    dst->pair_cardinality_est = src->pair_cardinality_est;
    dst->d8_cardinality_est   = src->d8_cardinality_est;
    dst->min_lb0              = src->min_lb0;
    dst->match4               = src->match4;
}

static TRS_NumericFeaturesV2 numeric_features_to_v2_sub64(
        const TRS_NumericFeatures* src,
        double sorted_gap_mode)
{
    TRS_NumericFeaturesV2 dst =
            init_numeric_features_v2(src->count, src->elt_width);
    numeric_features_to_v2_exact_fields(src, &dst);

    dst.sum_u             = (double)src->sum_u;
    dst.sum_s             = (double)src->sum_s;
    dst.delta_min_s       = (double)src->delta_min_s;
    dst.delta_max_s       = (double)src->delta_max_s;
    dst.mean_u            = decode_u64_fp(src->mean_u_fp);
    dst.mean_s            = decode_i64_fp(src->mean_s_fp);
    dst.mean_abs_dev_u    = decode_u64_fp(src->mean_abs_dev_u_fp);
    dst.mean_abs_dev_s    = decode_u64_fp(src->mean_abs_dev_s_fp);
    dst.mean_abs_delta_u  = decode_u64_fp(src->mean_abs_delta_u_fp);
    dst.zero_ratio        = decode_u64_fp(src->zero_ratio_fp);
    dst.delta_up_ratio    = decode_u64_fp(src->delta_up_ratio_fp);
    dst.delta_down_ratio  = decode_u64_fp(src->delta_down_ratio_fp);
    dst.sorted_gap_nmad   = decode_u64_fp(src->sorted_gap_cv_fp);
    dst.sorted_gap_mode   = sorted_gap_mode;
    dst.transition_gap_cv = decode_u64_fp(src->transition_gap_cv_fp);

    return dst;
}

/*
 * The specialized extraction scans below intentionally remain separate. Their
 * arithmetic is part of the trained feature contract: legacy paths produce Q32
 * fields and preserve specific wrapping and saturation behavior, V2 paths use
 * wide accumulators and doubles, and decision paths omit unconsumed fields.
 * Legacy signed delta extrema are zero-anchored; the num64 V2 path instead
 * initializes them from the first delta.
 * Width-specific paths also use different exact or estimated cardinality
 * strategies. Any consolidation must preserve operation order, intermediate
 * types, overflow behavior, and computed fields bit-for-bit. Perform that work
 * in the training repository, validate it there, then synchronize it here.
 */
static TRS_NumericFeaturesV2 NumericFeaturesV2_extract_num64(
        const uint64_t* data,
        size_t n_elements,
        const TRS_NumericExtractWorkspace* workspace)
{
    TRS_NumericFeaturesV2 dst = init_numeric_features_v2(n_elements, 8);
    if (n_elements == 0) {
        return dst;
    }
    ZL_ASSERT_NN(data);

    uint64_t min_u                   = data[0];
    uint64_t max_u                   = data[0];
    int64_t min_s                    = to_signed(data[0], 8);
    int64_t max_s                    = min_s;
    TRS_WideU128 sum_u_safe          = TRS_wide_u128_from_u64(data[0]);
    TRS_WideI128 sum_s_safe          = TRS_wide_i128_from_i64(min_s);
    uint64_t max_abs_delta_u         = 0;
    uint64_t max_abs_delta_s         = 0;
    TRS_WideU128 sum_abs_delta_u     = TRS_wide_u128_from_u64(0);
    uint64_t zero_count              = (data[0] == 0) ? 1 : 0;
    uint64_t current_run_length      = 1;
    uint64_t max_run_length          = 1;
    uint64_t delta_up_count          = 0;
    uint64_t delta_down_count        = 0;
    TRS_WideI128 delta_min_s_safe    = TRS_wide_i128_from_i64(0);
    TRS_WideI128 delta_max_s_safe    = TRS_wide_i128_from_i64(0);
    int have_delta                   = 0;
    uint64_t min_lb0                 = 64;
    uint64_t nonzero_count           = (data[0] != 0) ? 1 : 0;
    uint64_t prev_u                  = data[0];
    int64_t prev_s                   = min_s;
    int have_transition_pos          = 0;
    size_t last_transition_pos       = 0;
    size_t transition_gap_count      = 0;
    double transition_gap_sum        = 0.0;
    double transition_gap_square_sum = 0.0;
    TRS_NumericKmvEntry kmv_heap[TRS_NUMERIC_KMV_K];
    size_t kmv_n = 0;

    if (data[0] != 0) {
        min_lb0 = (uint64_t)ZL_ctz64(data[0]);
    }
    TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, data[0]);

    for (size_t i = 1; i < n_elements; i++) {
        uint64_t val_u            = data[i];
        int64_t val_s             = to_signed(val_u, 8);
        TRS_WideI128 delta_s_safe = TRS_wide_i128_from_i64(val_s);
        uint64_t abs_delta_u;
        uint64_t abs_delta_s;

        if (val_u < min_u)
            min_u = val_u;
        if (val_u > max_u)
            max_u = val_u;
        if (val_s < min_s)
            min_s = val_s;
        if (val_s > max_s)
            max_s = val_s;

        sum_u_safe = TRS_wide_u128_add_u64(sum_u_safe, val_u);
        sum_s_safe = TRS_wide_i128_add_i64(sum_s_safe, val_s);

        if (val_u == 0) {
            zero_count++;
        } else {
            nonzero_count++;
            {
                uint64_t ctz = (uint64_t)ZL_ctz64(val_u);
                if (ctz < min_lb0)
                    min_lb0 = ctz;
            }
        }

        abs_delta_u  = TRS_numeric_abs_diff_u64(val_u, prev_u);
        delta_s_safe = TRS_wide_i128_sub_i64(delta_s_safe, prev_s);
        abs_delta_s  = TRS_wide_i128_abs_to_u64(delta_s_safe);

        if (val_u != prev_u) {
            if (val_u > prev_u) {
                delta_up_count++;
            } else {
                delta_down_count++;
            }
            if (have_transition_pos) {
                double gap = (double)(i - last_transition_pos);
                transition_gap_sum += gap;
                transition_gap_square_sum += gap * gap;
                transition_gap_count++;
            }
            last_transition_pos = i;
            have_transition_pos = 1;
        }

        if (val_u == prev_u) {
            current_run_length++;
            if (current_run_length > max_run_length) {
                max_run_length = current_run_length;
            }
        } else {
            current_run_length = 1;
        }

        sum_abs_delta_u = TRS_wide_u128_add_u64(sum_abs_delta_u, abs_delta_u);
        if (abs_delta_u > max_abs_delta_u)
            max_abs_delta_u = abs_delta_u;
        if (abs_delta_s > max_abs_delta_s)
            max_abs_delta_s = abs_delta_s;

        if (!have_delta) {
            delta_min_s_safe = delta_s_safe;
            delta_max_s_safe = delta_s_safe;
            have_delta       = 1;
        } else {
            if (TRS_wide_i128_lt(delta_s_safe, delta_min_s_safe)) {
                delta_min_s_safe = delta_s_safe;
            }
            if (TRS_wide_i128_gt(delta_s_safe, delta_max_s_safe)) {
                delta_max_s_safe = delta_s_safe;
            }
        }

        TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, val_u);
        prev_u = val_u;
        prev_s = val_s;
    }

    dst.zero_count       = zero_count;
    dst.max_run_length   = max_run_length;
    dst.delta_up_count   = delta_up_count;
    dst.delta_down_count = delta_down_count;
    dst.min_u            = min_u;
    dst.max_u            = max_u;
    dst.min_s            = min_s;
    dst.max_s            = max_s;
    dst.max_abs_delta_u  = max_abs_delta_u;
    dst.max_abs_delta_s  = max_abs_delta_s;
    dst.range_u          = max_u - min_u;
    dst.range_s          = TRS_wide_i64_abs_diff_to_u64(max_s, min_s);
    dst.cardinality_est  = TRS_estimate_cardinality_u64(data, n_elements);
    dst.pair_cardinality_est =
            TRS_estimate_pair_cardinality_u64(data, n_elements);
    dst.min_lb0 = (nonzero_count == 0) ? 0 : min_lb0;
    dst.sorted_gap_nmad =
            decode_u64_fp(TRS_numeric_kmv_compute_gap_nmad_fp(kmv_heap, kmv_n));
    ZL_ASSERT(has_sorted_gap_workspace(workspace));
    dst.sorted_gap_mode = TRS_numeric_compute_sorted_gap_mode(
            data,
            n_elements,
            workspace->sorted_gap_buffer,
            workspace->sorted_gap_capacity);
    dst.transition_gap_cv = TRS_numeric_compute_transition_gap_cv(
            transition_gap_sum,
            transition_gap_square_sum,
            transition_gap_count);

    dst.sum_u = TRS_wide_u128_to_double(sum_u_safe);
    dst.sum_s = TRS_wide_i128_to_double(sum_s_safe);
    if (have_delta) {
        dst.delta_min_s = TRS_wide_i128_to_double(delta_min_s_safe);
        dst.delta_max_s = TRS_wide_i128_to_double(delta_max_s_safe);
    }
    dst.mean_u     = dst.sum_u / (double)n_elements;
    dst.mean_s     = dst.sum_s / (double)n_elements;
    dst.zero_ratio = (double)zero_count / (double)n_elements;

    if (n_elements > 1) {
        size_t n_transitions = n_elements - 1;
        dst.mean_abs_delta_u = TRS_wide_u128_to_double(sum_abs_delta_u)
                / (double)n_transitions;
        dst.delta_up_ratio   = (double)delta_up_count / (double)n_transitions;
        dst.delta_down_ratio = (double)delta_down_count / (double)n_transitions;
    }

    {
        double sum_abs_dev_u = 0.0;
        double sum_abs_dev_s = 0.0;

        for (size_t i = 0; i < n_elements; i++) {
            sum_abs_dev_u += fabs((double)data[i] - dst.mean_u);
            sum_abs_dev_s += fabs((double)to_signed(data[i], 8) - dst.mean_s);
        }

        dst.mean_abs_dev_u = sum_abs_dev_u / (double)n_elements;
        dst.mean_abs_dev_s = sum_abs_dev_s / (double)n_elements;
    }

    return dst;
}

TRS_NumericFeaturesV2 TRS_numericFeaturesV2_extract_num64_decision(
        const uint64_t* data,
        size_t n_elements,
        const TRS_NumericExtractWorkspace* workspace)
{
    TRS_NumericFeaturesV2 dst = init_numeric_features_v2(n_elements, 8);
    if (n_elements == 0) {
        return dst;
    }
    ZL_ASSERT_NN(data);

    uint64_t min_u               = data[0];
    uint64_t max_u               = data[0];
    int64_t min_s                = to_signed(data[0], 8);
    int64_t max_s                = min_s;
    TRS_WideU128 sum_abs_delta_u = TRS_wide_u128_from_u64(0);
    uint64_t zero_count          = (data[0] == 0) ? 1 : 0;
    uint64_t delta_up_count      = 0;
    uint64_t delta_down_count    = 0;
    uint64_t min_lb0             = 64;
    uint64_t nonzero_count       = (data[0] != 0) ? 1 : 0;
    uint64_t prev_u              = data[0];
    TRS_NumericKmvEntry kmv_heap[TRS_NUMERIC_KMV_K];
    size_t kmv_n = 0;

    if (data[0] != 0) {
        min_lb0 = (uint64_t)ZL_ctz64(data[0]);
    }
    TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, data[0]);

    for (size_t i = 1; i < n_elements; i++) {
        uint64_t val_u = data[i];
        int64_t val_s  = to_signed(val_u, 8);
        uint64_t abs_delta_u;

        if (val_u < min_u)
            min_u = val_u;
        if (val_u > max_u)
            max_u = val_u;
        if (val_s < min_s)
            min_s = val_s;
        if (val_s > max_s)
            max_s = val_s;

        if (val_u == 0) {
            zero_count++;
        } else {
            nonzero_count++;
            {
                uint64_t ctz = (uint64_t)ZL_ctz64(val_u);
                if (ctz < min_lb0)
                    min_lb0 = ctz;
            }
        }

        abs_delta_u     = TRS_numeric_abs_diff_u64(val_u, prev_u);
        sum_abs_delta_u = TRS_wide_u128_add_u64(sum_abs_delta_u, abs_delta_u);

        if (val_u != prev_u) {
            if (val_u > prev_u) {
                delta_up_count++;
            } else {
                delta_down_count++;
            }
        }

        TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, val_u);
        prev_u = val_u;
    }

    dst.zero_count       = zero_count;
    dst.delta_up_count   = delta_up_count;
    dst.delta_down_count = delta_down_count;
    dst.min_u            = min_u;
    dst.max_u            = max_u;
    dst.min_s            = min_s;
    dst.max_s            = max_s;
    dst.range_u          = max_u - min_u;
    dst.range_s          = TRS_wide_i64_abs_diff_to_u64(max_s, min_s);
    dst.cardinality_est  = TRS_estimate_cardinality_u64(data, n_elements);
    dst.pair_cardinality_est =
            TRS_estimate_pair_cardinality_u64(data, n_elements);
    dst.min_lb0 = (nonzero_count == 0) ? 0 : min_lb0;
    dst.sorted_gap_nmad =
            decode_u64_fp(TRS_numeric_kmv_compute_gap_nmad_fp(kmv_heap, kmv_n));
    ZL_ASSERT(has_sorted_gap_workspace(workspace));
    dst.sorted_gap_mode = TRS_numeric_compute_sorted_gap_mode(
            data,
            n_elements,
            workspace->sorted_gap_buffer,
            workspace->sorted_gap_capacity);
    dst.zero_ratio = (double)zero_count / (double)n_elements;

    if (n_elements > 1) {
        size_t n_transitions = n_elements - 1;
        dst.mean_abs_delta_u = TRS_wide_u128_to_double(sum_abs_delta_u)
                / (double)n_transitions;
        dst.delta_up_ratio   = (double)delta_up_count / (double)n_transitions;
        dst.delta_down_ratio = (double)delta_down_count / (double)n_transitions;
    }

    return dst;
}

static TRS_NumericFeatures NumericFeatures_extract_sub64(
        const uint32_t* data,
        size_t n_elements,
        size_t elt_width,
        const TRS_NumericExtractWorkspace* workspace)
{
    ZL_ASSERT_LE(n_elements, UINT32_MAX);
    TRS_NumericFeatures f = init_numeric_features(n_elements, elt_width);
    if (n_elements == 0)
        return f;
    ZL_ASSERT_NN(data);

    uint64_t min_u                   = (uint64_t)data[0];
    uint64_t max_u                   = min_u;
    uint64_t sum_u                   = min_u;
    int64_t min_s                    = to_signed((uint64_t)data[0], elt_width);
    int64_t max_s                    = min_s;
    int64_t sum_s                    = min_s;
    uint64_t max_abs_delta_u         = 0;
    uint64_t max_abs_delta_s         = 0;
    TRS_WideU128 sum_abs_delta_u     = TRS_wide_u128_from_u64(0);
    size_t zero_count                = (data[0] == 0) ? 1 : 0;
    size_t current_run_length        = 1;
    size_t max_run_length            = 1;
    size_t delta_up_count            = 0;
    size_t delta_down_count          = 0;
    int64_t delta_min_s              = 0;
    int64_t delta_max_s              = 0;
    int64_t delta_first_min_s        = INT64_MAX;
    int64_t delta_first_max_s        = INT64_MIN;
    uint64_t min_lb0                 = 64;
    size_t nonzero_count             = (data[0] != 0) ? 1 : 0;
    uint64_t prev_u                  = (uint64_t)data[0];
    int64_t prev_s                   = min_s;
    int have_transition_pos          = 0;
    size_t last_transition_pos       = 0;
    size_t transition_gap_count      = 0;
    double transition_gap_sum        = 0.0;
    double transition_gap_square_sum = 0.0;
    ZL_ASSERT(has_kmv_workspace(workspace));
    ZL_ASSERT(
            elt_width != 2
            || has_cardinality_workspace(
                    workspace, TRS_CARDINALITY_U16_BITMAP_WORDS));
    TRS_NumericKmvEntry* const kmv_heap = workspace->kmv_entries;
    size_t kmv_n                        = 0;
    uint64_t* const seen_u16            = workspace->cardinality_bitmap;
    int use_seen_u16 = (elt_width == 2 && data[0] <= UINT16_MAX);

    if (data[0] != 0) {
        min_lb0 = (uint64_t)ZL_ctz32((uint32_t)data[0]);
    }
    if (use_seen_u16) {
        memset(seen_u16,
               0,
               sizeof(*seen_u16) * TRS_CARDINALITY_U16_BITMAP_WORDS);
        seen_bitmap_set(seen_u16, TRS_CARDINALITY_U16_BITMAP_WORDS, data[0]);
    }
    TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, (uint64_t)data[0]);

    for (size_t i = 1; i < n_elements; i++) {
        uint64_t val_u = (uint64_t)data[i];
        int64_t val_s  = to_signed(data[i], elt_width);

        if (use_seen_u16) {
            if (val_u <= UINT16_MAX) {
                seen_bitmap_set(
                        seen_u16, TRS_CARDINALITY_U16_BITMAP_WORDS, data[i]);
            } else {
                use_seen_u16 = 0;
            }
        }

        if (val_u < min_u)
            min_u = val_u;
        if (val_u > max_u)
            max_u = val_u;
        if (val_s < min_s)
            min_s = val_s;
        if (val_s > max_s)
            max_s = val_s;

        sum_u += val_u;
        sum_s += val_s;

        if (val_u == 0) {
            zero_count++;
        } else {
            nonzero_count++;
            uint64_t ctz = (uint64_t)ZL_ctz32(data[i]);
            if (ctz < min_lb0)
                min_lb0 = ctz;
        }

        uint64_t abs_delta_u = TRS_numeric_abs_diff_u64(val_u, prev_u);
        int64_t delta_s      = val_s - prev_s;
        uint64_t abs_delta_s = TRS_wide_abs_i64_to_u64(delta_s);

        if (abs_delta_u > max_abs_delta_u)
            max_abs_delta_u = abs_delta_u;
        if (abs_delta_s > max_abs_delta_s)
            max_abs_delta_s = abs_delta_s;
        sum_abs_delta_u = TRS_wide_u128_add_u64(sum_abs_delta_u, abs_delta_u);
        if (delta_s < delta_min_s)
            delta_min_s = delta_s;
        if (delta_s > delta_max_s)
            delta_max_s = delta_s;
        if (delta_s < delta_first_min_s)
            delta_first_min_s = delta_s;
        if (delta_s > delta_first_max_s)
            delta_first_max_s = delta_s;

        if (val_u != prev_u) {
            if (val_u > prev_u) {
                delta_up_count++;
            } else {
                delta_down_count++;
            }
            if (have_transition_pos) {
                double gap = (double)(i - last_transition_pos);
                transition_gap_sum += gap;
                transition_gap_square_sum += gap * gap;
                transition_gap_count++;
            }
            last_transition_pos = i;
            have_transition_pos = 1;
        }

        if (val_u == prev_u) {
            current_run_length++;
            if (current_run_length > max_run_length) {
                max_run_length = current_run_length;
            }
        } else {
            current_run_length = 1;
        }

        TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, val_u);
        prev_u = val_u;
        prev_s = val_s;
    }

    f.min_u            = min_u;
    f.max_u            = max_u;
    f.sum_u            = sum_u;
    f.min_s            = min_s;
    f.max_s            = max_s;
    f.sum_s            = sum_s;
    f.max_run_length   = max_run_length;
    f.max_abs_delta_u  = max_abs_delta_u;
    f.max_abs_delta_s  = max_abs_delta_s;
    f.zero_count       = zero_count;
    f.delta_up_count   = delta_up_count;
    f.delta_down_count = delta_down_count;
    f.delta_min_s      = delta_min_s;
    f.delta_max_s      = delta_max_s;
    /* Fewer than two elements have no delta; report 0 like delta_min_s. */
    f.delta_first_min_s = n_elements > 1 ? delta_first_min_s : 0;
    f.delta_first_max_s = n_elements > 1 ? delta_first_max_s : 0;
    f.min_lb0           = (nonzero_count == 0) ? 0 : min_lb0;

    f.range_u   = max_u - min_u;
    f.range_s   = TRS_wide_i64_abs_diff_to_u64(max_s, min_s);
    f.mean_u_fp = TRS_wide_u128_scaled_div_to_u64(
            TRS_wide_u128_from_u64(sum_u), n_elements);
    f.mean_s_fp = TRS_wide_i128_scaled_div_to_i64(
            TRS_wide_i128_from_i64(sum_s), n_elements);
    f.zero_ratio_fp = (zero_count * TRS_NUMERIC_Q32_SCALE) / n_elements;

    size_t n_transitions = n_elements - 1;
    if (n_transitions > 0) {
        f.delta_up_ratio_fp =
                (delta_up_count * TRS_NUMERIC_Q32_SCALE) / n_transitions;
        f.delta_down_ratio_fp =
                (delta_down_count * TRS_NUMERIC_Q32_SCALE) / n_transitions;
        f.mean_abs_delta_u_fp =
                TRS_wide_u128_scaled_div_to_u64(sum_abs_delta_u, n_transitions);
    }

    uint64_t mean_u        = sum_u / n_elements;
    int64_t mean_s         = sum_s / (int64_t)n_elements;
    uint64_t sum_abs_dev_u = 0;
    uint64_t sum_abs_dev_s = 0;
    for (size_t i = 0; i < n_elements; i++) {
        sum_abs_dev_u += TRS_numeric_abs_diff_u64(data[i], mean_u);
        sum_abs_dev_s +=
                TRS_wide_abs_i64_to_u64(to_signed(data[i], elt_width) - mean_s);
    }

    f.mean_abs_dev_u_fp = TRS_wide_u128_scaled_div_to_u64(
            TRS_wide_u128_from_u64(sum_abs_dev_u), n_elements);
    f.mean_abs_dev_s_fp = TRS_wide_u128_scaled_div_to_u64(
            TRS_wide_u128_from_u64(sum_abs_dev_s), n_elements);
    f.cardinality_est = use_seen_u16
            ? TRS_estimate_cardinality_u16_bitmap(seen_u16, n_elements)
            : TRS_estimate_cardinality_u32(data, n_elements);
    f.pair_cardinality_est =
            TRS_estimate_pair_cardinality_u32(data, n_elements);
    f.sorted_gap_cv_fp = TRS_numeric_kmv_compute_gap_nmad_fp(kmv_heap, kmv_n);
    f.transition_gap_cv_fp =
            TRS_numeric_encode_double_fp(TRS_numeric_compute_transition_gap_cv(
                    transition_gap_sum,
                    transition_gap_square_sum,
                    transition_gap_count));
    return f;
}

static TRS_NumericFeatures NumericFeatures_extract_sub64_decision(
        const uint32_t* data,
        size_t n_elements,
        size_t elt_width,
        const TRS_NumericExtractWorkspace* workspace)
{
    ZL_ASSERT_LE(n_elements, UINT32_MAX);
    TRS_NumericFeatures f = init_numeric_features(n_elements, elt_width);
    if (n_elements == 0)
        return f;
    ZL_ASSERT_NN(data);

    uint64_t min_u = (uint64_t)data[0];
    uint64_t max_u = min_u;
    int64_t min_s  = to_signed((uint64_t)data[0], elt_width);
    int64_t max_s  = min_s;
    /* At most UINT32_MAX differences, each bounded by UINT32_MAX. */
    uint64_t sum_abs_delta_u  = 0;
    size_t zero_count         = (data[0] == 0) ? 1 : 0;
    size_t delta_up_count     = 0;
    size_t delta_down_count   = 0;
    int64_t delta_min_s       = 0;
    int64_t delta_max_s       = 0;
    int64_t delta_first_min_s = INT64_MAX;
    int64_t delta_first_max_s = INT64_MIN;
    uint64_t min_lb0          = 64;
    size_t nonzero_count      = (data[0] != 0) ? 1 : 0;
    uint64_t prev_u           = (uint64_t)data[0];
    int64_t prev_s            = min_s;
    ZL_ASSERT(has_kmv_workspace(workspace));
    ZL_ASSERT(
            elt_width != 2
            || has_cardinality_workspace(
                    workspace, TRS_CARDINALITY_U16_BITMAP_WORDS));
    TRS_NumericKmvEntry* const kmv_heap = workspace->kmv_entries;
    size_t kmv_n                        = 0;
    uint64_t* const seen_u16            = workspace->cardinality_bitmap;
    int use_seen_u16 = (elt_width == 2 && data[0] <= UINT16_MAX);

    if (data[0] != 0) {
        min_lb0 = (uint64_t)ZL_ctz32((uint32_t)data[0]);
    }
    if (use_seen_u16) {
        memset(seen_u16,
               0,
               sizeof(*seen_u16) * TRS_CARDINALITY_U16_BITMAP_WORDS);
        seen_bitmap_set(seen_u16, TRS_CARDINALITY_U16_BITMAP_WORDS, data[0]);
    }
    TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, (uint64_t)data[0]);

    for (size_t i = 1; i < n_elements; i++) {
        uint64_t val_u  = (uint64_t)data[i];
        int64_t val_s   = to_signed(data[i], elt_width);
        int64_t delta_s = val_s - prev_s;

        if (use_seen_u16) {
            if (val_u <= UINT16_MAX) {
                seen_bitmap_set(
                        seen_u16, TRS_CARDINALITY_U16_BITMAP_WORDS, data[i]);
            } else {
                use_seen_u16 = 0;
            }
        }

        if (val_u < min_u)
            min_u = val_u;
        if (val_u > max_u)
            max_u = val_u;
        if (val_s < min_s)
            min_s = val_s;
        if (val_s > max_s)
            max_s = val_s;

        if (val_u == 0) {
            zero_count++;
        } else {
            nonzero_count++;
            {
                uint64_t ctz = (uint64_t)ZL_ctz32(data[i]);
                if (ctz < min_lb0)
                    min_lb0 = ctz;
            }
        }

        if (delta_s < delta_min_s)
            delta_min_s = delta_s;
        if (delta_s > delta_max_s)
            delta_max_s = delta_s;
        if (delta_s < delta_first_min_s)
            delta_first_min_s = delta_s;
        if (delta_s > delta_first_max_s)
            delta_first_max_s = delta_s;

        if (val_u != prev_u) {
            if (val_u > prev_u) {
                delta_up_count++;
            } else {
                delta_down_count++;
            }
        }

        uint64_t abs_delta_u = TRS_numeric_abs_diff_u64(val_u, prev_u);
        sum_abs_delta_u += abs_delta_u;
        TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, val_u);
        prev_u = val_u;
        prev_s = val_s;
    }

    f.min_u            = min_u;
    f.max_u            = max_u;
    f.min_s            = min_s;
    f.max_s            = max_s;
    f.zero_count       = zero_count;
    f.delta_up_count   = delta_up_count;
    f.delta_down_count = delta_down_count;
    f.delta_min_s      = delta_min_s;
    f.delta_max_s      = delta_max_s;
    /* Fewer than two elements have no delta; report 0 like delta_min_s. */
    f.delta_first_min_s = n_elements > 1 ? delta_first_min_s : 0;
    f.delta_first_max_s = n_elements > 1 ? delta_first_max_s : 0;
    f.min_lb0           = (nonzero_count == 0) ? 0 : min_lb0;

    f.range_u       = max_u - min_u;
    f.range_s       = TRS_wide_i64_abs_diff_to_u64(max_s, min_s);
    f.zero_ratio_fp = (zero_count * TRS_NUMERIC_Q32_SCALE) / n_elements;

    {
        size_t n_transitions = n_elements - 1;
        if (n_transitions > 0) {
            f.delta_up_ratio_fp =
                    (delta_up_count * TRS_NUMERIC_Q32_SCALE) / n_transitions;
            f.delta_down_ratio_fp =
                    (delta_down_count * TRS_NUMERIC_Q32_SCALE) / n_transitions;
            f.mean_abs_delta_u_fp = TRS_wide_u128_scaled_div_to_u64(
                    TRS_wide_u128_from_u64(sum_abs_delta_u), n_transitions);
        }
    }

    f.cardinality_est = use_seen_u16
            ? TRS_estimate_cardinality_u16_bitmap(seen_u16, n_elements)
            : TRS_estimate_cardinality_u32(data, n_elements);
    f.pair_cardinality_est =
            TRS_estimate_pair_cardinality_u32(data, n_elements);
    f.sorted_gap_cv_fp = TRS_numeric_kmv_compute_gap_nmad_fp(kmv_heap, kmv_n);
    return f;
}

static TRS_NumericFeatures NumericFeatures_extract_num64(
        const uint64_t* data,
        size_t n_elements)
{
    ZL_ASSERT_LE(n_elements, UINT32_MAX);
    TRS_NumericFeatures f = init_numeric_features(n_elements, 8);
    if (n_elements == 0)
        return f;
    ZL_ASSERT_NN(data);

    /*
     * sum_u is the legacy modulo-2^64 feature. Derived means use the wide
     * accumulators; signed accumulation stays wide because its overflow is UB.
     */
    uint64_t min_u                   = data[0];
    uint64_t max_u                   = data[0];
    uint64_t sum_u                   = data[0];
    TRS_WideU128 sum_u_safe          = TRS_wide_u128_from_u64(data[0]);
    int64_t min_s                    = to_signed(data[0], 8);
    int64_t max_s                    = min_s;
    TRS_WideI128 sum_s_safe          = TRS_wide_i128_from_i64(min_s);
    uint64_t max_abs_delta_u         = 0;
    uint64_t max_abs_delta_s         = 0;
    TRS_WideU128 sum_abs_delta_u     = TRS_wide_u128_from_u64(0);
    size_t zero_count                = (data[0] == 0) ? 1 : 0;
    size_t current_run_length        = 1;
    size_t max_run_length            = 1;
    size_t delta_up_count            = 0;
    size_t delta_down_count          = 0;
    TRS_WideI128 delta_min_s_safe    = TRS_wide_i128_from_i64(0);
    TRS_WideI128 delta_max_s_safe    = TRS_wide_i128_from_i64(0);
    uint64_t min_lb0                 = 64;
    size_t nonzero_count             = (data[0] != 0) ? 1 : 0;
    uint64_t prev_u                  = data[0];
    int64_t prev_s                   = min_s;
    int have_transition_pos          = 0;
    size_t last_transition_pos       = 0;
    size_t transition_gap_count      = 0;
    double transition_gap_sum        = 0.0;
    double transition_gap_square_sum = 0.0;
    TRS_NumericKmvEntry kmv_heap[TRS_NUMERIC_KMV_K];
    size_t kmv_n = 0;

    if (data[0] != 0) {
        min_lb0 = (uint64_t)ZL_ctz64(data[0]);
    }
    TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, data[0]);

    for (size_t i = 1; i < n_elements; i++) {
        uint64_t val_u = data[i];
        int64_t val_s  = to_signed(data[i], 8);

        if (val_u < min_u)
            min_u = val_u;
        if (val_u > max_u)
            max_u = val_u;
        if (val_s < min_s)
            min_s = val_s;
        if (val_s > max_s)
            max_s = val_s;

        sum_u += val_u;
        sum_u_safe = TRS_wide_u128_add_u64(sum_u_safe, val_u);
        sum_s_safe = TRS_wide_i128_add_i64(sum_s_safe, val_s);

        if (val_u == 0) {
            zero_count++;
        } else {
            nonzero_count++;
            uint64_t ctz = (uint64_t)ZL_ctz64(val_u);
            if (ctz < min_lb0)
                min_lb0 = ctz;
        }

        uint64_t abs_delta_u      = TRS_numeric_abs_diff_u64(val_u, prev_u);
        TRS_WideI128 delta_s_safe = TRS_wide_i128_from_i64(val_s);
        delta_s_safe              = TRS_wide_i128_sub_i64(delta_s_safe, prev_s);
        uint64_t abs_delta_s      = TRS_wide_i128_abs_to_u64(delta_s_safe);

        if (abs_delta_u > max_abs_delta_u)
            max_abs_delta_u = abs_delta_u;
        if (abs_delta_s > max_abs_delta_s)
            max_abs_delta_s = abs_delta_s;
        sum_abs_delta_u = TRS_wide_u128_add_u64(sum_abs_delta_u, abs_delta_u);
        if (TRS_wide_i128_lt(delta_s_safe, delta_min_s_safe)) {
            delta_min_s_safe = delta_s_safe;
        }
        if (TRS_wide_i128_gt(delta_s_safe, delta_max_s_safe)) {
            delta_max_s_safe = delta_s_safe;
        }

        if (val_u != prev_u) {
            if (val_u > prev_u) {
                delta_up_count++;
            } else {
                delta_down_count++;
            }
            if (have_transition_pos) {
                double gap = (double)(i - last_transition_pos);
                transition_gap_sum += gap;
                transition_gap_square_sum += gap * gap;
                transition_gap_count++;
            }
            last_transition_pos = i;
            have_transition_pos = 1;
        }

        if (val_u == prev_u) {
            current_run_length++;
            if (current_run_length > max_run_length) {
                max_run_length = current_run_length;
            }
        } else {
            current_run_length = 1;
        }

        TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, val_u);
        prev_u = val_u;
        prev_s = val_s;
    }

    f.min_u            = min_u;
    f.max_u            = max_u;
    f.sum_u            = sum_u;
    f.min_s            = min_s;
    f.max_s            = max_s;
    f.sum_s            = TRS_wide_i128_saturate_to_i64(sum_s_safe);
    f.max_run_length   = max_run_length;
    f.max_abs_delta_u  = max_abs_delta_u;
    f.max_abs_delta_s  = max_abs_delta_s;
    f.zero_count       = zero_count;
    f.delta_up_count   = delta_up_count;
    f.delta_down_count = delta_down_count;
    f.delta_min_s      = TRS_wide_i128_saturate_to_i64(delta_min_s_safe);
    f.delta_max_s      = TRS_wide_i128_saturate_to_i64(delta_max_s_safe);
    f.min_lb0          = (nonzero_count == 0) ? 0 : min_lb0;

    f.range_u       = max_u - min_u;
    f.range_s       = TRS_wide_i64_abs_diff_to_u64(max_s, min_s);
    f.mean_u_fp     = TRS_wide_u128_scaled_div_to_u64(sum_u_safe, n_elements);
    f.mean_s_fp     = TRS_wide_i128_scaled_div_to_i64(sum_s_safe, n_elements);
    f.zero_ratio_fp = (zero_count * TRS_NUMERIC_Q32_SCALE) / n_elements;

    size_t n_transitions = n_elements - 1;
    if (n_transitions > 0) {
        f.delta_up_ratio_fp =
                (delta_up_count * TRS_NUMERIC_Q32_SCALE) / n_transitions;
        f.delta_down_ratio_fp =
                (delta_down_count * TRS_NUMERIC_Q32_SCALE) / n_transitions;
        f.mean_abs_delta_u_fp =
                TRS_wide_u128_scaled_div_to_u64(sum_abs_delta_u, n_transitions);
    }

    /* Arithmetic means remain within the range of their input type. */
    uint64_t mean_u =
            TRS_wide_u128_div_u64_to_u64(sum_u_safe, (uint64_t)n_elements);
    int64_t mean_s =
            TRS_wide_i128_div_u64_to_i64(sum_s_safe, (uint64_t)n_elements);
    TRS_WideU128 sum_abs_dev_u_safe = TRS_wide_u128_from_u64(0);
    TRS_WideU128 sum_abs_dev_s_safe = TRS_wide_u128_from_u64(0);
    for (size_t i = 0; i < n_elements; i++) {
        TRS_WideI128 dev_s = TRS_wide_i128_from_i64(to_signed(data[i], 8));
        dev_s              = TRS_wide_i128_sub_i64(dev_s, mean_s);
        sum_abs_dev_u_safe = TRS_wide_u128_add_u64(
                sum_abs_dev_u_safe, TRS_numeric_abs_diff_u64(data[i], mean_u));
        sum_abs_dev_s_safe = TRS_wide_u128_add_u64(
                sum_abs_dev_s_safe, TRS_wide_i128_abs_to_u64(dev_s));
    }

    f.mean_abs_dev_u_fp =
            TRS_wide_u128_scaled_div_to_u64(sum_abs_dev_u_safe, n_elements);
    f.mean_abs_dev_s_fp =
            TRS_wide_u128_scaled_div_to_u64(sum_abs_dev_s_safe, n_elements);
    f.cardinality_est = TRS_estimate_cardinality_u64(data, n_elements);
    f.pair_cardinality_est =
            TRS_estimate_pair_cardinality_u64(data, n_elements);
    f.sorted_gap_cv_fp = TRS_numeric_kmv_compute_gap_nmad_fp(kmv_heap, kmv_n);
    f.transition_gap_cv_fp =
            TRS_numeric_encode_double_fp(TRS_numeric_compute_transition_gap_cv(
                    transition_gap_sum,
                    transition_gap_square_sum,
                    transition_gap_count));
    return f;
}

int TRS_numericFeatures_extract(
        TRS_NumericFeatures* result,
        const uint64_t* data,
        size_t n_elements,
        size_t elt_width,
        const TRS_NumericExtractWorkspace* workspace)
{
    ZL_ASSERT_NN(result);
    ZL_ASSERT(
            elt_width == 1 || elt_width == 2 || elt_width == 4
            || elt_width == 8);
    if (result == NULL
        || (elt_width != 1 && elt_width != 2 && elt_width != 4
            && elt_width != 8)) {
        return 0;
    }
    if (elt_width == 8) {
        if (n_elements != 0 && data == NULL) {
            return 0;
        }
        *result = NumericFeatures_extract_num64(data, n_elements);
        return 1;
    }
    if (n_elements == 0) {
        *result = init_numeric_features(n_elements, elt_width);
        return 1;
    }
    ZL_ASSERT_NN(data);
    if (data == NULL || workspace == NULL || workspace->values == NULL
        || workspace->capacity < n_elements || !has_kmv_workspace(workspace)
        || (elt_width == 2
            && !has_cardinality_workspace(
                    workspace, TRS_CARDINALITY_U16_BITMAP_WORDS))) {
        return 0;
    }

    uint32_t* const buf      = workspace->values;
    uint64_t const max_value = elt_width == 1 ? UINT8_MAX
            : elt_width == 2                  ? UINT16_MAX
                                              : UINT32_MAX;
    for (size_t i = 0; i < n_elements; i++) {
        ZL_ASSERT_LE(data[i], max_value);
        buf[i] = (uint32_t)data[i];
    }

    *result = NumericFeatures_extract_sub64(
            buf, n_elements, elt_width, workspace);
    return 1;
}

static TRS_NumericFeatures NumericFeatures_extract_num8_from_bytes(
        const uint8_t* data,
        size_t n_bytes,
        const TRS_NumericExtractWorkspace* workspace)
{
    ZL_ASSERT_LE(n_bytes, UINT32_MAX);
    TRS_NumericFeatures f = init_numeric_features(n_bytes, 1);
    if (n_bytes == 0)
        return f;
    ZL_ASSERT_NN(data);

    uint64_t min_u                   = (uint64_t)data[0];
    uint64_t max_u                   = min_u;
    uint64_t sum_u                   = min_u;
    int64_t min_s                    = (int64_t)(int8_t)data[0];
    int64_t max_s                    = min_s;
    int64_t sum_s                    = min_s;
    uint64_t max_abs_delta_u         = 0;
    uint64_t max_abs_delta_s         = 0;
    TRS_WideU128 sum_abs_delta_u     = TRS_wide_u128_from_u64(0);
    size_t zero_count                = (data[0] == 0) ? 1 : 0;
    size_t current_run_length        = 1;
    size_t max_run_length            = 1;
    size_t delta_up_count            = 0;
    size_t delta_down_count          = 0;
    int64_t delta_min_s              = 0;
    int64_t delta_max_s              = 0;
    int64_t delta_first_min_s        = INT64_MAX;
    int64_t delta_first_max_s        = INT64_MIN;
    uint64_t min_lb0                 = 64;
    size_t nonzero_count             = (data[0] != 0) ? 1 : 0;
    uint64_t prev_u                  = (uint64_t)data[0];
    int64_t prev_s                   = min_s;
    int have_transition_pos          = 0;
    size_t last_transition_pos       = 0;
    size_t transition_gap_count      = 0;
    double transition_gap_sum        = 0.0;
    double transition_gap_square_sum = 0.0;
    ZL_ASSERT(has_kmv_workspace(workspace));
    ZL_ASSERT(has_cardinality_workspace(
            workspace, TRS_NUMERIC_CARDINALITY_SCRATCH_WORDS));
    TRS_NumericKmvEntry* const kmv_heap = workspace->kmv_entries;
    size_t kmv_n                        = 0;
    uint64_t* const seen_values         = workspace->cardinality_bitmap;
    uint64_t* const seen_pairs = seen_values + TRS_CARDINALITY_U8_BITMAP_WORDS;
    memset(seen_values,
           0,
           sizeof(*seen_values) * TRS_NUMERIC_CARDINALITY_SCRATCH_WORDS);

    if (data[0] != 0) {
        min_lb0 = (uint64_t)ZL_ctz32((uint32_t)data[0]);
    }
    seen_bitmap_set(seen_values, TRS_CARDINALITY_U8_BITMAP_WORDS, data[0]);
    TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, (uint64_t)data[0]);

    for (size_t i = 1; i < n_bytes; i++) {
        uint64_t val_u = (uint64_t)data[i];
        int64_t val_s  = (int64_t)(int8_t)data[i];
        uint64_t abs_delta_u;
        int64_t delta_s;
        uint64_t abs_delta_s;
        uint32_t pair_key = ((uint32_t)prev_u << 8) | (uint32_t)val_u;

        seen_bitmap_set(seen_values, TRS_CARDINALITY_U8_BITMAP_WORDS, data[i]);
        seen_bitmap_set(
                seen_pairs, TRS_CARDINALITY_U8_PAIR_BITMAP_WORDS, pair_key);

        if (val_u < min_u)
            min_u = val_u;
        if (val_u > max_u)
            max_u = val_u;
        if (val_s < min_s)
            min_s = val_s;
        if (val_s > max_s)
            max_s = val_s;

        sum_u += val_u;
        sum_s += val_s;

        if (val_u == 0) {
            zero_count++;
        } else {
            nonzero_count++;
            {
                uint64_t ctz = (uint64_t)ZL_ctz32(data[i]);
                if (ctz < min_lb0)
                    min_lb0 = ctz;
            }
        }

        abs_delta_u = TRS_numeric_abs_diff_u64(val_u, prev_u);
        delta_s     = val_s - prev_s;
        abs_delta_s = TRS_wide_abs_i64_to_u64(delta_s);

        if (abs_delta_u > max_abs_delta_u)
            max_abs_delta_u = abs_delta_u;
        if (abs_delta_s > max_abs_delta_s)
            max_abs_delta_s = abs_delta_s;
        sum_abs_delta_u = TRS_wide_u128_add_u64(sum_abs_delta_u, abs_delta_u);
        if (delta_s < delta_min_s)
            delta_min_s = delta_s;
        if (delta_s > delta_max_s)
            delta_max_s = delta_s;
        if (delta_s < delta_first_min_s)
            delta_first_min_s = delta_s;
        if (delta_s > delta_first_max_s)
            delta_first_max_s = delta_s;

        if (val_u != prev_u) {
            if (val_u > prev_u) {
                delta_up_count++;
            } else {
                delta_down_count++;
            }
            if (have_transition_pos) {
                double gap = (double)(i - last_transition_pos);
                transition_gap_sum += gap;
                transition_gap_square_sum += gap * gap;
                transition_gap_count++;
            }
            last_transition_pos = i;
            have_transition_pos = 1;
        }

        if (val_u == prev_u) {
            current_run_length++;
            if (current_run_length > max_run_length) {
                max_run_length = current_run_length;
            }
        } else {
            current_run_length = 1;
        }

        TRS_numeric_kmv_track_inline(kmv_heap, &kmv_n, val_u);
        prev_u = val_u;
        prev_s = val_s;
    }

    f.min_u            = min_u;
    f.max_u            = max_u;
    f.sum_u            = sum_u;
    f.min_s            = min_s;
    f.max_s            = max_s;
    f.sum_s            = sum_s;
    f.max_run_length   = max_run_length;
    f.max_abs_delta_u  = max_abs_delta_u;
    f.max_abs_delta_s  = max_abs_delta_s;
    f.zero_count       = zero_count;
    f.delta_up_count   = delta_up_count;
    f.delta_down_count = delta_down_count;
    f.delta_min_s      = delta_min_s;
    f.delta_max_s      = delta_max_s;
    /* Fewer than two elements have no delta; report 0 like delta_min_s. */
    f.delta_first_min_s = n_bytes > 1 ? delta_first_min_s : 0;
    f.delta_first_max_s = n_bytes > 1 ? delta_first_max_s : 0;
    f.min_lb0           = (nonzero_count == 0) ? 0 : min_lb0;

    f.range_u   = max_u - min_u;
    f.range_s   = TRS_wide_i64_abs_diff_to_u64(max_s, min_s);
    f.mean_u_fp = TRS_wide_u128_scaled_div_to_u64(
            TRS_wide_u128_from_u64(sum_u), n_bytes);
    f.mean_s_fp = TRS_wide_i128_scaled_div_to_i64(
            TRS_wide_i128_from_i64(sum_s), n_bytes);
    f.zero_ratio_fp = (zero_count * TRS_NUMERIC_Q32_SCALE) / n_bytes;

    {
        size_t n_transitions = n_bytes - 1;
        if (n_transitions > 0) {
            f.delta_up_ratio_fp =
                    (delta_up_count * TRS_NUMERIC_Q32_SCALE) / n_transitions;
            f.delta_down_ratio_fp =
                    (delta_down_count * TRS_NUMERIC_Q32_SCALE) / n_transitions;
            f.mean_abs_delta_u_fp = TRS_wide_u128_scaled_div_to_u64(
                    sum_abs_delta_u, n_transitions);
        }
    }

    {
        uint64_t mean_u        = sum_u / n_bytes;
        int64_t mean_s         = sum_s / (int64_t)n_bytes;
        uint64_t sum_abs_dev_u = 0;
        uint64_t sum_abs_dev_s = 0;
        for (size_t i = 0; i < n_bytes; i++) {
            sum_abs_dev_u +=
                    TRS_numeric_abs_diff_u64((uint64_t)data[i], mean_u);
            sum_abs_dev_s +=
                    TRS_wide_abs_i64_to_u64((int64_t)(int8_t)data[i] - mean_s);
        }

        f.mean_abs_dev_u_fp = TRS_wide_u128_scaled_div_to_u64(
                TRS_wide_u128_from_u64(sum_abs_dev_u), n_bytes);
        f.mean_abs_dev_s_fp = TRS_wide_u128_scaled_div_to_u64(
                TRS_wide_u128_from_u64(sum_abs_dev_s), n_bytes);
    }

    f.cardinality_est =
            TRS_estimate_cardinality_u8_bitmap(seen_values, n_bytes);
    f.pair_cardinality_est =
            TRS_estimate_pair_cardinality_u8_bitmap(seen_pairs, n_bytes);
    f.sorted_gap_cv_fp = TRS_numeric_kmv_compute_gap_nmad_fp(kmv_heap, kmv_n);
    f.transition_gap_cv_fp =
            TRS_numeric_encode_double_fp(TRS_numeric_compute_transition_gap_cv(
                    transition_gap_sum,
                    transition_gap_square_sum,
                    transition_gap_count));
    ZL_ASSERT(has_match4_workspace(workspace));
    f.match4 = TRS_numeric_compute_lz_matches_with_table(
            data,
            n_bytes,
            1,
            workspace->match4_table,
            workspace->match4_capacity);
    f.d8_cardinality_est = TRS_estimate_d8_cardinality(data, n_bytes, 1);
    return f;
}

/* ZL_Type_numeric stream construction rejects misaligned buffers in release
 * builds. The alignment assertions below verify that propagated invariant for
 * internal callers; alignment is therefore not a recoverable extraction
 * error. */
int TRS_numericFeatures_extract_from_bytes(
        TRS_NumericFeatures* result,
        const uint8_t* data_bytes,
        size_t n_bytes,
        size_t elt_width,
        const TRS_NumericExtractWorkspace* workspace)
{
    ZL_ASSERT_NN(result);
    ZL_ASSERT(
            elt_width == 1 || elt_width == 2 || elt_width == 4
            || elt_width == 8);
    if (result == NULL
        || (elt_width != 1 && elt_width != 2 && elt_width != 4
            && elt_width != 8)) {
        return 0;
    }
    ZL_ASSERT_EQ(n_bytes % elt_width, 0);
    if (n_bytes % elt_width != 0 || (n_bytes != 0 && data_bytes == NULL)) {
        return 0;
    }

    size_t n_elements = n_bytes / elt_width;
    TRS_NumericFeatures f;
    if (n_elements == 0) {
        *result = init_numeric_features(n_elements, elt_width);
        return 1;
    }
    if (!has_match4_workspace(workspace)) {
        return 0;
    }
    if (elt_width != 8 && !has_kmv_workspace(workspace)) {
        return 0;
    }
    if ((elt_width == 1
         && !has_cardinality_workspace(
                 workspace, TRS_NUMERIC_CARDINALITY_SCRATCH_WORDS))
        || (elt_width == 2
            && !has_cardinality_workspace(
                    workspace, TRS_CARDINALITY_U16_BITMAP_WORDS))) {
        return 0;
    }

    if (elt_width == 8) {
        ZL_ASSERT_EQ((uintptr_t)data_bytes % sizeof(uint64_t), (uintptr_t)0);
        const uint64_t* data64 = (const uint64_t*)(const void*)data_bytes;
        f = NumericFeatures_extract_num64(data64, n_elements);
    } else if (elt_width == 1) {
        *result = NumericFeatures_extract_num8_from_bytes(
                data_bytes, n_bytes, workspace);
        return 1;
    } else if (elt_width == 4) {
        ZL_ASSERT_EQ((uintptr_t)data_bytes % sizeof(uint32_t), (uintptr_t)0);
        f = NumericFeatures_extract_sub64(
                (const uint32_t*)(const void*)data_bytes,
                n_elements,
                elt_width,
                workspace);
    } else {
        ZL_ASSERT_EQ(elt_width, 2);
        if (workspace == NULL || workspace->values == NULL
            || workspace->capacity < n_elements) {
            return 0;
        }
        /* Widen sub-64-bit elements to uint32 via zero-extension. */
        uint32_t* const buf = workspace->values;

        for (size_t i = 0; i < n_elements; i++) {
            uint16_t val;
            memcpy(&val, data_bytes + i * sizeof(uint16_t), sizeof(val));
            buf[i] = val;
        }

        f = NumericFeatures_extract_sub64(
                buf, n_elements, elt_width, workspace);
    }

    f.match4 = TRS_numeric_compute_lz_matches_with_table(
            data_bytes,
            n_bytes,
            elt_width,
            workspace->match4_table,
            workspace->match4_capacity);
    f.d8_cardinality_est =
            TRS_estimate_d8_cardinality(data_bytes, n_bytes, elt_width);
    *result = f;
    return 1;
}

int TRS_numericFeatures_extract_sub64_decision_from_bytes(
        TRS_NumericFeatures* result,
        const uint8_t* data_bytes,
        size_t n_bytes,
        size_t elt_width,
        const TRS_NumericExtractWorkspace* workspace)
{
    ZL_ASSERT_NN(result);
    ZL_ASSERT_OR(elt_width == 2, elt_width == 4);
    if (result == NULL || (elt_width != 2 && elt_width != 4)) {
        return 0;
    }
    ZL_ASSERT_EQ(n_bytes % elt_width, 0);
    if (n_bytes % elt_width != 0 || (n_bytes != 0 && data_bytes == NULL)) {
        return 0;
    }

    size_t n_elements = n_bytes / elt_width;
    TRS_NumericFeatures f;

    if (n_elements == 0) {
        *result = init_numeric_features(n_elements, elt_width);
        return 1;
    }
    if (elt_width == 2
        && (!has_kmv_workspace(workspace)
            || !has_cardinality_workspace(
                    workspace, TRS_CARDINALITY_U16_BITMAP_WORDS))) {
        return 0;
    }

    if (elt_width == 4) {
        TRS_NumericKmvEntry kmv_entries[TRS_NUMERIC_KMV_K];
        const TRS_NumericExtractWorkspace local_workspace = {
            .kmv_entries  = kmv_entries,
            .kmv_capacity = TRS_NUMERIC_KMV_K,
        };
        ZL_ASSERT_EQ((uintptr_t)data_bytes % sizeof(uint32_t), (uintptr_t)0);
        f = NumericFeatures_extract_sub64_decision(
                (const uint32_t*)(const void*)data_bytes,
                n_elements,
                elt_width,
                &local_workspace);
    } else {
        ZL_ASSERT_EQ(elt_width, 2);
        if (workspace == NULL || workspace->values == NULL
            || workspace->capacity < n_elements) {
            return 0;
        }
        uint32_t* const buf = workspace->values;

        for (size_t i = 0; i < n_elements; i++) {
            uint16_t val16;
            memcpy(&val16, data_bytes + i * sizeof(uint16_t), sizeof(val16));
            uint32_t const val = val16;
            buf[i]             = val;
        }

        f = NumericFeatures_extract_sub64_decision(
                buf, n_elements, elt_width, workspace);
    }

    if (elt_width == 2) {
        f.d8_cardinality_est =
                TRS_estimate_d8_cardinality(data_bytes, n_bytes, elt_width);
    }
    *result = f;
    return 1;
}

int TRS_numericFeaturesV2_extract(
        TRS_NumericFeaturesV2* result,
        const uint64_t* data,
        size_t n_elements,
        size_t elt_width,
        const TRS_NumericExtractWorkspace* workspace)
{
    ZL_ASSERT_NN(result);
    ZL_ASSERT(
            elt_width == 1 || elt_width == 2 || elt_width == 4
            || elt_width == 8);
    if (result == NULL
        || (elt_width != 1 && elt_width != 2 && elt_width != 4
            && elt_width != 8)) {
        return 0;
    }
    if (n_elements != 0 && !has_sorted_gap_workspace(workspace)) {
        return 0;
    }
    if (elt_width == 8) {
        if (n_elements != 0 && data == NULL) {
            return 0;
        }
        *result = NumericFeaturesV2_extract_num64(data, n_elements, workspace);
        return 1;
    }
    TRS_NumericFeatures legacy;
    if (!TRS_numericFeatures_extract(
                &legacy, data, n_elements, elt_width, workspace)) {
        return 0;
    }
    double sorted_gap_mode = 0.0;
    if (n_elements != 0) {
        sorted_gap_mode = TRS_numeric_compute_sorted_gap_mode(
                data,
                n_elements,
                workspace->sorted_gap_buffer,
                workspace->sorted_gap_capacity);
    }
    *result = numeric_features_to_v2_sub64(&legacy, sorted_gap_mode);
    return 1;
}

int TRS_numericFeaturesV2_extract_from_bytes(
        TRS_NumericFeaturesV2* result,
        const uint8_t* data_bytes,
        size_t n_bytes,
        size_t elt_width,
        const TRS_NumericExtractWorkspace* workspace)
{
    ZL_ASSERT_NN(result);
    ZL_ASSERT(
            elt_width == 1 || elt_width == 2 || elt_width == 4
            || elt_width == 8);
    if (result == NULL
        || (elt_width != 1 && elt_width != 2 && elt_width != 4
            && elt_width != 8)) {
        return 0;
    }
    ZL_ASSERT_EQ(n_bytes % elt_width, 0);
    if (n_bytes % elt_width != 0 || (n_bytes != 0 && data_bytes == NULL)) {
        return 0;
    }

    size_t n_elements = n_bytes / elt_width;
    TRS_NumericFeaturesV2 dst;
    if (n_elements != 0
        && (!has_match4_workspace(workspace)
            || !has_sorted_gap_workspace(workspace))) {
        return 0;
    }

    if (elt_width == 8) {
        ZL_ASSERT_EQ((uintptr_t)data_bytes % sizeof(uint64_t), (uintptr_t)0);
        const uint64_t* data64 = (const uint64_t*)(const void*)data_bytes;
        dst = NumericFeaturesV2_extract_num64(data64, n_elements, workspace);
        if (n_elements != 0) {
            dst.match4 = TRS_numeric_compute_lz_matches_with_table(
                    data_bytes,
                    n_bytes,
                    elt_width,
                    workspace->match4_table,
                    workspace->match4_capacity);
        }
        dst.d8_cardinality_est =
                TRS_estimate_d8_cardinality(data_bytes, n_bytes, elt_width);
    } else {
        TRS_NumericFeatures legacy;
        if (!TRS_numericFeatures_extract_from_bytes(
                    &legacy, data_bytes, n_bytes, elt_width, workspace)) {
            return 0;
        }
        double sorted_gap_mode = 0.0;
        if (n_elements != 0) {
            sorted_gap_mode = TRS_numeric_compute_sorted_gap_mode_from_bytes(
                    data_bytes,
                    n_elements,
                    elt_width,
                    workspace->sorted_gap_buffer,
                    workspace->sorted_gap_capacity);
        }
        dst = numeric_features_to_v2_sub64(&legacy, sorted_gap_mode);
    }

    *result = dst;
    return 1;
}
