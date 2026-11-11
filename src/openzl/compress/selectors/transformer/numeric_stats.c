// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "numeric_stats.h"
#include "numeric_kmv.h"

#include "numeric_fixed_point.h"
#include "wide_arith.h"

#include "openzl/shared/mem.h"
#include "openzl/shared/pdqsort.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "openzl/common/assertion.h"

/* These band limits are part of the trained sorted-gap feature contract. */
#define TRS_SORTED_GAP_MAX_VALUES 512
#define TRS_SORTED_GAP_KEEP_WIDTH 384
#define TRS_SORTED_GAP_BATCH_VALUES 512
ZL_STATIC_ASSERT(
        TRS_NUMERIC_SORTED_GAP_BUFFER_ENTRIES
                == 2 * TRS_SORTED_GAP_MAX_VALUES + TRS_SORTED_GAP_BATCH_VALUES,
        "sorted-gap buffer must hold retained, batch, and output regions");
/* These parameters are part of the trained match4 feature contract. */
#define TRS_MATCH4_TABLE_BITS 12
#define TRS_MATCH4_WINDOW_BYTES 4
#define TRS_MATCH4_HASH_MULTIPLIER 2654435761U
ZL_STATIC_ASSERT(
        TRS_NUMERIC_MATCH4_TABLE_ENTRIES
                == ((size_t)1 << TRS_MATCH4_TABLE_BITS),
        "match4 table size must match its hash width");

static void
kmv_heap_sift_down(TRS_NumericKmvEntry* heap, size_t size, size_t index)
{
    while (1) {
        size_t largest = index;
        size_t left    = 2 * index + 1;
        size_t right   = 2 * index + 2;
        if (left < size && heap[left].hash > heap[largest].hash)
            largest = left;
        if (right < size && heap[right].hash > heap[largest].hash)
            largest = right;
        if (largest == index)
            break;
        TRS_NumericKmvEntry tmp = heap[index];
        heap[index]             = heap[largest];
        heap[largest]           = tmp;
        index                   = largest;
    }
}

void TRS_numeric_kmv_insert(
        TRS_NumericKmvEntry* heap,
        size_t* size,
        uint64_t hash,
        uint64_t value)
{
    if (*size < TRS_NUMERIC_KMV_K) {
        size_t index      = *size;
        heap[index].hash  = hash;
        heap[index].value = value;
        (*size)++;
        while (index > 0) {
            size_t parent = (index - 1) / 2;
            if (heap[parent].hash >= heap[index].hash)
                break;
            TRS_NumericKmvEntry tmp = heap[index];
            heap[index]             = heap[parent];
            heap[parent]            = tmp;
            index                   = parent;
        }
    } else {
        ZL_ASSERT_LT(hash, heap[0].hash);
        heap[0].hash  = hash;
        heap[0].value = value;
        kmv_heap_sift_down(heap, TRS_NUMERIC_KMV_K, 0);
    }
}

static int kmv_entry_cmp(const void* a, const void* b)
{
    uint64_t va = ((const TRS_NumericKmvEntry*)a)->value;
    uint64_t vb = ((const TRS_NumericKmvEntry*)b)->value;
    return (va > vb) - (va < vb);
}

uint64_t TRS_numeric_kmv_compute_gap_nmad_fp(
        TRS_NumericKmvEntry* heap,
        size_t kmv_size)
{
    ZL_ASSERT_LE(kmv_size, TRS_NUMERIC_KMV_K);

    if (kmv_size < 2)
        return 0;

    qsort(heap, kmv_size, sizeof(TRS_NumericKmvEntry), kmv_entry_cmp);

    size_t unique = 0;
    for (size_t i = 1; i < kmv_size; i++) {
        if (heap[i].value != heap[unique].value) {
            heap[++unique] = heap[i];
        }
    }
    size_t const distinct = unique + 1;
    if (distinct < 2)
        return 0;

    size_t const n_gaps = distinct - 1;
    uint64_t gaps[TRS_NUMERIC_KMV_K - 1];
    TRS_WideU128 sum_gaps = TRS_wide_u128_from_u64(0);
    for (size_t i = 0; i < n_gaps; i++) {
        gaps[i]  = heap[i + 1].value - heap[i].value;
        sum_gaps = TRS_wide_u128_add_u64(sum_gaps, gaps[i]);
    }

    /* The sorted gaps telescope to `max - min`, so their mean fits in u64. */
    uint64_t const mean_gap =
            TRS_wide_u128_div_u64_to_u64(sum_gaps, (uint64_t)n_gaps);
    ZL_ASSERT_LT(0, mean_gap);

    TRS_WideU128 sum_abs_dev = TRS_wide_u128_from_u64(0);
    for (size_t i = 0; i < n_gaps; i++) {
        sum_abs_dev = TRS_wide_u128_add_u64(
                sum_abs_dev, TRS_numeric_abs_diff_u64(gaps[i], mean_gap));
    }

    TRS_WideU128 const numerator = TRS_wide_u128_shl32(sum_abs_dev);
    TRS_WideU128 const denominator =
            TRS_wide_u128_mul_u64((uint64_t)n_gaps, mean_gap);
    /* This normalized deviation is bounded below 4 * 2^32. */
    return TRS_wide_u128_div_u128_to_u64(numerator, denominator);
}

typedef struct {
    uint64_t* retained_values;
    uint64_t* pending_values;
    uint64_t* output_values;
    size_t retained_count;
    size_t pending_count;
} SortedGapAccumulator;

static void sorted_gap_flush(SortedGapAccumulator* accumulator)
{
    ZL_ASSERT_LE(accumulator->pending_count, TRS_SORTED_GAP_BATCH_VALUES);
    pdqsort8(accumulator->pending_values, accumulator->pending_count);

    size_t pending_distinct = 0;
    for (size_t read = 0; read < accumulator->pending_count; ++read) {
        if (pending_distinct == 0
            || accumulator->pending_values[read]
                    != accumulator->pending_values[pending_distinct - 1]) {
            accumulator->pending_values[pending_distinct++] =
                    accumulator->pending_values[read];
        }
    }

    if (accumulator->retained_count == 0) {
        memcpy(accumulator->retained_values,
               accumulator->pending_values,
               pending_distinct * sizeof(*accumulator->retained_values));
        accumulator->retained_count = pending_distinct;
        accumulator->pending_count  = 0;
        return;
    }

    size_t retained_index = 0;
    size_t pending_index  = 0;
    size_t output_count   = 0;
    while (output_count < TRS_SORTED_GAP_MAX_VALUES
           && (retained_index < accumulator->retained_count
               || pending_index < pending_distinct)) {
        uint64_t value;
        if (pending_index == pending_distinct
            || (retained_index < accumulator->retained_count
                && accumulator->retained_values[retained_index]
                        < accumulator->pending_values[pending_index])) {
            value = accumulator->retained_values[retained_index++];
        } else if (
                retained_index == accumulator->retained_count
                || accumulator->pending_values[pending_index]
                        < accumulator->retained_values[retained_index]) {
            value = accumulator->pending_values[pending_index++];
        } else {
            value = accumulator->retained_values[retained_index++];
            ++pending_index;
        }
        if (output_count == 0
            || value != accumulator->output_values[output_count - 1]) {
            accumulator->output_values[output_count++] = value;
        }
    }

    uint64_t* const previous_retained = accumulator->retained_values;
    accumulator->retained_values      = accumulator->output_values;
    accumulator->output_values        = previous_retained;
    accumulator->retained_count       = output_count;
    accumulator->pending_count        = 0;
}

static void sorted_gap_add(SortedGapAccumulator* accumulator, uint64_t value)
{
    if (accumulator->retained_count == TRS_SORTED_GAP_MAX_VALUES
        && value >= accumulator
                            ->retained_values[TRS_SORTED_GAP_MAX_VALUES - 1]) {
        return;
    }

    ZL_ASSERT_LT(accumulator->pending_count, TRS_SORTED_GAP_BATCH_VALUES);
    accumulator->pending_values[accumulator->pending_count++] = value;
    if (accumulator->pending_count == TRS_SORTED_GAP_BATCH_VALUES) {
        sorted_gap_flush(accumulator);
    }
}

static size_t gap_frequency_in_suffix(
        const uint64_t* sorted_values,
        size_t gap_end,
        size_t end)
{
    uint64_t const gap = sorted_values[gap_end] - sorted_values[gap_end - 1];
    size_t frequency   = 0;
    for (size_t i = gap_end; i < end; ++i) {
        frequency += sorted_values[i] - sorted_values[i - 1] == gap;
    }
    return frequency;
}

static double compute_sorted_gap_mode(
        const uint64_t* smallest,
        size_t n_smallest)
{
    size_t const width = n_smallest < TRS_SORTED_GAP_KEEP_WIDTH
            ? n_smallest
            : TRS_SORTED_GAP_KEEP_WIDTH;
    if (width < 2)
        return 0.0;

    size_t const start   = n_smallest - width;
    size_t max_gap_count = 0;

    /*
     * The retained band contains at most 383 gaps. This bounded quadratic
     * scan avoids adding a hash table or allocation to selector feature
     * extraction. The earliest occurrence of each gap sees every later match,
     * so the maximum suffix frequency is the exact mode frequency.
     */
    for (size_t i = start + 1; i < n_smallest; i++) {
        size_t const gap_count =
                gap_frequency_in_suffix(smallest, i, n_smallest);
        if (gap_count > max_gap_count)
            max_gap_count = gap_count;
    }

    return (double)max_gap_count / (double)(width - 1);
}

double TRS_numeric_compute_sorted_gap_mode(
        const uint64_t* data,
        size_t n_elements,
        uint64_t* buffer,
        size_t buffer_capacity)
{
    ZL_ASSERT(n_elements == 0 || data != NULL);
    ZL_ASSERT(n_elements == 0 || buffer != NULL);
    ZL_ASSERT(
            n_elements == 0
            || buffer_capacity >= TRS_NUMERIC_SORTED_GAP_BUFFER_ENTRIES);
    if (n_elements == 0)
        return 0.0;

    SortedGapAccumulator accumulator = {
        .retained_values = buffer,
        .pending_values  = buffer + TRS_SORTED_GAP_MAX_VALUES,
        .output_values   = buffer + TRS_SORTED_GAP_MAX_VALUES
                + TRS_SORTED_GAP_BATCH_VALUES,
        .retained_count = 0,
        .pending_count  = 0,
    };

    for (size_t i = 0; i < n_elements; i++) {
        sorted_gap_add(&accumulator, data[i]);
    }
    if (accumulator.pending_count != 0)
        sorted_gap_flush(&accumulator);

    return compute_sorted_gap_mode(
            accumulator.retained_values, accumulator.retained_count);
}

static uint64_t
load_unsigned_value(const uint8_t* data, size_t index, size_t elt_width)
{
    const uint8_t* const src = data + index * elt_width;
    switch (elt_width) {
        case 1:
            return src[0];
        case 2: {
            uint16_t value;
            memcpy(&value, src, sizeof(value));
            return value;
        }
        case 4: {
            uint32_t value;
            memcpy(&value, src, sizeof(value));
            return value;
        }
        case 8: {
            uint64_t value;
            memcpy(&value, src, sizeof(value));
            return value;
        }
        default:
            return 0;
    }
}

double TRS_numeric_compute_sorted_gap_mode_from_bytes(
        const uint8_t* data,
        size_t n_elements,
        size_t elt_width,
        uint64_t* buffer,
        size_t buffer_capacity)
{
    ZL_ASSERT(n_elements == 0 || data != NULL);
    ZL_ASSERT(
            elt_width == 1 || elt_width == 2 || elt_width == 4
            || elt_width == 8);
    if ((n_elements != 0 && data == NULL)
        || (elt_width != 1 && elt_width != 2 && elt_width != 4
            && elt_width != 8)) {
        return 0.0;
    }
    ZL_ASSERT(n_elements == 0 || buffer != NULL);
    ZL_ASSERT(
            n_elements == 0
            || buffer_capacity >= TRS_NUMERIC_SORTED_GAP_BUFFER_ENTRIES);
    if (n_elements == 0)
        return 0.0;

    SortedGapAccumulator accumulator = {
        .retained_values = buffer,
        .pending_values  = buffer + TRS_SORTED_GAP_MAX_VALUES,
        .output_values   = buffer + TRS_SORTED_GAP_MAX_VALUES
                + TRS_SORTED_GAP_BATCH_VALUES,
        .retained_count = 0,
        .pending_count  = 0,
    };

    for (size_t i = 0; i < n_elements; i++) {
        sorted_gap_add(&accumulator, load_unsigned_value(data, i, elt_width));
    }
    if (accumulator.pending_count != 0)
        sorted_gap_flush(&accumulator);

    return compute_sorted_gap_mode(
            accumulator.retained_values, accumulator.retained_count);
}

static uint32_t
load_match4_sequence(const uint8_t* data, size_t index, size_t elt_width)
{
    if (ZL_isLittleEndian())
        return ZL_readLE32(data + index);

    uint8_t window[TRS_MATCH4_WINDOW_BYTES];
    for (size_t i = 0; i < sizeof(window); ++i) {
        window[i] = TRS_numeric_canonical_byte_from_big_endian(
                data, index + i, elt_width);
    }
    return ZL_readLE32(window);
}

uint64_t TRS_numeric_compute_lz_matches_with_table(
        const uint8_t* data,
        size_t n_bytes,
        size_t elt_width,
        uint32_t* table,
        size_t table_capacity)
{
    ZL_ASSERT(
            elt_width == 1 || elt_width == 2 || elt_width == 4
            || elt_width == 8);
    if (elt_width != 1 && elt_width != 2 && elt_width != 4 && elt_width != 8)
        return 0;
    ZL_ASSERT_EQ(n_bytes % elt_width, 0);
    if (n_bytes % elt_width != 0)
        return 0;
    if (n_bytes < TRS_MATCH4_WINDOW_BYTES)
        return 0;
    ZL_ASSERT_NN(table);
    ZL_ASSERT_GE(table_capacity, TRS_NUMERIC_MATCH4_TABLE_ENTRIES);
    if (table == NULL || table_capacity < TRS_NUMERIC_MATCH4_TABLE_ENTRIES)
        return 0;

    /*
     * Current models were trained without occupied bits, so an all-zero window
     * can match an empty slot. Change this only with model evaluation or
     * retraining.
     */
    memset(table, 0, sizeof(*table) * (size_t)TRS_NUMERIC_MATCH4_TABLE_ENTRIES);

    uint64_t matches = 0;
    for (size_t i = 0; i <= n_bytes - TRS_MATCH4_WINDOW_BYTES; i++) {
        uint32_t const sequence = load_match4_sequence(data, i, elt_width);
        uint32_t const hash     = (sequence * TRS_MATCH4_HASH_MULTIPLIER)
                >> (32 - TRS_MATCH4_TABLE_BITS);
        if (table[hash] == sequence)
            matches++;
        table[hash] = sequence;
    }
    return matches;
}

double TRS_numeric_compute_transition_gap_cv(
        double sum_gaps,
        double sum_gap_squares,
        size_t n_gaps)
{
    if (n_gaps == 0)
        return 0.0;

    double const mean_gap = sum_gaps / (double)n_gaps;
    if (mean_gap <= 0.0)
        return 0.0;

    double variance =
            (sum_gap_squares / (double)n_gaps) - (mean_gap * mean_gap);
    if (variance < 0.0)
        variance = 0.0;
    return sqrt(variance) / mean_gap;
}

uint64_t TRS_numeric_encode_double_fp(double value)
{
    if (value <= 0.0 || !isfinite(value))
        return 0;

    double const scaled  = value * (double)TRS_NUMERIC_Q32_SCALE;
    double const rounded = scaled + 0.5;
    if (rounded >= 0x1p64)
        return UINT64_MAX;
    return (uint64_t)rounded;
}

void TRS_numeric_kmv_track_value(
        TRS_NumericKmvEntry* heap,
        size_t* size,
        uint64_t value)
{
    TRS_numeric_kmv_track_inline(heap, size, value);
}
