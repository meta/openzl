// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cardinality.h"
#include "numeric_stats.h"

#include "openzl/shared/bits.h"
#include "openzl/shared/estimate_hashed.h"
#include "openzl/shared/mem.h"

/*
 * These hashes intentionally duplicate the shared estimator hashes used when
 * the Transformer models were trained. They are frozen model feature
 * definitions, not aliases: later shared-estimator changes must not propagate
 * here without regenerating features, retraining, and evaluation.
 */
#define XXH_INLINE_ALL
#define XXH_STATIC_LINKING_ONLY
#include "openzl/shared/xxhash.h"

#include "openzl/common/assertion.h"

typedef struct {
    const void* data;
} ArrayHashState;

typedef struct {
    const uint8_t* data;
    size_t elt_width;
} NumericByteHashState;

typedef struct {
    const uint64_t* words;
    size_t word_count;
    size_t word;
    uint64_t remaining;
} BitmapHashState;

static uint64_t linear_hash_u64(uint64_t value)
{
    uint64_t const hash = value * 0x9E3779B185EBCA87ULL;
    return hash ^ (hash << 47);
}

static uint64_t xxh3_hash_u64(uint64_t value)
{
    uint8_t little_endian[sizeof(value)];
    ZL_writeLE64(little_endian, value);
    return XXH3_64bits(little_endian, sizeof(little_endian));
}

static uint64_t cardinality_u64_linear_hash(void* state, size_t index)
{
    const ArrayHashState* array = (const ArrayHashState*)state;
    return linear_hash_u64(((uint64_t const*)array->data)[index]);
}

static uint64_t cardinality_u64_hll_hash(void* state, size_t index)
{
    const ArrayHashState* array = (const ArrayHashState*)state;
    uint64_t const value        = ((uint64_t const*)array->data)[index];
    return xxh3_hash_u64(value);
}

static uint64_t cardinality_u32_linear_hash(void* state, size_t index)
{
    const ArrayHashState* array = (const ArrayHashState*)state;
    return linear_hash_u64((uint64_t)((uint32_t const*)array->data)[index]);
}

static uint64_t cardinality_u32_hll_hash(void* state, size_t index)
{
    const ArrayHashState* array = (const ArrayHashState*)state;
    uint64_t const value = (uint64_t)((uint32_t const*)array->data)[index];
    return xxh3_hash_u64(value);
}

/* Pair and window features use XXH3 for both estimator paths, matching the
 * trained feature contract. */
static uint64_t pair_u64_hash(void* state, size_t index)
{
    const ArrayHashState* array = (const ArrayHashState*)state;
    const uint64_t* const data  = (const uint64_t*)array->data;
    uint8_t pair[2 * sizeof(uint64_t)];
    ZL_writeLE64(pair, data[index]);
    ZL_writeLE64(pair + sizeof(uint64_t), data[index + 1]);
    return XXH3_64bits(pair, sizeof(pair));
}

static uint64_t pair_u32_hash(void* state, size_t index)
{
    const ArrayHashState* array = (const ArrayHashState*)state;
    uint32_t const* data        = (uint32_t const*)array->data;
    uint8_t pair[2 * sizeof(uint64_t)];
    ZL_writeLE64(pair, (uint64_t)data[index]);
    ZL_writeLE64(pair + sizeof(uint64_t), (uint64_t)data[index + 1]);
    return XXH3_64bits(pair, sizeof(pair));
}

static uint64_t d8_hash(void* state, size_t index)
{
    const NumericByteHashState* array = (const NumericByteHashState*)state;
    return XXH3_64bits(array->data + index, 8);
}

static uint64_t d8_hash_big_endian(void* state, size_t index)
{
    const NumericByteHashState* array = (const NumericByteHashState*)state;
    uint8_t window[8];
    for (size_t i = 0; i < sizeof(window); ++i) {
        window[i] = TRS_numeric_canonical_byte_from_big_endian(
                array->data, index + i, array->elt_width);
    }
    return XXH3_64bits(window, sizeof(window));
}

static size_t bitmap_cardinality(const uint64_t* words, size_t word_count)
{
    size_t cardinality = 0;
    for (size_t word = 0; word < word_count; ++word) {
        cardinality += (size_t)ZL_popcount64(words[word]);
    }
    return cardinality;
}

static uint64_t bitmap_next_value(BitmapHashState* state)
{
    while (state->remaining == 0) {
        state->word++;
        if (state->word >= state->word_count) {
            ZL_ASSERT_FAIL(
                    "bitmap iterator exhausted before expected cardinality");
            return UINT64_MAX;
        }
        state->remaining = state->words[state->word];
    }

    unsigned const bit = (unsigned)ZL_ctz64(state->remaining);
    state->remaining &= state->remaining - 1;
    return (uint64_t)(state->word * 64 + bit);
}

static uint64_t bitmap_linear_hash(void* state, size_t index)
{
    (void)index;
    return linear_hash_u64(bitmap_next_value((BitmapHashState*)state));
}

static uint64_t bitmap_hll_hash(void* state, size_t index)
{
    (void)index;
    uint64_t const value = bitmap_next_value((BitmapHashState*)state);
    return xxh3_hash_u64(value);
}

static uint64_t bitmap_pair_u8_hash(void* state, size_t index)
{
    (void)index;
    uint32_t const key = (uint32_t)bitmap_next_value((BitmapHashState*)state);
    uint8_t pair[2 * sizeof(uint64_t)];
    ZL_writeLE64(pair, (uint64_t)(key >> 8));
    ZL_writeLE64(pair + sizeof(uint64_t), (uint64_t)(key & 0xFF));
    return XXH3_64bits(pair, sizeof(pair));
}

static uint64_t estimate_bitmap_cardinality(
        const uint64_t* words,
        size_t word_count,
        size_t cardinality_upper_bound,
        ZL_CardinalityHashFn linear_hash,
        ZL_CardinalityHashFn hll_hash)
{
    size_t const distinct = bitmap_cardinality(words, word_count);
    BitmapHashState state = {
        .words      = words,
        .word_count = word_count,
        .word       = 0,
        .remaining  = words[0],
    };
    /*
     * The estimator invokes exactly `distinct` callbacks. Each callback
     * consumes one set bit, so the iterator must end on the final set bit.
     */
    return ZL_estimateCardinality_hashedInline(
            &state, distinct, cardinality_upper_bound, linear_hash, hll_hash);
}

uint64_t TRS_estimate_cardinality_u64(const uint64_t* data, size_t n_elements)
{
    ArrayHashState state = { data };
    return ZL_estimateCardinality_hashedInline(
            &state,
            n_elements,
            n_elements,
            cardinality_u64_linear_hash,
            cardinality_u64_hll_hash);
}

uint64_t TRS_estimate_cardinality_u32(const uint32_t* data, size_t n_elements)
{
    ArrayHashState state = { data };
    return ZL_estimateCardinality_hashedInline(
            &state,
            n_elements,
            n_elements,
            cardinality_u32_linear_hash,
            cardinality_u32_hll_hash);
}

uint64_t TRS_estimate_pair_cardinality_u64(
        const uint64_t* data,
        size_t n_elements)
{
    if (n_elements < 2)
        return 0;
    ArrayHashState state = { data };
    return ZL_estimateCardinality_hashedInline(
            &state,
            n_elements - 1,
            n_elements - 1,
            pair_u64_hash,
            pair_u64_hash);
}

uint64_t TRS_estimate_pair_cardinality_u32(
        const uint32_t* data,
        size_t n_elements)
{
    if (n_elements < 2)
        return 0;
    ArrayHashState state = { data };
    return ZL_estimateCardinality_hashedInline(
            &state,
            n_elements - 1,
            n_elements - 1,
            pair_u32_hash,
            pair_u32_hash);
}

uint64_t TRS_estimate_cardinality_u8_bitmap(
        const uint64_t seen_values[TRS_CARDINALITY_U8_BITMAP_WORDS],
        size_t n_elements)
{
    return estimate_bitmap_cardinality(
            seen_values,
            TRS_CARDINALITY_U8_BITMAP_WORDS,
            n_elements,
            bitmap_linear_hash,
            bitmap_hll_hash);
}

uint64_t TRS_estimate_cardinality_u16_bitmap(
        const uint64_t seen_values[TRS_CARDINALITY_U16_BITMAP_WORDS],
        size_t n_elements)
{
    return estimate_bitmap_cardinality(
            seen_values,
            TRS_CARDINALITY_U16_BITMAP_WORDS,
            n_elements,
            bitmap_linear_hash,
            bitmap_hll_hash);
}

uint64_t TRS_estimate_pair_cardinality_u8_bitmap(
        const uint64_t seen_pairs[TRS_CARDINALITY_U8_PAIR_BITMAP_WORDS],
        size_t n_elements)
{
    if (n_elements < 2)
        return 0;
    return estimate_bitmap_cardinality(
            seen_pairs,
            TRS_CARDINALITY_U8_PAIR_BITMAP_WORDS,
            n_elements - 1,
            bitmap_pair_u8_hash,
            bitmap_pair_u8_hash);
}

uint64_t TRS_estimate_d8_cardinality(
        const uint8_t* data,
        size_t n_bytes,
        size_t elt_width)
{
    ZL_ASSERT(
            elt_width == 1 || elt_width == 2 || elt_width == 4
            || elt_width == 8);
    if (elt_width != 1 && elt_width != 2 && elt_width != 4 && elt_width != 8)
        return 0;
    ZL_ASSERT_EQ(n_bytes % elt_width, 0);
    if (n_bytes % elt_width != 0)
        return 0;
    if (n_bytes < 8)
        return 0;
    NumericByteHashState state = { data, elt_width };
    ZL_CardinalityHashFn const hash =
            ZL_isLittleEndian() ? d8_hash : d8_hash_big_endian;
    return ZL_estimateCardinality_hashedInline(
            &state, n_bytes - 7, n_bytes - 7, hash, hash);
}
