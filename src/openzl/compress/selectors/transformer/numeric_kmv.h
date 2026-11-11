// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_COMPRESS_SELECTORS_TRANSFORMER_NUMERIC_KMV_H
#define OPENZL_COMPRESS_SELECTORS_TRANSFORMER_NUMERIC_KMV_H

#include "numeric_stats.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

#define XXH_INLINE_ALL
#define XXH_STATIC_LINKING_ONLY
#include "openzl/shared/xxhash.h"

ZL_BEGIN_C_DECLS

void TRS_numeric_kmv_insert(
        TRS_NumericKmvEntry* heap,
        size_t* size,
        uint64_t hash,
        uint64_t value);

ZL_FORCE_INLINE void TRS_numeric_kmv_track_inline(
        TRS_NumericKmvEntry* heap,
        size_t* size,
        uint64_t value)
{
    uint8_t little_endian[sizeof(value)];
    ZL_writeLE64(little_endian, value);
    uint64_t const hash = XXH3_64bits(little_endian, sizeof(little_endian));
    if (*size < TRS_NUMERIC_KMV_K || hash < heap[0].hash) {
        TRS_numeric_kmv_insert(heap, size, hash, value);
    }
}

ZL_END_C_DECLS

#endif
