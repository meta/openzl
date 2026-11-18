// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_SHARED_ESTIMATE_HASHED_H
#define OPENZL_SHARED_ESTIMATE_HASHED_H

#include "openzl/shared/estimate.h"

#include "openzl/common/assertion.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/utils.h"

ZL_BEGIN_C_DECLS

typedef struct {
    size_t sizeLog;
    uint8_t* bits;
} LinearCount;

ZL_INLINE void LinearCount_bump(LinearCount* lc, uint64_t hash)
{
    size_t const index = (size_t)(hash >> (sizeof(hash) * 8 - lc->sizeLog));
    lc->bits[index]    = 1;
}

#define HLL_BUCKET_LOG 10
#define HLL_NB_BUCKETS (1 << HLL_BUCKET_LOG)

typedef struct {
    uint32_t buckets[HLL_NB_BUCKETS];
} HyperLogLog;

ZL_INLINE void HyperLogLog_bump(HyperLogLog* hll, uint64_t hash)
{
    size_t const bucket = (size_t)(hash >> (sizeof(hash) * 8 - HLL_BUCKET_LOG));
    uint64_t const update = hash & ~(hash - 1);
    hll->buckets[bucket] |= (uint32_t)update;
}

enum {
    HASHED_LINEAR_COUNT_SIZE_LOG = 13,
    HASHED_LINEAR_COUNT_BUCKETS  = 1 << HASHED_LINEAR_COUNT_SIZE_LOG,
};

uint64_t ZL_finishHashedLinearCount(LinearCount const* count);
uint64_t ZL_finishHashedHyperLogLog(HyperLogLog const* hll);

/* Fixed providers inline at the caller. Hash in order because some providers
 * consume one item of iterator state per call. */
ZL_FORCE_INLINE void ZL_fillHashedSketch(
        LinearCount* linear,
        HyperLogLog* hll,
        ZL_CardinalityHashFn hash,
        void* state,
        size_t count)
{
    size_t i = 0;
    for (; count - i >= 4; i += 4) {
        uint64_t hashes[4];
        for (size_t j = 0; j < 4; ++j)
            hashes[j] = hash(state, i + j);
        for (size_t j = 0; j < 4; ++j) {
            if (linear)
                LinearCount_bump(linear, hashes[j]);
            else
                HyperLogLog_bump(hll, hashes[j]);
        }
    }
    for (; i < count; ++i) {
        uint64_t const value = hash(state, i);
        if (linear)
            LinearCount_bump(linear, value);
        else
            HyperLogLog_bump(hll, value);
    }
}

ZL_FORCE_INLINE uint64_t ZL_estimateCardinality_hashedInline(
        void* state,
        size_t nbHashes,
        size_t cardinalityUpperBound,
        ZL_CardinalityHashFn linearCountHash,
        ZL_CardinalityHashFn hyperLogLogHash)
{
    if (nbHashes == 0 || cardinalityUpperBound == 0)
        return 0;
    ZL_ASSERT(linearCountHash != NULL && hyperLogLogHash != NULL);

    uint64_t estimate;
    if (cardinalityUpperBound <= ZL_ESTIMATE_CARDINALITY_16BITS) {
        uint8_t bits[HASHED_LINEAR_COUNT_BUCKETS] = { 0 };
        LinearCount count = { HASHED_LINEAR_COUNT_SIZE_LOG, bits };
        ZL_fillHashedSketch(&count, NULL, linearCountHash, state, nbHashes);
        estimate = ZL_finishHashedLinearCount(&count);
    } else {
        HyperLogLog hll = { { 0 } };
        ZL_fillHashedSketch(NULL, &hll, hyperLogLogHash, state, nbHashes);
        estimate = ZL_finishHashedHyperLogLog(&hll);
    }
    return ZL_MIN(estimate, cardinalityUpperBound);
}

ZL_END_C_DECLS

#endif
