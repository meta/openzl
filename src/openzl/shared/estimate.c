// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/shared/estimate.h"
#include "openzl/shared/estimate_hashed.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "openzl/common/assertion.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/data_stats.h"
#include "openzl/shared/hash.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/utils.h"
#include "openzl/shared/xxhash.h"

static double LinearCount_estimateValue(size_t nbZeros, size_t nbBuckets)
{
    ZL_ASSERT_NE(nbZeros, 0);
    return (double)nbBuckets * log((double)nbBuckets / (double)nbZeros);
}

static ZL_CardinalityEstimate LinearCount_estimateImpl(
        size_t nbZeros,
        size_t nbBuckets)
{
    ZL_CardinalityEstimate estimate;
    // We can lower bound by the number of non-zero buckets.
    estimate.lowerBound = nbBuckets - nbZeros;
    // No upper bound possible.
    estimate.upperBound = (size_t)(-1);
    if (nbZeros == 0) {
        // We've saturated the map, we have no idea what the cardinality is.
        estimate.estimateLowerBound = (uint64_t)-1;
        estimate.estimate           = (uint64_t)-1;
        estimate.estimateUpperBound = (uint64_t)-1;
    } else {
        double const cardinality =
                LinearCount_estimateValue(nbZeros, nbBuckets);
        double const t = (double)cardinality / (double)nbBuckets;
        // Double our error because calculating error from the estimated
        // cardinality isn't quite right. Also, never report less than 10%
        // error, because our hash function is weaker, which adds some error.
        // This is not rigorous.
        double const errorRatio = 2
                * ZL_MAX(0.1,
                         sqrt((double)nbBuckets) * pow(exp(t) - t - 1, 0.5)
                                 / cardinality);
        double const error = errorRatio * cardinality;
        estimate.estimateLowerBound =
                ZL_MIN((size_t)(cardinality - error), (size_t)cardinality - 1);
        estimate.estimate = (size_t)cardinality;
        estimate.estimateUpperBound =
                ZL_MAX((size_t)(cardinality + error), (size_t)cardinality + 1);
    }
    return estimate;
}

static void LinearCount_init(LinearCount* lc, uint8_t* bits, size_t sizeLog)
{
    lc->sizeLog = sizeLog;
    lc->bits    = bits;
    memset(lc->bits, 0, (size_t)1 << sizeLog);
}

static uint64_t LinearCount_hash(uint64_t value)
{
    // A pure multiplication overestimates when the values are dense.
    // E.g. cardinality=32K of 2-byte values. This is because consecutive
    // values are less likely to collide than random values. That is a
    // bummer, because it performs well when the values are sparse, and
    // performs ~40% faster than hash + xor + shift.
    // A multiplication only will estimate ~44K in that scenario, where this
    // method estimates ~32K.
    // For now, I've preferred a more accurate estimate at the cost of slower
    // speed, but we could decide to accept overestimates in the densely packed
    // values scenario.
    uint64_t const hash = value * 0x9E3779B185EBCA87ULL;
    return hash ^ (hash << 47);
}

static size_t LinearCount_nbZeros(LinearCount const* lc)
{
    size_t nbZeros = 0;
    int const size = 1 << lc->sizeLog;
    for (int i = 0; i < size; ++i) {
        nbZeros += lc->bits[i] == 0;
    }
    return nbZeros;
}

static ZL_CardinalityEstimate LinearCount_estimate(LinearCount const* lc)
{
    size_t const nbZeros   = LinearCount_nbZeros(lc);
    size_t const nbBuckets = (size_t)1 << lc->sizeLog;
    return LinearCount_estimateImpl(nbZeros, nbBuckets);
}

// < 13.1% estimated error for cardinalities <= 128
typedef struct {
    uint64_t bits;
} TinyLinearCount;

static void TinyLinearCount_init(TinyLinearCount* tlc)
{
    tlc->bits = 0;
}
static void TinyLinearCount_bump(TinyLinearCount* vlc, uint64_t hash)
{
    uint64_t const bit    = hash >> (sizeof(hash) * 8 - 8);
    uint64_t const update = (uint64_t)1 << (bit & 0x3F);
    vlc->bits |= update;
}

static size_t TinyLinearCount_nbZeros(TinyLinearCount const* vlc)
{
    return 64 - (size_t)ZL_popcount64(vlc->bits);
}

static ZL_CardinalityEstimate TinyLinearCount_estimate(
        TinyLinearCount const* count)
{
    size_t const nbBuckets = sizeof(*count) * 8;
    size_t const nbZeros   = TinyLinearCount_nbZeros(count);
    return LinearCount_estimateImpl(nbZeros, nbBuckets);
}

// The HyperLogLog is only used in edge cases where we're interested
// in very large cardinalities. We choose to optimize for accuracy over
// speed in this case.

#define HLL_ALPHA (0.7213 / (1.0 + (1.079 / (double)HLL_NB_BUCKETS)))

static void HyperLogLog_init(HyperLogLog* hll)
{
    memset(hll->buckets, 0, sizeof(hll->buckets));
}

static uint64_t HyperLogLog_hash(uint64_t value)
{
    /* Preserve historical little-endian hashes while making estimates
     * deterministic across host byte orders. */
    uint8_t littleEndian[sizeof(value)];
    ZL_writeLE64(littleEndian, value);
    return XXH3_64bits(littleEndian, sizeof(littleEndian));
}

#if 0
static void HyperLogLog_merge(HyperLogLog* hll1, HyperLogLog const* hll2)
{
    for (int b = 0; b < HLL_NB_BUCKETS; ++b)
        hll1->buckets[b] |= hll2->buckets[b];
}
#endif

static size_t HyperLogLog_nbZeros(HyperLogLog const* hll)
{
    size_t nbZeros = 0;
    for (int b = 0; b < HLL_NB_BUCKETS; ++b) {
        nbZeros += hll->buckets[b] == 0;
    }
    return nbZeros;
}

static double HyperLogLog_harmonicMean(HyperLogLog const* hll)
{
    double mean = 0;
    for (int b = 0; b < HLL_NB_BUCKETS; ++b) {
        ZL_ASSERT_NE(hll->buckets[b], 0);
        size_t const bits = (size_t)(32 - ZL_clz32(hll->buckets[b]));
        mean += pow(2.0, -(double)bits);
    }
    return 1.0 / mean;
}

static ZL_CardinalityEstimate
HyperLogLog_estimateImpl(size_t nbBuckets, double harmonicMean, double alpha)
{
    double const nbBuckets2  = (double)nbBuckets * (double)nbBuckets;
    double const cardinality = alpha * nbBuckets2 * harmonicMean;
    // Double our error because calculating error from the estimated cardinality
    // isn't quite right.
    double const errorRatio = 2 * 1.04 / sqrt((double)HLL_NB_BUCKETS);
    double const error      = cardinality * errorRatio;
    ZL_CardinalityEstimate estimate;
    // We know that every bucket has at least 1 hit (otherwise we would've used
    // the LC estimate).
    estimate.lowerBound         = nbBuckets;
    estimate.estimateLowerBound = (size_t)(cardinality - error);
    estimate.estimate           = (size_t)cardinality;
    estimate.estimateUpperBound = (size_t)(cardinality + error);
    estimate.upperBound         = (size_t)-1;
    return estimate;
}

static ZL_CardinalityEstimate HyperLogLog_estimate(HyperLogLog const* hll)
{
    size_t const nbZeros = HyperLogLog_nbZeros(hll);
    if (nbZeros == 0) {
        // Main estimate
        return HyperLogLog_estimateImpl(
                HLL_NB_BUCKETS, HyperLogLog_harmonicMean(hll), HLL_ALPHA);
    } else {
        // Small cardinality estimate
        return LinearCount_estimateImpl(nbZeros, HLL_NB_BUCKETS);
    }
}

ZL_FORCE_INLINE uint64_t
getElementFixed(void const* src, size_t elt, size_t kEltSize)
{
    if (kEltSize == 1) {
        return *((uint8_t const*)src + elt);
    } else if (kEltSize == 2) {
        return *((uint16_t const*)src + elt);
    } else if (kEltSize == 4) {
        return *((uint32_t const*)src + elt);
    } else {
        ZL_ASSERT_EQ(kEltSize, 8);
        return *((uint64_t const*)src + elt);
    }
}

ZL_FORCE_INLINE ZL_CardinalityEstimate ZS_estimateTiny_internal(
        void const* src,
        size_t nbElts,
        uint64_t cardinalityEarlyExit,
        size_t kEltSize)
{
    (void)cardinalityEarlyExit;
    TinyLinearCount count;
    TinyLinearCount_init(&count);
    for (size_t i = 0; i < nbElts; ++i) {
        TinyLinearCount_bump(
                &count, LinearCount_hash(getElementFixed(src, i, kEltSize)));
    }
    return TinyLinearCount_estimate(&count);
}

ZL_FORCE_INLINE ZL_CardinalityEstimate ZS_estimateLinear_internal(
        void const* src,
        size_t nbElts,
        uint64_t cardinalityEarlyExit,
        uint8_t bits[],
        size_t sizeLog,
        size_t kEltSize)
{
    (void)cardinalityEarlyExit;
    LinearCount count;
    LinearCount_init(&count, bits, sizeLog);
    size_t const prefix = nbElts & 3;
    size_t i;
    for (i = 0; i < prefix; ++i)
        LinearCount_bump(
                &count, LinearCount_hash(getElementFixed(src, i, kEltSize)));
    for (; i < nbElts; i += 4) {
        for (size_t u = 0; u < 4; ++u)
            LinearCount_bump(
                    &count,
                    LinearCount_hash(getElementFixed(src, i + u, kEltSize)));
    }
    ZL_ASSERT_EQ(i, nbElts);
    return LinearCount_estimate(&count);
}

ZL_FORCE_INLINE ZL_CardinalityEstimate ZS_estimateHLL_internal(
        void const* src,
        size_t nbElts,
        uint64_t cardinalityEarlyExit,
        size_t kEltSize)
{
    (void)cardinalityEarlyExit;
    HyperLogLog hll;
    HyperLogLog_init(&hll);
    size_t const prefix = nbElts & 3;
    size_t i;
    for (i = 0; i < prefix; ++i)
        HyperLogLog_bump(
                &hll, HyperLogLog_hash(getElementFixed(src, i, kEltSize)));
    for (; i < nbElts; i += 4) {
        for (size_t u = 0; u < 4; ++u)
            HyperLogLog_bump(
                    &hll,
                    HyperLogLog_hash(getElementFixed(src, i + u, kEltSize)));
    }
    ZL_ASSERT_EQ(i, nbElts);
    return HyperLogLog_estimate(&hll);
}

static size_t ZS_estimateLinearSizeLog(uint64_t cardinalityEarlyExit)
{
    size_t const highBit = (size_t)ZL_highbit32((uint32_t)cardinalityEarlyExit);
    size_t const nbBits =
            ZL_isPow2((uint32_t)cardinalityEarlyExit) ? highBit : highBit + 1;
    return ZL_MAX(5, ZL_MIN(nbBits, 13));
}

ZL_FORCE_INLINE ZL_CardinalityEstimate ZS_estimate_internal(
        void const* src,
        size_t nbElts,
        uint64_t cardinalityEarlyExit,
        size_t kEltSize)
{
    if (0 && cardinalityEarlyExit <= ZL_ESTIMATE_CARDINALITY_7BITS) {
        /**
         * TODO: This code is disabled because clang does an amazing job
         * auto-vectorizing the loop, and reaches speeds of 3.8B symbols/s.
         * However, gcc does a bad job with codegen and only hits 1.4B
         * symbols/s. The LinearCount estimator reaches 2.8B symbols/s on both
         * clang & gcc, so use that for now.
         *
         * The code is disabled through a dead branch so we don't get unused
         * warnings.
         */
        return ZS_estimateTiny_internal(
                src, nbElts, cardinalityEarlyExit, kEltSize);
    }
    if (cardinalityEarlyExit <= ZL_ESTIMATE_CARDINALITY_16BITS) {
        // TODO: We can get 10-20% faster speed by templating on sizeLog.
        // However, it is already very fast, and I'm not sure it is worth
        // the code size and code complexity.
        // The speed loss is likey from worse codegen than an actual fundamental
        // limitation, so it should be able to be gained back another way.
        uint8_t bits[1 << 13];
        size_t const sizeLog = ZS_estimateLinearSizeLog(cardinalityEarlyExit);
        return ZS_estimateLinear_internal(
                src, nbElts, cardinalityEarlyExit, bits, sizeLog, kEltSize);
    }
    return ZS_estimateHLL_internal(src, nbElts, cardinalityEarlyExit, kEltSize);
}

static ZL_CardinalityEstimate
ZS_estimate1(void const* src, size_t nbElts, uint64_t cardinalityEarlyExit)
{
    (void)cardinalityEarlyExit;
    uint8_t present[256];
    memset(present, 0, sizeof(present));
    uint8_t const* const ptr = (uint8_t const*)src;
    {
        size_t const prefix = nbElts & 3;
        size_t i;
        for (i = 0; i < prefix; ++i)
            present[ptr[i]] = 1;
        for (; i < nbElts; i += 4) {
            present[ptr[i + 0]] = 1;
            present[ptr[i + 1]] = 1;
            present[ptr[i + 2]] = 1;
            present[ptr[i + 3]] = 1;
        }
    }

    size_t cardinality = 0;
    for (size_t i = 0; i < sizeof(present); ++i)
        cardinality += present[i];

    ZL_CardinalityEstimate estimate;
    estimate.lowerBound         = cardinality;
    estimate.estimateLowerBound = cardinality;
    estimate.estimate           = cardinality;
    estimate.estimateUpperBound = cardinality;
    estimate.upperBound         = cardinality;
    return estimate;
}

static ZL_CardinalityEstimate
ZS_estimate2(void const* src, size_t nbElts, uint64_t cardinalityEarlyExit)
{
    return ZS_estimate_internal(src, nbElts, cardinalityEarlyExit, 2);
}

static ZL_CardinalityEstimate
ZS_estimate4(void const* src, size_t nbElts, uint64_t cardinalityEarlyExit)
{
    return ZS_estimate_internal(src, nbElts, cardinalityEarlyExit, 4);
}

static ZL_CardinalityEstimate
ZS_estimate8(void const* src, size_t nbElts, uint64_t cardinalityEarlyExit)
{
    return ZS_estimate_internal(src, nbElts, cardinalityEarlyExit, 8);
}

static ZL_CardinalityEstimate ZS_CardinalityEstimate_fixup(
        ZL_CardinalityEstimate estimate,
        uint64_t upperBound)
{
    ZL_ASSERT(estimate.lowerBound <= estimate.upperBound);
    ZL_ASSERT(estimate.estimateLowerBound <= estimate.estimate);
    ZL_ASSERT(estimate.estimate <= estimate.estimateUpperBound);

    /* Fix from the top down. */
    if (estimate.upperBound > upperBound)
        estimate.upperBound = upperBound;

    if (estimate.estimateUpperBound > estimate.upperBound)
        estimate.estimateUpperBound = estimate.upperBound;

    if (estimate.estimate > estimate.estimateUpperBound)
        estimate.estimate = estimate.estimateUpperBound;

    if (estimate.estimateLowerBound > estimate.estimate)
        estimate.estimateLowerBound = estimate.estimate;

    /* Move up the cardinalityEstimateLowerBound, if necessary. */
    if (estimate.estimateLowerBound < estimate.lowerBound)
        estimate.estimateLowerBound = estimate.lowerBound;

    return estimate;
}

ZL_CardinalityEstimate ZL_estimateCardinality_fixed(
        void const* src,
        size_t nbElts,
        size_t eltSize,
        uint64_t cardinalityEarlyExit)
{
    if (cardinalityEarlyExit == 0)
        cardinalityEarlyExit = (size_t)-1;

    uint64_t upperBound =
            eltSize == 8 ? (uint64_t)-1 : (uint64_t)1 << (eltSize * 8);
    upperBound           = ZL_MIN(upperBound, nbElts);
    cardinalityEarlyExit = ZL_MIN(cardinalityEarlyExit, upperBound);
    ZL_CardinalityEstimate estimate;
    if (nbElts == 0) {
        memset(&estimate, 0, sizeof(estimate));
        return estimate;
    }
    switch (eltSize) {
        default:
        case 1:
            estimate = ZS_estimate1(src, nbElts, cardinalityEarlyExit);
            break;
        case 2:
            estimate = ZS_estimate2(src, nbElts, cardinalityEarlyExit);
            break;
        case 4:
            estimate = ZS_estimate4(src, nbElts, cardinalityEarlyExit);
            break;
        case 8:
            estimate = ZS_estimate8(src, nbElts, cardinalityEarlyExit);
            break;
    }
    return ZS_CardinalityEstimate_fixup(estimate, upperBound);
}

/*
 * The hashed estimator's trained feature contract rounds LinearCount to the
 * nearest integer. The general structured estimator intentionally truncates.
 */
static uint64_t LinearCount_estimateRounded(size_t nbZeros, size_t nbBuckets)
{
    if (nbZeros == 0) {
        /*
         * This scalar API cannot represent an unknown estimate. Preserve its
         * compatibility contract by returning the occupied-bucket lower bound
         * when LinearCount saturates.
         */
        return nbBuckets;
    }
    return (uint64_t)(LinearCount_estimateValue(nbZeros, nbBuckets) + 0.5);
}

uint64_t ZL_finishHashedLinearCount(LinearCount const* count)
{
    return LinearCount_estimateRounded(
            LinearCount_nbZeros(count), HASHED_LINEAR_COUNT_BUCKETS);
}

uint64_t ZL_finishHashedHyperLogLog(HyperLogLog const* hll)
{
    size_t const nbZeros = HyperLogLog_nbZeros(hll);
    /* The linear-count correction rounds; the full HLL estimate truncates. */
    return nbZeros > 0
            ? LinearCount_estimateRounded(nbZeros, HLL_NB_BUCKETS)
            : HyperLogLog_estimateImpl(
                      HLL_NB_BUCKETS, HyperLogLog_harmonicMean(hll), HLL_ALPHA)
                      .estimate;
}

uint64_t ZL_estimateCardinality_hashed(
        void* state,
        size_t nbHashes,
        size_t cardinalityUpperBound,
        ZL_CardinalityHashFn linearCountHash,
        ZL_CardinalityHashFn hyperLogLogHash)
{
    return ZL_estimateCardinality_hashedInline(
            state,
            nbHashes,
            cardinalityUpperBound,
            linearCountHash,
            hyperLogLogHash);
}

static ZL_CardinalityEstimate ZS_estimateLinear_variable_internal(
        void const* const* srcs,
        size_t const* eltSizes,
        size_t nbElts,
        uint64_t cardinalityEarlyExit,
        uint8_t bits[],
        size_t sizeLog)
{
    (void)cardinalityEarlyExit;
    LinearCount count;
    LinearCount_init(&count, bits, sizeLog);
    for (size_t i = 0; i < nbElts; ++i) {
        LinearCount_bump(&count, XXH3_64bits(srcs[i], eltSizes[i]));
    }
    return LinearCount_estimate(&count);
}

static ZL_CardinalityEstimate ZS_estimateHLL_variable_internal(
        void const* const* srcs,
        size_t const* eltSizes,
        size_t nbElts,
        uint64_t cardinalityEarlyExit)
{
    (void)cardinalityEarlyExit;
    HyperLogLog hll;
    HyperLogLog_init(&hll);
    for (size_t i = 0; i < nbElts; ++i) {
        HyperLogLog_bump(&hll, XXH3_64bits(srcs[i], eltSizes[i]));
    }
    return HyperLogLog_estimate(&hll);
}

ZL_CardinalityEstimate ZL_estimateCardinality_variable(
        void const* const* srcs,
        size_t const* eltSizes,
        size_t nbElts,
        uint64_t cardinalityEarlyExit)
{
    if (cardinalityEarlyExit == 0)
        cardinalityEarlyExit = (size_t)-1;

    uint64_t const upperBound = nbElts;
    cardinalityEarlyExit      = ZL_MIN(cardinalityEarlyExit, upperBound);
    ZL_CardinalityEstimate estimate;

    if (nbElts == 0) {
        memset(&estimate, 0, sizeof(estimate));
        return estimate;
    }
    if (cardinalityEarlyExit <= ZL_ESTIMATE_CARDINALITY_16BITS) {
        uint8_t bits[1 << 13];
        size_t const sizeLog = ZS_estimateLinearSizeLog(cardinalityEarlyExit);
        estimate             = ZS_estimateLinear_variable_internal(
                srcs, eltSizes, nbElts, cardinalityEarlyExit, bits, sizeLog);
    } else {
        estimate = ZS_estimateHLL_variable_internal(
                srcs, eltSizes, nbElts, cardinalityEarlyExit);
    }

    return ZS_CardinalityEstimate_fixup(estimate, upperBound);
}

ZL_ElementRange
ZL_computeUnsignedRange(void const* src, size_t nbElts, size_t eltSize)
{
    switch (eltSize) {
        case 1:
            return ZL_computeUnsignedRange8(src, nbElts);
        case 2:
            return ZL_computeUnsignedRange16(src, nbElts);
        case 4:
            return ZL_computeUnsignedRange32(src, nbElts);
        case 8:
            return ZL_computeUnsignedRange64(src, nbElts);
        default:
            ZL_ASSERT_FAIL("Unsupported");
            return (ZL_ElementRange){ 0, 0 }; // TODO: error handling
    }
}

ZL_ElementRange ZL_computeUnsignedRange64(uint64_t const* src, size_t srcSize)
{
    uint64_t min = (uint64_t)-1;
    uint64_t max = 0;
    for (size_t i = 0; i < srcSize; ++i) {
        if (src[i] < min)
            min = src[i];
        if (src[i] > max)
            max = src[i];
    }
    if (srcSize == 0)
        min = 0;
    ZL_ASSERT_LE(min, max);
    return (ZL_ElementRange){ min, max };
}

ZL_ElementRange ZL_computeUnsignedRange32(uint32_t const* src, size_t srcSize)
{
    uint32_t min = (uint32_t)-1;
    uint32_t max = 0;
    for (size_t i = 0; i < srcSize; ++i) {
        if (src[i] < min)
            min = src[i];
        if (src[i] > max)
            max = src[i];
    }
    if (srcSize == 0)
        min = 0;
    ZL_ASSERT_LE(min, max);
    return (ZL_ElementRange){ (uint64_t)min, (uint64_t)max };
}

ZL_ElementRange ZL_computeUnsignedRange16(uint16_t const* src, size_t srcSize)
{
    uint16_t min = (uint16_t)-1;
    uint16_t max = 0;
    for (size_t i = 0; i < srcSize; ++i) {
        if (src[i] < min)
            min = src[i];
        if (src[i] > max)
            max = src[i];
    }
    if (srcSize == 0)
        min = 0;
    ZL_ASSERT_LE(min, max);
    return (ZL_ElementRange){ (uint64_t)min, (uint64_t)max };
}

ZL_ElementRange ZL_computeUnsignedRange8(uint8_t const* src, size_t srcSize)
{
    uint8_t min = (uint8_t)-1;
    uint8_t max = 0;
    for (size_t i = 0; i < srcSize; ++i) {
        if (src[i] < min)
            min = src[i];
        if (src[i] > max)
            max = src[i];
    }
    if (srcSize == 0)
        min = 0;
    ZL_ASSERT_LE(min, max);
    return (ZL_ElementRange){ (uint64_t)min, (uint64_t)max };
}

static size_t
ZS_elementHash(void const* src, size_t kHashLog, size_t kHashLength)
{
    if (kHashLength == 1) {
        size_t const hash = *(uint8_t const*)src;
        ZL_ASSERT_LE(kHashLog, 8);
        return hash & ((1u << kHashLog) - 1);
    }
    return ZL_hashPtr(src, (uint32_t)kHashLog, (uint32_t)kHashLength);
}
static size_t
ZS_elementMatches(void const* ip, void const* match, size_t kHashLength)
{
    switch (kHashLength) {
        default:
        case 1:
            return *(uint8_t const*)ip == *(uint8_t const*)match;
        case 2:
            return *(uint16_t const*)ip == *(uint16_t const*)match;
        case 3:
            return ZL_read24(ip) == ZL_read24(match);
        case 4:
            return *(uint32_t const*)ip == *(uint32_t const*)match;
        case 8:
            return *(uint64_t const*)ip == *(uint64_t const*)match;
    }
}

static bool isPeak(uint32_t const* freqTable, size_t pos, size_t size)
{
    ZL_ASSERT_LT(pos, size);
    ZL_ASSERT_NE(pos, 0);
    ZL_ASSERT_GE(size, 9);
    uint32_t sum;
    if (pos < 8) {
        // For small offsets, we only look at the immediate neighbors.
        // This is because natural offsets are more frequent.
        sum = freqTable[pos - 1] + freqTable[pos + 1];
    } else {
        // For larger offsets look at 2 neighbors.
        sum = freqTable[pos - 2] + freqTable[pos - 1];
        if (pos + 2 >= size)
            sum *= 2;
        else
            sum += freqTable[pos + 1] + freqTable[pos + 2];
    }

    return freqTable[pos] > sum;
}

ZL_FORCE_INLINE ZL_DimensionalityEstimate
ZS_estimateDimensionalityImpl(void const* src, size_t nbElts, size_t kEltSize)
{
    // Tables sized to fit entirely in L1 cache.
    // Using a chain table for simplicity, and because everything fits in L1.
    // TODO: Optimize
    size_t const kHashLog     = kEltSize == 1 ? 8 : 10;
    size_t const kChainLog    = 10;
    uint32_t const kMaxOffset = 1u << kChainLog;
    uint32_t kChainMask       = (1u << kChainLog) - 1;
    uint32_t* const hashTable = calloc((size_t)1 << kHashLog, sizeof(uint32_t));
    uint32_t* const chainTable =
            calloc((size_t)1 << kChainLog, sizeof(uint32_t));
    uint32_t* const freqTable =
            calloc(((size_t)1 << kChainLog) + 1, sizeof(uint32_t));
    uint32_t matches = 0;

    ZL_REQUIRE_NN(hashTable);
    ZL_REQUIRE_NN(chainTable);
    ZL_REQUIRE_NN(freqTable);

    //> Find all matching elements and store the offsets in freqTable.
    //> Offsets are all multiples of kEltSize so that we can fit offsets
    //> of up to 1024 into our chain table.

    uint8_t const* const istart = src;
    uint8_t const* const iend   = istart + nbElts * kEltSize;
    uint32_t pos                = 0;
    for (uint8_t const* ip = istart; ip != iend; ip += kEltSize, ++pos) {
        size_t const hash       = ZS_elementHash(ip, kHashLog, kEltSize);
        uint32_t match          = hashTable[hash];
        uint32_t const minMatch = pos > kMaxOffset ? pos - kMaxOffset : 1;
        while (match >= minMatch) {
            uint32_t const nextMatch = chainTable[match & kChainMask];
            if (ZS_elementMatches(ip, istart + match * kEltSize, kEltSize)) {
                ++freqTable[pos - match];
                ++matches;
            }
            match = nextMatch;
        }

        chainTable[pos & kChainMask] = match;
        hashTable[hash]              = pos;
    }

    //> Compute the stride:
    //> Peaks are offsets that are more frequent than their neighbors.
    //> For each peak:
    //>   1. Compute the share of the peaks at multiples of the peak,
    //>      excluding multiples that are 2x more frequent than the offset
    //>      itself (because that is probably the real stride). We only
    //>      include peaks to exclude natural matches. E.g. stride=2
    //>      would include every even offset in its multiples.
    //>   2. If the share is > than the max share, set the stride to the
    //>      current offset.

    ZL_DimensionalityEstimate estimate;
    {
        uint32_t const minShare = matches >> 4;
        size_t stride           = 0;
        size_t strideMatches    = 0;
        for (size_t o = 2; o <= kMaxOffset; ++o) {
            if (!isPeak(freqTable, o, kMaxOffset + 1))
                continue;
            uint32_t share = freqTable[o];
            for (size_t m = 2; o * m <= kMaxOffset; ++m) {
                if (freqTable[o * m] >= 2 * freqTable[o]) {
                    // Heuristic: Don't add to the count if the next peak is
                    // much stronger. This avoids accidentally picking up a
                    // divisor of the dimensionality as the stride.
                    continue;
                }
                if (!isPeak(freqTable, o * m, kMaxOffset + 1))
                    continue;
                share += freqTable[o * m];
            }
            if (share > strideMatches) {
                stride        = o;
                strideMatches = share;
            }
        }

        //> If there are no peaks then report no dimensionality.
        //> If share of matches at the peak is > 1/16 (heuristic)
        //> then report there is likely dimensionality.
        //> Else, report that there many be dimensionality.

        if (stride == 0)
            estimate.dimensionality = ZL_DimensionalityStatus_none;
        else if (strideMatches > minShare)
            estimate.dimensionality = ZL_DimensionalityStatus_likely2D;
        else
            estimate.dimensionality = ZL_DimensionalityStatus_possibly2D;

        //> Report the estimated stride and the ratio of stride matches
        //> to total matches.

        estimate.stride        = stride;
        estimate.strideMatches = strideMatches;
        estimate.totalMatches  = matches;
    }

    free(freqTable);
    free(chainTable);
    free(hashTable);

    return estimate;
}

ZL_DimensionalityEstimate
ZL_estimateDimensionality(void const* src, size_t nbElts, size_t eltSize)
{
    switch (eltSize) {
        case 1:
            return ZL_estimateDimensionality1(src, nbElts);
        case 2:
            return ZL_estimateDimensionality2(src, nbElts);
        case 3:
            return ZS_estimateDimensionality3(src, nbElts);
        case 4:
            return ZL_estimateDimensionality4(src, nbElts);
        case 8:
            return ZL_estimateDimensionality8(src, nbElts);
        default:
            ZL_REQUIRE_FAIL("Unsupported");
    }
}

ZL_DimensionalityEstimate ZL_estimateDimensionality1(
        void const* src,
        size_t srcSize)
{
    return ZS_estimateDimensionalityImpl(src, srcSize, 1);
}

ZL_DimensionalityEstimate ZL_estimateDimensionality2(
        void const* src,
        size_t srcSize)
{
    return ZS_estimateDimensionalityImpl(src, srcSize, 2);
}

ZL_DimensionalityEstimate ZS_estimateDimensionality3(
        void const* src,
        size_t srcSize)
{
    return ZS_estimateDimensionalityImpl(src, srcSize, 3);
}

ZL_DimensionalityEstimate ZL_estimateDimensionality4(
        void const* src,
        size_t srcSize)
{
    return ZS_estimateDimensionalityImpl(src, srcSize, 4);
}

ZL_DimensionalityEstimate ZL_estimateDimensionality8(
        void const* src,
        size_t srcSize)
{
    return ZS_estimateDimensionalityImpl(src, srcSize, 8);
}

/**
 * @param selector If bit i is set, include histogram i in the sum.
 * @returns The entropy of the sum of all the selected histograms.
 */
static double combinedHistogramEntropy(
        const uint16_t srcHistograms[8][256],
        size_t srcSize,
        int selector)
{
    uint32_t histogram[256] = { 0 };
    for (size_t h = 0; h < 8; ++h) {
        if (selector & (1 << h)) {
            for (size_t i = 0; i < 256; ++i) {
                histogram[i] += srcHistograms[h][i];
            }
        }
    }
    return ZL_calculateEntropyU8(histogram, srcSize) * (double)srcSize;
}

size_t ZL_guessFloatWidth(void const* src, size_t srcSize)
{
    if (srcSize % 2 != 0) {
        return 1;
    }

    const bool canBeFloat32 = (srcSize % 4) == 0;
    const bool canBeFloat64 = (srcSize % 8) == 0;

    srcSize = ZL_MIN(srcSize, 32768);
    srcSize &= ~(size_t)0x7;

    uint16_t histograms[8][256];
    memset(histograms, 0, sizeof(histograms));

    const uint8_t* const src8 = src;
    for (size_t i = 0; i < srcSize; i += 8) {
        for (size_t h = 0; h < 8; ++h) {
            ++histograms[h][src8[i + h]];
        }
    }

    const double u8Entropy =
            combinedHistogramEntropy(histograms, srcSize, 0xFF);
    const double f16Entropy =
            combinedHistogramEntropy(histograms, srcSize / 2, 0x55)
            + combinedHistogramEntropy(histograms, srcSize / 2, 0xAA);

    if (u8Entropy * 0.99 <= f16Entropy) {
        return 1;
    }

    if (!canBeFloat32) {
        return 2;
    }

    const double f32Entropy =
            combinedHistogramEntropy(histograms, srcSize / 4, 0x11)
            + combinedHistogramEntropy(histograms, srcSize / 4, 0x22)
            + combinedHistogramEntropy(histograms, srcSize / 4, 0x44)
            + combinedHistogramEntropy(histograms, srcSize / 4, 0x88);

    if (f16Entropy * 0.99 <= f32Entropy) {
        return 2;
    }

    if (!canBeFloat64) {
        return 4;
    }

    double f64Entropy = 0;
    for (size_t h = 0; h < 8; ++h) {
        f64Entropy += combinedHistogramEntropy(histograms, srcSize / 8, 1 << h);
    }

    if (f32Entropy * 0.99 <= f64Entropy) {
        return 4;
    }

    return 8;
}
