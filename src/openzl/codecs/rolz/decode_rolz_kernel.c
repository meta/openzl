// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/rolz/decode_rolz_kernel.h"
#include "openzl/codecs/rolz/decode_decoder.h"
#include "openzl/common/allocation.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/utils.h"

ZL_Report ZL_rolzDecompress(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    ZS_decoderCtx* const ctx = ZS_rolzDecoder->ctx_create();
    if (ctx == NULL) {
        ZL_ERR(GENERIC);
    }
    ZL_Report const dstSize = ZS_rolzDecoder->decompress(
            ctx, (uint8_t*)dst, dstCapacity, (uint8_t const*)src, srcSize);
    ZS_rolzDecoder->ctx_release(ctx);
    if (ZL_isError(dstSize))
        ZL_ERR(GENERIC);
    return dstSize;
}

ZL_Report ZS_fastLzDecompress(
        void* dst,
        size_t dstCapacity,
        void const* src,
        size_t srcSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    ZS_decoderCtx* const ctx = ZS_fastLzDecoder->ctx_create();
    if (ctx == NULL) {
        ZL_ERR(GENERIC);
    }
    ZL_Report const dstSize = ZS_fastLzDecoder->decompress(
            ctx, (uint8_t*)dst, dstCapacity, (uint8_t const*)src, srcSize);
    ZS_fastLzDecoder->ctx_release(ctx);
    if (ZL_isError(dstSize))
        ZL_ERR(GENERIC);
    return dstSize;
}

uint32_t usesHist[HIST_MAX + 1];
uint32_t ageHist[HIST_MAX + 1];
uint32_t indexHist[HIST_MAX + 1];

static uint32_t total(uint32_t* hist)
{
    uint32_t sum = 0;
    for (uint32_t i = 0; i <= HIST_MAX; ++i) {
        sum += hist[i];
    }
    return ZL_MAX(sum, 1);
}

static void printRunningFraction(uint32_t* hist, uint32_t size)
{
    uint32_t const sum = total(hist);
    uint32_t curr      = 0;
    for (uint32_t i = 0; i <= size; ++i) {
        curr += hist[i];
        ZL_LOG(V, "%u: %u\t(%u / %u)", i, (curr * 100) / sum, hist[i], sum);
    }
}

static void stars(uint32_t* hist, size_t n)
{
    size_t const sum       = total(hist);
    size_t const kMaxStars = 512;
    for (size_t i = 0; i < n; ++i) {
        ZL_RLOG(V, "%2u: ", (uint32_t)i);
        size_t const stars = (hist[i] * kMaxStars) / sum;
        for (size_t s = 0; s < stars; ++s) {
            ZL_RLOG(V, "*");
        }
        ZL_RLOG(V, "\n");
    }
}

void printUsesHist(void)
{
    ZL_LOG(V, "LZ usage");
    stars(usesHist, ZL_MIN(HIST_MAX, 24));
    // ZL_LOG(V, "USES B4 EVICTION\nUSES: COUNT");
    // for (uint32_t i = 0; i <= ZL_MIN(HIST_MAX, 64); ++i) {
    //   ZL_LOG(V, "%u: %u", i, usesHist[i]);
    // }
}

void printIndexHist(void)
{
    ZL_LOG(V, "RZ usage");
    stars(indexHist, ZL_MIN(HIST_MAX, 24));
    // ZL_LOG(V, "INDICES\nINDEX: COUNT");
    // for (uint32_t i = 0; i <= ZL_MIN(HIST_MAX, 64); ++i) {
    //   ZL_LOG(V, "%u: %u", i, indexHist[i]);
    // }
}

void printAgeHist(void)
{
    ZL_LOG(V, "Log age of evicted (higher is better)\nAGE: COUNT");
    printRunningFraction(ageHist, ZL_MIN(HIST_MAX, 21));
    // for (uint32_t i = 0; i <= HIST_MAX; ++i) {
    //   ZL_LOG(V, "%u: %u", i, ageHist[i]);
    // }
}

int ZS_rolzDTable_init(
        ZS_rolzDTable* dtable,
        ZS_window const* window,
        uint32_t numEntries,
        uint32_t ctxBits)
{
    uint32_t const numCtx = 1u << ctxBits;
    if (!ZL_isPow2(numEntries))
        return 1;
    dtable->ctxBits   = ctxBits;
    dtable->entryMask = numEntries - 1;
    dtable->entryLog  = (uint32_t)ZL_highbit32(numEntries);
    ZL_ASSERT_EQ((1u << dtable->entryLog), numEntries);
    dtable->anchor = window->dictLimit + ((dtable->ctxBits + 7) >> 3);
    dtable->table  = (ZS_rolzDEntry*)ZL_calloc(
            numCtx * numEntries * sizeof(*dtable->table));
    dtable->head = (uint32_t*)ZL_calloc(numCtx * sizeof(*dtable->head));

    if (dtable->table == NULL || dtable->head == NULL) {
        ZS_rolzDTable_destroy(dtable);
        return 1;
    }
    return 0;
}

void ZS_rolzDTable_destroy(ZS_rolzDTable* dtable)
{
    ZL_free(dtable->head);
    ZL_free(dtable->table);
}

void ZS_rolzDTable_reset(ZS_rolzDTable* dtable, ZS_window const* window)
{
    dtable->anchor = window->dictLimit + ((dtable->ctxBits + 7) >> 3);
    // Not strictly necessary
    // memset(dtable->head, 0, 256 * sizeof(*dtable->head));
}

ZL_Report ZS_RolzDTable2_init(
        ZS_RolzDTable2* table,
        uint32_t contextDepth,
        uint32_t contextLog,
        uint32_t rowLog,
        uint32_t minLength,
        bool predictMatchLength)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    table->contextDepth       = contextDepth;
    table->contextLog         = contextLog;
    table->rowLog             = rowLog;
    table->minLength          = minLength;
    table->rowMask            = (1u << rowLog) - 1;
    table->predictMatchLength = predictMatchLength;
    size_t const tableSize    = sizeof(*table->table) << (contextLog + rowLog);
    table->table              = (ZS_RolzDEntry2*)ZL_calloc(tableSize);
    if (table->table == NULL)
        ZL_ERR(GENERIC);
    return ZL_returnSuccess();
}

void ZS_RolzDTable2_destroy(ZS_RolzDTable2* table)
{
    ZL_free(table->table);
    table->table = NULL;
}
