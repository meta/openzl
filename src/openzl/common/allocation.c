// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/common/allocation.h"
#include "openzl/common/assertion.h"
#include "openzl/common/limits.h"
#include "openzl/common/map.h"
#include "openzl/shared/overflow.h"
#include "openzl/shared/utils.h"

#include <stdalign.h>
#include <stddef.h>
#include <stdlib.h> // malloc, free
#include <string.h> // memset

#if ZL_POISON_SUPPORTED
#    define ZL_REDZONE_SIZE 128
#else
#    define ZL_REDZONE_SIZE 0
#endif

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION

__attribute__((weak)) bool ZS2_malloc_should_fail(size_t size);

#endif

void* ZL_malloc(size_t s)
{
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    if (ZS2_malloc_should_fail && ZS2_malloc_should_fail(s)) {
        return NULL;
    }
    // Cap allocations at 100MB for fuzzer builds
    if (s > 100 * 1024 * 1024) {
        return NULL;
    }
#endif
    return malloc(s);
}

void* ZL_calloc(size_t s)
{
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    if (ZS2_malloc_should_fail && ZS2_malloc_should_fail(s)) {
        return NULL;
    }
    // Cap allocations at 100MB for fuzzer builds
    if (s > 100 * 1024 * 1024) {
        return NULL;
    }
#endif
    return calloc(1, s);
}

void* ZL_realloc(void* ptr, size_t size)
{
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    if (ZS2_malloc_should_fail && ZS2_malloc_should_fail(size)) {
        return NULL;
    }
#endif
    return realloc(ptr, size);
}

void ZL_free(void* p)
{
    free(p);
}

void ZL_zeroes(void* p, size_t s)
{
    memset(p, 0, s);
}

/*
 * Arena internals and API:
 */

void* ALLOC_Arena_malloc(Arena* arena, size_t size)
{
    ZL_ASSERT_NN(arena);
    ZL_ASSERT_NN(arena->malloc);
    return arena->malloc(arena, size);
}
void* ALLOC_Arena_calloc(Arena* arena, size_t size)
{
    ZL_ASSERT_NN(arena);
    ZL_ASSERT_NN(arena->calloc);
    return arena->calloc(arena, size);
}
void* ALLOC_Arena_realloc(Arena* arena, void* ptr, size_t newSize)
{
    ZL_ASSERT_NN(arena);
    ZL_ASSERT_NN(arena->realloc);
    return arena->realloc(arena, ptr, newSize);
}
void ALLOC_Arena_free(Arena* arena, void* ptr)
{
    ZL_ASSERT_NN(arena);
    ZL_ASSERT_NN(arena->free);
    arena->free(arena, ptr);
}
void ALLOC_Arena_freeAll(Arena* arena)
{
    ZL_DLOG(BLOCK, "ALLOC_Arena_freeAll (address:%p)", arena);
    if (arena == NULL)
        return;
    ZL_ASSERT_NN(arena->freeAll);
    arena->freeAll(arena);
}
void ALLOC_Arena_freeArena(Arena* arena)
{
    ZL_DLOG(OBJ, "ALLOC_Arena_freeArena (address:%p)", arena);
    if (arena == NULL)
        return;
    ZL_ASSERT_NN(arena->freeArena);
    arena->freeArena(arena);
}

size_t ALLOC_Arena_memAllocated(const Arena* arena)
{
    if (arena == NULL)
        return 0;
    ZL_ASSERT_NN(arena->memAllocated);
    return arena->memAllocated(arena);
}

size_t ALLOC_Arena_memUsed(const Arena* arena)
{
    if (arena == NULL)
        return 0;
    ZL_ASSERT_NN(arena->memUsed);
    return arena->memUsed(arena);
}

#define ALLOC_ARENA_SIZE_MAX (1024ULL * 1024 * 1024) // 1 GiB

static bool ALLOC_Arena_canAllocate(
        const Arena* arena,
        size_t releasedSize,
        size_t requestedSize)
{
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    size_t allocatedSize = ALLOC_Arena_memAllocated(arena);
    if (releasedSize > allocatedSize) {
        return false;
    }
    allocatedSize -= releasedSize;
    return allocatedSize <= ALLOC_ARENA_SIZE_MAX
            && requestedSize <= ALLOC_ARENA_SIZE_MAX - allocatedSize;
#else
    (void)arena;
    (void)releasedSize;
    (void)requestedSize;
    return true;
#endif
}

/*============================================================================
 * RawAllocator: For internal allocator use for allocs that MUST NOT be part
 * of the allocation failure injection mechanism.
 *===========================================================================*/

static void* raw_malloc(Arena* arena, size_t size)
{
    (void)arena;
    return malloc(size);
}
static void* raw_calloc(Arena* arena, size_t size)
{
    (void)arena;
    return calloc(1, size);
}
static void* raw_realloc(Arena* arena, void* ptr, size_t size)
{
    (void)arena;
    return realloc(ptr, size);
}
static void raw_free(Arena* arena, void* ptr)
{
    (void)arena;
    free(ptr);
}
static Arena kRawAllocator = {
    .malloc  = raw_malloc,
    .calloc  = raw_calloc,
    .realloc = raw_realloc,
    .free    = raw_free,
};

/*
 * ==============================================================
 * HeapArena:
 * ==============================================================
 */

typedef struct {
    size_t index;
    size_t size;
#if ZL_POISON_SUPPORTED
    uint8_t redzone[ZL_REDZONE_SIZE];
#endif
} HeapMeta;

DECLARE_VECTOR_POINTERS_TYPE(HeapMeta)

typedef struct {
    Arena base;
    VECTOR_POINTERS(HeapMeta) ptrs;
} HeapArena;

/**
 * Implements the shared code for malloc and calloc.
 * @param meta A pointer to a HeapMeta object, or NULL if allocation failed.
 *             It should be (sizeof(HeapMeta) + size) bytes.
 * @param size The user requested size of the allocation.
 * @returns The pointer or NULL on failure. On failure @p meta is freed.
 */
static void*
ALLOC_HeapArena_allocImpl(HeapArena* arena, HeapMeta* meta, size_t size)
{
    ZL_STATIC_ASSERT(
            sizeof(size_t) == 8,
            "OpenZL currently only supports 64-bit platforms");

    if (meta == NULL) {
        return NULL;
    }
    ZL_ASSERT_EQ((uintptr_t)meta % alignof(HeapMeta), 0);
    meta->index = VECTOR_SIZE(arena->ptrs);
    meta->size  = size;
    ZL_POISON_MEMORY(meta->redzone, sizeof(meta->redzone));

    if (!VECTOR_PUSHBACK(arena->ptrs, meta)) {
        ZL_LOG(ERROR, "Failed to push ptr into HeapArena");
        ZL_free(meta);
        return NULL;
    }
    ZL_STATIC_ASSERT(
            sizeof(HeapMeta) % 16 == 0,
            "sizeof(HeapMeta) must be a multiple of 16 to guarantee alignment");
    return (void*)(meta + 1);
}

static void* ALLOC_HeapArena_malloc(Arena* arena, size_t size)
{
    ZL_ASSERT_NN(arena);
    HeapArena* const heapArena = ZL_CONTAINER_OF(arena, HeapArena, base);
    size_t allocSize;
    if (ZL_overflowAddST(size, sizeof(HeapMeta), &allocSize)) {
        return NULL;
    }
    if (!ALLOC_Arena_canAllocate(arena, 0, size)) {
        return NULL;
    }
    void* const ptr = ZL_malloc(allocSize);
    return ALLOC_HeapArena_allocImpl(heapArena, ptr, size);
}

static void* ALLOC_HeapArena_calloc(Arena* arena, size_t size)
{
    ZL_ASSERT_NN(arena);
    HeapArena* heapArena = ZL_CONTAINER_OF(arena, HeapArena, base);
    size_t allocSize;
    if (ZL_overflowAddST(size, sizeof(HeapMeta), &allocSize)) {
        return NULL;
    }
    if (!ALLOC_Arena_canAllocate(arena, 0, size)) {
        return NULL;
    }
    void* const ptr = ZL_calloc(allocSize);
    return ALLOC_HeapArena_allocImpl(heapArena, ptr, size);
}

static void* ALLOC_HeapArena_realloc(Arena* arena, void* ptr, size_t newSize)
{
    ZL_ASSERT_NN(arena);
    if (ptr == NULL) {
        return ALLOC_HeapArena_malloc(arena, newSize);
    }
    HeapArena* heapArena    = ZL_CONTAINER_OF(arena, HeapArena, base);
    HeapMeta* const oldMeta = (HeapMeta*)ptr - 1;
    size_t allocSize;
    if (ZL_overflowAddST(newSize, sizeof(HeapMeta), &allocSize)) {
        return NULL;
    }
    if (!ALLOC_Arena_canAllocate(arena, oldMeta->size, newSize)) {
        return NULL;
    }
    HeapMeta* const newMeta = ZL_realloc(oldMeta, allocSize);
    if (newMeta == NULL) {
        return NULL;
    }
    newMeta->size = newSize;
    ZL_POISON_MEMORY(newMeta->redzone, sizeof(newMeta->redzone));
    VECTOR_AT(heapArena->ptrs, newMeta->index) = newMeta;
    return (void*)(newMeta + 1);
}

static void ALLOC_HeapArena_freeSegment(Arena* arena, void* ptr)
{
    if (ptr == NULL)
        return;
    ZL_ASSERT_NN(arena);
    HeapArena* const heapArena = ZL_CONTAINER_OF(arena, HeapArena, base);
    HeapMeta* const meta       = (HeapMeta*)ptr - 1;

    const size_t index   = meta->index;
    const size_t numPtrs = VECTOR_SIZE(heapArena->ptrs);
    ZL_ASSERT_LT(index, numPtrs);
    ZL_ASSERT_EQ(VECTOR_AT(heapArena->ptrs, index), meta);

    // Unconditionally move last element to this position
    // This is a no-op when index == numPtrs - 1, but avoid a branch
    HeapMeta* const last = VECTOR_AT(heapArena->ptrs, numPtrs - 1);
    ZL_ASSERT_EQ(last->index, numPtrs - 1);
    VECTOR_AT(heapArena->ptrs, index) = last;
    last->index                       = index;

    // Remove last element
    VECTOR_POPBACK(heapArena->ptrs);

    ZL_free(meta);
}

static void ALLOC_HeapArena_freeAllSegments(Arena* arena)
{
    ZL_DLOG(BLOCK, "ALLOC_HeapArena_freeAllSegments (address:%p)", arena);
    if (arena == NULL)
        return;
    HeapArena* const heapArena = ZL_CONTAINER_OF(arena, HeapArena, base);
    const size_t numPtrs       = VECTOR_SIZE(heapArena->ptrs);
    for (size_t i = 0; i < numPtrs; ++i) {
        ZL_free(VECTOR_AT(heapArena->ptrs, i));
    }
    VECTOR_CLEAR(heapArena->ptrs);
}

static void ALLOC_HeapArena_destroy(HeapArena* arena)
{
    ZL_ASSERT_NN(arena);
    ALLOC_HeapArena_freeAllSegments(&arena->base);
    ZL_ASSERT_EQ(VECTOR_SIZE(arena->ptrs), 0);
    VECTOR_DESTROY(arena->ptrs);
}

static void ALLOC_HeapArena_freeArena(Arena* arena)
{
    ZL_DLOG(OBJ, "ALLOC_HeapArena_freeArena (address:%p)", arena);
    if (arena == NULL) {
        return;
    }
    HeapArena* const heapArena = ZL_CONTAINER_OF(arena, HeapArena, base);
    ALLOC_HeapArena_destroy(heapArena);

    ZL_free(arena);
}

static size_t ALLOC_HeapArena_countMem(const HeapArena* heapArena)
{
    if (heapArena == NULL)
        return 0;
    size_t totalMem      = 0;
    const size_t numPtrs = VECTOR_SIZE(heapArena->ptrs);
    for (size_t i = 0; i < numPtrs; ++i) {
        totalMem += VECTOR_AT(heapArena->ptrs, i)->size;
    }
    return totalMem;
}

static size_t ALLOC_HeapArena_memAllocated(const Arena* arena)
{
    if (arena == NULL)
        return 0;
    const HeapArena* const heapArena =
            ZL_CONST_CONTAINER_OF(arena, HeapArena, base);
    return ALLOC_HeapArena_countMem(heapArena);
}

static size_t ALLOC_HeapArena_memUsed(const Arena* arena)
{
    return ALLOC_Arena_memAllocated(arena);
}

static const Arena kHeapArena = {
    .malloc       = ALLOC_HeapArena_malloc,
    .calloc       = ALLOC_HeapArena_calloc,
    .realloc      = ALLOC_HeapArena_realloc,
    .free         = ALLOC_HeapArena_freeSegment,
    .freeAll      = ALLOC_HeapArena_freeAllSegments,
    .freeArena    = ALLOC_HeapArena_freeArena,
    .memAllocated = ALLOC_HeapArena_memAllocated,
    .memUsed      = ALLOC_HeapArena_memUsed,
};
static void ALLOC_HeapArena_init(HeapArena* a)
{
    a->base = kHeapArena;
    VECTOR_INIT(a->ptrs, ZL_CONTAINER_SIZE_LIMIT);
}

Arena* ALLOC_HeapArena_create(void)
{
    HeapArena* const arena = ZL_malloc(sizeof(HeapArena));
    if (arena) {
        ALLOC_HeapArena_init(arena);
    }
    return &arena->base;
}

/*
 * ==============================================================
 * StackArena:
 * ==============================================================
 * This allocator tries to create one Primary Buffer per Arena
 * and assigns Slices from it to future requests from this Arena.
 * It is designed to reduce stress on malloc/free,
 * in scenario where Arenas are maintained and re-used over multiple sessions.
 *
 * When there is not enough room in the primary buffer for a new slice,
 * create additional buffers with malloc/free, as usual.
 * Primary Buffer is then resized dynamically as needed.
 */

typedef struct StackArena_s StackArena;

#if ZL_POISON_SUPPORTED

ZL_DECLARE_MAP_TYPE(AsanOnlySizeMap, uintptr_t, size_t);

static void
AsanOnlySizeMap_setPtrSize(AsanOnlySizeMap* map, void* ptr, size_t size)
{
    ZL_REQUIRE_NN(map);
    ZL_REQUIRE_NN(ptr);
    const AsanOnlySizeMap_Insert insert = AsanOnlySizeMap_insertVal(
            map, (AsanOnlySizeMap_Entry){ (uintptr_t)ptr, size });
    ZL_REQUIRE(insert.inserted, "ptr already exists in map");
    ZL_REQUIRE(!insert.badAlloc);
}

static void AsanOnlySizeMap_erasePtrSize(AsanOnlySizeMap* map, void* ptr)
{
    ZL_REQUIRE_NN(map);
    ZL_REQUIRE_NN(ptr);
    const bool erased = AsanOnlySizeMap_eraseVal(map, (uintptr_t)ptr);
    ZL_REQUIRE(erased, "ptr not found in map");
}

static size_t AsanOnlySizeMap_getPtrSize(AsanOnlySizeMap* map, void* ptr)
{
    ZL_REQUIRE_NN(map);
    ZL_REQUIRE_NN(ptr);
    const AsanOnlySizeMap_Entry* const entry =
            AsanOnlySizeMap_findVal(map, (uintptr_t)ptr);
    ZL_REQUIRE_NN(entry, "ptr not found in map");
    return entry->val;
}

static void AsanOnlySizeMap_unpoisonAllPtrs(AsanOnlySizeMap* map)
{
    AsanOnlySizeMap_Iter iter = AsanOnlySizeMap_iter(map);
    for (const AsanOnlySizeMap_Entry* entry;
         (entry = AsanOnlySizeMap_Iter_next(&iter));) {
        ZL_unpoisonMemory((const void*)entry->key, entry->val);
    }
}

#else

typedef int AsanOnlySizeMap;

static void
AsanOnlySizeMap_setPtrSize(AsanOnlySizeMap* map, void* ptr, size_t size)
{
    (void)map;
    (void)ptr;
    (void)size;
}

static void AsanOnlySizeMap_erasePtrSize(AsanOnlySizeMap* map, void* ptr)
{
    (void)map;
    (void)ptr;
}

static void AsanOnlySizeMap_clear(AsanOnlySizeMap* map)
{
    (void)map;
}

static void AsanOnlySizeMap_destroy(AsanOnlySizeMap* map)
{
    (void)map;
}

static AsanOnlySizeMap AsanOnlySizeMap_createInArena(
        Arena* arena,
        uint32_t limit)
{
    (void)arena;
    (void)limit;
    return 0;
}

static void AsanOnlySizeMap_unpoisonAllPtrs(AsanOnlySizeMap* map)
{
    (void)map;
}

// Not defined if ASAN is not supported
// static size_t AsanOnlySizeMap_getPtrSize(AsanOnlySizeMap* map, void* ptr);

#endif

struct StackArena_s {
    Arena base; // must be first member
    void* primaryBuffer;
    size_t pBuffCapacity;
    size_t pBuffUsed;
    size_t wouldHaveNeeded; // Amount allocated outside of @primaryBuffer
                            // during the current session, i.e. how much larger
                            // @primaryBuffer should have been. Reset when the
                            // next session sizes the buffer from it.
    size_t sessionUsage;    // Tracks amount of memory requested from this arena
                            // before a reset
    size_t wasted;          // pBuffCapacity * nbTimesUsedWastefully,
                            // to trigger a size-down event
    HeapArena heapBackup;
    AsanOnlySizeMap asanOnlySizeMap;
};

// Track allocations in the primary buffer to poison memory
static void
ALLOC_StackArena_trackMalloc(StackArena* pba, void* ptr, size_t requestSize)
{
    AsanOnlySizeMap_setPtrSize(&pba->asanOnlySizeMap, ptr, requestSize);
    ZL_unpoisonMemory(ptr, requestSize);
}

// Track frees in the primary buffer to poison memory
static void ALLOC_StackArena_trackFree(StackArena* pba, void* ptr)
{
    ZL_POISON_MEMORY(
            ptr, AsanOnlySizeMap_getPtrSize(&pba->asanOnlySizeMap, ptr));
    AsanOnlySizeMap_erasePtrSize(&pba->asanOnlySizeMap, ptr);
}

static void* ALLOC_StackArena_malloc(Arena* arena, size_t requestSize)
{
    ZL_ASSERT_NN(arena);
    StackArena* const pba = ZL_CONTAINER_OF(arena, StackArena, base);
    ZL_ASSERT_GE(pba->pBuffCapacity, pba->pBuffUsed);
    size_t pBuffAvailable         = pba->pBuffCapacity - pba->pBuffUsed;
    static const size_t alignment = 16; // could become a variable in the future
    size_t size, neededSize;
    /* Requests close to SIZE_MAX would wrap around into small slices */
    if (ZL_overflowAddST(requestSize, ZL_REDZONE_SIZE, &size)
        || ZL_overflowAddST(size, alignment - 1, &neededSize)) {
        return NULL;
    }
    pba->sessionUsage += neededSize;

    /* On the first request of a session, resize the primary buffer if it is
     * too small to serve that request, or if the previous session had to
     * spill into the heap backup. */
    if ((pba->pBuffUsed == 0)
        && ((pBuffAvailable < neededSize) || (pba->wouldHaveNeeded > 0))) {
        /* let's resize the primary buffer, which is too small */
        ZL_free(pba->primaryBuffer);
        // We are going to reuse primaryBuffer for all future allocation in this
        // arena, so make it worthwhile (essentially 1 page - classic malloc
        // metadata), this will save a bunch of bump up if many small buffers
        // are requested.
        static const size_t pBuffSizeMin =
                4080; // could become a compilation constant in the future
        static const size_t pBuffSizeMax = ALLOC_STACK_SIZE_MAX;
        /* The previous session needed its whole buffer plus whatever spilled
         * to the backup. Consuming @wouldHaveNeeded here, rather than carrying
         * it forward, is what lets the buffer shrink again: a session that did
         * not spill leaves it at 0, so the next one keeps (or sizes down) the
         * current buffer instead of re-growing to a stale high-water mark. */
        size_t const prevSessionNeed =
                pba->pBuffCapacity + pba->wouldHaveNeeded;
        size_t const toAllocate =
                ZL_MAX(ZL_MAX(prevSessionNeed, neededSize), pBuffSizeMin);
        pba->wouldHaveNeeded = 0;
        if (toAllocate <= pBuffSizeMax) {
            if (ALLOC_Arena_canAllocate(
                        arena, pba->pBuffCapacity, toAllocate)) {
                pba->primaryBuffer = ZL_malloc(toAllocate);
            } else {
                pba->primaryBuffer = NULL;
            }
        } else {
            /* request too large : do not allocate primaryBuffer
             * request will be taken care of by HeapArena backup */
            pba->primaryBuffer = NULL;
        }
        if (pba->primaryBuffer) {
            ZL_poisonMemory(pba->primaryBuffer, toAllocate);
            pba->pBuffCapacity = toAllocate;
            pba->pBuffUsed     = neededSize;
            ZL_ASSERT_EQ(
                    (size_t)pba->primaryBuffer % alignment,
                    0); /* note : this alignment method will have to change if
                           requested alignment is larger than base allocation of
                           primaryBuffer */
            ALLOC_StackArena_trackMalloc(pba, pba->primaryBuffer, requestSize);
            return pba->primaryBuffer;
        }
        /* allocation failed */
        pba->pBuffCapacity = 0;
        pba->pBuffUsed     = 0;
        pBuffAvailable     = 0;
    }

    /* second+ request, or allocation of primary buffer failed */
    size_t const start = ((pba->pBuffUsed + (alignment - 1)) / alignment)
            * alignment; /* ensure start position is properly aligned */
    /* Even an empty slice must start within the primary buffer:
     * its end address belongs to the heap backup for free() and realloc() */
    if ((pBuffAvailable >= neededSize)
        && (start + ZL_MAX(size, 1) <= pba->pBuffCapacity)) {
        /* enough space in primary buffer -> assign a slice */
        ZL_ASSERT_GE(start, pba->pBuffUsed);
        ZL_ASSERT_LE(start + size, pba->pBuffCapacity);
        ZL_ASSERT_NN(pba->primaryBuffer);
        ZL_ASSERT_EQ(
                (size_t)pba->primaryBuffer % alignment,
                0); /* note : this alignment method will have to change if
                       requested alignment is larger than base allocation of
                       primaryBuffer */
        void* const r = (char*)pba->primaryBuffer + start;
        /* The first slice of a session moves pBuffUsed even when empty,
         * since pBuffUsed == 0 identifies the first request of a session;
         * that byte fits in the alignment - 1 slack of neededSize */
        pba->pBuffUsed = start + size + (pba->pBuffUsed == 0);
        ALLOC_StackArena_trackMalloc(pba, r, requestSize);
        return r;
    }

    /* not enough space in primaryBuffer :
     * assign backup heap memory for this session
     * and track necessary space, for next session */
    if (!ALLOC_Arena_canAllocate(arena, 0, requestSize)) {
        return NULL;
    }
    pba->wouldHaveNeeded += neededSize;
    return ALLOC_Arena_malloc(&pba->heapBackup.base, requestSize);
}

static void* ALLOC_StackArena_calloc(Arena* arena, size_t size)
{
    ZL_ASSERT_NN(arena);
    StackArena* pba = ZL_CONTAINER_OF(arena, StackArena, base);
    /* note: could be optimized later, by a call to `calloc()` instead of
     * `malloc()` when reaching heapBackup */
    void* const r = ALLOC_Arena_malloc(&pba->base, size);
    if (r == NULL)
        return NULL;
    ZL_zeroes(r, size);
    return r;
}

static bool ALLOC_StackArena_inPrimaryBuffer(StackArena* arena, void* ptr)
{
    if (arena->primaryBuffer == NULL) {
        return false;
    }
    void* const pBuffEnd = (char*)arena->primaryBuffer + arena->pBuffCapacity;
    return (ptr >= arena->primaryBuffer && ptr < pBuffEnd);
}

static void* ALLOC_StackArena_realloc(Arena* arena, void* ptr, size_t newSize)
{
    ZL_ASSERT_NN(arena);
    StackArena* const pba = ZL_CONTAINER_OF(arena, StackArena, base);
    if (ALLOC_StackArena_inPrimaryBuffer(pba, ptr)) {
        // TODO: If ptr was the most recently allocated pointer, we could
        // realloc in-place. This optimization would make sense if we also
        // supported freeing the most recently allocated pointer.
        //
        // For now, we always copy into the heap arena.
        if (!ALLOC_Arena_canAllocate(arena, 0, newSize)) {
            return NULL;
        }
        void* newPtr = ALLOC_Arena_malloc(&pba->heapBackup.base, newSize);
        if (newPtr == NULL) {
            return NULL;
        }
        // The old size isn't stored so copy based on the new size, and the
        // largest size it could've possibly been.
        ZL_ASSERT_NN(ptr);
        char* const pBuffEnd = (char*)pba->primaryBuffer + pba->pBuffCapacity;
        const size_t toCopy  = ZL_MIN((size_t)(pBuffEnd - (char*)ptr), newSize);

        ZL_UNPOISON_MEMORY(pba->primaryBuffer, pba->pBuffCapacity);
        memcpy(newPtr, ptr, toCopy);
        ZL_POISON_MEMORY(pba->primaryBuffer, pba->pBuffCapacity);
        AsanOnlySizeMap_unpoisonAllPtrs(&pba->asanOnlySizeMap);

        // Mark the old pointer as freed.
        ALLOC_StackArena_trackFree(pba, ptr);
        return newPtr;
    } else {
        const size_t oldSize = ptr == NULL ? 0 : ((HeapMeta*)ptr - 1)->size;
        if (!ALLOC_Arena_canAllocate(arena, oldSize, newSize)) {
            return NULL;
        }
        return ALLOC_HeapArena_realloc(&pba->heapBackup.base, ptr, newSize);
    }
}

static void ALLOC_StackArena_freeSegment(Arena* arena, void* ptr)
{
    if (ptr == NULL)
        return;
    ZL_ASSERT_NN(arena);
    StackArena* const pba = ZL_CONTAINER_OF(arena, StackArena, base);
    if (ALLOC_StackArena_inPrimaryBuffer(pba, ptr)) {
        // This ptr owns a slice within primaryBuffer -> don't free anything
        // Just mark it as freed in ASAN mode
        ALLOC_StackArena_trackFree(pba, ptr);
        return;
    }
    /* this ptr is presumed tracked by heapBackup */
    ALLOC_HeapArena_freeSegment(&pba->heapBackup.base, ptr);
}

#define PBUFF_USAGE_MIN(s) ((s) / 2) // arbitrary threshold

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
#    define PBUFF_SIZEDOWN_THRESHOLD (1U << 14)
#else
#    define PBUFF_SIZEDOWN_THRESHOLD (1U << 30) // arbitrary threshold
#endif

static void ALLOC_StackArena_freeAllSegments(Arena* arena)
{
    ZL_DLOG(BLOCK, "ALLOC_StackArena_freeAllSegments (address:%p)", arena);
    if (arena == NULL)
        return;
    StackArena* const pba = ZL_CONTAINER_OF(arena, StackArena, base);
    pba->pBuffUsed        = 0;
    ALLOC_HeapArena_freeAllSegments(&pba->heapBackup.base);

    // Reset ASAN size map & poison the primary buffer
    AsanOnlySizeMap_clear(&pba->asanOnlySizeMap);
    if (pba->primaryBuffer) {
        ZL_poisonMemory(pba->primaryBuffer, pba->pBuffCapacity);
    }

    if (pba->sessionUsage < PBUFF_USAGE_MIN(pba->pBuffCapacity)) {
        pba->wasted += pba->pBuffCapacity;
    } else {
        pba->wasted = 0;
    }
    if (pba->wasted > PBUFF_SIZEDOWN_THRESHOLD) {
        // size down the Primary buffer -> it has been too big for a while
        pba->pBuffCapacity /= 2;
        // Use realloc to improve odd of keeping current buffer in place.
        void* const newPBuffer =
                ZL_realloc(pba->primaryBuffer, pba->pBuffCapacity);
        if (newPBuffer != NULL) {
            pba->primaryBuffer = newPBuffer;
            // Poison the new buffer
            ZL_poisonMemory(pba->primaryBuffer, pba->pBuffCapacity);
        } else {
            // Failed realloc: Just give up the primary buffer
            ZL_free(pba->primaryBuffer);
            pba->primaryBuffer = NULL;
            pba->pBuffCapacity = 0;
        }
        pba->wasted = 0;
    }
    pba->pBuffUsed    = 0;
    pba->sessionUsage = 0;
}

static void ALLOC_StackArena_freeArena(Arena* arena)
{
    if (arena == NULL)
        return;
    ZL_ASSERT_NN(arena->freeAll);
    arena->freeAll(arena); /* note: freeAll doesn't free the primaryBuffer */
    StackArena* const pba = ZL_CONTAINER_OF(arena, StackArena, base);
    ALLOC_HeapArena_destroy(&pba->heapBackup);
    AsanOnlySizeMap_destroy(&pba->asanOnlySizeMap);
    ZL_free(pba->primaryBuffer);
    ZL_free(pba);
}

static size_t ALLOC_StackArena_memAllocated(const Arena* arena)
{
    if (arena == NULL)
        return 0;
    const StackArena* const stackArena =
            ZL_CONST_CONTAINER_OF(arena, StackArena, base);
    return stackArena->pBuffCapacity
            + ALLOC_HeapArena_countMem(&stackArena->heapBackup);
}

static size_t ALLOC_StackArena_memUsed(const Arena* arena)
{
    if (arena == NULL)
        return 0;
    const StackArena* const stackArena =
            ZL_CONST_CONTAINER_OF(arena, StackArena, base);
    return stackArena->pBuffUsed
            + ALLOC_HeapArena_countMem(&stackArena->heapBackup);
}

static const Arena kStackArena = {
    .malloc       = ALLOC_StackArena_malloc,
    .calloc       = ALLOC_StackArena_calloc,
    .realloc      = ALLOC_StackArena_realloc,
    .free         = ALLOC_StackArena_freeSegment,
    .freeAll      = ALLOC_StackArena_freeAllSegments,
    .freeArena    = ALLOC_StackArena_freeArena,
    .memAllocated = ALLOC_StackArena_memAllocated,
    .memUsed      = ALLOC_StackArena_memUsed,
};
Arena* ALLOC_StackArena_create(void)
{
    StackArena* const arena = ZL_calloc(sizeof(*arena));
    if (arena == NULL)
        return NULL;
    arena->base = kStackArena;
    ALLOC_HeapArena_init(&arena->heapBackup);
    arena->asanOnlySizeMap = AsanOnlySizeMap_createInArena(
            &kRawAllocator, ZL_CONTAINER_SIZE_LIMIT);
    return &arena->base;
}
