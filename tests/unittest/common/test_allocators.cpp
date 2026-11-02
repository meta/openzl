// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <cstdint>
#include <random>
#include <tuple>
#include "openzl/common/allocation.h"

namespace {

// NOTE: These tests are designed to be coupled with ASAN to verify correct
// memory and pointers usage

class ArenaImplementation {
   public:
    virtual Arena* create()                  = 0;
    virtual void free(Arena* a)              = 0;
    virtual size_t maxAllocatorAllocations() = 0;
    virtual std::string name()               = 0;
    virtual ~ArenaImplementation()           = default;
};

class AllocatorTest : public testing::TestWithParam<ArenaImplementation*> {
   public:
    static Arena* createArena()
    {
        auto& impl = GetParam();
        return impl->create();
    }
    static size_t maxAllocatorAllocations()
    {
        auto& impl = GetParam();
        return impl->maxAllocatorAllocations();
    }
    static void freeArena(Arena* arena)
    {
        auto& impl = GetParam();
        return impl->free(arena);
    }
};

TEST_P(AllocatorTest, creationDestruction)
{
    auto arena = createArena();
    ASSERT_NE(arena, nullptr);

    freeArena(arena);
}

TEST_P(AllocatorTest, allocate1AndFree)
{
    auto arena = createArena();

    void* ptr = ALLOC_Arena_malloc(arena, 100);
    ASSERT_NE(ptr, nullptr);
    ALLOC_Arena_free(arena, ptr);

    freeArena(arena);
}

TEST_P(AllocatorTest, allocateBigAndInit)
{
    auto arena = createArena();

    // Larger than PBuff max size
    size_t const bigSize = ALLOC_STACK_SIZE_MAX + (4 << 10);
    void* const ptr      = ALLOC_Arena_malloc(arena, bigSize);
    ASSERT_NE(ptr, nullptr);

    // This will crash (asan) if buffer is too small
    memset(ptr, 1, bigSize);

    ALLOC_Arena_free(arena, ptr);

    freeArena(arena);
}

// Sizes read from corrupted frames can be close to SIZE_MAX:
// such requests must fail, not wrap around into a small allocation.
// These margins overflow the arenas' own size computations, even without the
// ASAN redzones that GCC builds may lack, so the system allocator is never
// reached (under ASAN, it aborts instead of returning NULL).
TEST_P(AllocatorTest, allocateHugeFails)
{
    auto arena = createArena();

    for (size_t const margin : { 0u, 1u, 14u }) {
        size_t const hugeSize = SIZE_MAX - margin;
        EXPECT_EQ(ALLOC_Arena_malloc(arena, hugeSize), nullptr) << margin;
        EXPECT_EQ(ALLOC_Arena_calloc(arena, hugeSize), nullptr) << margin;
    }

    // The arena remains usable
    void* const ptr = ALLOC_Arena_malloc(arena, 100);
    ASSERT_NE(ptr, nullptr);
    ALLOC_Arena_free(arena, ptr);

    freeArena(arena);
}

TEST_P(AllocatorTest, allocate5AndFreeAll)
{
    auto arena = createArena();

    for (int i = 0; i < 5; i++) {
        void* ptr = ALLOC_Arena_malloc(arena, 100);
        ASSERT_NE(ptr, nullptr);
    }
    ALLOC_Arena_freeAll(arena);

    freeArena(arena);
}

TEST_P(AllocatorTest, allocateManyAndDestroy)
{
    auto arena = createArena();

    // Would fail for 101
    for (size_t i = 0; i < maxAllocatorAllocations(); i++) {
        void* ptr = ALLOC_Arena_malloc(arena, 100);
        ASSERT_NE(ptr, nullptr);
    }
    ALLOC_Arena_freeAll(arena);

    freeArena(arena);
}

TEST_P(AllocatorTest, allocateMultipleFreeAndAllocateAgain)
{
    auto arena = createArena();

    // Would fail for 101
    std::vector<void*> pointers;
    for (size_t i = 0; i < maxAllocatorAllocations(); i++) {
        void* ptr = ALLOC_Arena_malloc(arena, 100);
        ASSERT_NE(ptr, nullptr);
        pointers.push_back(ptr);
    }
    ASSERT_GT((int)ALLOC_Arena_memAllocated(arena), (int)0);
    for (size_t i = 0; i < 50; i++) {
        ALLOC_Arena_free(arena, pointers.back());
        pointers.pop_back();
    }
    for (size_t i = 0; i < 50; i++) {
        void* ptr = ALLOC_Arena_malloc(arena, 100);
        ASSERT_NE(ptr, nullptr);
        pointers.push_back(ptr);
    }
    pointers.resize(0);
    ALLOC_Arena_freeAll(arena);
    for (size_t i = 0; i < maxAllocatorAllocations(); i++) {
        void* ptr = ALLOC_Arena_malloc(arena, 100);
        ASSERT_NE(ptr, nullptr);
        pointers.push_back(ptr);
    }

    freeArena(arena);
    ASSERT_EQ((int)ALLOC_Arena_memAllocated(nullptr), (int)0);
}

TEST_P(AllocatorTest, reallocWithFree)
{
    Arena* arena = createArena();

    char* ptr = (char*)ALLOC_Arena_realloc(arena, NULL, 5);
    ASSERT_NE(ptr, nullptr);
    ptr[4] = '4';
    ptr    = (char*)ALLOC_Arena_realloc(arena, ptr, 6);
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(ptr[4], '4');
    ptr[3] = '3';
    ptr[5] = '5';
    ptr    = (char*)ALLOC_Arena_realloc(arena, ptr, 4);
    ASSERT_EQ(ptr[3], '3');

    ALLOC_Arena_free(arena, ptr);
    freeArena(arena);
}

TEST_P(AllocatorTest, reallocWithoutFree)
{
    Arena* arena = createArena();

    char* ptr = (char*)ALLOC_Arena_realloc(arena, NULL, 5);
    ASSERT_NE(ptr, nullptr);
    ptr[4] = '4';
    ptr    = (char*)ALLOC_Arena_realloc(arena, ptr, 6);
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(ptr[4], '4');
    freeArena(arena);
}

TEST_P(AllocatorTest, reallocAfterMalloc)
{
    Arena* arena = createArena();

    char* ptr0 = (char*)ALLOC_Arena_malloc(arena, 5);
    ASSERT_NE(ptr0, nullptr);
    ptr0[4] = '4';

    char* ptr1 = (char*)ALLOC_Arena_malloc(arena, 5);
    ASSERT_NE(ptr1, nullptr);
    ptr1[1] = '1';

    ptr0 = (char*)ALLOC_Arena_realloc(arena, ptr0, 10);
    ASSERT_NE(ptr0, nullptr);
    ASSERT_EQ(ptr0[4], '4');

    ptr1 = (char*)ALLOC_Arena_realloc(arena, ptr1, 2);
    ASSERT_NE(ptr1, nullptr);
    ASSERT_EQ(ptr1[1], '1');

    ALLOC_Arena_free(arena, ptr0);

    freeArena(arena);
}

TEST_P(AllocatorTest, mallocAndReallocWithSeveralIterations)
{
    Arena* arena = createArena();

    std::unordered_map<char*, size_t> ptrs;

    std::mt19937 gen(0xdeadbeef);

    auto randomPtr = [&ptrs, &gen] {
        std::uniform_int_distribution<size_t> idx(0, ptrs.size() - 1);
        auto iter = ptrs.begin();
        std::advance(iter, idx(gen));
        return iter->first;
    };

    auto cmpPtrVal = [](char* ptr, char expect, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            if (ptr[i] != expect) {
                return false;
            }
        }
        return true;
    };
    auto setPtrVal = [](char* ptr, size_t size) {
        if (size > 0) {
            memset(ptr, (char)size, size);
        }
    };
    auto testPtrVal = [cmpPtrVal](char* ptr, size_t size) {
        return cmpPtrVal(ptr, size, size);
    };

    std::uniform_int_distribution<size_t> action(0, 3);
    std::uniform_int_distribution<size_t> allocSize(1, 100);
    for (size_t iter = 0; iter < 10; ++iter) {
        for (size_t i = 0; i < maxAllocatorAllocations(); ++i) {
            switch (action(gen)) {
                case 0:
                    // Free a pointer
                    if (ptrs.size() > 0) {
                        auto ptr = randomPtr();
                        ASSERT_TRUE(testPtrVal(ptr, ptrs[ptr]));
                        ALLOC_Arena_free(arena, ptr);
                        ptrs.erase(ptr);
                    }
                    break;
                case 1: {
                    // Mallocs a pointer
                    const size_t size = allocSize(gen);
                    auto ptr          = (char*)ALLOC_Arena_malloc(arena, size);
                    ASSERT_NE(ptr, nullptr);
                    ASSERT_EQ(ptrs.count(ptr), 0) << ptrs[ptr];
                    ptrs[ptr] = size;
                    setPtrVal(ptr, size);
                    break;
                }
                case 2: {
                    // Callocs a pointer
                    const size_t size = allocSize(gen);
                    auto ptr          = (char*)ALLOC_Arena_calloc(arena, size);
                    ASSERT_NE(ptr, nullptr);
                    ASSERT_EQ(ptrs.count(ptr), 0);
                    ptrs[ptr] = size;
                    ASSERT_TRUE(cmpPtrVal(ptr, 0, size));
                    setPtrVal(ptr, size);
                    break;
                }
                case 3:
                    // Reallocs a pointer
                    if (ptrs.size() > 0) {
                        auto oldPtr          = randomPtr();
                        const size_t oldSize = ptrs[oldPtr];
                        ASSERT_TRUE(testPtrVal(oldPtr, oldSize));
                        const size_t newSize = allocSize(gen);
                        auto newPtr          = (char*)ALLOC_Arena_realloc(
                                arena, oldPtr, newSize);
                        ASSERT_NE(newPtr, nullptr);
                        if (newPtr != oldPtr) {
                            ASSERT_EQ(ptrs.count(newPtr), 0);
                        }
                        ptrs.erase(oldPtr);
                        ASSERT_TRUE(cmpPtrVal(
                                newPtr,
                                (char)oldSize,
                                std::min(oldSize, newSize)));
                        ptrs[newPtr] = newSize;
                        setPtrVal(newPtr, newSize);
                    }
            }
        }
        ALLOC_Arena_freeAll(arena);
        ptrs.clear();
    }

    freeArena(arena);
}

TEST_P(AllocatorTest, redzoneIsPoisonedWithMallocAndFree)
{
    if (!ZL_POISON_SUPPORTED) {
        return;
    }

    Arena* arena = createArena();

    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            // Allocate a buffer
            char* ptr = (char*)ALLOC_Arena_malloc(arena, 100);
            ASSERT_NE(ptr, nullptr);

            ASSERT_FALSE(ZL_addressIsPoisoned(ptr));
            ASSERT_FALSE(ZL_addressIsPoisoned(ptr + 99));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 100));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));

            // Free the buffer
            ALLOC_Arena_free(arena, ptr);

            ASSERT_TRUE(ZL_addressIsPoisoned(ptr));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 99));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 100));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));
        }

        ALLOC_Arena_freeAll(arena);
    }

    freeArena(arena);
}

TEST_P(AllocatorTest, redzoneIsPoisonedWithMallocThenFreeAll)
{
    if (!ZL_POISON_SUPPORTED) {
        return;
    }

    Arena* arena = createArena();

    for (size_t i = 0; i < 5; ++i) {
        std::array<char*, 5> ptrs;
        for (size_t j = 0; j < 5; ++j) {
            // Allocate a buffer
            char* ptr = (char*)ALLOC_Arena_malloc(arena, 100);
            ASSERT_NE(ptr, nullptr);

            ASSERT_FALSE(ZL_addressIsPoisoned(ptr));
            ASSERT_FALSE(ZL_addressIsPoisoned(ptr + 99));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 100));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));

            ptrs[j] = ptr;
        }

        ALLOC_Arena_freeAll(arena);

        for (char* ptr : ptrs) {
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 99));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 100));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));
        }
    }

    freeArena(arena);
}

TEST_P(AllocatorTest, redzoneIsPoisonedWithMallocThenRealloc)
{
    if (!ZL_POISON_SUPPORTED) {
        return;
    }

    Arena* arena = createArena();

    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            // Allocate a buffer with realloc
            char* ptr = (char*)ALLOC_Arena_malloc(arena, 100);
            ASSERT_NE(ptr, nullptr);

            ASSERT_FALSE(ZL_addressIsPoisoned(ptr));
            ASSERT_FALSE(ZL_addressIsPoisoned(ptr + 99));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 100));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));

            // Resize the buffer larger
            ptr = (char*)ALLOC_Arena_realloc(arena, ptr, 200);
            ASSERT_NE(ptr, nullptr);

            ASSERT_FALSE(ZL_addressIsPoisoned(ptr));
            ASSERT_FALSE(ZL_addressIsPoisoned(ptr + 199));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 200));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));

            // Resize the buffer smaller
            ptr = (char*)ALLOC_Arena_realloc(arena, ptr, 50);
            ASSERT_NE(ptr, nullptr);

            ASSERT_FALSE(ZL_addressIsPoisoned(ptr));
            ASSERT_FALSE(ZL_addressIsPoisoned(ptr + 49));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 50));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));

            // Free the buffer
            ALLOC_Arena_free(arena, ptr);

            ASSERT_TRUE(ZL_addressIsPoisoned(ptr));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 49));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 99));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 199));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 200));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));
        }
        ALLOC_Arena_freeAll(arena);
    }

    freeArena(arena);
}

TEST_P(AllocatorTest, redzoneIsPoisonedWithReallocThenRealloc)
{
    if (!ZL_POISON_SUPPORTED) {
        return;
    }

    Arena* arena = createArena();

    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            // Allocate a buffer with realloc
            char* ptr = (char*)ALLOC_Arena_realloc(arena, nullptr, 100);
            ASSERT_NE(ptr, nullptr);

            ASSERT_FALSE(ZL_addressIsPoisoned(ptr));
            ASSERT_FALSE(ZL_addressIsPoisoned(ptr + 99));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 100));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));

            // Resize the buffer larger
            ptr = (char*)ALLOC_Arena_realloc(arena, ptr, 200);
            ASSERT_NE(ptr, nullptr);

            ASSERT_FALSE(ZL_addressIsPoisoned(ptr));
            ASSERT_FALSE(ZL_addressIsPoisoned(ptr + 199));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 200));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));

            // Resize the buffer smaller
            ptr = (char*)ALLOC_Arena_realloc(arena, ptr, 50);
            ASSERT_NE(ptr, nullptr);

            ASSERT_FALSE(ZL_addressIsPoisoned(ptr));
            ASSERT_FALSE(ZL_addressIsPoisoned(ptr + 49));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 50));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));

            // Free the buffer
            ALLOC_Arena_free(arena, ptr);

            ASSERT_TRUE(ZL_addressIsPoisoned(ptr));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 49));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 99));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 199));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr + 200));
            ASSERT_TRUE(ZL_addressIsPoisoned(ptr - 1));
        }
        ALLOC_Arena_freeAll(arena);
    }

    freeArena(arena);
}

// Instantiate tests for HeapArena

class HeapArenaImplementation : public ArenaImplementation {
    Arena* create() override
    {
        return ALLOC_HeapArena_create();
    }
    void free(Arena* a) override
    {
        ALLOC_Arena_freeArena(a);
    }
    size_t maxAllocatorAllocations() override
    {
        return 1000;
    }
    std::string name() override
    {
        return "HeapArena";
    }
};
auto heapArena = HeapArenaImplementation();

auto heapTests = testing::Values(&heapArena);

INSTANTIATE_TEST_SUITE_P(HeapArenaTests, AllocatorTest, heapTests, [](auto p) {
    return p.param->name();
});

// Instantiate tests for StackArena

class StackArenaImplementation : public ArenaImplementation {
    Arena* create() override
    {
        return ALLOC_StackArena_create();
    }
    void free(Arena* a) override
    {
        ALLOC_Arena_freeArena(a);
    }
    size_t maxAllocatorAllocations() override
    {
        return 1000;
    }
    std::string name() override
    {
        return "StackArena";
    }
};
auto pbufferArena = StackArenaImplementation();

auto pbufferTests = testing::Values(&pbufferArena);

INSTANTIATE_TEST_SUITE_P(
        StackArenaTests,
        AllocatorTest,
        pbufferTests,
        [](auto p) { return p.param->name(); });

// StackArena promises that a Primary Buffer oversized by one exceptional
// session is sized back down to suit the general case. Its owner (e.g. a
// pooled ZL_CCtx) may live for the lifetime of the process, so a buffer that
// stays pinned at the high-water mark is retained memory that is never
// reclaimed.
TEST(StackArenaSizingTest, primaryBufferShrinksAfterOneExceptionalSession)
{
    Arena* const arena = ALLOC_StackArena_create();
    ASSERT_NE(arena, nullptr);

    constexpr size_t kExceptionalAlloc = 16u << 20;
    constexpr size_t kTypicalAlloc     = 1024;
    // Every under-using session adds pBuffCapacity to the `wasted` counter and
    // a size-down needs that to exceed PBUFF_SIZEDOWN_THRESHOLD (1GiB), so at
    // 16MiB a halving takes ~65 sessions. This affords two of them.
    constexpr size_t kTypicalSessions = 256;

    ALLOC_Arena_malloc(arena, kExceptionalAlloc);
    const size_t exceptionalSize = ALLOC_Arena_memAllocated(arena);
    ALLOC_Arena_freeAll(arena);
    ASSERT_GE(exceptionalSize, kExceptionalAlloc);

    for (size_t i = 0; i < kTypicalSessions; i++) {
        ALLOC_Arena_malloc(arena, kTypicalAlloc);
        ALLOC_Arena_freeAll(arena);
    }

    // Measure from inside a session: a re-grow happens on the first allocation
    // after freeAll, so measuring between sessions would miss it.
    ALLOC_Arena_malloc(arena, kTypicalAlloc);
    const size_t steadyStateSize = ALLOC_Arena_memAllocated(arena);
    ALLOC_Arena_freeAll(arena);

    EXPECT_LE(steadyStateSize, exceptionalSize / 2);

    ALLOC_Arena_freeArena(arena);
}

// Slices are 16-byte aligned. Without ASAN redzones, an empty request could get
// the address right past the end of the Primary Buffer, which free() and
// realloc() then take for a heap backup pointer.
TEST(StackArenaSliceTest, emptySliceIsNotPastTheEndOfPrimaryBuffer)
{
    Arena* const arena = ALLOC_StackArena_create();
    ASSERT_NE(arena, nullptr);

    // First session sizes the Primary Buffer
    ALLOC_Arena_malloc(arena, 100);
    const size_t capacity = ALLOC_Arena_memAllocated(arena);
    ALLOC_Arena_freeAll(arena);

    // Leave 15 bytes free, i.e. less than one aligned slot
    char* const full  = (char*)ALLOC_Arena_malloc(arena, capacity - 15);
    char* const empty = (char*)ALLOC_Arena_malloc(arena, 0);
    ASSERT_NE(full, nullptr);
    ASSERT_NE(empty, nullptr);
    EXPECT_NE(empty, full + capacity);

    ALLOC_Arena_free(arena, empty);
    ALLOC_Arena_free(arena, full);
    ALLOC_Arena_freeArena(arena);
}

// The first slice of a session must be recorded even when it's empty,
// otherwise the next request is taken for the first of the session.
TEST(StackArenaSliceTest, emptyFirstSliceIsRecorded)
{
    Arena* const arena = ALLOC_StackArena_create();
    ASSERT_NE(arena, nullptr);

    ALLOC_Arena_malloc(arena, 100);
    ALLOC_Arena_freeAll(arena);

    void* const empty = ALLOC_Arena_malloc(arena, 0);
    void* const next  = ALLOC_Arena_malloc(arena, 64);
    ASSERT_NE(empty, nullptr);
    ASSERT_NE(next, nullptr);
    EXPECT_NE(empty, next);

    ALLOC_Arena_freeArena(arena);
}

} // namespace
