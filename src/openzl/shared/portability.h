// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef ZS_COMMON_PORTABILITY_H
#define ZS_COMMON_PORTABILITY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "openzl/zl_portability.h"

#if defined(__cplusplus)
#    define ZL_C_DECL extern "C"
#    define ZL_BEGIN_C_DECLS \
        ZL_C_DECL            \
        {
#    define ZL_END_C_DECLS }
#else
#    define ZL_C_DECL
#    define ZL_BEGIN_C_DECLS
#    define ZL_END_C_DECLS
#endif

ZL_BEGIN_C_DECLS

/// MSVC-specific warnings disabled
#ifdef _MSC_VER
#    ifndef _CRT_SECURE_NO_WARNINGS
#        define _CRT_SECURE_NO_WARNINGS // Suppress MSVC secure CRT warnings
                                        // (e.g., strncpy, sprintf)
#    endif
#    pragma warning(disable : 4127) // conditional expression is constant
#    pragma warning(disable : 4702) // unreachable code
#endif

/// Attributes
#if defined(_MSC_VER) && !defined(__clang__)
#    define ZL_FORCE_INLINE_ATTR __forceinline
#    define ZL_FORCE_NOINLINE_ATTR __declspec(noinline)
#    define ZL_FLATTEN_ATTR // MSVC doesn't have equivalent to flatten
#    define ZL_ALIGNED(n) __declspec(align(n))
#    define ZL_UNUSED // MSVC doesn't need unused attribute
#    define ZL_NORETURN __declspec(noreturn)
#elif defined(__GNUC__) || defined(__clang__)
#    define ZL_FORCE_INLINE_ATTR __attribute__((always_inline))
#    define ZL_FORCE_NOINLINE_ATTR __attribute__((noinline))
#    define ZL_FLATTEN_ATTR __attribute__((flatten))
#    define ZL_ALIGNED(n) __attribute__((aligned(n)))
#    define ZL_UNUSED __attribute__((unused))
#    define ZL_NORETURN __attribute__((noreturn))
#else
#    define ZL_FORCE_INLINE_ATTR
#    define ZL_FORCE_NOINLINE_ATTR
#    define ZL_FLATTEN_ATTR
#    define ZL_ALIGNED(n)
#    define ZL_UNUSED
#    define ZL_NORETURN
#endif

/// Keywords
#define ZL_INLINE_KEYWORD static inline

/// Helper combinations
#if defined(_MSC_VER)
#    define ZL_FORCE_INLINE static __forceinline
#    define ZL_FORCE_NOINLINE static __declspec(noinline)
#else
#    define ZL_FORCE_INLINE ZL_INLINE ZL_FORCE_INLINE_ATTR
#    define ZL_FORCE_NOINLINE static ZL_FORCE_NOINLINE_ATTR
#endif

/// Compiler builtins
#ifndef ZL_LIKELY
#    if defined(__GNUC__) || defined(__clang__)
#        define ZL_LIKELY(x) __builtin_expect((x), 1)
#    else
#        define ZL_LIKELY(x) (x)
#    endif
#endif

#ifndef ZL_UNLIKELY
#    if defined(__GNUC__) || defined(__clang__)
#        define ZL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#    else
#        define ZL_UNLIKELY(x) (x)
#    endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#    define ZL_PREFETCH_L1(ptr) \
        __builtin_prefetch((ptr), 0 /* rw==read */, 3 /* locality */)
#    define ZL_PREFETCH_L2(ptr) \
        __builtin_prefetch((ptr), 0 /* rw==read */, 2 /* locality */)
#else
#    define ZL_PREFETCH_L1(ptr) ((void)(ptr))
#    define ZL_PREFETCH_L2(ptr) ((void)(ptr))
#endif

#if defined(__GNUC__) || defined(__ICCARM__) || ZL_HAS_ATTRIBUTE(__target__)
#    define ZL_TARGET_ATTRIBUTE(target) __attribute__((__target__(target)))
#else
#    define ZL_TARGET_ATTRIBUTE(target)
#endif

#ifdef __has_builtin
#    define ZL_HAS_BUILTIN(x) __has_builtin(x)
#else
#    define ZL_HAS_BUILTIN(x) 0
#endif

#ifdef __has_attribute
#    define ZL_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#    define ZL_HAS_ATTRIBUTE(x) 0
#endif

#ifdef __has_feature
#    define ZL_HAS_FEATURE(x) __has_feature(x)
#else
#    define ZL_HAS_FEATURE(x) 0
#endif

#ifdef __has_include
#    define ZL_HAS_INCLUDE(x) __has_include(x)
#else
#    define ZL_HAS_INCLUDE(x) 0
#endif

#ifndef ZL_ADDRESS_SANITIZER
#    if ZL_HAS_FEATURE(address_sanitizer)
#        define ZL_ADDRESS_SANITIZER 1
#    else
#        define ZL_ADDRESS_SANITIZER 0
#    endif
#endif

#if ZL_ADDRESS_SANITIZER
/* Not all platforms that support asan provide sanitizers/asan_interface.h.
 * We therefore declare the functions we need ourselves, rather than trying to
 * include the header file... */

/**
 * Marks a memory region (<c>[addr, addr+size)</c>) as unaddressable.
 *
 * This memory must be previously allocated by your program. Instrumented
 * code is forbidden from accessing addresses in this region until it is
 * unpoisoned. This function is not guaranteed to poison the entire region -
 * it could poison only a subregion of <c>[addr, addr+size)</c> due to ASan
 * alignment restrictions.
 *
 * \note This function is not thread-safe because no two threads can poison or
 * unpoison memory in the same memory region simultaneously.
 *
 * \param addr Start of memory region.
 * \param size Size of memory region. */
void __asan_poison_memory_region(void const volatile* addr, size_t size);

/**
 * Marks a memory region (<c>[addr, addr+size)</c>) as addressable.
 *
 * This memory must be previously allocated by your program. Accessing
 * addresses in this region is allowed until this region is poisoned again.
 * This function could unpoison a super-region of <c>[addr, addr+size)</c> due
 * to ASan alignment restrictions.
 *
 * \note This function is not thread-safe because no two threads can
 * poison or unpoison memory in the same memory region simultaneously.
 *
 * \param addr Start of memory region.
 * \param size Size of memory region. */
void __asan_unpoison_memory_region(void const volatile* addr, size_t size);

/**
 * Checks if an address is poisoned.
 *
 *  Returns 1 if <c><i>addr</i></c> is poisoned (that is, 1-byte read/write
 *  access to this address would result in an error report from ASan).
 *  Otherwise returns 0.
 *
 *  \param addr Address to check.
 *
 *  \retval 1 Address is poisoned.
 *  \retval 0 Address is not poisoned.
 */
int __asan_address_is_poisoned(void const volatile* addr);
#endif

#define ZL_POISON_SUPPORTED ZL_ADDRESS_SANITIZER

ZL_INLINE void ZL_poisonMemory(const void* ptr, size_t size)
{
#if ZL_ADDRESS_SANITIZER
    __asan_poison_memory_region(ptr, size);
#else
    (void)ptr;
    (void)size;
#endif
}

ZL_INLINE void ZL_unpoisonMemory(const void* ptr, size_t size)
{
#if ZL_ADDRESS_SANITIZER
    __asan_unpoison_memory_region(ptr, size);
#else
    (void)ptr;
    (void)size;
#endif
}

ZL_INLINE bool ZL_addressIsPoisoned(const void* ptr)
{
#if ZL_ADDRESS_SANITIZER
    return __asan_address_is_poisoned(ptr);
#else
    (void)ptr;
    return false;
#endif
}

#if ZL_POISON_SUPPORTED
#    define ZL_POISON_MEMORY(ptr, size) ZL_poisonMemory(ptr, size)
#    define ZL_UNPOISON_MEMORY(ptr, size) ZL_unpoisonMemory(ptr, size)
#else
#    define ZL_POISON_MEMORY(ptr, size)
#    define ZL_UNPOISON_MEMORY(ptr, size)
#endif

#define ZL_ARCH_FLAG_X86 (1 << 0)
#define ZL_ARCH_FLAG_X86_64 ((1 << 1) | ZL_ARCH_FLAG_X86)
#define ZL_ARCH_FLAG_I386 ((1 << 2) | ZL_ARCH_FLAG_X86)

#define ZL_ARCH_FLAG_ARM (1 << 3)
#define ZL_ARCH_FLAG_ARM64 ((1 << 4) | ZL_ARCH_FLAG_ARM)
#define ZL_ARCH_FLAG_ARM32 ((1 << 5) | ZL_ARCH_FLAG_ARM)

#define ZL_ARCH_FLAG_PPC (1 << 6)
#define ZL_ARCH_FLAG_PPC64 ((1 << 7) | ZL_ARCH_FLAG_PPC)
#define ZL_ARCH_FLAG_PPC64LE ((1 << 8) | ZL_ARCH_FLAG_PPC64)
#define ZL_ARCH_FLAG_PPC64BE ((1 << 9) | ZL_ARCH_FLAG_PPC64)

#define ZL_ARCH_FLAG_UNKNOWN (1 << 30)

#if defined(__x86_64__) || defined(_M_X64)
#    define ZL_ARCH_FLAGS ZL_ARCH_FLAG_X86_64
#elif defined(__i386__)
#    define ZL_ARCH_FLAGS ZL_ARCH_FLAG_I386
#elif defined(__aarch64__) || defined(__arm64__)
#    define ZL_ARCH_FLAGS ZL_ARCH_FLAG_ARM64
#elif defined(__arm__)
#    define ZL_ARCH_FLAGS ZL_ARCH_FLAG_ARM32
#elif defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
#    if defined(_LITTLE_ENDIAN) || defined(__LITTLE_ENDIAN__) \
            || (defined(__BYTE_ORDER__)                       \
                && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
#        define ZL_ARCH_FLAGS ZL_ARCH_FLAG_PPC64LE
#    else
#        define ZL_ARCH_FLAGS ZL_ARCH_FLAG_PPC64BE
#    endif
#else
#    define ZL_ARCH_FLAGS ZL_ARCH_FLAG_UNKNOWN
#endif

#define ZL_ARCH_X86 ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_X86) != 0)
#define ZL_ARCH_X86_64 ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_X86_64) != 0)
#define ZL_ARCH_I386 ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_I386) != 0)
#define ZL_ARCH_ARM ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_ARM) != 0)
#define ZL_ARCH_ARM64 ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_ARM64) != 0)
#define ZL_ARCH_ARM32 ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_ARM32) != 0)
#define ZL_ARCH_PPC ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_PPC) != 0)
#define ZL_ARCH_PPC64 ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_PPC64) != 0)
#define ZL_ARCH_PPC64LE ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_PPC64LE) != 0)
#define ZL_ARCH_PPC64BE ((ZL_ARCH_FLAGS & ZL_ARCH_FLAG_PPC64BE) != 0)

// Enforce 64-bit compilation
#if (ZL_ARCH_FLAGS == ZL_ARCH_FLAG_I386) \
        || (ZL_ARCH_FLAGS == ZL_ARCH_FLAG_ARM32)
#    error "This codebase requires a 64-bit platform. 32-bit compilation is not supported yet."
#endif

// Error on unknown architectures for now.
// TODO: This should probably be removed before open source.
// But I don't want us to accidentally miss an architecture, or
// mess up our detection silently.
#ifndef ZL_NO_ERROR_ON_UNKNOWN_ARCH
#    ifdef ZL_ARCH_UNKNOWN
#        error Unknown architecture please set the correct ZS_ARCH_* macro above!
#    endif
#endif

#if defined(__BMI2__)
#    define ZL_HAS_BMI2 1
#else
#    define ZL_HAS_BMI2 0
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#    define ZL_HAS_NEON 1
#else
#    define ZL_HAS_NEON 0
#endif

#if ZL_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
#    define ZL_HAS_NEON_DOTPROD 1
#else
#    define ZL_HAS_NEON_DOTPROD 0
#endif

#if ZL_HAS_NEON_DOTPROD && defined(__ARM_FEATURE_MATMUL_INT8)
#    define ZL_HAS_NEON_I8MM 1
#else
#    define ZL_HAS_NEON_I8MM 0
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) \
        && ZL_HAS_INCLUDE(<arm_sve.h>)
#    define ZL_HAS_SVE 1
#else
#    define ZL_HAS_SVE 0
#endif

#if ZL_HAS_SVE && defined(__ARM_FEATURE_SVE2)
#    define ZL_HAS_SVE2 1
#else
#    define ZL_HAS_SVE2 0
#endif

#if ZL_HAS_SVE2 && defined(__ARM_FEATURE_SVE2_BITPERM)
#    define ZL_HAS_SVE2_BITPERM 1
#else
#    define ZL_HAS_SVE2_BITPERM 0
#endif

#if defined(__AVX2__)
#    define ZL_HAS_AVX2 1
#    if defined(__GNUC__) || defined(__clang__)
#        include <immintrin.h>
#    endif
#    if defined(_MSC_VER)
#        include <intrin.h>
// MSVC doesn't define __m256i_u and __m128i_u types like GCC/Clang do.
// These types are used for unaligned SIMD loads/stores. In MSVC, we just
// use the regular __m256i and __m128i types for both aligned and unaligned.
typedef __m256i __m256i_u;
typedef __m128i __m128i_u;
#    endif
#else
#    define ZL_HAS_AVX2 0
#endif

#if defined(__SSSE3__)
#    define ZL_HAS_SSSE3 1
#    include <immintrin.h>
#else
#    define ZL_HAS_SSSE3 0
#endif

#if defined(__SSE4_2__)
#    define ZL_HAS_SSE42 1
#    include <immintrin.h>
#else
#    define ZL_HAS_SSE42 0
#endif

#ifndef ZL_FALLTHROUGH
#    if ZL_HAS_C_ATTRIBUTE(fallthrough)
#        define ZL_FALLTHROUGH [[fallthrough]]
#    elif ZL_HAS_CPP_ATTRIBUTE(fallthrough)
#        define ZL_FALLTHROUGH [[fallthrough]]
#    elif ZL_HAS_ATTRIBUTE(__fallthrough__)
/* Leading semicolon is to satisfy gcc-11 with -pedantic. Without the semicolon
 * gcc complains about: a label can only be part of a statement and a
 * declaration is not a statement.
 */
#        define ZL_FALLTHROUGH \
            ;                  \
            __attribute__((__fallthrough__))
#    else
#        define ZL_FALLTHROUGH
#    endif
#endif

// Detect IEEE 754 floating point support.
// Apple doesn't define __STDC_IEC_559__, but supports IEEE 754.
// MinGW doesn't define __STDC_IEC_559__, but supports IEEE 754.
// MSVC on x86/x64 supports IEEE 754.
#if (defined(__STDC_IEC_559__) && __STDC_IEC_559__) \
        || (defined(__STDC_IEC_60559_BFP__)         \
            && __STDC_IEC_60559_BFP__ >= 202311L)   \
        || defined(__APPLE__) || defined(__MINGW32__) || defined(_MSC_VER)
#    define ZL_HAS_IEEE_754 1
#else
#    define ZL_HAS_IEEE_754 0
#endif

#define ZL_PRAGMA(x) _Pragma(#x)

#ifdef __clang__
#    define ZL_UNROLL_LOOP(n) ZL_PRAGMA(clang loop unroll_count(n))
#elif defined(__GNUC__)
#    define ZL_UNROLL_LOOP(n) ZL_PRAGMA(GCC unroll n)
#else
#    define ZL_UNROLL_LOOP(n)
#endif

ZL_END_C_DECLS

#endif // ZS_COMMON_PORTABILITY_H
