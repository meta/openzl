// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * \file
 *
 * This file defines and configures a number of macros that can be used to
 * validate expectations, and print useful information when those expectations
 * are violated.
 */

#ifndef ZSTRONG_COMMON_ASSERTION_H
#define ZSTRONG_COMMON_ASSERTION_H

#include "openzl/zl_errors.h"
#include "openzl/zl_macro_helpers.h"

#include "openzl/common/debug_level.h"
#include "openzl/common/logging.h"
#include "openzl/shared/portability.h"

/**
 * The ZL_STATIC_ASSERT macro evaluates its argument at compile-time, and
 * triggers a compilation error if the argument evaluates to false. This
 * process leaves no runtime artifact.
 *
 * The argument expression must be a constexpr.
 */
#define ZL_STATIC_ASSERT(expr, msg) ZL_STATIC_ASSERT_IMPL(expr, msg)

/**
 * ASSERTIONS:
 *
 * We have a rich assortment of assertion macros to choose from.
 *
 * These macros detect when an expectation is violated, print an error message,
 * and then abort the process.
 *
 * ASSERT vs REQUIRE:
 *
 * A key distinction is drawn between macros with "ASSERT" in the name and
 * macros with "REQUIRE" in the name. "ASSERT" macros mirror the C stdlib's
 * assert() function in that they can be disabled for production builds. By
 * default, this is driven in the same way, by the presence or absence of the
 * NDEBUG macro, although it can be overridden. In comparison, "REQUIRE" macros
 * are active even in production builds. (Ok, technically they can be disabled
 * with `-DZS_ENABLE_REQUIRE=0`, but who would do that??)
 *
 * Note that when an assertion of either variety is disabled, its arguments are
 * not evaluated (although they are still referenced, to protect against
 * surprise `-Wunused` spam).
 *
 * ASSERT-style calls are therefore "cheaper", in that they aren't present in
 * production builds. Which means they should be preferred when sufficient.
 * But on the other hand, they don't actually protect production builds. So
 * they should only be used to enforce checks that validate the developer's
 * understanding of program invariants. They should not be used to enforce
 * things that are known to be possible to fail.
 *
 * By process of elimination, REQUIRE-style calls represent things that could
 * actually go wrong in production. But ZL_REQUIRE() and friends are very heavy
 * weapons for such things, since they abort the process. In practice, it seems
 * like most REQUIRE-style callsites would be better represented by checks
 * that, if they fail, return an error. REQUIRE-style calls are therefore best
 * understood as a transitional tool marking such situations while we don't
 * have good error handling conventions and tools developed. At some point, we
 * should plan to migrate the majority of them over to some pattern that
 * propagates errors up the call chain rather than just aborting.
 *
 * ARGUMENTS:
 *
 * The other big dimension along which variants exist is how many arguments are
 * being evaluated:
 *
 *        | Assert                      | Require
 * -------+-----------------------------+------------------------------
 * 0-args | ZL_ASSERT_FAIL(...)         | ZL_REQUIRE_FAIL(...)
 * 1-arg  | ZL_ASSERT(expr, ...)        | ZL_REQUIRE(expr, ...)
 * 2-args | ZS_ASSERT_??(lhs, rhs, ...) | ZS_REQUIRE_??(lhs, rhs, ...)
 *
 * The zero-arg variants unconditionally fail, and are useful for enforcing that
 * a particular codepath is unreachable. In particular, their implementations
 * are unconditional, which means that the compiler will correctly understand
 * them in the context of evaluating switch statements for `-Wfallthrough`,
 * which some compilers fail to do with `ZL_REQUIRE(false)`.
 *
 * The single-arg variants evaluate the provided expression and fire if it
 * evaluates to false.
 *
 * The two-arg variants evaluate a binary operator expression and fire if it
 * evaluates to false, where the macro name's suffix (`??`) represents a binary
 * comparator and is one of `EQ` (==), `NE` (!=), `GE` (>=), `LE` (<=),
 * `GT` (>), `LT` (<), `AND` (&&), and `OR` (||). The two-arg variants'
 * arguments are cast to long long ints.
 *
 * These variants are provided because they conveniently print out the operands
 * when they fail, rather than just saying the expression evaluated to false.
 *
 * Additionally, a helper `NN` is provided which only takes one argument (plus
 * optional message params), which represents "Not Null" and resolves to
 * `..._NE(expr, NULL)`.
 *
 * The additional arguments to all of the macro variants will be forwarded to
 * a printf-like call, to display a meaningful message to the user if the
 * assertion fails. In the case of the 0-arg variants, a message is required
 * (since ISO C99 does not allow empty varargs in macros). For all other
 * variants, it is optional.
 */
#define ZL_ASSERT(...) ZS_MACRO_PAD1(ZL_ASSERT_UNARY_IMPL, __VA_ARGS__)
#define ZL_REQUIRE(...) ZS_MACRO_PAD1(ZL_REQUIRE_UNARY_IMPL, __VA_ARGS__)

#define ZL_ASSERT_FAIL(...) ZL_ASSERT_NULLARY_IMPL(__VA_ARGS__)
#define ZL_REQUIRE_FAIL(...) ZL_REQUIRE_NULLARY_IMPL(__VA_ARGS__)

#define ZL_ASSERT_EQ(...) ZS_MACRO_PAD3(ZL_ASSERT_BINARY_IMPL, ==, __VA_ARGS__)
#define ZL_ASSERT_NE(...) ZS_MACRO_PAD3(ZL_ASSERT_BINARY_IMPL, !=, __VA_ARGS__)
#define ZL_ASSERT_GE(...) ZS_MACRO_PAD3(ZL_ASSERT_BINARY_IMPL, >=, __VA_ARGS__)
#define ZL_ASSERT_LE(...) ZS_MACRO_PAD3(ZL_ASSERT_BINARY_IMPL, <=, __VA_ARGS__)
#define ZL_ASSERT_GT(...) ZS_MACRO_PAD3(ZL_ASSERT_BINARY_IMPL, >, __VA_ARGS__)
#define ZL_ASSERT_LT(...) ZS_MACRO_PAD3(ZL_ASSERT_BINARY_IMPL, <, __VA_ARGS__)
#define ZL_ASSERT_AND(...) ZS_MACRO_PAD3(ZL_ASSERT_BINARY_IMPL, &&, __VA_ARGS__)
#define ZL_ASSERT_OR(...) ZS_MACRO_PAD3(ZL_ASSERT_BINARY_IMPL, ||, __VA_ARGS__)

#define ZL_ASSERT_NN(...) ZS_MACRO_PAD1(ZL_ASSERT_NN_IMPL, __VA_ARGS__)
#define ZL_ASSERT_NULL(...) ZS_MACRO_PAD1(ZL_ASSERT_NULL_IMPL, __VA_ARGS__)

// Note : below macros seem to transparently cast large positive numbers as
// negative values (??)
#define ZL_REQUIRE_EQ(...) \
    ZS_MACRO_PAD3(ZL_REQUIRE_BINARY_IMPL, ==, __VA_ARGS__)
#define ZL_REQUIRE_NE(...) \
    ZS_MACRO_PAD3(ZL_REQUIRE_BINARY_IMPL, !=, __VA_ARGS__)
#define ZL_REQUIRE_GE(...) \
    ZS_MACRO_PAD3(ZL_REQUIRE_BINARY_IMPL, >=, __VA_ARGS__)
#define ZL_REQUIRE_LE(...) \
    ZS_MACRO_PAD3(ZL_REQUIRE_BINARY_IMPL, <=, __VA_ARGS__)
#define ZL_REQUIRE_GT(...) ZS_MACRO_PAD3(ZL_REQUIRE_BINARY_IMPL, >, __VA_ARGS__)
#define ZL_REQUIRE_LT(...) ZS_MACRO_PAD3(ZL_REQUIRE_BINARY_IMPL, <, __VA_ARGS__)
#define ZL_REQUIRE_AND(...) \
    ZS_MACRO_PAD3(ZL_REQUIRE_BINARY_IMPL, &&, __VA_ARGS__)
#define ZL_REQUIRE_OR(...) \
    ZS_MACRO_PAD3(ZL_REQUIRE_BINARY_IMPL, ||, __VA_ARGS__)

#define ZL_REQUIRE_NN(...) ZS_MACRO_PAD1(ZL_REQUIRE_NN_IMPL, __VA_ARGS__)
#define ZL_REQUIRE_NULL(...) ZS_MACRO_PAD1(ZL_REQUIRE_NULL_IMPL, __VA_ARGS__)

// There are also ZL_ASSERT_SUCCESS() and ZL_REQUIRE_SUCCESS() macroes defined
// in the error headers which check whether the provided ZL_RESULT_OF(T)
// contains an error.

/**
 * Unlike ZL_ASSERT(), which becomes a no-op in optimized builds, ZL_ABORT()
 * never returns. By default, it kills your process.
 *
 * An application which must not be killed, such as an R package, can build
 * OpenZL with -DZL_ABORT_FUNC=name to call its own `void name(void)` instead.
 * That function must not return either.
 */
#include <stdlib.h>
#ifdef ZL_ABORT_FUNC
ZL_BEGIN_C_DECLS
ZL_NORETURN void ZL_ABORT_FUNC(void);
ZL_END_C_DECLS
#    define ZL_ABORT() ZL_ABORT_FUNC()
#else
#    define ZL_ABORT() abort()
#endif

/************************************************************
 * These macros control whether these features are enabled. *
 ************************************************************/

#ifndef ZL_ENABLE_STATIC_ASSERT
#    define ZL_ENABLE_STATIC_ASSERT (ZL_DBG_LVL >= 1)
#endif

#ifndef ZL_ENABLE_ASSERT
#    define ZL_ENABLE_ASSERT (ZL_DBG_LVL >= 3)
#endif

#ifndef ZL_ENABLE_REQUIRE
#    define ZL_ENABLE_REQUIRE 1
#endif

/**********************************
 * Implementation details follow. *
 **********************************/

#define ZL_ASSERT_NN_IMPL(expr, ...) \
    ZL_ASSERT_BINARY_IMPL(           \
            !=, (const char*)(const void*)expr, (const char*)0, __VA_ARGS__)
#define ZL_REQUIRE_NN_IMPL(expr, ...) \
    ZL_REQUIRE_BINARY_IMPL(           \
            !=, (const char*)(const void*)expr, (const char*)0, __VA_ARGS__)

#define ZL_ASSERT_NULL_IMPL(expr, ...) \
    ZL_ASSERT_BINARY_IMPL(             \
            ==, (const char*)(const void*)expr, (const char*)0, __VA_ARGS__)
#define ZL_REQUIRE_NULL_IMPL(expr, ...) \
    ZL_REQUIRE_BINARY_IMPL(             \
            ==, (const char*)(const void*)expr, (const char*)0, __VA_ARGS__)

#if ZL_ENABLE_ASSERT
#    define ZL_ASSERT_NULLARY_IMPL(...) \
        ZL_ASSERTION_NULLARY_IMPL("Assertion", __VA_ARGS__)
#else
#    define ZL_ASSERT_NULLARY_IMPL(...) \
        ZL_LOG_IF(0, ALWAYS, "Error: " __VA_ARGS__)
#endif // ZL_ENABLE_ASSERT

#if ZL_ENABLE_REQUIRE
#    define ZL_REQUIRE_NULLARY_IMPL(...) \
        ZL_ASSERTION_NULLARY_IMPL("Requirement", __VA_ARGS__)
#else
#    define ZL_REQUIRE_NULLARY_IMPL(...) \
        ZL_LOG_IF(0, ALWAYS, "Error: " __VA_ARGS__)
#endif // ZL_ENABLE_REQUIRE

#if ZL_ENABLE_ASSERT
#    define ZL_ASSERT_UNARY_IMPL(expr, ...) \
        ZL_ASSERTION_UNARY_IMPL(expr, "Assertion", __VA_ARGS__)
#else
#    define ZL_ASSERT_UNARY_IMPL(expr, ...) \
        ZL_LOG_IF(0 && (expr), ALWAYS, "Error: " __VA_ARGS__)
#endif // ZL_ENABLE_ASSERT

#if ZL_ENABLE_REQUIRE
#    define ZL_REQUIRE_UNARY_IMPL(expr, ...) \
        ZL_ASSERTION_UNARY_IMPL(expr, "Requirement", __VA_ARGS__)
#else
#    define ZL_REQUIRE_UNARY_IMPL(expr, ...) \
        ZL_LOG_IF(0 && (expr), ALWAYS, "Error: " __VA_ARGS__)
#endif // ZL_ENABLE_REQUIRE

#if ZL_ENABLE_ASSERT
#    define ZL_ASSERT_BINARY_IMPL(op, lhs, rhs, ...) \
        ZL_ASSERTION_BINARY_IMPL("Assertion", op, lhs, rhs, __VA_ARGS__)
#else
#    define ZL_ASSERT_BINARY_IMPL(op, lhs, rhs, ...) \
        ZL_LOG_IF(0 && ((lhs)op(rhs)), ALWAYS, "Error: " __VA_ARGS__)
#endif // ZL_ENABLE_ASSERT

#if ZL_ENABLE_REQUIRE
#    define ZL_REQUIRE_BINARY_IMPL(op, lhs, rhs, ...) \
        ZL_ASSERTION_BINARY_IMPL("Requirement", op, lhs, rhs, __VA_ARGS__)
#else
#    define ZL_REQUIRE_BINARY_IMPL(op, lhs, rhs, ...) \
        ZL_LOG_IF(0 && ((lhs)op(rhs)), ALWAYS, "Error: " __VA_ARGS__)
#endif // ZL_ENABLE_REQUIRE

// Zero arg impl
#define ZL_ASSERTION_NULLARY_IMPL(req_str, ...)                \
    do {                                                       \
        ZL_LOG(ALWAYS, "%s unconditionally failed.", req_str); \
        ZL_LOG_IFNONEMPTY(ALWAYS, "Error: ", __VA_ARGS__);     \
        ZL_ABORT();                                            \
    } while (0)

// Single arg impl
#define ZL_ASSERTION_UNARY_IMPL(expr, req_str, ...)            \
    do {                                                       \
        if (ZL_UNLIKELY(!(expr))) {                            \
            ZL_LOG(ALWAYS, "%s `%s' failed", req_str, #expr);  \
            ZL_LOG_IFNONEMPTY(ALWAYS, "Error: ", __VA_ARGS__); \
            ZL_ABORT();                                        \
        }                                                      \
    } while (0)

// Controls whether to use the verbose non-standards-compliant binary assertion
// implementation (if enabled) or the fallback standards-compliant version (if
// disabled).
#ifndef ZL_ENABLE_ASSERTION_ARG_PRINTING
#    if defined(__GNUC__)
#        define ZL_ENABLE_ASSERTION_ARG_PRINTING 1
#    else
#        define ZL_ENABLE_ASSERTION_ARG_PRINTING 0
#    endif
#endif

// Binary arg impl
#if ZL_ENABLE_ASSERTION_ARG_PRINTING
/* We can only evaluate the lhs and rhs arguments once (they might have
 * side-effects). But we use their values twice. So we need to store the
 * results of their evaluation in temporary variables. But it's very difficult
 * to know what type those temporaries should have.
 *
 * So we do this weird typeof(...) dance to get temporaries that preserve or
 * exceed the signedness / precision / pointerness of the arguments. When the
 * the arguments are arithmetic types, the `L + (L - R)` construct resolves to
 * a suitably promoted type. When the arguments are pointers it just resoves to
 * the pointer type (`T* + (T* - T*)` becomes `T* + ptrdiff_t`, which is `T*`).
 *
 * Annoyingly, we can't use lhs and rhs directly in the typeof expression,
 * because clang has a warning (-Wnull-pointer-subtraction) that fires if one
 * of the arguments is a null pointer. So we use surrogate variables.
 */
#    define ZL_ASSERTION_BINARY_IMPL(req_str, op, lhs, rhs, ...)                                   \
        do {                                                                                       \
            const __typeof__(lhs) __dbg_lhs_type = (__typeof__(lhs))0;                             \
            const __typeof__(rhs) __dbg_rhs_type = (__typeof__(rhs))0;                             \
            const __typeof__((__dbg_lhs_type) + ((__dbg_lhs_type) - (__dbg_rhs_type))) __dbg_lhs = \
                    (__typeof__((__dbg_lhs_type) + ((__dbg_lhs_type) - (__dbg_rhs_type))))(lhs);   \
            const __typeof__((__dbg_lhs_type) + ((__dbg_lhs_type) - (__dbg_rhs_type))) __dbg_rhs = \
                    (__typeof__((__dbg_lhs_type) + ((__dbg_lhs_type) - (__dbg_rhs_type))))(rhs);   \
            if (0 && ((lhs)op(rhs))) /* enforce type comparability */ {                            \
            }                                                                                      \
            if (ZL_UNLIKELY(!(__dbg_lhs op __dbg_rhs))) {                                          \
                ZL_LOG(ALWAYS,                                                                     \
                       ZS_GENERIC_PRINTF_BUILD_FORMAT_2_ARG(                                       \
                               __dbg_lhs,                                                          \
                               "%s `%s %s %s' failed where:\n\tlhs = ",                            \
                               "\n\trhs = ",                                                       \
                               ""),                                                                \
                       req_str,                                                                    \
                       #lhs,                                                                       \
                       #op,                                                                        \
                       #rhs,                                                                       \
                       ZS_GENERIC_PRINTF_CAST(__dbg_lhs),                                          \
                       ZS_GENERIC_PRINTF_CAST(__dbg_rhs));                                         \
                ZL_LOG_IFNONEMPTY(ALWAYS, "Error: ", __VA_ARGS__);                                 \
                ZL_ABORT();                                                                        \
            }                                                                                      \
        } while (0)
#else
/* Otherwise, if we don't have typeof(), it's impossible to store the results
 * safely. So we just evaluate and use them directly in the comparison, and
 * give up the ability to print their values.
 */
#    define ZL_ASSERTION_BINARY_IMPL(req_str, op, lhs, rhs, ...)   \
        do {                                                       \
            if (ZL_UNLIKELY(!((lhs)op(rhs)))) {                    \
                ZL_LOG(ALWAYS,                                     \
                       "%s `%s %s %s' failed",                     \
                       req_str,                                    \
                       #lhs,                                       \
                       #op,                                        \
                       #rhs);                                      \
                ZL_LOG_IFNONEMPTY(ALWAYS, "Error: ", __VA_ARGS__); \
                ZL_ABORT();                                        \
            }                                                      \
        } while (0)
#endif

/* Static Assertions Implementation */

#if ZL_ENABLE_STATIC_ASSERT
#    if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
/* Use the C11 _Static_assert keyword directly rather than the static_assert
 * macro from <assert.h>, which is optional and missing on some platforms.
 */
#        define ZL_STATIC_ASSERT_IMPL(expr, msg) _Static_assert(expr, msg)
#    else

/* This fallback static assert idiom was copied from the following URL, where
 * it was released under a GNU All-Permissive License:
 * http://www.pixelbeat.org/programming/gcc/static_assert.html
 */

#        define ZL_STATIC_ASSERT_CONCAT_INNER(a, b) a##b
#        define ZL_STATIC_ASSERT_CONCAT(a, b) \
            ZL_STATIC_ASSERT_CONCAT_INNER(a, b)
#        ifdef __COUNTER__
#            define ZL_STATIC_ASSERT_IMPL(expr, msg)            \
                enum {                                          \
                    ZL_STATIC_ASSERT_CONCAT(                    \
                            static_assert_,                     \
                            ZL_STATIC_ASSERT_CONCAT(            \
                                    __LINE__,                   \
                                    ZL_STATIC_ASSERT_CONCAT(    \
                                            _, __COUNTER__))) = \
                            1 / (int)(!!(expr))                 \
                }
#        else
/* This can't be used twice on the same line so ensure if using in headers
 * that the headers are not included twice (by wrapping in #ifndef...#endif)
 * Note it doesn't cause an issue when used on same line of separate modules
 * compiled with gcc -combine -fwhole-program.
 */
#            define ZL_STATIC_ASSERT_IMPL(expr, msg)                    \
                enum {                                                  \
                    ZL_STATIC_ASSERT_CONCAT(static_assert_, __LINE__) = \
                            1 / (int)(!!(expr))                         \
                }
#        endif

#    endif
#else
#    define ZL_STATIC_ASSERT_IMPL(expr, msg)
#endif // ZL_ENABLE_STATIC_ASSERT

#endif // ZSTRONG_COMMON_ASSERTION_H
