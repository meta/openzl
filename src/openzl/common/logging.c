// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/common/logging.h"

#include <stdarg.h> // va_list, va_start, va_end

int ZL_g_logLevel = ZL_LOG_LVL;

/* By default, logs are written to stderr.
 * An application can build OpenZL with -DZL_LOG_VPRINTF_FUNC=name
 * to send them to its own `void name(const char* fmt, va_list args)` instead,
 * such as -DZL_LOG_VPRINTF_FUNC=REvprintf for an R package. */
#ifdef ZL_LOG_VPRINTF_FUNC
void ZL_LOG_VPRINTF_FUNC(const char* fmt, va_list args);
#else
static void ZL_stderrVPrintf(const char* fmt, va_list args)
{
    vfprintf(stderr, fmt, args);
}
#    define ZL_LOG_VPRINTF_FUNC ZL_stderrVPrintf
#endif

static void ZL_logPrintf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    ZL_LOG_VPRINTF_FUNC(fmt, args);
    va_end(args);
}

// Note: requires -Wno-format-nonliteral to build without a warning.

void ZL_LOG_func(
        const char* file,
        const char* func,
        int line,
        const char* fmt,
        ...)
{
    va_list args;
    va_start(args, fmt);
    ZL_VLOG_func(file, func, line, fmt, args);
    va_end(args);
}

void ZL_LOG_func_if_nonempty(
        const char* file,
        const char* func,
        int line,
        const char* prefix,
        const char* fmt,
        ...)
{
    va_list args;
    va_start(args, fmt);
    ZL_VLOG_func_if_nonempty(file, func, line, prefix, fmt, args);
    va_end(args);
}

void ZL_VLOG_func(
        const char* file,
        const char* func,
        int line,
        const char* fmt,
        va_list args)
{
    (void)func;
    ZL_logPrintf("%s:%d: ", file, line);
    ZL_LOG_VPRINTF_FUNC(fmt, args);
    ZL_logPrintf("\n");
}

void ZL_VLOG_func_if_nonempty(
        const char* file,
        const char* func,
        int line,
        const char* prefix,
        const char* fmt,
        va_list args)
{
    (void)func;
    va_list args_copy;
    va_copy(args_copy, args);
    int len = vsnprintf(NULL, 0, fmt, args_copy);
    va_end(args_copy);
    if (len > 0) {
        ZL_logPrintf("%s:%d: %s", file, line, prefix);
        ZL_LOG_VPRINTF_FUNC(fmt, args);
        ZL_logPrintf("\n");
    }
}

// Note: requires -Wno-format-nonliteral to build without a warning.
void ZL_RLOG_func(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    ZL_VRLOG_func(fmt, args);
    va_end(args);
}

void ZL_VRLOG_func(const char* fmt, va_list args)
{
    ZL_LOG_VPRINTF_FUNC(fmt, args);
}
