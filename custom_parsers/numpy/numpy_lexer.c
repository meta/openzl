// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/numpy/numpy_lexer.h"

#include <string.h>

#include "custom_parsers/pickle_lexer.h"
#include "custom_parsers/zip_lexer.h"
#include "openzl/shared/mem.h"

#define NP_MAX_DEPTH 32

static const char kMagic[6] = { '\x93', 'N', 'U', 'M', 'P', 'Y' };

typedef struct {
    const char* ptr;
    const char* end;
} NP_Cursor;

/* ------------------------------------------------------------------------
 * Python literal scanning, for the subset that headers use. Strings are not
 * unescaped: keys and descrs are compared on their raw bytes.
 * ------------------------------------------------------------------------ */

static void NP_skipSpaces(NP_Cursor* c)
{
    while (c->ptr < c->end
           && (*c->ptr == ' ' || *c->ptr == '\t' || *c->ptr == '\n'
               || *c->ptr == '\r')) {
        ++c->ptr;
    }
}

static bool NP_consume(NP_Cursor* c, char expected)
{
    NP_skipSpaces(c);
    if (c->ptr < c->end && *c->ptr == expected) {
        ++c->ptr;
        return true;
    }
    return false;
}

static bool NP_consumeWord(NP_Cursor* c, const char* word)
{
    NP_skipSpaces(c);
    const size_t size = strlen(word);
    if ((size_t)(c->end - c->ptr) < size || memcmp(c->ptr, word, size) != 0) {
        return false;
    }
    c->ptr += size;
    return true;
}

/// Reads a string quoted with ' or "; on success, (*str, *size) is its raw
/// content.
static bool NP_readString(NP_Cursor* c, const char** str, size_t* size)
{
    NP_skipSpaces(c);
    if (c->ptr >= c->end || (*c->ptr != '\'' && *c->ptr != '"')) {
        return false;
    }
    const char quote        = *c->ptr++;
    const char* const begin = c->ptr;
    while (c->ptr < c->end) {
        const char ch = *c->ptr;
        if (ch == quote) {
            *str  = begin;
            *size = (size_t)(c->ptr - begin);
            ++c->ptr;
            return true;
        }
        if (ch == '\n') {
            return false;
        }
        if (ch == '\\') {
            if (c->end - c->ptr < 2) {
                return false;
            }
            ++c->ptr;
        }
        ++c->ptr;
    }
    return false;
}

static bool NP_strEquals(const char* str, size_t size, const char* literal)
{
    return size == strlen(literal) && memcmp(str, literal, size) == 0;
}

static bool NP_isDigit(char ch)
{
    return ch >= '0' && ch <= '9';
}

/// Reads a non-negative integer that fits in 63 bits, with the L suffix of the
/// longs written by Python 2.
static bool NP_readUInt(NP_Cursor* c, uint64_t* value)
{
    NP_skipSpaces(c);
    if (c->ptr >= c->end || !NP_isDigit(*c->ptr)) {
        return false;
    }
    uint64_t v = 0;
    while (c->ptr < c->end && NP_isDigit(*c->ptr)) {
        const uint64_t digit = (uint64_t)(*c->ptr - '0');
        if (v > (UINT64_C(0x7FFFFFFFFFFFFFFF) - digit) / 10) {
            return false;
        }
        v = v * 10 + digit;
        ++c->ptr;
    }
    if (c->ptr < c->end && *c->ptr == 'L') {
        ++c->ptr;
    }
    *value = v;
    return true;
}

static bool NP_isLiteralChar(char ch)
{
    return NP_isDigit(ch) || (ch >= 'a' && ch <= 'z')
            || (ch >= 'A' && ch <= 'Z') || ch == '-' || ch == '+' || ch == '.'
            || ch == '_';
}

/// Skips a string, a bracketed container or a bare literal. Lenient, as
/// skipped values are never interpreted.
static bool NP_skipValue(NP_Cursor* c, int depth)
{
    if (depth > NP_MAX_DEPTH) {
        return false;
    }
    NP_skipSpaces(c);
    if (c->ptr >= c->end) {
        return false;
    }
    const char open = *c->ptr;
    if (open == '\'' || open == '"') {
        const char* str;
        size_t size;
        return NP_readString(c, &str, &size);
    }
    const char close = open == '(' ? ')'
            : open == '['          ? ']'
            : open == '{'          ? '}'
                                   : 0;
    if (close == 0) {
        const char* const begin = c->ptr;
        while (c->ptr < c->end && NP_isLiteralChar(*c->ptr)) {
            ++c->ptr;
        }
        return c->ptr > begin;
    }
    ++c->ptr;
    for (;;) {
        if (NP_consume(c, close)) {
            return true;
        }
        if (!NP_skipValue(c, depth + 1)) {
            return false;
        }
        // ',' separates items, ':' keys from values
        if (!NP_consume(c, ',') && !NP_consume(c, ':')) {
            NP_skipSpaces(c);
            if (c->ptr >= c->end || *c->ptr != close) {
                return false;
            }
        }
    }
}

/* ------------------------------------------------------------------------
 * .npy header
 * ------------------------------------------------------------------------ */

/// Reads a little-endian descr such as '<f4', '|u1' or '<M8[ns]' into the kind
/// and item size of @p t. Others leave the kind at 0: big-endian ('>') and
/// native ('=') byte orders, types without a size, and objects, which are
/// pickled whatever size their descr states ('|O', or '|O8' in older releases).
static void NP_parseDescr(const char* s, size_t size, ZL_NumPyToken* t)
{
    t->kind     = 0;
    t->itemSize = 0;
    if (size < 3 || (s[0] != '<' && s[0] != '|')) {
        return;
    }
    const char kind = s[1];
    if (!((kind >= 'a' && kind <= 'z') || (kind >= 'A' && kind <= 'Z'))
        || kind == 'O') {
        return;
    }
    size_t pos        = 2;
    uint64_t itemSize = 0;
    while (pos < size && NP_isDigit(s[pos])) {
        itemSize = itemSize * 10 + (uint64_t)(s[pos] - '0');
        if (itemSize > UINT32_MAX) {
            return;
        }
        ++pos;
    }
    // unit of datetimes and timedeltas
    if (pos < size && s[pos] == '[') {
        while (pos < size && s[pos] != ']') {
            ++pos;
        }
        if (pos == size) {
            return;
        }
        ++pos;
    }
    if (pos != size || itemSize == 0) {
        return;
    }
    // unicode strings count characters of 4 bytes
    if (kind == 'U') {
        itemSize *= 4;
        if (itemSize > UINT32_MAX) {
            return;
        }
    }
    t->kind     = kind;
    t->itemSize = (uint32_t)itemSize;
}

/// Reads a tuple of dimensions. More than ZL_NUMPY_MAX_DIMS are counted in
/// @p ndims but not stored.
static bool NP_readShape(NP_Cursor* c, ZL_NumPyToken* t, uint64_t* ndims)
{
    if (!NP_consume(c, '(')) {
        return false;
    }
    *ndims = 0;
    for (;;) {
        if (NP_consume(c, ')')) {
            return true;
        }
        uint64_t dim;
        if (!NP_readUInt(c, &dim)) {
            return false;
        }
        if (*ndims < ZL_NUMPY_MAX_DIMS) {
            t->shape[*ndims] = dim;
        }
        ++*ndims;
        if (NP_consume(c, ')')) {
            return true;
        }
        if (!NP_consume(c, ',')) {
            return false;
        }
    }
}

/// Reads the header dict into @p t. Unknown keys are skipped; descrs that are
/// not strings, such as the lists of structured types, leave the kind at 0.
static bool NP_readHeader(NP_Cursor* c, ZL_NumPyToken* t)
{
    bool hasDescr  = false;
    bool hasOrder  = false;
    bool hasShape  = false;
    uint64_t ndims = 0;
    if (!NP_consume(c, '{')) {
        return false;
    }
    while (!NP_consume(c, '}')) {
        const char* key;
        size_t keySize;
        if (!NP_readString(c, &key, &keySize) || !NP_consume(c, ':')) {
            return false;
        }
        if (NP_strEquals(key, keySize, "descr")) {
            const NP_Cursor start = *c;
            const char* descr;
            size_t descrSize;
            if (NP_readString(c, &descr, &descrSize)) {
                NP_parseDescr(descr, descrSize, t);
            } else {
                *c = start;
                if (!NP_skipValue(c, 1)) {
                    return false;
                }
                t->kind = 0;
            }
            hasDescr = true;
        } else if (NP_strEquals(key, keySize, "fortran_order")) {
            if (NP_consumeWord(c, "True")) {
                t->fortranOrder = true;
            } else if (NP_consumeWord(c, "False")) {
                t->fortranOrder = false;
            } else {
                return false;
            }
            hasOrder = true;
        } else if (NP_strEquals(key, keySize, "shape")) {
            if (!NP_readShape(c, t, &ndims)) {
                return false;
            }
            hasShape = true;
        } else if (!NP_skipValue(c, 1)) {
            return false;
        }
        if (!NP_consume(c, ',')) {
            if (!NP_consume(c, '}')) {
                return false;
            }
            break;
        }
    }
    NP_skipSpaces(c);
    if (c->ptr != c->end || !hasDescr || !hasOrder || !hasShape) {
        return false;
    }
    if (ndims > ZL_NUMPY_MAX_DIMS) {
        t->kind = 0;
        ndims   = 0;
    }
    t->ndims = (uint32_t)ndims;
    return true;
}

/// Reads the magic and the header of the .npy file at the start of @p src.
/// @returns false if @p src does not start with a valid .npy header, else
/// fills @p headerSize with the size of the magic and the header, and @p array
/// with the description of the array, without its position.
static bool NP_readNpy(
        const char* src,
        size_t srcSize,
        size_t* headerSize,
        ZL_NumPyToken* array)
{
    if (srcSize < 10 || memcmp(src, kMagic, sizeof(kMagic)) != 0) {
        return false;
    }
    const uint8_t major = (uint8_t)src[6];
    size_t start;
    size_t size;
    if (major == 1) {
        start = 10;
        size  = ZL_readLE16(src + 8);
    } else if (major == 2 || major == 3) {
        if (srcSize < 12) {
            return false;
        }
        start = 12;
        size  = ZL_readLE32(src + 8);
    } else {
        return false;
    }
    if (size > ZL_NUMPY_MAX_HEADER_SIZE || size > srcSize - start) {
        return false;
    }
    memset(array, 0, sizeof(*array));
    array->type = ZL_NumPyTokenType_array;
    NP_Cursor c = { src + start, src + start + size };
    *headerSize = start + size;
    return NP_readHeader(&c, array);
}

/// @returns the byte size of prod(shape) elements of @p t, or UINT64_MAX when
/// it overflows.
static uint64_t NP_byteSize(const ZL_NumPyToken* t)
{
    uint64_t bytes = t->itemSize;
    for (uint32_t d = 0; d < t->ndims; ++d) {
        if (t->shape[d] != 0 && bytes > UINT64_MAX / t->shape[d]) {
            return UINT64_MAX;
        }
        bytes *= t->shape[d];
    }
    return bytes;
}

/* ------------------------------------------------------------------------
 * Tokens
 * ------------------------------------------------------------------------ */

typedef struct {
    ZL_NumPyToken* tokens; // NULL when counting
    size_t capacity;
    size_t nbTokens;
    ZL_NumPyTokenType lastType;
    ZL_ErrorContext* errCtx;
} NP_Out;

/// Appends @p t, or extends the last token with it when both are headers or
/// both are gaps. Tokens are emitted in input order, so they are adjacent.
static ZL_Report NP_emit(NP_Out* out, const ZL_NumPyToken* t)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(out->errCtx);
    if (t->size == 0) {
        return ZL_returnSuccess();
    }
    if (out->nbTokens > 0 && t->type != ZL_NumPyTokenType_array
        && t->type == out->lastType) {
        if (out->tokens != NULL) {
            out->tokens[out->nbTokens - 1].size += t->size;
        }
        return ZL_returnSuccess();
    }
    if (out->tokens != NULL) {
        ZL_ERR_IF_GE(
                out->nbTokens,
                out->capacity,
                parameter_invalid,
                "too many tokens for the capacity");
        out->tokens[out->nbTokens] = *t;
    }
    ++out->nbTokens;
    out->lastType = t->type;
    return ZL_returnSuccess();
}

static ZL_Report
NP_emitBytes(NP_Out* out, ZL_NumPyTokenType type, const char* ptr, size_t size)
{
    const ZL_NumPyToken t = { .ptr = ptr, .size = size, .type = type };
    return NP_emit(out, &t);
}

/// Emits the header of the .npy file in @p src, of @p headerSize bytes, then
/// its array, then a gap for the bytes past the array. When they are too few
/// for its shape, the array keeps them all, with a kind of 0.
static ZL_Report NP_emitNpy(
        NP_Out* out,
        const char* src,
        size_t srcSize,
        size_t headerSize,
        ZL_NumPyToken* array)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(out->errCtx);
    ZL_ERR_IF_ERR(NP_emitBytes(out, ZL_NumPyTokenType_header, src, headerSize));
    const size_t available = srcSize - headerSize;
    size_t size            = available;
    if (array->kind != 0) {
        const uint64_t expected = NP_byteSize(array);
        if (expected <= (uint64_t)available) {
            size = (size_t)expected;
        } else {
            array->kind = 0;
        }
    }
    array->ptr  = src + headerSize;
    array->size = size;
    ZL_ERR_IF_ERR(NP_emit(out, array));
    return NP_emitBytes(
            out,
            ZL_NumPyTokenType_gap,
            src + headerSize + size,
            available - size);
}

static ZL_Report NP_emitZipToken(NP_Out* out, const ZS2_ZipToken* zt)
{
    switch (zt->type) {
        case ZS2_ZipTokenType_CompressedData: {
            size_t headerSize;
            ZL_NumPyToken array;
            if (zt->compressionMethod == 0
                && NP_readNpy(zt->ptr, zt->size, &headerSize, &array)) {
                array.name     = zt->filename;
                array.nameSize = zt->filenameSize;
                return NP_emitNpy(out, zt->ptr, zt->size, headerSize, &array);
            }
            return NP_emitBytes(out, ZL_NumPyTokenType_gap, zt->ptr, zt->size);
        }
        case ZS2_ZipTokenType_LocalFileHeader:
        case ZS2_ZipTokenType_DataDescriptor:
        case ZS2_ZipTokenType_CentralDirectory:
        case ZS2_ZipTokenType_Zip64EndOfCentralDirectoryRecord:
        case ZS2_ZipTokenType_Zip64EndOfCentralDirectoryLocator:
        case ZS2_ZipTokenType_EndOfCentralDirectoryRecord:
            return NP_emitBytes(
                    out, ZL_NumPyTokenType_header, zt->ptr, zt->size);
        case ZS2_ZipTokenType_Unknown:
        default:
            return NP_emitBytes(out, ZL_NumPyTokenType_gap, zt->ptr, zt->size);
    }
}

static ZL_Report NP_lexNpz(NP_Out* out, const char* src, size_t srcSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(out->errCtx);
    ZS2_ZipLexer lexer;
    ZL_ERR_IF(
            ZL_isError(ZS2_ZipLexer_init(&lexer, src, srcSize)),
            graph_parser_malformedInput,
            "neither a .npy file, a pickle of protocol 2 to 5, "
            "nor a zip archive");
    ZL_ERR_IF_GT(
            ZS2_ZipLexer_numFiles(&lexer),
            ZL_NUMPY_MAX_MEMBERS,
            graph_parser_unhandledInput,
            "too many members");
    while (!ZS2_ZipLexer_finished(&lexer)) {
        ZS2_ZipToken zipTokens[32];
        const ZL_Report nbZipTokens = ZS2_ZipLexer_lex(&lexer, zipTokens, 32);
        ZL_ERR_IF(
                ZL_isError(nbZipTokens),
                graph_parser_malformedInput,
                "malformed zip archive");
        for (size_t i = 0; i < ZL_validResult(nbZipTokens); ++i) {
            ZL_ERR_IF_ERR(NP_emitZipToken(out, &zipTokens[i]));
        }
    }
    return ZL_returnSuccess();
}

/// Describes the pickled array @p pt into @p t, with a kind of 0 when its bytes
/// are not exactly prod(shape) little-endian elements of its dtype.
static void NP_describePickled(const ZL_PickleToken* pt, ZL_NumPyToken* t)
{
    char descr[32];
    t->kind         = 0;
    t->fortranOrder = pt->fortranOrder;
    if (pt->byteOrder == 0 || pt->typeStrSize >= sizeof(descr)
        || pt->ndims > ZL_NUMPY_MAX_DIMS || pt->ndims > ZL_PICKLE_MAX_DIMS) {
        return;
    }
    descr[0] = pt->byteOrder;
    for (size_t i = 0; i < pt->typeStrSize; ++i) {
        descr[1 + i] = pt->typeStr[i];
    }
    NP_parseDescr(descr, pt->typeStrSize + 1, t);
    t->ndims = pt->ndims;
    for (uint32_t d = 0; d < pt->ndims; ++d) {
        t->shape[d] = pt->shape[d];
    }
    if (t->kind != 0 && NP_byteSize(t) != (uint64_t)pt->size) {
        t->kind = 0;
    }
}

static ZL_Report NP_lexPickle(NP_Out* out, const char* src, size_t srcSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(out->errCtx);
    ZL_PickleLexer lexer;
    ZL_PickleLexer_init(&lexer, src, srcSize);
    while (!ZL_PickleLexer_finished(&lexer)) {
        ZL_PickleToken pickleTokens[8];
        const size_t nbPickleTokens =
                ZL_PickleLexer_lex(&lexer, pickleTokens, 8);
        for (size_t i = 0; i < nbPickleTokens; ++i) {
            const ZL_PickleToken* const pt = &pickleTokens[i];
            ZL_NumPyToken t = { .ptr = pt->ptr, .size = pt->size };
            switch (pt->type) {
                case ZL_PickleTokenType_array:
                    if (pt->size >= ZL_NUMPY_MIN_PICKLED_ARRAY_SIZE) {
                        t.type = ZL_NumPyTokenType_array;
                        NP_describePickled(pt, &t);
                    } else {
                        t.type = ZL_NumPyTokenType_header;
                    }
                    break;
                case ZL_PickleTokenType_opcodes:
                    t.type = ZL_NumPyTokenType_header;
                    break;
                case ZL_PickleTokenType_trailing:
                default:
                    t.type = ZL_NumPyTokenType_gap;
                    break;
            }
            ZL_ERR_IF_ERR(NP_emit(out, &t));
        }
    }
    return ZL_returnSuccess();
}

ZL_Report ZL_NumPy_lex(
        const void* src,
        size_t srcSize,
        ZL_NumPyToken* tokens,
        size_t tokensCapacity,
        ZL_ErrorContext* errCtx)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(errCtx);
    NP_Out out = {
        .tokens   = tokens,
        .capacity = tokensCapacity,
        .errCtx   = errCtx,
    };
    const char* const s = (const char*)src;
    if (srcSize >= sizeof(kMagic) && memcmp(s, kMagic, sizeof(kMagic)) == 0) {
        size_t headerSize;
        ZL_NumPyToken array;
        ZL_ERR_IF_NOT(
                NP_readNpy(s, srcSize, &headerSize, &array),
                graph_parser_malformedInput,
                "malformed .npy header");
        ZL_ERR_IF_ERR(NP_emitNpy(&out, s, srcSize, headerSize, &array));
    } else if (ZL_PickleLexer_isPickle(s, srcSize)) {
        ZL_ERR_IF_ERR(NP_lexPickle(&out, s, srcSize));
    } else {
        ZL_ERR_IF_ERR(NP_lexNpz(&out, s, srcSize));
    }
    return ZL_returnValue(out.nbTokens);
}
