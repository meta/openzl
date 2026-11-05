// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/pickle_lexer.h"

#include <string.h>

#include "openzl/common/assertion.h"
#include "openzl/shared/mem.h"

/// Longer tuples are never shapes, dtype states nor array states.
#define PK_MAX_TUPLE 16
/// Longer strings are not type strings of dtypes.
#define PK_MAX_TYPESTR 32
/// Memory orders of arrays, as _frombuffer takes them.
#define PK_MEMORY_ORDERS "CF"
/// Byte orders of dtypes, as their states hold them.
#define PK_BYTE_ORDERS "<>|"

typedef enum {
    PK_unknown = 0,
    PK_none,
    PK_int,
    PK_string,     // str or bytes
    PK_dtypeClass, // numpy.dtype
    PK_dtypeArgs,  // (typestr, flag, flag)
    PK_dtype,
    PK_dtypeState, // (version, byteorder, ...)
    PK_shape,      // tuple of non-negative integers
} PK_Type;

/// Opcodes, as named by the pickle module.
enum {
    PK_MARK             = '(',
    PK_STOP             = '.',
    PK_POP              = '0',
    PK_POP_MARK         = '1',
    PK_DUP              = '2',
    PK_FLOAT            = 'F',
    PK_INT              = 'I',
    PK_BININT           = 'J',
    PK_BININT1          = 'K',
    PK_LONG             = 'L',
    PK_BININT2          = 'M',
    PK_NONE             = 'N',
    PK_PERSID           = 'P',
    PK_BINPERSID        = 'Q',
    PK_REDUCE           = 'R',
    PK_STRING           = 'S',
    PK_BINSTRING        = 'T',
    PK_SHORT_BINSTRING  = 'U',
    PK_UNICODE          = 'V',
    PK_BINUNICODE       = 'X',
    PK_APPEND           = 'a',
    PK_BUILD            = 'b',
    PK_GLOBAL           = 'c',
    PK_DICT             = 'd',
    PK_EMPTY_DICT       = '}',
    PK_APPENDS          = 'e',
    PK_GET              = 'g',
    PK_BINGET           = 'h',
    PK_INST             = 'i',
    PK_LONG_BINGET      = 'j',
    PK_LIST             = 'l',
    PK_EMPTY_LIST       = ']',
    PK_OBJ              = 'o',
    PK_PUT              = 'p',
    PK_BINPUT           = 'q',
    PK_LONG_BINPUT      = 'r',
    PK_SETITEM          = 's',
    PK_TUPLE            = 't',
    PK_EMPTY_TUPLE      = ')',
    PK_SETITEMS         = 'u',
    PK_BINFLOAT         = 'G',
    PK_BINBYTES         = 'B',
    PK_SHORT_BINBYTES   = 'C',
    PK_PROTO            = 0x80,
    PK_NEWOBJ           = 0x81,
    PK_EXT1             = 0x82,
    PK_EXT2             = 0x83,
    PK_EXT4             = 0x84,
    PK_TUPLE1           = 0x85,
    PK_TUPLE2           = 0x86,
    PK_TUPLE3           = 0x87,
    PK_NEWTRUE          = 0x88,
    PK_NEWFALSE         = 0x89,
    PK_LONG1            = 0x8a,
    PK_LONG4            = 0x8b,
    PK_SHORT_BINUNICODE = 0x8c,
    PK_BINUNICODE8      = 0x8d,
    PK_BINBYTES8        = 0x8e,
    PK_EMPTY_SET        = 0x8f,
    PK_ADDITEMS         = 0x90,
    PK_FROZENSET        = 0x91,
    PK_NEWOBJ_EX        = 0x92,
    PK_STACK_GLOBAL     = 0x93,
    PK_MEMOIZE          = 0x94,
    PK_FRAME            = 0x95,
    PK_BYTEARRAY8       = 0x96,
    PK_NEXT_BUFFER      = 0x97,
    PK_READONLY_BUFFER  = 0x98,
};

typedef ZL_PickleLexer_Value PK_Value;

static PK_Value PK_make(PK_Type type, uint64_t a, uint64_t b)
{
    PK_Value v;
    memset(&v, 0, sizeof(v));
    v.type = (uint8_t)type;
    v.a    = a;
    v.b    = b;
    return v;
}

static PK_Value PK_unknownValue(void)
{
    return PK_make(PK_unknown, 0, 0);
}

bool ZL_PickleLexer_isPickle(const void* src, size_t srcSize)
{
    const uint8_t* const s = (const uint8_t*)src;
    return srcSize >= 2 && s[0] == PK_PROTO && s[1] >= 2 && s[1] <= 5;
}

/* ------------------------------------------------------------------------
 * Reading the arguments of opcodes
 * ------------------------------------------------------------------------ */

/// Reads a little-endian unsigned integer of @p size bytes, at most 8.
static bool PK_readLE(ZL_PickleLexer* L, size_t size, uint64_t* value)
{
    if (size > L->srcSize - L->pos) {
        return false;
    }
    *value = ZL_readLE64_N(L->src + L->pos, size);
    L->pos += size;
    return true;
}

static bool PK_skip(ZL_PickleLexer* L, uint64_t size)
{
    if (size > L->srcSize - L->pos) {
        return false;
    }
    L->pos += (size_t)size;
    return true;
}

/// Reads the line up to the next '\n', which it skips.
static bool PK_readLine(ZL_PickleLexer* L, const char** line, size_t* size)
{
    const char* const begin = L->src + L->pos;
    const char* const nl    = memchr(begin, '\n', L->srcSize - L->pos);
    if (nl == NULL) {
        return false;
    }
    *line = begin;
    *size = (size_t)(nl - begin);
    L->pos += *size + 1;
    return true;
}

/// Reads a decimal integer that fits in 63 bits, with an optional '-'.
static bool PK_parseInt(const char* s, size_t size, int64_t* value)
{
    const bool negative = size > 0 && s[0] == '-';
    size_t pos          = negative ? 1 : 0;
    if (pos == size) {
        return false;
    }
    int64_t v = 0;
    for (; pos < size; ++pos) {
        if (s[pos] < '0' || s[pos] > '9' || v > (INT64_MAX - 9) / 10) {
            return false;
        }
        v = v * 10 + (s[pos] - '0');
    }
    *value = negative ? -v : v;
    return true;
}

/* ------------------------------------------------------------------------
 * Stack, marks and memo
 * ------------------------------------------------------------------------ */

/// @returns the depth of the bottom of the current frame, above the last mark.
static uint64_t PK_frameBottom(const ZL_PickleLexer* L)
{
    return L->nbMarks > 0 ? L->marks[L->nbMarks - 1] : 0;
}

static bool PK_frameIsEmpty(const ZL_PickleLexer* L)
{
    return L->depth == PK_frameBottom(L);
}

static void PK_push(ZL_PickleLexer* L, PK_Value v)
{
    v.depth                                = L->depth;
    L->window[L->depth % ZL_PICKLE_WINDOW] = v;
    ++L->depth;
}

/// @returns the object at @p depth, unknown when it left the window.
static PK_Value PK_at(const ZL_PickleLexer* L, uint64_t depth)
{
    const PK_Value* const v = &L->window[depth % ZL_PICKLE_WINDOW];
    return v->depth == depth ? *v : PK_unknownValue();
}

/// Pops the top @p n objects of the current frame. When @p items is not NULL,
/// it receives them, from the deepest one.
static bool PK_popN(ZL_PickleLexer* L, size_t n, PK_Value* items)
{
    if (L->depth - PK_frameBottom(L) < n) {
        return false;
    }
    L->depth -= n;
    for (size_t i = 0; items != NULL && i < n; ++i) {
        items[i] = PK_at(L, L->depth + i);
    }
    return true;
}

static bool PK_pushMark(ZL_PickleLexer* L)
{
    if (L->nbMarks == ZL_PICKLE_MAX_MARKS) {
        return false;
    }
    L->marks[L->nbMarks++] = L->depth;
    return true;
}

/// Pops the objects above the last mark, then the mark. When @p items is not
/// NULL, it receives the first PK_MAX_TUPLE of them.
static bool PK_popMark(ZL_PickleLexer* L, PK_Value* items, uint64_t* nbItems)
{
    if (L->nbMarks == 0) {
        return false;
    }
    const uint64_t bottom = L->marks[--L->nbMarks];
    const uint64_t n      = L->depth - bottom;
    for (uint64_t i = 0; items != NULL && i < n && i < PK_MAX_TUPLE; ++i) {
        items[i] = PK_at(L, bottom + i);
    }
    if (nbItems != NULL) {
        *nbItems = n;
    }
    L->depth = bottom;
    return true;
}

static ZL_PickleLexer_MemoEntry*
PK_memoFind(ZL_PickleLexer_MemoEntry* table, size_t size, uint64_t key)
{
    for (size_t i = 0; i < size; ++i) {
        if (table[i].used && table[i].key == key) {
            return &table[i];
        }
    }
    return NULL;
}

/// @returns the entry of @p key, else the least recently used one.
static ZL_PickleLexer_MemoEntry*
PK_memoSlot(ZL_PickleLexer_MemoEntry* table, size_t size, uint64_t key)
{
    ZL_PickleLexer_MemoEntry* const found = PK_memoFind(table, size, key);
    if (found != NULL) {
        return found;
    }
    ZL_PickleLexer_MemoEntry* oldest = &table[0];
    for (size_t i = 1; i < size; ++i) {
        if (table[i].lastUse < oldest->lastUse) {
            oldest = &table[i];
        }
    }
    return oldest;
}

/// @returns true if @p v is a string of one of the characters of @p chars.
static bool
PK_isCharOf(const ZL_PickleLexer* L, const PK_Value* v, const char* chars)
{
    return v->type == PK_string && v->b == 1 && L->src[v->a] != '\0'
            && strchr(chars, L->src[v->a]) != NULL;
}

/// @returns true for the strings that arrays and dtypes are made of: memory
/// orders, byte orders, type strings such as "f4" or "U10", and the module and
/// name of numpy.dtype.
static bool PK_isArrayPart(const ZL_PickleLexer* L, const PK_Value* v)
{
    if (v->type != PK_string || v->b == 0 || v->b > 8) {
        return false;
    }
    const char* const s = L->src + v->a;
    const size_t size   = (size_t)v->b;
    if (size == 1) {
        return PK_isCharOf(L, v, PK_MEMORY_ORDERS PK_BYTE_ORDERS);
    }
    if (size == 5
        && (memcmp(s, "numpy", 5) == 0 || memcmp(s, "dtype", 5) == 0)) {
        return true;
    }
    if (!((s[0] >= 'a' && s[0] <= 'z') || (s[0] >= 'A' && s[0] <= 'Z'))) {
        return false;
    }
    for (size_t i = 1; i < size; ++i) {
        if (s[i] < '0' || s[i] > '9') {
            return false;
        }
    }
    return true;
}

/// Stores the object on top of the stack under @p key, if it is a dtype, its
/// class, or a string arrays and dtypes are made of. Pickles never reuse keys,
/// so the other objects clear no entry.
static bool PK_memoPut(ZL_PickleLexer* L, uint64_t key)
{
    if (PK_frameIsEmpty(L)) {
        return false;
    }
    if (key >= L->memoCount) {
        L->memoCount = key + 1;
    }
    const PK_Value v = PK_at(L, L->depth - 1);
    ZL_PickleLexer_MemoEntry* e;
    if (v.type == PK_dtype || v.type == PK_dtypeClass) {
        e = PK_memoSlot(L->memoDtypes, ZL_PICKLE_MEMO_DTYPES, key);
    } else if (PK_isArrayPart(L, &v)) {
        e = PK_memoSlot(L->memoParts, ZL_PICKLE_MEMO_PARTS, key);
    } else {
        return true;
    }
    e->key     = key;
    e->used    = true;
    e->lastUse = ++L->memoClock;
    e->value   = v;
    return true;
}

static void PK_memoGet(ZL_PickleLexer* L, uint64_t key)
{
    ZL_PickleLexer_MemoEntry* e =
            PK_memoFind(L->memoDtypes, ZL_PICKLE_MEMO_DTYPES, key);
    if (e == NULL) {
        e = PK_memoFind(L->memoParts, ZL_PICKLE_MEMO_PARTS, key);
    }
    if (e == NULL) {
        PK_push(L, PK_unknownValue());
        return;
    }
    e->lastUse = ++L->memoClock;
    PK_push(L, e->value);
}

/* ------------------------------------------------------------------------
 * Tokens
 * ------------------------------------------------------------------------ */

/// Covers the bytes up to @p end with a token of @p type.
static void PK_emit(ZL_PickleLexer* L, ZL_PickleTokenType type, size_t end)
{
    if (end <= L->emitted) {
        return;
    }
    ZL_ASSERT_LT(L->nbOut, L->outCapacity);
    ZL_PickleToken* const t = &L->out[L->nbOut++];
    memset(t, 0, sizeof(*t));
    t->ptr     = L->src + L->emitted;
    t->size    = end - L->emitted;
    t->type    = type;
    L->emitted = end;
}

static const ZL_PickleLexer_Shape* PK_shapeOf(
        const ZL_PickleLexer* L,
        const PK_Value* shape)
{
    const ZL_PickleLexer_Shape* const s = &L->shapes[shape->a];
    return s->generation == shape->b ? s : NULL;
}

/// Emits the array whose bytes are the string @p bytes.
static void PK_emitArray(
        ZL_PickleLexer* L,
        const PK_Value* bytes,
        const PK_Value* dtype,
        const PK_Value* shape,
        bool fortranOrder)
{
    const ZL_PickleLexer_Shape* const s = PK_shapeOf(L, shape);
    if (s == NULL || bytes->b == 0 || bytes->a < L->emitted) {
        return;
    }
    PK_emit(L, ZL_PickleTokenType_opcodes, (size_t)bytes->a);
    PK_emit(L, ZL_PickleTokenType_array, (size_t)(bytes->a + bytes->b));
    ZL_PickleToken* const t = &L->out[L->nbOut - 1];
    t->typeStr              = L->src + dtype->a;
    t->typeStrSize          = (size_t)dtype->b;
    t->byteOrder            = dtype->byteOrder;
    t->fortranOrder         = fortranOrder;
    t->ndims                = s->ndims;
    for (uint32_t d = 0; d < s->ndims && d < ZL_PICKLE_MAX_DIMS; ++d) {
        t->shape[d] = s->dims[d];
    }
}

/* ------------------------------------------------------------------------
 * Objects
 * ------------------------------------------------------------------------ */

static bool PK_isInt(const PK_Value* v)
{
    return v->type == PK_int;
}

static bool PK_isBytes(const PK_Value* v)
{
    return v->type == PK_string && !v->unicode;
}

static PK_Value
PK_newShape(ZL_PickleLexer* L, const PK_Value* dims, uint64_t ndims)
{
    const size_t slot             = (size_t)(L->nbShapes % ZL_PICKLE_NB_SHAPES);
    ZL_PickleLexer_Shape* const s = &L->shapes[slot];
    s->generation                 = ++L->nbShapes;
    s->ndims                      = (uint32_t)ndims;
    for (uint64_t d = 0; d < ndims && d < ZL_PICKLE_MAX_DIMS; ++d) {
        s->dims[d] = dims[d].a;
    }
    return PK_make(PK_shape, slot, s->generation);
}

/// @returns the tuple of @p items, emitting the array it describes, if any.
static PK_Value PK_tuple(ZL_PickleLexer* L, const PK_Value* items, uint64_t n)
{
    if (n > PK_MAX_TUPLE) {
        return PK_unknownValue();
    }
    bool isShape = true;
    for (uint64_t i = 0; i < n; ++i) {
        isShape &= PK_isInt(&items[i]) && (int64_t)items[i].a >= 0;
    }
    if (isShape) {
        return PK_newShape(L, items, n);
    }
    // state of _reconstruct: (version, shape, dtype, fortran, bytes)
    if (n == 5 && PK_isInt(&items[0]) && items[1].type == PK_shape
        && items[2].type == PK_dtype && PK_isInt(&items[3])
        && PK_isBytes(&items[4])) {
        PK_emitArray(L, &items[4], &items[2], &items[1], items[3].a != 0);
        return PK_unknownValue();
    }
    // arguments of _frombuffer: (bytes, dtype, shape, order)
    if (n == 4 && PK_isBytes(&items[0]) && items[1].type == PK_dtype
        && items[2].type == PK_shape
        && PK_isCharOf(L, &items[3], PK_MEMORY_ORDERS)) {
        PK_emitArray(
                L, &items[0], &items[1], &items[2], L->src[items[3].a] == 'F');
        return PK_unknownValue();
    }
    if (n == 3 && items[0].type == PK_string && items[0].b <= PK_MAX_TYPESTR
        && PK_isInt(&items[1]) && PK_isInt(&items[2])) {
        return PK_make(PK_dtypeArgs, items[0].a, items[0].b);
    }
    // state of dtype: (version, byteorder, subarray, names, ...)
    if (n >= 2 && PK_isInt(&items[0])
        && PK_isCharOf(L, &items[1], PK_BYTE_ORDERS)) {
        PK_Value state = PK_make(PK_dtypeState, 0, 0);
        const bool plain =
                n < 4 || (items[2].type == PK_none && items[3].type == PK_none);
        state.byteOrder = plain ? L->src[items[1].a] : 0;
        return state;
    }
    return PK_unknownValue();
}

static bool PK_tupleOf(ZL_PickleLexer* L, size_t n)
{
    PK_Value items[3];
    if (!PK_popN(L, n, items)) {
        return false;
    }
    PK_push(L, PK_tuple(L, items, n));
    return true;
}

static bool PK_tupleFromMark(ZL_PickleLexer* L)
{
    PK_Value items[PK_MAX_TUPLE];
    uint64_t n;
    if (!PK_popMark(L, items, &n)) {
        return false;
    }
    PK_push(L, PK_tuple(L, items, n));
    return true;
}

static PK_Value PK_global(
        const char* module,
        size_t moduleSize,
        const char* name,
        size_t nameSize)
{
    const bool isDtype = moduleSize == 5 && memcmp(module, "numpy", 5) == 0
            && nameSize == 5 && memcmp(name, "dtype", 5) == 0;
    return isDtype ? PK_make(PK_dtypeClass, 0, 0) : PK_unknownValue();
}

/* ------------------------------------------------------------------------
 * Opcodes
 * ------------------------------------------------------------------------ */

static bool PK_pushInt(ZL_PickleLexer* L, size_t size, bool isSigned)
{
    uint64_t v;
    if (!PK_readLE(L, size, &v)) {
        return false;
    }
    if (isSigned && size < 8 && size > 0 && (v >> (8 * size - 1)) != 0) {
        v |= ~(uint64_t)0 << (8 * size);
    }
    PK_push(L, PK_make(PK_int, v, 0));
    return true;
}

/// LONG1: a two's complement integer of up to 255 bytes.
static bool PK_pushLong1(ZL_PickleLexer* L)
{
    uint64_t size;
    if (!PK_readLE(L, 1, &size)) {
        return false;
    }
    if (size > 8) {
        PK_push(L, PK_unknownValue());
        return PK_skip(L, size);
    }
    return PK_pushInt(L, (size_t)size, true);
}

/// A string whose size is coded on @p lengthSize bytes. Sizes of BINSTRING
/// are signed.
static bool PK_pushString(
        ZL_PickleLexer* L,
        size_t lengthSize,
        bool unicode,
        bool signedLength)
{
    uint64_t size;
    if (!PK_readLE(L, lengthSize, &size)
        || (signedLength && size > (uint64_t)INT32_MAX)
        || size > L->srcSize - L->pos) {
        return false;
    }
    PK_Value v = PK_make(PK_string, L->pos, size);
    v.unicode  = unicode;
    L->pos += (size_t)size;
    PK_push(L, v);
    return true;
}

/// Opcodes of protocol 0 whose argument is a line.
static bool PK_textOpcode(ZL_PickleLexer* L, uint8_t op)
{
    const char* line;
    size_t size;
    if (!PK_readLine(L, &line, &size)) {
        return false;
    }
    int64_t v;
    switch (op) {
        case PK_INT:
            if (!PK_parseInt(line, size, &v)) {
                PK_push(L, PK_unknownValue());
                return true;
            }
            PK_push(L, PK_make(PK_int, (uint64_t)v, 0));
            return true;
        case PK_PUT:
            return PK_parseInt(line, size, &v) && v >= 0
                    && PK_memoPut(L, (uint64_t)v);
        case PK_GET:
            if (!PK_parseInt(line, size, &v) || v < 0) {
                return false;
            }
            PK_memoGet(L, (uint64_t)v);
            return true;
        default: // FLOAT, LONG, STRING, UNICODE, PERSID
            PK_push(L, PK_unknownValue());
            return true;
    }
}

/// GLOBAL and INST: a module and a name on two lines.
static bool PK_textGlobal(ZL_PickleLexer* L, bool isInst)
{
    const char* module;
    size_t moduleSize;
    const char* name;
    size_t nameSize;
    if (!PK_readLine(L, &module, &moduleSize)
        || !PK_readLine(L, &name, &nameSize)) {
        return false;
    }
    if (isInst) {
        if (!PK_popMark(L, NULL, NULL)) {
            return false;
        }
        PK_push(L, PK_unknownValue());
        return true;
    }
    PK_push(L, PK_global(module, moduleSize, name, nameSize));
    return true;
}

static bool PK_stackGlobal(ZL_PickleLexer* L)
{
    PK_Value items[2];
    if (!PK_popN(L, 2, items)) {
        return false;
    }
    if (items[0].type != PK_string || items[1].type != PK_string) {
        PK_push(L, PK_unknownValue());
        return true;
    }
    PK_push(L,
            PK_global(
                    L->src + items[0].a,
                    (size_t)items[0].b,
                    L->src + items[1].a,
                    (size_t)items[1].b));
    return true;
}

static bool PK_reduce(ZL_PickleLexer* L)
{
    PK_Value items[2];
    if (!PK_popN(L, 2, items)) {
        return false;
    }
    PK_Value result = PK_unknownValue();
    if (items[0].type == PK_dtypeClass && items[1].type == PK_dtypeArgs) {
        result    = PK_make(PK_dtype, items[1].a, items[1].b);
        result.id = ++L->nbDtypes;
    }
    PK_push(L, result);
    return true;
}

/// Sets the byte order of a dtype, and of its copies in the memo.
static bool PK_build(ZL_PickleLexer* L)
{
    PK_Value state;
    if (!PK_popN(L, 1, &state) || PK_frameIsEmpty(L)) {
        return false;
    }
    PK_Value* const top = &L->window[(L->depth - 1) % ZL_PICKLE_WINDOW];
    if (top->depth != L->depth - 1 || top->type != PK_dtype
        || state.type != PK_dtypeState) {
        return true;
    }
    top->byteOrder = state.byteOrder;
    for (size_t i = 0; i < ZL_PICKLE_MEMO_DTYPES; ++i) {
        PK_Value* const v = &L->memoDtypes[i].value;
        if (L->memoDtypes[i].used && v->type == PK_dtype && v->id == top->id) {
            v->byteOrder = state.byteOrder;
        }
    }
    return true;
}

/// Pops objects, then pushes the unknown object made of them.
static bool PK_combine(ZL_PickleLexer* L, size_t n)
{
    if (!PK_popN(L, n, NULL)) {
        return false;
    }
    PK_push(L, PK_unknownValue());
    return true;
}

static bool PK_combineFromMark(ZL_PickleLexer* L, bool needsItem)
{
    uint64_t n;
    if (!PK_popMark(L, NULL, &n) || (needsItem && n == 0)) {
        return false;
    }
    PK_push(L, PK_unknownValue());
    return true;
}

/// Pops objects added to the container below them.
static bool PK_addTo(ZL_PickleLexer* L, size_t n)
{
    return PK_popN(L, n, NULL) && !PK_frameIsEmpty(L);
}

static bool PK_addFromMark(ZL_PickleLexer* L)
{
    return PK_popMark(L, NULL, NULL) && !PK_frameIsEmpty(L);
}

static bool PK_pop(ZL_PickleLexer* L)
{
    if (!PK_frameIsEmpty(L)) {
        --L->depth;
        return true;
    }
    return PK_popMark(L, NULL, NULL);
}

static bool PK_dup(ZL_PickleLexer* L)
{
    if (PK_frameIsEmpty(L)) {
        return false;
    }
    PK_push(L, PK_at(L, L->depth - 1));
    return true;
}

static void PK_resetStack(ZL_PickleLexer* L)
{
    L->depth     = 0;
    L->nbMarks   = 0;
    L->memoCount = 0;
    memset(L->window, 0, sizeof(L->window));
    memset(L->memoDtypes, 0, sizeof(L->memoDtypes));
    memset(L->memoParts, 0, sizeof(L->memoParts));
}

/// Ends the pickle. Another one may follow.
static bool PK_stop(ZL_PickleLexer* L)
{
    L->stopPos = L->pos;
    if (!ZL_PickleLexer_isPickle(L->src + L->pos, L->srcSize - L->pos)) {
        return false;
    }
    PK_resetStack(L);
    return true;
}

static bool PK_memoOpcode(ZL_PickleLexer* L, uint8_t op)
{
    uint64_t key;
    switch (op) {
        case PK_BINPUT:
            return PK_readLE(L, 1, &key) && PK_memoPut(L, key);
        case PK_LONG_BINPUT:
            return PK_readLE(L, 4, &key) && PK_memoPut(L, key);
        case PK_MEMOIZE:
            return PK_memoPut(L, L->memoCount);
        case PK_BINGET:
        case PK_LONG_BINGET:
            if (!PK_readLE(L, op == PK_BINGET ? 1 : 4, &key)) {
                return false;
            }
            PK_memoGet(L, key);
            return true;
        default:
            return false;
    }
}

/// Executes the next opcode. @returns false when the lexer cannot go on,
/// from L->stopPos.
static bool PK_step(ZL_PickleLexer* L)
{
    L->stopPos = L->pos;
    if (L->pos >= L->srcSize) {
        return false;
    }
    const uint8_t op = (uint8_t)L->src[L->pos++];
    uint64_t arg;
    switch (op) {
        case PK_MARK:
            return PK_pushMark(L);
        case PK_STOP:
            return PK_stop(L);
        case PK_POP:
            return PK_pop(L);
        case PK_POP_MARK:
            return PK_popMark(L, NULL, NULL);
        case PK_DUP:
            return PK_dup(L);
        case PK_NONE:
            PK_push(L, PK_make(PK_none, 0, 0));
            return true;
        case PK_NEWTRUE:
        case PK_NEWFALSE:
            PK_push(L, PK_make(PK_int, op == PK_NEWTRUE, 0));
            return true;
        case PK_BININT1:
            return PK_pushInt(L, 1, false);
        case PK_BININT2:
            return PK_pushInt(L, 2, false);
        case PK_BININT:
            return PK_pushInt(L, 4, true);
        case PK_LONG1:
            return PK_pushLong1(L);
        case PK_LONG4:
            if (!PK_readLE(L, 4, &arg) || arg > (uint64_t)INT32_MAX) {
                return false;
            }
            PK_push(L, PK_unknownValue());
            return PK_skip(L, arg);
        case PK_BINFLOAT:
            PK_push(L, PK_unknownValue());
            return PK_skip(L, 8);
        case PK_SHORT_BINSTRING:
            return PK_pushString(L, 1, false, false);
        case PK_BINSTRING:
            return PK_pushString(L, 4, false, true);
        case PK_SHORT_BINBYTES:
            return PK_pushString(L, 1, false, false);
        case PK_BINBYTES:
            return PK_pushString(L, 4, false, false);
        case PK_BINBYTES8:
        case PK_BYTEARRAY8:
            return PK_pushString(L, 8, false, false);
        case PK_SHORT_BINUNICODE:
            return PK_pushString(L, 1, true, false);
        case PK_BINUNICODE:
            return PK_pushString(L, 4, true, false);
        case PK_BINUNICODE8:
            return PK_pushString(L, 8, true, false);
        case PK_INT:
        case PK_LONG:
        case PK_FLOAT:
        case PK_STRING:
        case PK_UNICODE:
        case PK_PERSID:
        case PK_PUT:
        case PK_GET:
            return PK_textOpcode(L, op);
        case PK_GLOBAL:
            return PK_textGlobal(L, false);
        case PK_INST:
            return PK_textGlobal(L, true);
        case PK_STACK_GLOBAL:
            return PK_stackGlobal(L);
        case PK_EMPTY_TUPLE:
            PK_push(L, PK_tuple(L, NULL, 0));
            return true;
        case PK_TUPLE1:
        case PK_TUPLE2:
        case PK_TUPLE3:
            return PK_tupleOf(L, (size_t)(op - PK_TUPLE1 + 1));
        case PK_TUPLE:
            return PK_tupleFromMark(L);
        case PK_EMPTY_LIST:
        case PK_EMPTY_DICT:
        case PK_EMPTY_SET:
        case PK_NEXT_BUFFER:
            PK_push(L, PK_unknownValue());
            return true;
        case PK_LIST:
        case PK_DICT:
        case PK_FROZENSET:
            return PK_combineFromMark(L, false);
        case PK_OBJ:
            return PK_combineFromMark(L, true);
        case PK_APPEND:
            return PK_addTo(L, 1);
        case PK_SETITEM:
            return PK_addTo(L, 2);
        case PK_APPENDS:
        case PK_SETITEMS:
        case PK_ADDITEMS:
            return PK_addFromMark(L);
        case PK_REDUCE:
            return PK_reduce(L);
        case PK_BUILD:
            return PK_build(L);
        case PK_NEWOBJ:
            return PK_combine(L, 2);
        case PK_NEWOBJ_EX:
            return PK_combine(L, 3);
        case PK_BINPERSID:
            return PK_combine(L, 1);
        case PK_READONLY_BUFFER:
            return !PK_frameIsEmpty(L);
        case PK_BINPUT:
        case PK_LONG_BINPUT:
        case PK_MEMOIZE:
        case PK_BINGET:
        case PK_LONG_BINGET:
            return PK_memoOpcode(L, op);
        case PK_PROTO:
            return PK_readLE(L, 1, &arg) && arg <= 5;
        case PK_FRAME:
            return PK_readLE(L, 8, &arg);
        case PK_EXT1:
        case PK_EXT2:
        case PK_EXT4:
            if (!PK_readLE(
                        L,
                        op == PK_EXT1           ? 1
                                : op == PK_EXT2 ? 2
                                                : 4,
                        &arg)) {
                return false;
            }
            PK_push(L, PK_unknownValue());
            return true;
        default:
            return false;
    }
}

/* ------------------------------------------------------------------------
 * Public functions
 * ------------------------------------------------------------------------ */

void ZL_PickleLexer_init(ZL_PickleLexer* lexer, const void* src, size_t srcSize)
{
    memset(lexer, 0, sizeof(*lexer));
    lexer->src     = (const char*)src;
    lexer->srcSize = srcSize;
}

size_t ZL_PickleLexer_lex(
        ZL_PickleLexer* lexer,
        ZL_PickleToken* out,
        size_t outCapacity)
{
    ZL_ASSERT_GE(outCapacity, 2);
    lexer->out         = out;
    lexer->outCapacity = outCapacity;
    lexer->nbOut       = 0;
    // An opcode emits at most 2 tokens
    while (!lexer->finished && lexer->nbOut + 2 <= outCapacity) {
        if (!PK_step(lexer)) {
            PK_emit(lexer, ZL_PickleTokenType_opcodes, lexer->stopPos);
            PK_emit(lexer, ZL_PickleTokenType_trailing, lexer->srcSize);
            lexer->finished = true;
        }
    }
    return lexer->nbOut;
}

bool ZL_PickleLexer_finished(const ZL_PickleLexer* lexer)
{
    return lexer->finished;
}
