// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CUSTOM_PARSERS_PICKLE_LEXER_H
#define CUSTOM_PARSERS_PICKLE_LEXER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "openzl/shared/portability.h"

ZL_BEGIN_C_DECLS

/**
 * Lexer for Python pickles of protocols 2 to 5, which finds the bytes of the
 * NumPy arrays they hold.
 *
 * A pickle is a program for a stack machine: each opcode pushes or combines
 * objects, and STOP ends the program. NumPy pickles an array as a call that
 * rebuilds it from its shape, its dtype and its bytes, held by a bytes object:
 *   _reconstruct(ndarray, (0,), b'b'), then BUILD (1, shape, dtype, fortran,
 *   bytes)
 * or, from protocol 5,
 *   _frombuffer(bytes, dtype, shape, order)
 * and a dtype as dtype(typestr, False, True), then BUILD (3, byteorder, ...).
 * Below protocol 3, Python 3 pickles bytes objects as text, so the arrays of
 * its protocol 2 pickles are missed.
 *
 * The lexer follows the stack symbolically, never resolving nor calling
 * anything, with bounded state: the objects it does not track, or that left
 * its window, are unknown, and an array built from them is missed. Missing an
 * array only costs compression, as the tokens cover the input whatever the
 * lexer recognizes. Concatenated pickles, as written by successive calls to
 * `pickle.dump`, are lexed one after the other.
 */

#define ZL_PICKLE_MAX_DIMS 8

typedef enum {
    ZL_PickleTokenType_opcodes,  // opcodes and their arguments, but array bytes
    ZL_PickleTokenType_array,    // the bytes of a NumPy array
    ZL_PickleTokenType_trailing, // bytes past the last pickle, or from the
                                 // first opcode the lexer could not read
} ZL_PickleTokenType;

typedef struct {
    /// Pointer to the beginning of the token in the source buffer.
    const char* ptr;
    /// Size of the token in bytes, never 0.
    size_t size;
    ZL_PickleTokenType type;

    /// The following fields are only valid for array tokens.

    /// Type string of the dtype, such as "f4", "u1", "U3" or "M8". Points into
    /// the source buffer, not NUL-terminated.
    const char* typeStr;
    size_t typeStrSize;
    /// Byte order of the dtype, '<', '>' or '|', or 0 for dtypes with fields
    /// or subarrays.
    char byteOrder;
    bool fortranOrder;
    /// Number of dimensions, of which the first ZL_PICKLE_MAX_DIMS are in
    /// @p shape.
    uint32_t ndims;
    uint64_t shape[ZL_PICKLE_MAX_DIMS];
} ZL_PickleToken;

typedef struct ZL_PickleLexer_s ZL_PickleLexer;

/// @returns true if @p src starts with the PROTO opcode of protocol 2 to 5,
/// which pickles of these protocols start with.
bool ZL_PickleLexer_isPickle(const void* src, size_t srcSize);

void ZL_PickleLexer_init(
        ZL_PickleLexer* lexer,
        const void* src,
        size_t srcSize);

/**
 * Lexes the next tokens of the input into @p out, of at least 2 tokens.
 * Tokens cover the input in order.
 * @returns the number of tokens written, 0 once the input has been lexed.
 */
size_t ZL_PickleLexer_lex(
        ZL_PickleLexer* lexer,
        ZL_PickleToken* out,
        size_t outCapacity);

/// @returns true once the whole input has been lexed.
bool ZL_PickleLexer_finished(const ZL_PickleLexer* lexer);

/// @section Implementation details

#define ZL_PICKLE_WINDOW 64
#define ZL_PICKLE_MAX_MARKS 64
#define ZL_PICKLE_MEMO_DTYPES 16
#define ZL_PICKLE_MEMO_PARTS 16
#define ZL_PICKLE_NB_SHAPES 8

/// An object of the pickle, as far as the lexer tracks it.
typedef struct {
    /// Position of the object on the stack, from its bottom.
    uint64_t depth;
    /// Content, depending on the type: the value of an integer; the offset and
    /// size of the bytes of a string; the offset and size of the type string
    /// of a dtype; the slot and generation of a shape in the ring.
    uint64_t a;
    uint64_t b;
    uint32_t id; // of a dtype, to update its copies in the memo
    uint8_t type;
    bool unicode;   // strings: str rather than bytes
    char byteOrder; // dtypes and dtype states
} ZL_PickleLexer_Value;

typedef struct {
    uint64_t key;
    uint64_t lastUse;
    bool used;
    ZL_PickleLexer_Value value;
} ZL_PickleLexer_MemoEntry;

typedef struct {
    uint64_t generation;
    uint32_t ndims;
    uint64_t dims[ZL_PICKLE_MAX_DIMS];
} ZL_PickleLexer_Shape;

struct ZL_PickleLexer_s {
    const char* src;
    size_t srcSize;
    size_t pos;     // of the next opcode
    size_t emitted; // bytes covered by the tokens written so far
    size_t stopPos; // where the bytes the lexer cannot read start
    bool finished;

    ZL_PickleToken* out;
    size_t outCapacity;
    size_t nbOut;

    /// The top of the stack: the object at depth d is in window[d % size].
    uint64_t depth;
    ZL_PickleLexer_Value window[ZL_PICKLE_WINDOW];
    /// Stack depths of the marks.
    uint64_t marks[ZL_PICKLE_MAX_MARKS];
    size_t nbMarks;
    /// The memo, restricted to the dtypes and their class, and to the strings
    /// arrays and dtypes are made of, so that other objects never evict them.
    /// Entries are replaced when least recently used.
    ZL_PickleLexer_MemoEntry memoDtypes[ZL_PICKLE_MEMO_DTYPES];
    ZL_PickleLexer_MemoEntry memoParts[ZL_PICKLE_MEMO_PARTS];
    uint64_t memoClock;
    uint64_t memoCount;
    ZL_PickleLexer_Shape shapes[ZL_PICKLE_NB_SHAPES];
    uint64_t nbShapes;
    uint32_t nbDtypes;
};

ZL_END_C_DECLS

#endif
