// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CUSTOM_PARSERS_NUMPY_NUMPY_LEXER_H
#define CUSTOM_PARSERS_NUMPY_NUMPY_LEXER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "openzl/shared/portability.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS

/**
 * Lexer for NumPy files: .npy arrays and .npz archives.
 *
 * A .npy file starts with the magic "\x93NUMPY", a major and a minor version
 * byte, and the size of the header, little-endian, on 2 bytes in version 1
 * and 4 bytes in versions 2 and 3. The header is a Python dict literal padded
 * with spaces and a newline, such as
 *   {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }
 * where descr is the element type, as written by `numpy.dtype.str`, or as a
 * list of fields for structured types. The array bytes follow, in C order, or
 * in Fortran order when fortran_order is True.
 * A .npz file is a zip archive of .npy members: `np.savez` stores them, and
 * `np.savez_compressed` deflates them.
 */

#define ZL_NUMPY_MAX_DIMS 8
/// Archives holding more members are rejected, which bounds the memory needed
/// to parse them.
#define ZL_NUMPY_MAX_MEMBERS ((size_t)1 << 17)
#define ZL_NUMPY_MAX_HEADER_SIZE ((size_t)1 << 20)

typedef enum {
    ZL_NumPyTokenType_header, // .npy magic and header, zip records
    ZL_NumPyTokenType_array,  // the bytes of an array
    ZL_NumPyTokenType_gap, // anything else: deflated or non-.npy members, bytes
                           // past an array
} ZL_NumPyTokenType;

typedef struct {
    /// Pointer to the beginning of the token in the source buffer.
    const char* ptr;
    /// Size of the token in bytes, never 0.
    size_t size;
    ZL_NumPyTokenType type;

    /// The following fields are only valid for array tokens.

    /// Name of the .npz member, NULL in a .npy file. Points into the file, not
    /// NUL-terminated.
    const char* name;
    size_t nameSize;
    /// Kind of the elements, the letter of descr ('f' for floats, 'i' for
    /// signed integers...), or 0 when the token does not hold exactly
    /// prod(shape) little-endian elements of `itemSize` bytes: structured
    /// types, big-endian types, truncated arrays, or more than
    /// ZL_NUMPY_MAX_DIMS dimensions.
    char kind;
    uint32_t itemSize;
    bool fortranOrder;
    uint32_t ndims;
    uint64_t shape[ZL_NUMPY_MAX_DIMS];
} ZL_NumPyToken;

/**
 * Lexes the .npy file or .npz archive in @p src into tokens that cover it in
 * order. In a .npy file, the header is followed by the array, then by a gap
 * holding the bytes past it, if any. In a .npz archive, each stored .npy
 * member becomes a header and an array, between the header tokens of the zip
 * records. Empty arrays produce no token. Adjacent header tokens, and adjacent
 * gaps, are merged.
 *
 * When @p tokens is NULL, the tokens are counted without being written, so
 * that the caller can size @p tokens before lexing again.
 *
 * @returns the number of tokens written to @p tokens, or counted when
 *          @p tokens is NULL. Errors are graph_parser_malformedInput for an
 *          input that is neither a .npy file nor a zip archive, and
 *          graph_parser_unhandledInput for an archive of more than
 *          ZL_NUMPY_MAX_MEMBERS members.
 */
ZL_Report ZL_NumPy_lex(
        const void* src,
        size_t srcSize,
        ZL_NumPyToken* tokens,
        size_t tokensCapacity,
        ZL_ErrorContext* errCtx);

ZL_END_C_DECLS

#endif
