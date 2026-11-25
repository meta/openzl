// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CUSTOM_PARSERS_ZIP_LEXER_H
#define CUSTOM_PARSERS_ZIP_LEXER_H

#include <stddef.h>
#include <stdint.h>

#include "openzl/shared/portability.h"
#include "openzl/zl_errors.h"

ZL_BEGIN_C_DECLS

typedef struct ZS2_ZipLexer_s ZS2_ZipLexer;

/**
 * The type of a token in a zip file, corresponding to the Zip spec.
 */
typedef enum {
    ZS2_ZipTokenType_LocalFileHeader,
    ZS2_ZipTokenType_CompressedData,
    ZS2_ZipTokenType_DataDescriptor,
    ZS2_ZipTokenType_CentralDirectory,
    ZS2_ZipTokenType_Zip64EndOfCentralDirectoryRecord,
    ZS2_ZipTokenType_Zip64EndOfCentralDirectoryLocator,
    ZS2_ZipTokenType_EndOfCentralDirectoryRecord,
    ZS2_ZipTokenType_Unknown,
} ZS2_ZipTokenType;

typedef struct {
    /// Pointer to the beginning of the token in the source buffer.
    const char* ptr;
    size_t size;           //< Size of the token in bytes.
    ZS2_ZipTokenType type; //< Type of the token.
    /// Compression method for the file, or 0 if not a file.
    uint16_t compressionMethod;
    /// Size of the filename, or 0 if not a file.
    uint16_t filenameSize;
    /// Pointer to the filename, or NULL if not a file.
    /// @warning Not zero terminated.
    const char* filename;
} ZS2_ZipToken;

/**
 * Initializes a Zip lexer on the given input buffer. The lexer allows for
 * garbage data before & after the zip file.
 *
 * @param opCtx The operation context to use for error reporting or NULL.
 *
 * @returns Success if the input buffer may be a valid zip file,
 *          or an error code if the input file is definitely not
 *          a supported zip file.
 *
 * @note This lexer supports all zip files whose central directory
 *       is listed in order of occurrence in the file.
 */
ZL_Report ZS2_ZipLexer_init(
        ZS2_ZipLexer* lexer,
        const void* src,
        size_t srcSize,
        ZL_OperationContext* opCtx);

/**
 * Lexes the next @p outCapacity tokens from the input buffer.
 *
 * @returns The number of tokens lexed, or an error code.
 *          Upon success, the input may be a valid zip file,
 *          and upon error the input is definitely not a supported
 *          zip file. Once it returns a value < @p outCapacity,
 *          the input has been fully lexed, and it will return 0
 *          on subsequent calls.
 */
ZL_Report
ZS2_ZipLexer_lex(ZS2_ZipLexer* lexer, ZS2_ZipToken* out, size_t outCapacity);

/// @returns true if the lexer has finished lexing the source.
bool ZS2_ZipLexer_finished(const ZS2_ZipLexer* lexer);

size_t ZS2_ZipLexer_expectedNumTokens(const ZS2_ZipLexer* lexer);

/**
 * @returns the number of files in the zip file.
 * @note If the zip file is corrupt, this may report an incorrect value,
 * however it is validated that the number of files is at least plausible,
 * and is no more than the source size / 76.
 */
size_t ZS2_ZipLexer_numFiles(const ZS2_ZipLexer* lexer);

/// @returns true if the input buffer is likely a zip file.
bool ZS2_isLikelyZipFile(const void* src, size_t srcSize);

/// @section Implementation details

typedef struct {
    const char* localFileHeaderPtr;
    size_t localFileHeaderSize;
    const char* compressedDataPtr;
    uint64_t compressedDataSize;
    const char* dataDescriptorPtr;
    size_t dataDescriptorSize;

    uint16_t compressionMethod;
    uint16_t filenameSize;
    const char* filename; //< Not zero terminated, may be NULL.
} ZS2_ZipLexer_FileState;

struct ZS2_ZipLexer_s {
    /// Pointer to the current position in the input buffer.
    /// Everything before this pointer has already been lexed.
    const char* srcPtr;
    /// Pointer to the end of the input buffer.
    const char* srcEnd;
    /// Pointer to the beginning of the zip file.
    /// It may not be the beginning of the input buffer, if the lexer detects
    /// that there is garbage before the zip file.
    const char* zipBegin;
    /// Pointer to the beginning of the current Central Directory File Header
    const char* cdfhPtr;
    /// Pointer to the end of the Central Directory File Headers
    const char* cdfhEnd;
    /// Index of the current Central Directory File Header
    size_t cdfhIdx;
    /// Number of Central Directory File Headers
    size_t cdfhNum;

    // Pointers & sizes for the trailing metadata sections.
    // They are set to NULL if they are not present, or if
    // they have already been lexed.
    const char* centralDirectoryPtr;
    const char* zip64EndOfCentralDirectoryRecordPtr;
    uint64_t zip64EndOfCentralDirectoryRecordSize;
    const char* zip64EndOfCentralDirectoryLocatorPtr;
    const char* endOfCentralDirectoryRecordPtr;
    uint32_t endOfCentralDirectoryRecordSize;

    ZS2_ZipLexer_FileState fileState;

    ZL_OperationContext* opCtx;
};

ZL_END_C_DECLS

#endif
