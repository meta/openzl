// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_ZS2_DECOMPRESS_H
#define ZSTRONG_ZS2_DECOMPRESS_H

// basic definitions
#include "openzl/zl_common_types.h"
#include "openzl/zl_errors.h"        // ZL_Report, ZL_isError()
#include "openzl/zl_introspection.h" // ZL_DecompressIntrospectionHooks
#include "openzl/zl_opaque_types.h"  // ZL_DCtx
#include "openzl/zl_output.h"

#if defined(__cplusplus)
extern "C" {
#endif

// ----------------------------------------
// Simple API
// No custom decoders allowed (only standard ones)
// ----------------------------------------
// One-pass compression and decompression

/**
 * @brief Decompresses a frame hosting a single serialized output.
 *
 * @param dst Destination buffer for decompressed data
 * @param dstCapacity Size of destination buffer in bytes
 * @param src Source compressed data
 * @param srcSize Size of source data in bytes
 * @return Either an error (check with ZL_isError()) or decompressed size on
 * success
 */
ZL_Report
ZL_decompress(void* dst, size_t dstCapacity, const void* src, size_t srcSize);

/**
 * @brief Gets the decompressed size of content from a single-output frame.
 *
 * Useful to determine the size of buffer to allocate.
 *
 * @param compressed Pointer to compressed data
 * @param cSize Size of compressed frame
          or at a minimum the size of the complete frame header
 * @return Decompressed size in bytes, or error code
 *
 * @note Huge content sizes (> 4 GB) can't be represented on 32-bit systems
 * @note For String type, @return size of all strings concatenated
 */
ZL_Report ZL_getDecompressedSize(const void* compressed, size_t cSize);

/**
 * @brief Gets the size of the compressed frame.
 *        This method could be useful when the compressed frame only represents
 *        a first portion of a larger buffer.
 *
 * @param compressed Pointer to compressed data.
 *                   Must point at the start of a compressed frame.
 * @param testedSize Size of data @p compressed points to.
 *        To be useful, at a minimum, this size must be:
 *        - frame header + chunk  header for version < ZL_CHUNK_VERSION_MIN.
 *        - >= compressedSize for frames of versions >= ZL_CHUNK_VERSION_MIN.
 * @return Compressed frame size in bytes, or error code
 *
 * @note Huge content sizes (> 4 GB) can't be represented on 32-bit systems
 */
ZL_Report ZL_getCompressedSize(const void* compressed, size_t testedSize);

// ------------------------------------------------
// One-pass decompression with advanced parameters
// ------------------------------------------------

/**
 * @brief Creates a new decompression context.
 *
 * @return Pointer to new context, or NULL on error
 */
ZL_DCtx* ZL_DCtx_create(void);

/**
 * @brief Frees a decompression context.
 *
 * @param dctx Decompression context to free
 */
void ZL_DCtx_free(ZL_DCtx* dctx);

/**
 * @brief Returns the memory currently retained by @p dctx.
 *
 * @return 0 when @p dctx is NULL.
 */
size_t ZL_DCtx_sizeof(const ZL_DCtx* dctx);

/**
 * @brief Attach a dict loader to the decompression context.
 *
 * The dict loader is referenced (not owned) by the DCtx. The caller must
 * ensure the dict loader outlives the DCtx.
 */
void ZL_DCtx_refDictLoader(ZL_DCtx* dctx, ZL_DictLoader* loader);

/**
 * @brief Global decompression parameters.
 */
typedef enum {
    /**
     * @brief Keep parameters across decompression sessions.
     *
     * By default, parameters are reset between decompression sessions.
     * Setting this parameter to 1 keeps the parameters across sessions.
     */
    ZL_DParam_stickyParameters = 1,

    /**
     * @brief Enable checking the checksum of the compressed frame.
     *
     * The following two parameters control whether checksums are checked during
     * decompression. These checks can be disabled to achieve faster speeds in
     * exchange for the risk of data corruption going unnoticed.
     *
     * Disabling these checks is more effective when decompression speed is
     * already fast. Expected improvements: ~20-30% for speeds > 2GB/s, 10-15%
     * for speeds between 1GB/s and 2GB/s, and 1-5% for speeds < 1GB/s.
     *
     * Valid values use the ZS2_GPARAM_* format.
     * @note Default 0 currently means check the checksum, might change in
     * future
     */
    ZL_DParam_checkCompressedChecksum = 2,

    /**
     * @brief Enable checking the checksum of the uncompressed content.
     *
     * Valid values use the ZS2_GPARAM_* format.
     * @note Default 0 currently means check the checksum, might change in
     * future
     */
    ZL_DParam_checkContentChecksum = 3,

    /**
     * @brief Enable codec fusion during decompression.
     *
     * Codec fusion combines multiple adjacent codec nodes into a single
     * optimized decoder. Setting this to ZL_TernaryParam_disable causes each
     * codec in the graph to be decoded individually, which can be useful for
     * debugging or testing codec correctness without fusion.
     *
     * Valid values use the ZL_TernaryParam format defaulting to enabled.
     */
    ZL_DParam_enableCodecFusion = 4,

} ZL_DParam;

/**
 * @brief Sets global parameters via the decompression context.
 *
 * @param dctx Decompression context
 * @param gdparam Parameter to set
 * @param value Value to set for the parameter
 * @return Error code or success
 *
 * @note By default, parameters are reset at end of decompression session.
 *       To preserve them across sessions, set stickyParameters=1
 */
ZL_Report ZL_DCtx_setParameter(ZL_DCtx* dctx, ZL_DParam gdparam, int value);

/**
 * @brief Reads a parameter's configured value in the decompression context.
 *
 * @param dctx Decompression context
 * @param gdparam Parameter to query
 * @return The value set for the given parameter, or 0 if unset/nonexistent
 */
int ZL_DCtx_getParameter(const ZL_DCtx* dctx, ZL_DParam gdparam);

/**
 * @brief Resets parameters selection from a blank slate.
 *
 * Useful when unsure if ZL_DParam_stickyParameters is set to 1.
 *
 * @param dctx Decompression context
 * @return Error code or success
 */
ZL_Report ZL_DCtx_resetParameters(ZL_DCtx* dctx);

/**
 * @brief Sets the Stream Arena type for ZL_Input* buffers (experimental).
 *
 * This frees the previous Stream Arena and creates a new one.
 * This choice remains sticky until set again.
 *
 * @param dctx Decompression context
 * @param sat Stream Arena type to set
 * @return Error code or success
 *
 * @note Default Stream Arena is StackArena. It keeps its buffer across chunks
 * and decompressions, which avoids reallocating and page-faulting stream
 * buffers every time, at the cost of retaining that memory between
 * decompressions and of a higher peak, since streams are only released at the
 * end of each chunk. Select ZL_DataArenaType_heap to return stream memory to
 * the allocator after each chunk, and free each stream as soon as it is
 * consumed.
 */
#include "openzl/zl_data.h" // ZL_DataArenaType
ZL_Report ZL_DCtx_setStreamArena(ZL_DCtx* dctx, ZL_DataArenaType sat);

/**
 * @brief Gets a verbose error string containing context about the error.
 *
 * Useful for debugging and submitting bug reports to OpenZL developers.
 *
 * @param dctx Decompression context
 * @param report Error report to get context for
 * @return Error context string
 *
 * @note String is stored within the dctx and is only valid for the lifetime of
 * the dctx
 */
char const* ZL_DCtx_getErrorContextString(
        ZL_DCtx const* dctx,
        ZL_Report report);

/**
 * @brief Gets error context string from error code.
 *
 * @param dctx Decompression context
 * @param error Error code to get context for
 * @return Error context string
 *
 * @note String is stored within the dctx and is only valid for the lifetime of
 * the dctx
 */
char const* ZL_DCtx_getErrorContextString_fromError(
        ZL_DCtx const* dctx,
        ZL_Error error);

/**
 * @brief Gets the array of warnings from the previous operation.
 *
 * @param dctx Decompression context
 * @return Array of warnings encountered during the previous operation
 *
 * @note Array and errors' lifetimes are valid until the next non-const call on
 * the DCtx
 */
ZL_Error_Array ZL_DCtx_getWarnings(ZL_DCtx const* dctx);

/**
 * @brief Decompresses data with explicit state management.
 *
 * Same as ZL_decompress(), but with explicit state management.
 * The state can be used to store advanced parameters.
 *
 * @param dctx Decompression context
 * @param dst Destination buffer for decompressed data
 * @param dstCapacity Size of destination buffer in bytes
 * @param compressed Source compressed data
 * @param cSize Size of compressed data in bytes
 * @return Error code or decompressed size on success
 */
ZL_Report ZL_DCtx_decompress(
        ZL_DCtx* dctx,
        void* dst,
        size_t dstCapacity,
        const void* compressed,
        size_t cSize);

// ----------------------------------------------------
// Querying compressed frames
// ----------------------------------------------------

/**
 * @brief Gets the number of outputs stored in a compressed frame.
 *
 * Doesn't need the whole frame, just enough to read the requested information.
 *
 * @param compressed Pointer to compressed data
 * @param cSize Size of compressed data
 * @return Number of outputs, or error on failure (invalid frame, size too
 * small, etc.)
 */
ZL_Report ZL_getNumOutputs(const void* compressed, size_t cSize);

/* For single-output frames:
 * we already have ZL_getDecompressedSize(),
 * so we only need one other prototype: ZL_getOutputType().
 */

/**
 * @brief Gets the output type for single-output frames.
 *
 * Only works for single-output frames. Provides the type of the single output.
 *
 * @param stPtr Pointer to store the output type
 * @param compressed Pointer to compressed data
 * @param cSize Size of compressed data
 * @return Error code or success
 *
 * @note Only works for single-output frames
 */
ZL_Report
ZL_getOutputType(ZL_Type* stPtr, const void* compressed, size_t cSize);

// @note: The following API is more suitable for multi-outputs frames

/**
 * @brief Frame information object for querying frame metadata.
 *
 * The frame header is parsed once, and all information is stored in the
 * ZL_FrameInfo object. Then the object can be queried for various properties.
 */
typedef struct ZL_FrameInfo ZL_FrameInfo;

/**
 * @brief Creates a frame information object from compressed data.
 *
 * @param compressed Pointer to compressed data
 * @param cSize Size of compressed data
 * @return Pointer to frame info object, or NULL on error
 */
ZL_FrameInfo* ZL_FrameInfo_create(const void* compressed, size_t cSize);

/**
 * @brief Frees a frame information object.
 *
 * @param fi Frame information object to free
 */
void ZL_FrameInfo_free(ZL_FrameInfo* fi);

/**
 * @brief Gets the format version of the frame.
 *
 * @param fi Frame information object
 * @return The version number, or error if unsupported or invalid
 */
ZL_Report ZL_FrameInfo_getFormatVersion(const ZL_FrameInfo* fi);

/**
 * @brief Gets the number of regenerated outputs on decompression.
 *
 * @param fi Frame information object
 * @return The number of outputs on decompression
 */
ZL_Report ZL_FrameInfo_getNumOutputs(const ZL_FrameInfo* fi);

/**
 * @brief Gets the output type for a specific output ID.
 *
 * @param fi Frame information object
 * @param outputID Output ID (starts at 0, single output has ID 0)
 * @return Output type, or error code
 */
ZL_Report ZL_FrameInfo_getOutputType(const ZL_FrameInfo* fi, int outputID);

/**
 * @brief Gets the decompressed size of a specific output.
 *
 * @param fi Frame information object
 * @param outputID Output ID (starts at 0)
 * @return Size of specified output in bytes, or error code
 */
ZL_Report ZL_FrameInfo_getDecompressedSize(
        const ZL_FrameInfo* fi,
        int outputID);

/**
 * @brief Gets the number of elements in a specific output.
 *
 * @param fi Frame information object
 * @param outputID Output ID (starts at 0)
 * @return Number of elements in specified output, or error code
 */
ZL_Report ZL_FrameInfo_getNumElts(const ZL_FrameInfo* fi, int outputID);

ZL_RESULT_DECLARE_TYPE(ZL_Comment);

/**
 * @brief Gets the comment stored in the FrameInfo.
 *
 * @returns The comment or an error. If no comment is present it
 * returns a comment with `size == 0`. The buffer returned is owned by @p zfi
 */
ZL_RESULT_OF(ZL_Comment) ZL_FrameInfo_getComment(const ZL_FrameInfo* zfi);

/**
 * @brief Gets the dict bundle ID referenced by the frame, if any.
 *
 * @returns A pointer to the bundle ID stored in @p zfi, or NULL if the frame
 * does not reference a dict bundle.
 */
const ZL_BundleID* ZL_FrameInfo_getBundleID(const ZL_FrameInfo* zfi);

// ----------------------------------------------------
// Decompression of Typed content
// ----------------------------------------------------

/**
 * @brief Information about a decompressed typed output.
 */
typedef struct {
    ZL_Type type;                  /**< Type of the output */
    uint32_t fixedWidth;           /**< width of elements in bytes */
    uint64_t decompressedByteSize; /**< Decompressed size in bytes */
    uint64_t numElts;              /**< Number of elements */
} ZL_OutputInfo;

/**
 * @brief Decompresses typed content from frames with a single typed output
 *        into a pre-allocated buffer @p dst .
 *        Information about output type is written into @p outputInfo .
 *
 * @param dctx Decompression context
 * @param outputInfo Output metadata filled on success
 * @param dst Destination buffer (must be large enough)
 * @param dstByteCapacity Size of destination buffer in bytes
 * @param compressed Pointer to compressed data
 * @param cSize Size of compressed data in bytes
 * @return Error code or success
 *
 * @note Only works for frames with a single typed output,
 *       and even then, does not work for String type.
 * @note For Numeric type, dst must be correctly aligned (if unknown, assume
 * 64-bit numeric)
 * @note Numeric values are written using host's endianness
 * @note On error, @p outputInfo content is undefined and should not be read
 */
ZL_Report ZL_DCtx_decompressTyped(
        ZL_DCtx* dctx,
        ZL_OutputInfo* outputInfo,
        void* dst,
        size_t dstByteCapacity,
        const void* compressed,
        size_t cSize);

/* ZL_TypedBuffer interface */

/**
 * @brief Typed buffer object that can own and auto-size their buffers.
 *
 * TypedBuffer uses standard C malloc/free to allocate buffers.
 * Future versions could allow insertion of a custom allocator.
 */
typedef ZL_Output ZL_TypedBuffer;

/**
 * @brief Creates an empty typed buffer object.
 *
 * On creation, the TypedBuffer object is empty.
 * By default, when provided as an empty shell,
 * it will automatically allocate and fill its own buffer
 * during invocation of ZL_DCtx_decompressTBuffer().
 * A TypedBuffer object is not reusable and must be freed after usage.
 * Releasing the object also releases its owned buffers.
 *
 * @return Pointer to new typed buffer, or NULL on error
 */
ZL_TypedBuffer* ZL_TypedBuffer_create(void);

/**
 * @brief Frees a typed buffer object.
 *
 * @param tbuffer Typed buffer to free
 */
void ZL_TypedBuffer_free(ZL_TypedBuffer* tbuffer);

/**
 * @brief Creates a pre-allocated typed buffer for serial data.
 *
 * The object will use the referenced buffer, and not allocate its own one.
 * It will be filled during invocation of ZL_DCtx_decompressTBuffer().
 * Releasing the ZL_TypedBuffer object will not release the referenced buffer.
 *
 * @param buffer buffer to write into
 * @param bufferCapacity maximum size to write into buffer
 * @return Pointer to wrapped typed buffer, or NULL on error
 */
ZL_TypedBuffer* ZL_TypedBuffer_createWrapSerial(
        void* buffer,
        size_t bufferCapacity);

/**
 * @brief Creates a pre-allocated typed buffer for struct data.
 *
 * @param structBuffer buffer to write struct into.
 *        Its size must be at least @p structWidth * @p structCapacity
 * @param structWidth Width of each struct in bytes
 * @param structCapacity Maximum number of struct that can be written into @p
 * structBuffer
 * @return Pointer to wrapped typed buffer, or NULL on error
 */
ZL_TypedBuffer* ZL_TypedBuffer_createWrapStruct(
        void* structBuffer,
        size_t structWidth,
        size_t numStructs);

/**
 * @brief Creates a pre-allocated typed buffer for numeric data.
 *
 * @param numArray buffer to write the array of numeric values into
 *        The array size must be >= @p numWidth * @p numCapacity.
 *        It must also be correctly aligned for the numeric width requested.
 * @param numWidth Width of numeric values in bytes
 * @param numCapacity Maximum number of numeric value that can be written into
 * @p numArray
 * @return Pointer to wrapped typed buffer, or NULL on error
 */
ZL_TypedBuffer* ZL_TypedBuffer_createWrapNumeric(
        void* numArray,
        size_t numWidth,
        size_t numCapacity);

/**
 * @brief Creates a pre-allocated ZL_TypedBuffer for String data.
 *
 * @param stringBuffer The buffer where the concatenation of all Strings will be
 * written.
 * @param stringBufferCapacity Size of stringBuffer
 * @param lenBuffer Buffer for the array of lengths, which are 32-bit unsigned.
 * @param maxNumStrings Maximum number of strings that can be written.
 * @return Pointer to wrapped typed buffer, or NULL on error
 */
ZL_TypedBuffer* ZL_TypedBuffer_createWrapString(
        void* stringBuffer,
        size_t stringBufferCapacity,
        uint32_t* lenBuffer,
        size_t maxNumStrings);

/**
 * @brief Decompresses a frame with a single typed output into a TypedBuffer.
 *
 * This prototype works for frames with a single typed output only.
 * Data will be decompressed into the output buffer.
 *
 * @param dctx Decompression context
 * @param output TypedBuffer object (must be created first)
 * @param compressed Pointer to compressed data
 * @param cSize Size of compressed data in bytes
 * @return Error code or size in bytes of the main buffer inside output
 *
 * @note Output is filled on success, but undefined on error, and can only be
 * free() in this case.
 * @note ZL_TypedBuffer object is required to decompress String type
 */
ZL_Report ZL_DCtx_decompressTBuffer(
        ZL_DCtx* dctx,
        ZL_TypedBuffer* output,
        const void* compressed,
        size_t cSize);

/**
 * @brief Decompresses a frame into multiple TypedBuffers.
 *
 * @param dctx Decompression context
 * @param outputs Array of ZL_TypedBuffer* objects (must be created using
 *                ZL_TypedBuffer_create or pre-allocated creation methods)
 * @param nbOutputs Exact number of outputs expected from the frame
 *                  (can be obtained from ZL_FrameInfo_getNumOutputs())
 * @param compressed Pointer to compressed data
 * @param cSize Size of compressed data in bytes
 * @return Error code or number of decompressed TypedBuffers
 *
 * @note Requires exact number of outputs (not permissive)
 * @note @p outputs are filled on success, but undefined on error
 * @note TypedBuffer lifetimes are controlled by the caller
 */
ZL_Report ZL_DCtx_decompressMultiTBuffer(
        ZL_DCtx* dctx,
        ZL_TypedBuffer* outputs[],
        size_t nbOutputs,
        const void* compressed,
        size_t cSize);

/** Once decompression is completed, the ZL_TypedBuffer object can be queried.
 * Here are its accessors: */

/**
 * @brief Gets the type of the typed buffer.
 *
 * @param tbuffer Typed buffer object
 * @return Type of the buffer
 */
ZL_Type ZL_TypedBuffer_type(const ZL_TypedBuffer* tbuffer);

/**
 * @brief Gets the number of bytes written into the internal buffer.
 *
 * @param tbuffer Typed buffer object
 * @return Number of bytes in the internal buffer.
 *
 * @note Generally equals eltWidth * nbElts,
 *       but for String type equals sum(stringLens)
 * @note ZL_DCtx_decompressTBuffer() returns the same value
 */
size_t ZL_TypedBuffer_byteSize(const ZL_TypedBuffer* tbuffer);

/**
 * @brief Gets direct access to the internal buffer for reading operation.
 *
 * @param tbuffer Typed buffer object
 * @return Pointer to the beginning of buffer (for String type, points at the
 * beginning of the first string).
 *
 * @warning Users must pay attention to buffer boundaries
 * @note Buffer size is provided by ZL_TypedBuffer_byteSize()
 */
const void* ZL_TypedBuffer_rPtr(const ZL_TypedBuffer* tbuffer);

/**
 * @brief Gets the number of elements in the typed buffer.
 *
 * @param tbuffer Typed buffer object
 * @return Number of elements in the buffer
 *
 * @note for Serial type, this is the number of bytes.
 */
size_t ZL_TypedBuffer_numElts(const ZL_TypedBuffer* tbuffer);

/**
 * @brief Gets the size of each element for fixed-size types.
 *
 * @param tbuffer Typed buffer object
 * @return Size of each element in bytes, or 0 for String type
 *
 * @note Not valid for String type (returns 0)
 */
size_t ZL_TypedBuffer_eltWidth(const ZL_TypedBuffer* tbuffer);

/**
 * @brief Gets pointer to the array of string lengths.
 *
 * @param tbuffer Typed buffer object
 * @return Pointer to beginning of string lengths array, or NULL if incorrect
 * type
 *
 * @note **Only valid for String type**
 * @note Array size equals ZL_TypedBuffer_numElts()
 */
const uint32_t* ZL_TypedBuffer_rStringLens(const ZL_TypedBuffer* tbuffer);

// -----------------------------
// Advanced & unstable functions
// -----------------------------

/**
 * @brief Gets the size of the OpenZL header.
 *
 * Useful to determine header overhead.
 *
 * @param src Source compressed data
 * @param srcSize Size of source data
 * @return Header size in bytes, or error code
 *
 * @note This is a temporary function, not guaranteed to remain in future
 * versions
 */
ZL_Report ZL_getHeaderSize(const void* src, size_t srcSize);

// -----------------------------
// Decompression Introspection
// -----------------------------

/**
 * Attach introspection hooks to the DCtx. Hooks allow code to run at specific
 * DWAYPOINTs during decompression. A hook set to NULL will simply be skipped.
 * There can only be one set of hooks attached at a time; calling this again
 * will overwrite the previous hooks. The caller is responsible for maintaining
 * the lifetime of the objects referenced by the hooks.
 *
 * @note This will only do something if the library is compiled with the
 * ALLOW_INTROSPECTION option. Otherwise, all the hooks will be no-ops.
 */
ZL_Report ZL_DCtx_attachDecompressIntrospectionHooks(
        ZL_DCtx* dctx,
        const ZL_DecompressIntrospectionHooks* hooks);

/**
 * Detach any decompression introspection hooks currently attached to the DCtx.
 */
ZL_Report ZL_DCtx_detachAllDecompressIntrospectionHooks(ZL_DCtx* dctx);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // ZSTRONG_ZS2_DECOMPRESS_H
