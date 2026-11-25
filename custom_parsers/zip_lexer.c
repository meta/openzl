// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "custom_parsers/zip_lexer.h"

#include "openzl/shared/mem.h"
#include "openzl/shared/utils.h"

static const uint16_t kZip64InfoID                           = 0x0001;
static const uint16_t kDataDescriptorMask                    = (1 << 3);
static const size_t kMinLocalHeaderSize                      = 30;
static const size_t kMinCentralDirectoryFileHeaderSize       = 46;
static const size_t kMinEndOfCentralDirectoryRecordSize      = 22;
static const size_t kMinZip64EndOfCentralDirectoryRecordSize = 56;
static const size_t kZip64EndOfCentralDirectoryLocatorSize   = 20;

/// @returns true if the pointer is likely to point to an End of Central
/// Directory Record
static bool isEOCD(const char* ptr)
{
    return ZL_readLE32(ptr) == 0x06054b50;
}

typedef struct {
    uint64_t diskNumber;
    uint64_t centralDirectoryDiskNumber;
    uint64_t centralDirectoryRecordCountOnDisk;
    uint64_t centralDirectoryRecordCount;
    uint64_t centralDirectorySize;
    uint64_t centralDirectoryOffset;
} ZS2_ZipLexer_EndOfCentralDirectory;

/// @returns true if the EOCD requires Zip64
static bool ZS2_ZipLexer_EndOfCentralDirectory_needZip64(
        const ZS2_ZipLexer_EndOfCentralDirectory* eocd)
{
    const uint64_t kNeg1 = (uint64_t)-1;
    if (eocd->diskNumber == kNeg1) {
        return true;
    }
    if (eocd->centralDirectoryDiskNumber == kNeg1) {
        return true;
    }
    if (eocd->centralDirectoryRecordCountOnDisk == kNeg1) {
        return true;
    }
    if (eocd->centralDirectoryRecordCount == kNeg1) {
        return true;
    }
    if (eocd->centralDirectorySize == kNeg1) {
        return true;
    }
    if (eocd->centralDirectoryOffset == kNeg1) {
        return true;
    }
    return false;
}

static uint64_t read16OrNeg1(const void* ptr)
{
    uint16_t val = ZL_readLE16(ptr);
    if (val == (uint16_t)-1) {
        return (uint64_t)-1;
    } else {
        return (uint64_t)val;
    }
}

static uint64_t read32OrNeg1(const void* ptr)
{
    uint32_t val = ZL_readLE32(ptr);
    if (val == (uint32_t)-1) {
        return (uint64_t)-1;
    } else {
        return (uint64_t)val;
    }
}

/// @returns true if the extra fields of LFH or CDFH has a Zip64 entry
static bool hasZip64Info(const char* extraFieldPtr, size_t extraFieldLength)
{
    const char* const extraFieldEnd = extraFieldPtr + extraFieldLength;
    for (; extraFieldPtr < extraFieldEnd;) {
        if ((extraFieldEnd - extraFieldPtr) < 4) {
            return false;
        }
        const uint16_t id = ZL_readLE16(extraFieldPtr);
        const size_t size = ZL_readLE16(extraFieldPtr + 2);
        extraFieldPtr += 4;
        if ((size_t)(extraFieldEnd - extraFieldPtr) < size) {
            return false;
        }
        if (id == kZip64InfoID) {
            return true;
        }
        extraFieldPtr += size;
    }
    return false;
}

/**
 * Reads a 64-bit field from @p offsetTrusted (does not include the 4-byte
 * header).
 *
 * @p offsetTrusted An offset to read from that is guaranteed to be <= 16.
 */
static ZL_RESULT_OF(uint64_t) readZip64Info(
        ZL_OperationContext* opCtx,
        const char* zip64InfoPtr,
        size_t zip64InfoSize,
        size_t offsetTrusted)
{
    ZL_RESULT_DECLARE_SCOPE(uint64_t, opCtx);
    const char* const zip64InfoEnd = zip64InfoPtr + zip64InfoSize;
    ZL_ASSERT_LE(offsetTrusted, 16);
    for (;;) {
        ZL_ERR_IF_LT(zip64InfoEnd - zip64InfoPtr, 4, corruption);
        const uint16_t id = ZL_readLE16(zip64InfoPtr);
        const size_t size = ZL_readLE16(zip64InfoPtr + 2);
        zip64InfoPtr += 4;
        ZL_ERR_IF_LT((size_t)(zip64InfoEnd - zip64InfoPtr), size, corruption);
        if (id == kZip64InfoID) {
            ZL_ERR_IF_GT(offsetTrusted + 8, size, corruption);
            return ZL_WRAP_VALUE(ZL_readLE64(zip64InfoPtr + offsetTrusted));
        }
        zip64InfoPtr += size;
    }
}

/**
 * Reads the central directory starting at @p cdfhPtr and fills the
 * @p localFileHeaderOffset and @p compressedSize. The compressed size is read
 * from the CDFH entry because the local file header might not have it.
 * @returns The size of the CDFH entry or an error.
 */
static ZL_Report readCentralDirectoryFileHeader(
        ZL_OperationContext* opCtx,
        const char* cdfhPtr,
        const char* cdfhEnd,
        uint64_t* localFileHeaderOffset,
        uint64_t* compressedSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(opCtx);
    // suppress -Wmaybe-uninitialized
    *localFileHeaderOffset = 0;

    const size_t remaining = (size_t)(cdfhEnd - cdfhPtr);
    ZL_ERR_IF_LT(remaining, kMinCentralDirectoryFileHeaderSize, corruption);

    ZL_ERR_IF_NE(ZL_readLE32(cdfhPtr), 0x02014b50, corruption);
    const size_t filenameLength    = ZL_readLE16(cdfhPtr + 28);
    const size_t extraFieldLength  = ZL_readLE16(cdfhPtr + 30);
    const size_t fileCommentLength = ZL_readLE16(cdfhPtr + 32);
    const size_t cdfhLength        = kMinCentralDirectoryFileHeaderSize
            + filenameLength + extraFieldLength + fileCommentLength;
    ZL_ERR_IF_LT(remaining, cdfhLength, corruption);
    const char* const extraFieldPtr =
            cdfhPtr + kMinCentralDirectoryFileHeaderSize + filenameLength;

    // Zip64 extra field stores values in order: uncompressed size,
    // compressed size, header offset. Each field is only present when
    // the corresponding CDFH field is 0xFFFFFFFF.
    size_t zip64Offset = 0;
    if (ZL_readLE32(cdfhPtr + 24) == (uint32_t)-1) {
        zip64Offset += 8;
    }

    *compressedSize = read32OrNeg1(cdfhPtr + 20);
    if (*compressedSize == (uint64_t)-1) {
        ZL_TRY_LET(
                uint64_t,
                size,
                readZip64Info(
                        opCtx, extraFieldPtr, extraFieldLength, zip64Offset));
        *compressedSize = size;
        zip64Offset += 8;
    }

    *localFileHeaderOffset = read32OrNeg1(cdfhPtr + 42);
    if (*localFileHeaderOffset == (uint64_t)-1) {
        ZL_TRY_LET(
                uint64_t,
                offset,
                readZip64Info(
                        opCtx, extraFieldPtr, extraFieldLength, zip64Offset));
        *localFileHeaderOffset = offset;
    }

    return ZL_returnValue(cdfhLength);
}

/**
 * Reads the EOCD64 starting at @p eocd64Ptr and fills @p eocd on success.
 * @returns The size of the EOCD64 or an error.
 */
static ZL_Report readEOCD64(
        const char* eocd64Ptr,
        const char* eocd64End,
        ZS2_ZipLexer_EndOfCentralDirectory* eocd)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    // suppress -Wmaybe-uninitialized
    eocd->diskNumber                        = 0;
    eocd->centralDirectoryDiskNumber        = 0;
    eocd->centralDirectoryRecordCountOnDisk = 0;
    eocd->centralDirectoryRecordCount       = 0;
    eocd->centralDirectorySize              = 0;
    eocd->centralDirectoryOffset            = 0;

    ZL_ERR_IF_LT(
            (size_t)(eocd64End - eocd64Ptr),
            kMinZip64EndOfCentralDirectoryRecordSize,
            corruption);

    ZL_ERR_IF_NE(
            ZL_readLE32(eocd64Ptr),
            0x06064b50,
            corruption,
            "Zip64 End of Central Directory signature incorrect");
    const uint64_t eocd64Size = ZL_readLE64(eocd64Ptr + 4);
    ZL_ERR_IF_GT(
            eocd64Size, (uint64_t)(eocd64End - (eocd64Ptr + 12)), corruption);
    ZL_ERR_IF_LT(
            eocd64Size,
            kMinZip64EndOfCentralDirectoryRecordSize - 12,
            corruption);

    eocd->diskNumber                        = ZL_readLE32(eocd64Ptr + 16);
    eocd->centralDirectoryDiskNumber        = ZL_readLE32(eocd64Ptr + 20);
    eocd->centralDirectoryRecordCountOnDisk = ZL_readLE64(eocd64Ptr + 24);
    eocd->centralDirectoryRecordCount       = ZL_readLE64(eocd64Ptr + 32);
    eocd->centralDirectorySize              = ZL_readLE64(eocd64Ptr + 40);
    eocd->centralDirectoryOffset            = ZL_readLE64(eocd64Ptr + 48);

    return ZL_returnValue(eocd64Size);
}

/**
 * Validate the central directory by locating the first local file header and
 * validating its signature.
 */
static bool ZS2_ZipLexer_validateCentralDirectory(
        const ZS2_ZipLexer* lexer,
        const char* zipBegin,
        const char* centralDirectoryPtr)
{
    // Locate the local file header by reading the CDFH
    uint64_t localFileHeaderOffset;
    uint64_t compressedSize;
    if (ZL_isError(readCentralDirectoryFileHeader(
                lexer->opCtx,
                centralDirectoryPtr,
                lexer->srcEnd,
                &localFileHeaderOffset,
                &compressedSize))) {
        return false;
    }

    // Validate the first local file header points to a valid signature
    if (localFileHeaderOffset > (size_t)(lexer->srcEnd - zipBegin)) {
        return false;
    }
    const char* const lfhPtr = zipBegin + localFileHeaderOffset;
    if ((size_t)(lexer->srcEnd - lfhPtr) < kMinLocalHeaderSize) {
        return false;
    }
    if (ZL_readLE32(lfhPtr) != 0x04034b50) {
        return false;
    }
    return true;
}

/**
 * Validates the EOCD64 record by using it to locate the central directory and
 * validating the central directory.
 */
static bool ZS2_ZipLexer_validateEOCD64(
        const ZS2_ZipLexer* lexer,
        const char* zipBegin,
        const char* eocd64Ptr)
{
    ZS2_ZipLexer_EndOfCentralDirectory eocd;
    if (ZL_isError(readEOCD64(eocd64Ptr, lexer->srcEnd, &eocd))) {
        return false;
    }
    if (eocd.centralDirectoryOffset > (size_t)(lexer->srcEnd - zipBegin)) {
        return false;
    }
    return ZS2_ZipLexer_validateCentralDirectory(
            lexer, zipBegin, zipBegin + eocd.centralDirectoryOffset);
}

/**
 * Determines the beginning of the zip file by comparing two methods of finding
 * the same record signature. We use the known size of the record and subtract
 * it from where the following field begins, and we use the record offset and
 * add it to `lexer->zipBegin`. Then we search for the signature in the range of
 * possible values. Once we find the signature, we validate the record using
 * the validation function.
 *
 * In Zip64 files, we use the EOCD64 as the record, otherwise we use the Central
 * Directory as the record.
 *
 * Adjusts `lexer->zipBegin` based on this process.
 */
static ZL_Report ZS2_ZipLexer_findZipBegin(
        ZS2_ZipLexer* lexer,
        uint32_t recordSignature,
        size_t recordOffset,
        size_t minRecordSize,
        const char* maxRecordEnd,
        bool (*validate)(const ZS2_ZipLexer*, const char*, const char*))
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    ZL_ERR_IF_LT(minRecordSize, 4, corruption);
    ZL_ERR_IF_GT(
            minRecordSize,
            (size_t)(maxRecordEnd - lexer->zipBegin),
            corruption);
    const char* const maxRecordBegin = maxRecordEnd - minRecordSize;

    ZL_ERR_IF_GT(
            recordOffset,
            (size_t)(maxRecordBegin - lexer->zipBegin),
            corruption);
    const char* const minRecordBegin = lexer->zipBegin + recordOffset;

    // We must search from min to max because the central directory signature
    // shows up multiple times within the central directory.
    const char* recordBegin = minRecordBegin;
    for (;;) {
        ZL_ASSERT_LE(minRecordBegin, maxRecordBegin);
        ZL_ASSERT_GE((size_t)(lexer->srcEnd - maxRecordBegin), minRecordSize);
        // Filter out any records that don't have the correct signature
        if (ZL_readLE32(recordBegin) == recordSignature) {
            const char* const zipBegin = recordBegin - recordOffset;
            // Run the more expensive validation to make sure it matches.
            // The validation uses a pointer found in the record to look
            // for another record, and validates its signature, so it is
            // unlikely to have false positives.
            if (validate(lexer, zipBegin, recordBegin)) {
                break;
            }
        }
        ZL_ERR_IF_EQ(recordBegin, maxRecordBegin, corruption);
        ++recordBegin;
    }
    ZL_ASSERT_LE(recordOffset, (size_t)(recordBegin - lexer->zipBegin));
    lexer->zipBegin = recordBegin - recordOffset;
    return ZL_returnSuccess();
}

/**
 * Find the EOCD64 given the pointer to the EOCD, fill out the @p eocd struct,
 * and set some state in the lexer to record where the Zip64 EOCD record and
 * locator are. Additionally, fill `lexer->zipBegin`.
 */
static ZL_Report ZS2_ZipLexer_findEOCD64(
        ZS2_ZipLexer* lexer,
        const char* eocdPtr,
        ZS2_ZipLexer_EndOfCentralDirectory* eocd)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    ZL_ASSERT(ZS2_ZipLexer_EndOfCentralDirectory_needZip64(eocd));
    ZL_ASSERT_GE(eocdPtr, lexer->zipBegin);
    ZL_ASSERT_LE(eocdPtr, lexer->srcEnd);
    const size_t eocdOffset = (size_t)(eocdPtr - lexer->zipBegin);

    ZL_ERR_IF_LT(
            eocdOffset, kZip64EndOfCentralDirectoryLocatorSize, corruption);
    const char* const locatorPtr =
            eocdPtr - kZip64EndOfCentralDirectoryLocatorSize;

    ZL_ERR_IF_NE(
            ZL_readLE32(locatorPtr),
            0x07064b50,
            corruption,
            "Zip64 End of Central Directory Locator signature incorrect");
    ZL_ERR_IF_NE(
            ZL_readLE32(locatorPtr + 4),
            0,
            corruption,
            "Only single disk supported");
    ZL_ERR_IF_NE(
            ZL_readLE32(locatorPtr + 16),
            1,
            corruption,
            "Only single disk supported");

    const uint64_t eocd64Offset = ZL_readLE64(locatorPtr + 8);
    ZL_ERR_IF_GT(
            eocd64Offset, (size_t)(locatorPtr - lexer->zipBegin), corruption);

    ZL_ERR_IF_ERR(ZS2_ZipLexer_findZipBegin(
            lexer,
            0x06064b50,
            eocd64Offset,
            kMinZip64EndOfCentralDirectoryRecordSize,
            locatorPtr,
            ZS2_ZipLexer_validateEOCD64));

    const char* eocd64Ptr = lexer->zipBegin + eocd64Offset;
    ZL_TRY_LET(size_t, eocd64Size, readEOCD64(eocd64Ptr, locatorPtr, eocd));

    lexer->zip64EndOfCentralDirectoryRecordPtr  = eocd64Ptr;
    lexer->zip64EndOfCentralDirectoryRecordSize = eocd64Size + 12;
    lexer->zip64EndOfCentralDirectoryLocatorPtr = locatorPtr;

    return ZL_returnSuccess();
}

/**
 * Find the EOCD in the Zip file, and if Zip64 also find the EOCD64.
 * Fill out the lexer state to find the Central Directory, as well as
 * record the location of the EOCD and other Zip sections for later lexing.
 */
static ZL_Report ZS2_ZipLexer_parseEOCD(ZS2_ZipLexer* lexer, size_t eocdOffset)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    ZL_ERR_IF_GT(
            eocdOffset, (size_t)(lexer->srcEnd - lexer->zipBegin), corruption);
    const char* const eocdPtr = lexer->zipBegin + eocdOffset;
    ZL_ERR_IF_LT(
            (size_t)(lexer->srcEnd - eocdPtr),
            kMinEndOfCentralDirectoryRecordSize,
            corruption);
    ZL_ERR_IF(!isEOCD(eocdPtr), corruption);

    ZS2_ZipLexer_EndOfCentralDirectory eocd;
    eocd.diskNumber                        = read16OrNeg1(eocdPtr + 4);
    eocd.centralDirectoryDiskNumber        = read16OrNeg1(eocdPtr + 6);
    eocd.centralDirectoryRecordCountOnDisk = read16OrNeg1(eocdPtr + 8);
    eocd.centralDirectoryRecordCount       = read16OrNeg1(eocdPtr + 10);
    eocd.centralDirectorySize              = read32OrNeg1(eocdPtr + 12);
    eocd.centralDirectoryOffset            = read32OrNeg1(eocdPtr + 16);

    // Check the comment isn't too long, but allow garbage at the end of the zip
    // file.
    const uint32_t commentLength = ZL_readLE16(eocdPtr + 20);
    ZL_ERR_IF_GT(
            commentLength,
            lexer->srcEnd - (eocdPtr + kMinEndOfCentralDirectoryRecordSize),
            corruption);

    if (ZS2_ZipLexer_EndOfCentralDirectory_needZip64(&eocd)) {
        ZL_ERR_IF_ERR(ZS2_ZipLexer_findEOCD64(lexer, eocdPtr, &eocd));
    } else {
        if (eocd.centralDirectoryRecordCount > 0) {
            // If there are any entries in the Central Directory, we need to
            // adjust the beginning of the zip file in case there is garbage at
            // the beginning.
            ZL_ERR_IF_ERR(ZS2_ZipLexer_findZipBegin(
                    lexer,
                    0x02014b50,
                    eocd.centralDirectoryOffset,
                    eocd.centralDirectorySize,
                    eocdPtr,
                    ZS2_ZipLexer_validateCentralDirectory));
        }
        lexer->zip64EndOfCentralDirectoryRecordPtr  = NULL;
        lexer->zip64EndOfCentralDirectoryRecordSize = 0;
        lexer->zip64EndOfCentralDirectoryLocatorPtr = NULL;
    }
    // Compute srcSize now that we've adjusted lexer->zipBegin
    const size_t srcSize = (size_t)(lexer->srcEnd - lexer->zipBegin);

    ZL_ERR_IF_NE(eocd.diskNumber, 0, corruption, "Only single disk supported");
    ZL_ERR_IF_NE(
            eocd.centralDirectoryDiskNumber,
            0,
            corruption,
            "Only single disk supported");

    ZL_ERR_IF_GT(eocd.centralDirectoryOffset, srcSize, corruption);
    ZL_ERR_IF_GT(
            eocd.centralDirectorySize,
            srcSize - eocd.centralDirectoryOffset,
            corruption);

    // Sanity check the number of files, so we don't report an
    // impossible number.
    const size_t kMinBytesPerFile =
            kMinLocalHeaderSize + kMinCentralDirectoryFileHeaderSize;
    const size_t maxNumFilesPossible = srcSize / kMinBytesPerFile;
    ZL_ERR_IF_GT(
            eocd.centralDirectoryRecordCount, maxNumFilesPossible, corruption);

    lexer->cdfhPtr = lexer->zipBegin + eocd.centralDirectoryOffset;
    lexer->cdfhEnd = lexer->cdfhPtr + eocd.centralDirectorySize;

    lexer->cdfhIdx = 0;
    lexer->cdfhNum = eocd.centralDirectoryRecordCount;

    lexer->centralDirectoryPtr            = lexer->cdfhPtr;
    lexer->endOfCentralDirectoryRecordPtr = eocdPtr;
    lexer->endOfCentralDirectoryRecordSize =
            (uint32_t)kMinEndOfCentralDirectoryRecordSize + commentLength;

    return ZL_returnSuccess();
}

static ZL_Report ZS2_ZipLexer_setFileState(ZS2_ZipLexer* lexer);

static ZL_Report ZS2_ZipLexer_initWithEOCD(
        ZS2_ZipLexer* lexer,
        const void* src,
        size_t srcSize,
        size_t eocdOffset)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    lexer->zipBegin = src;
    lexer->srcPtr   = src;
    lexer->srcEnd   = lexer->zipBegin + srcSize;

    memset(&lexer->fileState, 0, sizeof(lexer->fileState));

    ZL_ERR_IF_ERR(ZS2_ZipLexer_parseEOCD(lexer, eocdOffset));

    if (lexer->cdfhIdx < lexer->cdfhNum) {
        // Proactively initialize the file state to catch more invalid zip files
        // in the init() function.
        ZL_ERR_IF_ERR(ZS2_ZipLexer_setFileState(lexer));
    }
    return ZL_returnSuccess();
}

static bool ZS2_ZipLexer_tryInit(
        ZS2_ZipLexer* lexer,
        const void* src,
        size_t srcSize,
        size_t reverseOffset)
{
    ZL_ASSERT_LE(reverseOffset, srcSize);
    ZL_ASSERT_GE(reverseOffset, kMinEndOfCentralDirectoryRecordSize);
    const char* const srcEnd = (const char*)src + srcSize;
    if (!isEOCD(srcEnd - reverseOffset)) {
        return false;
    }
    const ZL_Report ret = ZS2_ZipLexer_initWithEOCD(
            lexer, src, srcSize, srcSize - reverseOffset);
    if (ZL_isError(ret)) {
        return false;
    }
    return true;
}

ZL_Report ZS2_ZipLexer_init(
        ZS2_ZipLexer* lexer,
        const void* src,
        size_t srcSize,
        ZL_OperationContext* opCtx)
{
    lexer->opCtx = opCtx;
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    const size_t minReverseOffset = kMinEndOfCentralDirectoryRecordSize;
    // Maximum allowed offset if there is no garbage at the end
    const size_t maxLegalReverseOffset =
            ZL_MIN(kMinEndOfCentralDirectoryRecordSize + 65535, srcSize);
    const size_t maxReverseOffset = srcSize;

    ZL_ERR_IF_LT(
            srcSize,
            kMinEndOfCentralDirectoryRecordSize,
            GENERIC,
            "Zip file too small");
    ZL_ASSERT_GE(maxReverseOffset, minReverseOffset);

    if (ZS2_ZipLexer_tryInit(lexer, src, srcSize, minReverseOffset)) {
        return ZL_returnSuccess();
    } else if (ZS2_ZipLexer_tryInit(
                       lexer, src, srcSize, maxLegalReverseOffset)) {
        return ZL_returnSuccess();
    }

    size_t reverseOffset;
    for (reverseOffset = minReverseOffset + 1; reverseOffset < maxReverseOffset;
         ++reverseOffset) {
        if (ZS2_ZipLexer_tryInit(lexer, src, srcSize, reverseOffset)) {
            break;
        }
    }
    ZL_ERR_IF_GE(reverseOffset, maxReverseOffset, GENERIC, "EOCD not found");
    return ZL_returnSuccess();
}

/**
 * Emits an unknown token. This is used for bytes in the Zip file that are
 * otherwise unaccounted for. The Zip format, read loosely, allows for gaps in
 * the file in many places, e.g. between files. This is not used for comment
 * fields.
 *
 * @post `lexer->srcPtr == nextPtr`
 */
static ZL_Report ZS2_ZipLexer_lexUnknown(
        ZS2_ZipLexer* lexer,
        ZS2_ZipToken* out,
        const char* nextPtr)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    ZL_ASSERT_GT(nextPtr, lexer->srcPtr);
    ZL_ASSERT_LE(nextPtr, lexer->srcEnd);

    memset(out, 0, sizeof(*out));
    out->type = ZS2_ZipTokenType_Unknown;
    out->ptr  = lexer->srcPtr;
    out->size = (size_t)(nextPtr - lexer->srcPtr);

    lexer->srcPtr = nextPtr;

    return ZL_returnSuccess();
}

/**
 * If `lexer->srcPtr < *sectionPtrPtr` then emit an unknown token.
 * Else If `lexer->srcPtr > *sectionPtrPtr` then emit an error.
 * Else then emit a token of type @p type and size @p sectionSize.
 *
 * @p sectionPtrPtr Pointer to a pointer containing the beginning of the
 * section. This pointer is set to `NULL`. This is intended to be used as
 * a signal to say if the section has already been consumed.
 *
 * @pre `*sectionPtrPtr != NULL`
 * @post `lexer->srcPtr = *sectionPtrPtr + sectionSize`
 * @post `*sectionPtrPtr == NULL`
 */
static ZL_Report ZS2_ZipLexer_lexSection(
        ZS2_ZipLexer* lexer,
        ZS2_ZipToken* out,
        ZS2_ZipTokenType type,
        const char** sectionPtrPtr,
        size_t sectionSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    // Ensure the srcPtr is at the beginning of the section
    if (lexer->srcPtr < *sectionPtrPtr) {
        return ZS2_ZipLexer_lexUnknown(lexer, out, *sectionPtrPtr);
    }
    ZL_ERR_IF_GT(lexer->srcPtr, *sectionPtrPtr, corruption);

    ZL_ASSERT_LE(sectionSize, (size_t)(lexer->srcEnd - *sectionPtrPtr));

    out->type = type;
    out->ptr  = lexer->srcPtr;
    out->size = sectionSize;

    lexer->srcPtr += sectionSize;
    *sectionPtrPtr = NULL;

    return ZL_returnSuccess();
}

/// Handles emitting tokens for all sections after the files.
static ZL_Report ZS2_ZipLexer_lexTail(ZS2_ZipLexer* lexer, ZS2_ZipToken* out)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    ZL_ERR_IF_NE(lexer->cdfhPtr, lexer->cdfhEnd, corruption);

    memset(out, 0, sizeof(*out));

    if (lexer->centralDirectoryPtr != NULL) {
        ZL_ASSERT_LE(lexer->centralDirectoryPtr, lexer->cdfhEnd);
        const size_t centralDirectorySize =
                (size_t)(lexer->cdfhEnd - lexer->centralDirectoryPtr);
        return ZS2_ZipLexer_lexSection(
                lexer,
                out,
                ZS2_ZipTokenType_CentralDirectory,
                &lexer->centralDirectoryPtr,
                centralDirectorySize);
    }

    if (lexer->zip64EndOfCentralDirectoryRecordPtr != NULL) {
        return ZS2_ZipLexer_lexSection(
                lexer,
                out,
                ZS2_ZipTokenType_Zip64EndOfCentralDirectoryRecord,
                &lexer->zip64EndOfCentralDirectoryRecordPtr,
                lexer->zip64EndOfCentralDirectoryRecordSize);
    }

    if (lexer->zip64EndOfCentralDirectoryLocatorPtr != NULL) {
        return ZS2_ZipLexer_lexSection(
                lexer,
                out,
                ZS2_ZipTokenType_Zip64EndOfCentralDirectoryLocator,
                &lexer->zip64EndOfCentralDirectoryLocatorPtr,
                kZip64EndOfCentralDirectoryLocatorSize);
    }

    if (lexer->endOfCentralDirectoryRecordPtr != NULL) {
        return ZS2_ZipLexer_lexSection(
                lexer,
                out,
                ZS2_ZipTokenType_EndOfCentralDirectoryRecord,
                &lexer->endOfCentralDirectoryRecordPtr,
                lexer->endOfCentralDirectoryRecordSize);
    }

    return ZS2_ZipLexer_lexUnknown(lexer, out, lexer->srcEnd);
}

/**
 * Reads the next central directory file header & advances to the next entry.
 * Fills the @p localFileHeaderOffset and @p compressedSize on success.
 * @warning Does not validate the values read.
 */
static ZL_Report ZS2_ZipLexer_readNextCentralDirectoryFileHeader(
        ZS2_ZipLexer* lexer,
        uint64_t* localFileHeaderOffset,
        uint64_t* compressedSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    ZL_ASSERT_LT(lexer->cdfhIdx, lexer->cdfhNum);
    ZL_TRY_LET(
            size_t,
            cdfhLength,
            readCentralDirectoryFileHeader(
                    lexer->opCtx,
                    lexer->cdfhPtr,
                    lexer->cdfhEnd,
                    localFileHeaderOffset,
                    compressedSize));
    lexer->cdfhIdx += 1;
    lexer->cdfhPtr += cdfhLength;
    return ZL_returnSuccess();
}

/// The file state is empty if all the sections have been consumed.
static bool ZS2_ZipLexer_FileState_empty(
        const ZS2_ZipLexer_FileState* fileState)
{
    return fileState->localFileHeaderPtr == NULL
            && fileState->compressedDataPtr == NULL
            && fileState->dataDescriptorPtr == NULL;
}

/**
 * Fills the `lexer->fileState` for the next file, and advances
 * the `cdfhIdx` and `cdfhPtr` to the next file.
 *
 * @pre `ZS2_ZipLexer_FileState_empty(&lexer->fileState)`
 */
static ZL_Report ZS2_ZipLexer_setFileState(ZS2_ZipLexer* lexer)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    ZL_ASSERT(ZS2_ZipLexer_FileState_empty(&lexer->fileState));

    // Read fields from the CDFH. The compressed size is read from the CDFH,
    // because it may be in the DataDescriptor in the LFH.
    uint64_t localFileHeaderOffset;
    uint64_t compressedSize;
    ZL_ERR_IF_ERR(ZS2_ZipLexer_readNextCentralDirectoryFileHeader(
            lexer, &localFileHeaderOffset, &compressedSize));

    ZL_ASSERT_LE(lexer->zipBegin, lexer->srcEnd);
    ZL_ERR_IF_GT(
            localFileHeaderOffset,
            (uint64_t)(lexer->srcEnd - lexer->zipBegin),
            corruption);
    const char* const lfhPtr = lexer->zipBegin + localFileHeaderOffset;
    ZL_ERR_IF_LT(
            (size_t)(lexer->srcEnd - lfhPtr), kMinLocalHeaderSize, corruption);

    ZL_ERR_IF_NE(ZL_readLE32(lfhPtr), 0x04034b50, corruption);
    const uint16_t generalPurposeBits = ZL_readLE16(lfhPtr + 6);
    const uint16_t filenameLength     = ZL_readLE16(lfhPtr + 26);
    const size_t extraFieldLength     = ZL_readLE16(lfhPtr + 28);
    const size_t lfhSize =
            kMinLocalHeaderSize + filenameLength + extraFieldLength;
    const char* const extraFieldPtr =
            lfhPtr + kMinLocalHeaderSize + filenameLength;
    ZL_ERR_IF_LT((size_t)(lexer->srcEnd - lfhPtr), lfhSize, corruption);

    const bool hasDataDescriptor =
            (generalPurposeBits & kDataDescriptorMask) != 0;

    const char* const compressedDataPtr = lfhPtr + lfhSize;
    ZL_ERR_IF_GT(
            compressedSize,
            (size_t)(lexer->srcEnd - compressedDataPtr),
            corruption);

    lexer->fileState.compressionMethod = ZL_readLE16(lfhPtr + 8);
    lexer->fileState.filename          = lfhPtr + kMinLocalHeaderSize;
    lexer->fileState.filenameSize      = filenameLength;

    lexer->fileState.localFileHeaderPtr  = lfhPtr;
    lexer->fileState.localFileHeaderSize = lfhSize;
    lexer->fileState.compressedDataPtr   = compressedDataPtr;
    lexer->fileState.compressedDataSize  = compressedSize;

    if (hasDataDescriptor) {
        const char* const dataDescriptorPtr =
                compressedDataPtr + compressedSize;
        ZL_ERR_IF_LT(lexer->srcEnd - dataDescriptorPtr, 4, corruption);
        const uint32_t signature = ZL_readLE32(dataDescriptorPtr);
        const bool hasSignature  = signature == 0x08074b50;
        const bool isZip64 = hasZip64Info(extraFieldPtr, extraFieldLength);
        const size_t dataDescriptorSize =
                (hasSignature ? 4u : 0u) + 4u + (isZip64 ? 16u : 8u);
        ZL_ERR_IF_LT(
                (size_t)(lexer->srcEnd - dataDescriptorPtr),
                dataDescriptorSize,
                corruption);

        lexer->fileState.dataDescriptorPtr  = dataDescriptorPtr;
        lexer->fileState.dataDescriptorSize = dataDescriptorSize;
    } else {
        lexer->fileState.dataDescriptorPtr  = NULL;
        lexer->fileState.dataDescriptorSize = 0;
    }

    return ZL_returnSuccess();
}

static ZL_Report ZS2_ZipLexer_lexFile(ZS2_ZipLexer* lexer, ZS2_ZipToken* out)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    ZL_ASSERT_LE(lexer->cdfhIdx, lexer->cdfhNum);
    ZL_ASSERT_LE(lexer->cdfhPtr, lexer->cdfhEnd);

    // If the current file state is empty, fill the file state for the next
    // file.
    if (ZS2_ZipLexer_FileState_empty(&lexer->fileState)) {
        ZL_ASSERT_LT(lexer->cdfhIdx, lexer->cdfhNum);
        ZL_ERR_IF_ERR(ZS2_ZipLexer_setFileState(lexer));
        ZL_ASSERT_NN(lexer->fileState.localFileHeaderPtr);
        ZL_ASSERT_NN(lexer->fileState.compressedDataPtr);
    }

    // Set the file-related fields
    out->compressionMethod = lexer->fileState.compressionMethod;
    out->filenameSize      = lexer->fileState.filenameSize;
    out->filename          = lexer->fileState.filename;

    if (lexer->fileState.localFileHeaderPtr != NULL) {
        return ZS2_ZipLexer_lexSection(
                lexer,
                out,
                ZS2_ZipTokenType_LocalFileHeader,
                &lexer->fileState.localFileHeaderPtr,
                lexer->fileState.localFileHeaderSize);
    }

    if (lexer->fileState.compressedDataPtr != NULL) {
        return ZS2_ZipLexer_lexSection(
                lexer,
                out,
                ZS2_ZipTokenType_CompressedData,
                &lexer->fileState.compressedDataPtr,
                lexer->fileState.compressedDataSize);
    }

    if (lexer->fileState.dataDescriptorPtr != NULL) {
        return ZS2_ZipLexer_lexSection(
                lexer,
                out,
                ZS2_ZipTokenType_DataDescriptor,
                &lexer->fileState.dataDescriptorPtr,
                lexer->fileState.dataDescriptorSize);
    }

    ZL_ERR(logicError);
}

static ZL_Report ZS2_ZipLexer_lexOne(ZS2_ZipLexer* lexer, ZS2_ZipToken* out)
{
    // If we've already parsed every file, move on to lexing the trailing
    // metadata sections.
    if (lexer->cdfhIdx == lexer->cdfhNum
        && ZS2_ZipLexer_FileState_empty(&lexer->fileState)) {
        return ZS2_ZipLexer_lexTail(lexer, out);
    }

    return ZS2_ZipLexer_lexFile(lexer, out);
}

ZL_Report
ZS2_ZipLexer_lex(ZS2_ZipLexer* lexer, ZS2_ZipToken* out, size_t outCapacity)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(lexer->opCtx);
    size_t entries;
    for (entries = 0; !ZS2_ZipLexer_finished(lexer) && entries < outCapacity;
         ++entries) {
        ZL_ERR_IF_ERR(ZS2_ZipLexer_lexOne(lexer, out + entries));
    }
    return ZL_returnValue(entries);
}

bool ZS2_ZipLexer_finished(const ZS2_ZipLexer* lexer)
{
    return lexer->srcPtr == lexer->srcEnd;
}

size_t ZS2_ZipLexer_expectedNumTokens(const ZS2_ZipLexer* lexer)
{
    return lexer->cdfhNum * 4 + 4;
}

size_t ZS2_ZipLexer_numFiles(const ZS2_ZipLexer* lexer)
{
    return lexer->cdfhNum;
}

bool ZS2_isLikelyZipFile(const void* src, size_t srcSize)
{
    ZS2_ZipLexer lexer;
    const ZL_Report report = ZS2_ZipLexer_init(&lexer, src, srcSize, NULL);
    return !ZL_isError(report);
}
