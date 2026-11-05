// Copyright (c) Meta Platforms, Inc. and affiliates.

// Main decompression function

#include <stdint.h>
#include "openzl/common/allocation.h"      // ZL_calloc, ZL_free
#include "openzl/common/assertion.h"       // ZS_ASSERT_*
#include "openzl/common/buffer_internal.h" // ZL_RCursor
#include "openzl/common/errors_internal.h" // ZS2_RET_IF_ERR
#include "openzl/common/limits.h"
#include "openzl/common/logging.h"
#include "openzl/common/operation_context.h"
#include "openzl/common/stream.h" // ZL_Data
#include "openzl/common/vector.h"
#include "openzl/common/wire_format.h"            // TransformType_e
#include "openzl/decompress/dctx2.h"              // DCTX_* declarations
#include "openzl/decompress/decode_frameheader.h" // DFH_*
#include "openzl/decompress/decoder_fusion.h"
#include "openzl/decompress/dictx.h"       // struct ZL_Decoder_s
#include "openzl/decompress/dtransforms.h" // DTransforms_manager, TransformID
#include "openzl/decompress/gdparams.h"
#include "openzl/dict/dict_constants.h" // ZL_DICT_INDEX_NONE
#include "openzl/shared/mem.h"          // ZL_readLE32, etc.
#include "openzl/shared/xxhash.h"       // XXH3_64bits
#include "openzl/zl_buffer.h"           // ZL_RBuffer
#include "openzl/zl_data.h"
#include "openzl/zl_decompress.h" // ZL_TypedDecoderDesc
#include "openzl/zl_dict.h"       // ZL_Dict, ZL_DictBundle
#include "openzl/zl_dictloader.h" // ZL_DictLoader_fetchDictBundle
#include "openzl/zl_dtransform.h" //
#include "openzl/zl_errors.h"
#include "openzl/zl_opaque_types.h" // ZL_IDType
#include "openzl/zl_version.h"

// --------------------------
// Struct Definitions
// --------------------------

/**
 * The AppendToOutputOptimization allows decoders that act like concat to
 * append inputs directly to the output buffer. This reduces memory usage and
 * can also elide a copy in some cases.
 *
 * Transforms are concat-like if:
 * - They have one or more encoded streams
 * - They produce exactly one decoded stream
 * - The decoder produces the decoded stream by concatenating all the encoded
 *   streams.
 */
typedef struct ZL_AppendToOutputOptimization_s {
    /// Pointer to the inputs array
    /// @note Inputs are listed in reverse order!
    struct ZL_DCtx_DataInfo* inputInfos;
    /// Index of the next input to append to the head of the output.
    /// Starts at 0.
    size_t headInputIdx;
    /// Index of the next input to prepend to the tail of the output.
    /// Starts at `nbInputs`.
    size_t tailInputIdx;
    size_t nbInputs; ///< Number of input streams

    /// Pointer to the output
    struct ZL_DCtx_DataInfo* outputInfo;
    /// Head inputs get appended to this pointer
    uint8_t* outputHeadPtr;
    /// Tail inputs get prepended to this pointer
    uint8_t* outputTailPtr;
} ZL_AppendToOutputOptimization;

typedef struct {
    ZL_DCtx_DataInfo* ptr;
    size_t size;
} ZL_DCtx_DataInfoArray;

struct ZL_DCtx_s {
    DTransforms_manager dtm;
    ZL_DecoderFusionState fusion; ///< Registered decoder fusions
    DFH_Struct dfh;
    ZL_DCtx_DataInfoArray dataInfo;
    ZL_Data** outputs;
    size_t nbOutputs;
    ZL_RBuffer thstream;
    size_t currentStreamNb;
    size_t streamEndPos;
    bool preserveStreams;
    Arena* chunkArena; ///< Lives for the lifetime of the chunk decompression
    Arena* decoderWkspArena; ///< Lives for the lifetime of a single decoder
    /// Lives for the lifetime of a single decoder fusion
    Arena* fusionWkspArena;
    Arena* streamArena;
    ZL_OperationContext opCtx;
    GDParams requestedGDParams;  // As user-selected at DCtx level
    GDParams appliedGDParams;    // Used at decompression time; DCtx > default
    ZL_DictLoader* dictLoader;   // Referenced, not owned
    const ZL_DictBundle* bundle; // Current frame's resolved bundle, or NULL
}; // typedef'd to ZL_DCtx within zs2_decompress.h

// --------------------------
// Lifetime management
// --------------------------
ZL_DCtx* ZL_DCtx_create(void)
{
    ZL_DCtx* const dctx = ZL_calloc(sizeof(ZL_DCtx));
    if (!dctx) {
        return NULL;
    }

    ZL_OC_init(&dctx->opCtx);
    ZL_OC_startOperation(&dctx->opCtx, ZL_Operation_decompress);

    dctx->chunkArena = ALLOC_StackArena_create();
    if (!dctx->chunkArena) {
        ZL_DCtx_free(dctx);
        return NULL;
    }
    dctx->decoderWkspArena = ALLOC_StackArena_create();
    if (!dctx->decoderWkspArena) {
        ZL_DCtx_free(dctx);
        return NULL;
    }
    dctx->fusionWkspArena = ALLOC_StackArena_create();
    if (!dctx->fusionWkspArena) {
        ZL_DCtx_free(dctx);
        return NULL;
    }
    // Streams are reallocated at every chunk: keeping their buffers across
    // chunks and decompressions avoids page faults (see ZL_DCtx_setStreamArena)
    dctx->streamArena = ALLOC_StackArena_create();
    if (!dctx->streamArena) {
        ZL_DCtx_free(dctx);
        return NULL;
    }
    if (ZL_isError(DTM_init(
                &dctx->dtm,
                ZL_GET_OPERATION_CONTEXT(dctx),
                ZL_CUSTOM_TRANSFORM_LIMIT))) {
        ZL_DCtx_free(dctx);
        return NULL;
    }
    if (ZL_isError(ZL_DecoderFusionState_init(&dctx->fusion))) {
        ZL_DCtx_free(dctx);
        return NULL;
    }
    DFH_init(&dctx->dfh);
    return dctx;
}

ZL_Report ZL_DCtx_setStreamArena(ZL_DCtx* dctx, ZL_DataArenaType sat)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_NN(dctx);
    Arena* newArena = NULL;
    switch (sat) {
        case ZL_DataArenaType_heap:
            newArena = ALLOC_HeapArena_create();
            break;
        case ZL_DataArenaType_stack:
            newArena = ALLOC_StackArena_create();
            break;
        default:
            ZL_ERR_IF(1, parameter_invalid, "Stream Arena type is invalid");
    }
    ZL_ERR_IF_NULL(newArena, allocation);
    ALLOC_Arena_freeArena(dctx->streamArena);
    dctx->streamArena = newArena;
    return ZL_returnSuccess();
}

void DCTX_preserveStreams(ZL_DCtx* dctx)
{
    ZL_ASSERT_NN(dctx);
    dctx->preserveStreams = true;
}

static void DCTX_freeStreams(ZL_DCtx* dctx)
{
    ZL_DLOG(SEQ, "DCTX_freeStreams");
    ZL_ASSERT_NN(dctx);
    size_t const nbStreams = dctx->dataInfo.size;
    ZL_DLOG(SEQ, "free %zu Streams", nbStreams);
    ALLOC_Arena_freeAll(dctx->streamArena);
}

void ZL_DCtx_free(ZL_DCtx* dctx)
{
    if (dctx == NULL)
        return;
    DCTX_freeStreams(dctx);
    DTM_destroy(&dctx->dtm);
    ZL_DecoderFusionState_destroy(&dctx->fusion);
    DFH_destroy(&dctx->dfh);
    ALLOC_Arena_freeArena(dctx->decoderWkspArena);
    ALLOC_Arena_freeArena(dctx->fusionWkspArena);
    ALLOC_Arena_freeArena(dctx->streamArena);
    ALLOC_Arena_freeArena(dctx->chunkArena);
    ZL_OC_destroy(&dctx->opCtx);
    ZL_free(dctx);
}

size_t ZL_DCtx_sizeof(const ZL_DCtx* dctx)
{
    if (dctx == NULL) {
        return 0;
    }
    return sizeof(*dctx) + DFH_sizeof(&dctx->dfh) + DTM_sizeof(&dctx->dtm)
            + ZL_OC_sizeof(&dctx->opCtx)
            + ALLOC_Arena_memAllocated(dctx->fusion.arena)
            + ALLOC_Arena_memAllocated(dctx->decoderWkspArena)
            + ALLOC_Arena_memAllocated(dctx->fusionWkspArena)
            + ALLOC_Arena_memAllocated(dctx->streamArena)
            + ALLOC_Arena_memAllocated(dctx->chunkArena);
}

ZL_Report DCTX_registerDecoderFusion(
        ZL_DCtx* dctx,
        const ZL_DecoderFusionDesc* fusion)
{
    return ZL_DecoderFusionState_registerFusion(&dctx->fusion, fusion);
}

void DCTX_clearDecoderFusions(ZL_DCtx* dctx)
{
    ZL_DecoderFusionState_clearFusions(&dctx->fusion);
}

void ZL_DCtx_refDictLoader(ZL_DCtx* dctx, ZL_DictLoader* loader)
{
    ZL_ASSERT_NN(dctx);
    dctx->dictLoader = loader;
}

ZL_Report ZL_DCtx_setParameter(ZL_DCtx* dctx, ZL_DParam gdparam, int value)
{
    ZL_ASSERT_NN(dctx);
    return GDParams_setParameter(&dctx->requestedGDParams, gdparam, value);
}

int ZL_DCtx_getParameter(const ZL_DCtx* dctx, ZL_DParam gdparam)
{
    ZL_ASSERT_NN(dctx);
    return GDParams_getParameter(&dctx->requestedGDParams, gdparam);
}

ZL_Report ZL_DCtx_resetParameters(ZL_DCtx* dctx)
{
    ZL_zeroes(&dctx->requestedGDParams, sizeof(dctx->requestedGDParams));
    return ZL_returnSuccess();
}

ZL_Report DCtx_setAppliedParameters(ZL_DCtx* dctx)
{
    ZL_ASSERT_NN(dctx);
    GDParams* const dst      = &dctx->appliedGDParams;
    const GDParams* const p1 = &dctx->requestedGDParams;
    const GDParams* const p2 = &GDParams_default;

    *dst = *p1;
    GDParams_applyDefaults(dst, p2);

    return GDParams_finalize(dst);
}

int DCtx_getAppliedGParam(const ZL_DCtx* dctx, ZL_DParam gdparam)
{
    ZL_ASSERT_NN(dctx);
    return GDParams_getParameter(&dctx->appliedGDParams, gdparam);
}

// --------------------------
// Accessors
// --------------------------

unsigned ZL_DCtx_getFrameFormatVersion(const ZL_DCtx* dctx)
{
    return dctx->dfh.formatVersion;
}

size_t ZL_DCtx_getNumStreams(const ZL_DCtx* dctx)
{
    return dctx->dataInfo.size;
}

const ZL_Data* ZL_DCtx_getConstStream(const ZL_DCtx* dctx, ZL_IDType streamID)
{
    if (streamID >= ZL_DCtx_getNumStreams(dctx)) {
        return NULL;
    }
    return dctx->dataInfo.ptr[streamID].data;
}

ZL_IDType DCTX_getStreamProducerNodeIdx(const ZL_DCtx* dctx, ZL_IDType streamID)
{
    if (dctx == NULL || dctx->dataInfo.ptr == NULL
        || streamID >= dctx->dataInfo.size) {
        return ZL_PRODUCER_STORE;
    }
    return dctx->dataInfo.ptr[streamID].producerNodeIdx;
}

ZL_Report ZL_DCtx_registerPipeDecoder(
        ZL_DCtx* dctx,
        const ZL_PipeDecoderDesc* ctd)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_NN(dctx);
    ZL_ASSERT_NN(ctd);
    ZL_ERR_IF_ERR(DTM_registerDPipeTransform(&dctx->dtm, ctd));
    return ZL_returnSuccess();
}

ZL_Report ZL_DCtx_registerSplitDecoder(
        ZL_DCtx* dctx,
        const ZL_SplitDecoderDesc* ctd)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_NN(dctx);
    ZL_ASSERT_NN(ctd);
    ZL_ERR_IF_ERR(DTM_registerDSplitTransform(&dctx->dtm, ctd));
    return ZL_returnSuccess();
}

ZL_Report ZL_DCtx_registerTypedDecoder(
        ZL_DCtx* dctx,
        const ZL_TypedDecoderDesc* dttd)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_NN(dctx);
    ZL_ASSERT_NN(dttd);
    // WARNING: Must not fail before this line otherwise opaque will be leaked
    ZL_ERR_IF_ERR(DTM_registerDTypedTransform(&dctx->dtm, dttd));
    return ZL_returnSuccess();
}

ZL_Report ZL_DCtx_registerVODecoder(
        ZL_DCtx* dctx,
        const ZL_VODecoderDesc* dvotd)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_DLOG(BLOCK,
            "ZL_DCtx_registerVODecoder '%s'",
            STR_REPLACE_NULL(dvotd->name));
    ZL_ASSERT_NN(dctx);
    ZL_ASSERT_NN(dvotd);
    // WARNING: Must not fail before this line otherwise opaque will be leaked
    ZL_ERR_IF_ERR(DTM_registerDVOTransform(&dctx->dtm, dvotd));
    return ZL_returnSuccess();
}

ZL_Report ZL_DCtx_registerMIDecoder(
        ZL_DCtx* dctx,
        const ZL_MIDecoderDesc* dmitd)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_NN(dctx);
    ZL_ASSERT_NN(dmitd);
    // WARNING: Must not fail before this line otherwise opaque will be leaked
    ZL_ERR_IF_ERR(DTM_registerDMITransform(&dctx->dtm, dmitd));
    return ZL_returnSuccess();
}

// ------------------------------
// ZL_AppendToOutputOptimization
// ------------------------------

/**
 * Hook to try to initialize the AppendToOutputOptimization.
 *
 * @param node The decoder information
 * @param inputInfos The inputs array
 * @param nbInputs Number of inputs to the decoder
 * @param[inout] outputInfo The output info entry. Must be empty. It will be
 * filled if the optimization is enabled.
 * @param outputData The output data from `dctx->outputs` for the decompression
 * operation that this decoder is regenerating.
 *
 * @returns An error, or non-zero if the optimization is enabled.
 */
static ZL_Report ZL_AppendToOutputOptimization_register(
        ZL_DCtx* dctx,
        const DFH_NodeInfo* node,
        ZL_DCtx_DataInfo* inputInfos,
        size_t nbInputs,
        ZL_DCtx_DataInfo* outputInfo,
        ZL_Data* outputData)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    if (dctx->preserveStreams) {
        // Does not work with stream preservation, so disable the optimization.
        return ZL_returnValue(0);
    }
    if (!STREAM_hasBuffer(outputData)) {
        // Can only optimize when there is already an output buffer
        // TODO: We could inspect the frame & find the output size
        return ZL_returnValue(0);
    }
    if (nbInputs < 1) {
        return ZL_returnValue(0);
    }
    if (node->nbRegens != 1) {
        return ZL_returnValue(0);
    }
    if (ZL_Data_type(outputData) != 0
        && ZL_Data_type(outputData) != ZL_Type_serial) {
        // Only works with serial streams.
        // type=0 means that the type of the output stream is not yet set, and
        // will be set by the decoder.
        return ZL_returnValue(0);
    }
    if (node->trpid.trt != trt_standard) {
        return ZL_returnValue(0);
    }

    uint8_t* outputPtr          = ZL_Data_wPtr(outputData);
    const size_t outputCapacity = STREAM_byteCapacity(outputData);

    switch (node->trpid.trid) {
        default:
            return ZL_returnValue(0);
        case ZL_StandardTransformID_convert_serial_to_struct:
            break;
        case ZL_StandardTransformID_splitn:
            break;
    }
    ZL_AppendToOutputOptimization* append = ALLOC_Arena_calloc(
            dctx->chunkArena, sizeof(ZL_AppendToOutputOptimization));
    ZL_ERR_IF_NULL(append, allocation);
    append->inputInfos   = inputInfos;
    append->nbInputs     = nbInputs;
    append->headInputIdx = 0;
    append->tailInputIdx = nbInputs;

    append->outputInfo    = outputInfo;
    append->outputHeadPtr = outputPtr;
    append->outputTailPtr = outputPtr + outputCapacity;

    ZL_ERR_IF_ERR(STREAM_typeAttachedBuffer(
            outputData, ZL_Type_serial, 1, outputCapacity));

    // Everything succeeded, make stateful changes to infos
    for (size_t i = 0; i < nbInputs; ++i) {
        ZL_ASSERT_NULL(inputInfos[i].appendOpt);
        inputInfos[i].appendOpt = append;
    }

    ZL_ASSERT_NULL(outputInfo->data);
    ZL_ASSERT_NULL(outputInfo->appendOpt);
    outputInfo->data      = outputData;
    outputInfo->appendOpt = append;

    return ZL_returnValue(1);
}

/**
 * Try to commit a single input at @p inputIdx into the output data.
 * Inputs that have been committed will be committed to the output buffer.
 * Once an input is committed to the output, it is freed.
 *
 * @returns An error, or non-zero if the stream was successfully committed.
 */
static ZL_Report ZL_AppendToOutputOptimization_commitInput(
        ZL_AppendToOutputOptimization* append,
        size_t inputIdx)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    // Append to head if equal to the headInputIdx.
    const bool head = (inputIdx == append->headInputIdx);
    // Inputs are listed in reverse order
    const size_t infoInputPos         = append->nbInputs - (inputIdx + 1);
    ZL_DCtx_DataInfo* const inputInfo = &append->inputInfos[infoInputPos];
    ZL_ASSERT_NN(inputInfo);
    ZL_ASSERT_EQ(inputInfo->appendOpt, append);
    ZL_Data* const input = inputInfo->data;
    if (input == NULL || !STREAM_isCommitted(input)) {
        return ZL_returnValue(0);
    }
    const uint8_t* const inputPtr = ZL_Data_rPtr(input);
    const size_t inputSize        = STREAM_byteSize(input);
    ZL_ASSERT_LE(append->outputHeadPtr, append->outputTailPtr);
    const size_t outputCapacity =
            (size_t)(append->outputTailPtr - append->outputHeadPtr);
    ZL_ERR_IF_GT(inputSize, outputCapacity, dstCapacity_tooSmall);

    uint8_t* const outputBegin = ZL_Data_wPtr(append->outputInfo->data);
    uint8_t const* outputEnd =
            outputBegin + STREAM_byteCapacity(append->outputInfo->data);

    ZL_ASSERT(
            inputPtr == append->outputHeadPtr
                    || (inputPtr + inputSize) <= outputBegin
                    || inputPtr >= outputEnd,
            "Input must either be equal to head pointer, or not overlapping output!");
    if (inputPtr == append->outputHeadPtr) {
        ZL_ASSERT(head);
        ZL_LOG(STREAM,
               "AppendToOutputOptimization: Append %zu directly to output buffer head for input %zu",
               inputSize,
               inputIdx);
        append->outputHeadPtr += inputSize;
    } else if (head) {
        ZL_LOG(STREAM,
               "AppendToOutputOptimization: Copied %zu into output buffer head for input %zu",
               inputSize,
               inputIdx);
        if (inputSize > 0) {
            memcpy(append->outputHeadPtr, inputPtr, inputSize);
        }
        append->outputHeadPtr += inputSize;
    } else {
        ZL_LOG(STREAM,
               "AppendToOutputOptimization: Copied %zu into output buffer tail for input %zu",
               inputSize,
               inputIdx);
        if (inputSize > 0) {
            memcpy(append->outputTailPtr - inputSize, inputPtr, inputSize);
        }
        append->outputTailPtr -= inputSize;
    }
    STREAM_free(input);
    inputInfo->data = NULL;

    return ZL_returnValue(1);
}

/**
 * Commits all inputs starting from `headInputIdx` by appending to the head of
 * the output buffer, and all inputs starting from `tailInputIdx` by prepending
 * to the tail of the output buffer. Stops when an uncommitted input is reached.
 */
static ZL_Report ZL_AppendToOutputOptimization_commitInputs(
        ZL_AppendToOutputOptimization* append)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    for (; append->headInputIdx < append->tailInputIdx;) {
        // First check if we can commit head inputs.
        {
            ZL_TRY_LET(
                    size_t,
                    success,
                    ZL_AppendToOutputOptimization_commitInput(
                            append, append->headInputIdx));
            if (success) {
                ++append->headInputIdx;
                continue;
            }
        }
        // Next check if we can commit tail inputs.
        {
            ZL_TRY_LET(
                    size_t,
                    success,
                    ZL_AppendToOutputOptimization_commitInput(
                            append, append->tailInputIdx - 1));
            if (success) {
                --append->tailInputIdx;
                continue;
            }
        }
        // All inputs that are ready have been committed.
        break;
    }
    return ZL_returnSuccess();
}

/**
 * Called on the output before running codecs that have `appendOpt` set on their
 * output. This hook does two things:
 * 1. All inputs that can be appended to the head of the output or prepended to
 *    the tail of the output are committed, and the input data is freed.
 * 2. If the codec is being replaced by the optimization, it ensures all input
 *    streams are committed, and commits the output stream.
 */
static ZL_Report ZL_AppendToOutputOptimization_preTransformHook(
        ZL_DCtx_DataInfo* info)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    ZL_AppendToOutputOptimization* append = info->appendOpt;
    ZL_ASSERT_NN(append);

    ZL_ERR_IF_ERR(ZL_AppendToOutputOptimization_commitInputs(append));

    if (info == info->appendOpt->outputInfo) {
        ZL_ERR_IF_NE(
                append->headInputIdx,
                append->tailInputIdx,
                corruption,
                "Not all input streams committed!");
        ZL_ASSERT_LE(append->outputHeadPtr, append->outputTailPtr);
        uint8_t* const outputBegin = ZL_Data_wPtr(info->data);
        uint8_t* const outputEnd =
                outputBegin + STREAM_byteCapacity(info->data);
        const size_t tailSize = (size_t)(outputEnd - append->outputTailPtr);
        if (append->outputTailPtr > append->outputHeadPtr) {
            ZL_LOG(FRAME,
                   "AppendToOutputOptimization: Moving tail of size %zu up %zu bytes to head",
                   tailSize,
                   (size_t)(append->outputTailPtr - append->outputHeadPtr));
            memmove(append->outputHeadPtr, append->outputTailPtr, tailSize);
        }
        append->outputHeadPtr += tailSize;

        const size_t outputSize = (size_t)(append->outputHeadPtr - outputBegin);
        ZL_ERR_IF_ERR(ZL_Data_commit(info->data, outputSize));
        ZL_LOG(FRAME,
               "AppendToOutputOptimization: Successfully committed %zu bytes",
               outputSize);
        memset(append, 0, sizeof(*append));
        return ZL_returnValue(1);
    } else {
        return ZL_returnValue(0);
    }
}

/**
 * Called when a new stream is requested by a codec is producing an input stream
 * to the AppendToOutputOptimization. It first commits any outstanding streams
 * to the output buffer. Then if the input can be appended to the head of the
 * output, point the input directly at the output, to elide a copy.
 */
static ZL_Report ZL_AppendToOutputOptimization_newStreamHook(
        ZL_DCtx_DataInfo* const info,
        ZL_Type type,
        size_t eltWidth,
        size_t eltsCapacity)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    ZL_ERR_IF_EQ(
            type,
            ZL_Type_string,
            corruption,
            "Strings not supported (already validated cannot be string)");
    ZL_ERR_IF(
            type == ZL_Type_numeric,
            corruption,
            "Numeric not supported (already validated cannot be int)");

    ZL_AppendToOutputOptimization* append = info->appendOpt;
    ZL_ASSERT_NN(append);

    ZL_ASSERT_GE(info, append->inputInfos);
    ZL_ASSERT_LT(info, append->inputInfos + append->nbInputs);
    // Inputs are listed in reverse order
    const size_t inputIdx =
            append->nbInputs - ((size_t)(info - append->inputInfos) + 1);
    if (inputIdx != append->headInputIdx) {
        // This only works if every previous stream has been committed because
        // we need to know the offset to write to.
        ZL_LOG(STREAM,
               "AppendToOutputOptimization: Skipping optimization for input %zu because it arrived out of order",
               inputIdx);
        return ZL_returnValue(0);
    }

    ZL_ERR_IF_ERR(ZL_AppendToOutputOptimization_commitInputs(append));

    size_t bytesNeeded;
    ZL_ERR_IF(
            ZL_overflowMulST(eltWidth, eltsCapacity, &bytesNeeded),
            integerOverflow);

    ZL_ASSERT_LE(append->outputHeadPtr, append->outputTailPtr);
    const size_t outputCapacity =
            (size_t)(append->outputTailPtr - append->outputHeadPtr);
    if (bytesNeeded > outputCapacity) {
        // A new stream needs to be allocated in this case.
        // Decompression may still succeed if the transform is over-reserving
        // output space.
        ZL_LOG(STREAM,
               "AppendToOutputOptimization: Skipping optimization for input %zu because it requested too much memory (%zu > %zu)",
               bytesNeeded,
               outputCapacity,
               inputIdx);
        return ZL_returnValue(0);
    }

    ZL_ERR_IF_ERR(STREAM_attachWritableBuffer(
            info->data, append->outputHeadPtr, type, eltWidth, eltsCapacity));

    return ZL_returnValue(1);
}

/**
 * Runs all node validation except checks that require the input and output
 * streams to be materialized.
 */
static ZL_Report DCTX_validateNodeStatic(
        ZL_DCtx* dctx,
        const DTransform* dt,
        const DFH_NodeInfo* nodeInfo)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);

    if (dctx->dfh.formatVersion < 9) {
        ZL_ERR_IF_EQ(
                nodeInfo->numInputStreams,
                0,
                formatVersion_unsupported,
                "0 output streams not supported until format version 9");
    }
    ZL_ERR_IF_GT(
            nodeInfo->numInputStreams,
            ZL_transformOutStreamsLimit(dctx->dfh.formatVersion),
            formatVersion_unsupported);

    ZL_ERR_IF_EQ(
            nodeInfo->nbRegens,
            0,
            corruption,
            "Graph inconsistency: Transform has no regenerated streams");
    // Validate that nb of regen streams is compatible
    ZL_ERR_IF_NOT(
            DT_isNbRegensCompatible(dt, nodeInfo->nbRegens),
            nodeRegen_countIncorrect,
            "Transform '%s'(%u) is assigned %u streams to regenerate, but its signature specifies %zu streams",
            DT_getTransformName(dt),
            nodeInfo->trpid.trid,
            nodeInfo->nbRegens,
            dt->miGraphDesc.nbInputs);

    // Validate that we only have variable output streams when we have a
    // non-zero number of variable output stream types. This is required
    // to ensure that all non-variable transforms get the right number
    // of streams.
    if (dt->miGraphDesc.nbVOs == 0) {
        ZL_ERR_IF_NE(
                nodeInfo->nbVOs,
                0,
                corruption,
                "Transform id=%u isn't accepting VO streams, "
                "but %u VO streams are nonetheless assigned to it in this graph.",
                dt->miGraphDesc.CTid,
                nodeInfo->nbVOs);
    }

    // We already validated the input streams during decode frame header.
    // Though they may not all be filled, so we have to check that.
    const size_t inputStreamEndIdx =
            nodeInfo->inputStreamBaseIdx + nodeInfo->numInputStreams;
    ZL_ERR_IF_GT(inputStreamEndIdx, dctx->dataInfo.size, graph_invalid);
    ZL_ASSERT_LE(inputStreamEndIdx, dctx->dataInfo.size);
    // Check that output streams are not used as input streams
    ZL_ERR_IF_GT(
            inputStreamEndIdx,
            dctx->dataInfo.size - dctx->nbOutputs,
            graph_invalid,
            "Graph inconsistency: Output stream depends on another output stream");

    return ZL_returnSuccess();
}

/* Reference streams stored in the frame.
 * Allocate them at their position in the graph.
 * @return size read from input
 */
static ZL_Report fillStoredStreams(
        ZL_DCtx* dctx,
        const void* src,
        size_t srcSize,
        size_t startPos)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);

    ZL_DLOG(SEQ,
            "fillStoredStreams (srcSize=%zu, startPos=%zu)",
            srcSize,
            startPos);
    ZL_ASSERT_EQ(dctx->dataInfo.size, 0);

    // Start by referencing the Transforms' Header stream
    size_t srcPos = startPos;
    {
        size_t const thsize = dctx->dfh.totalTHSize;
        dctx->thstream      = (ZL_RBuffer){
                 .start = (const char*)src + srcPos,
                 .size  = thsize,
        };
        srcPos += thsize;
    }
    ZL_ERR_IF_GT(srcPos, srcSize, srcSize_tooSmall);

    // Next we'll reference each of the stored streams in the frame
    size_t const nbTransforms    = dctx->dfh.nbDTransforms;
    size_t const nbStoredStreams = dctx->dfh.nbStoredStreams;
    size_t const nbRegenStreams  = dctx->dfh.nbRegens;
    size_t const totalNbStreams  = nbStoredStreams + nbRegenStreams;
    size_t const firstOutputIdx  = totalNbStreams - dctx->nbOutputs;
    ZL_ERR_IF_GT(
            totalNbStreams,
            ZL_runtimeStreamLimit(dctx->dfh.formatVersion),
            temporaryLibraryLimitation,
            "too many Streams defined in this Frame");

    ALLOC_ARENA_MALLOC_CHECKED(
            ZL_DCtx_DataInfo, dataInfo, totalNbStreams, dctx->chunkArena);
    dctx->dataInfo.ptr  = dataInfo;
    dctx->dataInfo.size = totalNbStreams;

    ZL_ASSERT_LE(nbTransforms, ZL_runtimeNodeLimit(dctx->dfh.formatVersion));
    ZL_ASSERT_EQ(VECTOR_SIZE(dctx->dfh.nodes), nbTransforms);
    DFH_NodeInfo* nodeInfo = VECTOR_DATA(dctx->dfh.nodes);

    // Initialize the dataInfo array
    for (size_t stream = 0; stream < totalNbStreams; ++stream) {
        dataInfo[stream].data            = NULL;
        dataInfo[stream].appendOpt       = NULL;
        dataInfo[stream].producerNodeIdx = ZL_PRODUCER_STORE;
    }

    // Index of the stream being analyzed,
    // can be either stored or regenerated stream.
    size_t streamIdx = 0;
    // Stored stream index keeps track of the current stored stream number.
    // We'll assign each stored stream into the correct stream index.
    size_t storedStreamIdx = 0;

    // Loop over each transform and identify regenerated streams.
    ZL_DLOG(FRAME,
            "frame header defines %zu transforms and %zu stored streams",
            nbTransforms,
            nbStoredStreams);
    for (size_t transformIdx = 0; transformIdx < nbTransforms; ++transformIdx) {
        const DFH_NodeInfo* node = &VECTOR_AT(dctx->dfh.nodes, transformIdx);
        ZL_TRY_LET(
                DTrPtr,
                wrappedTr,
                DTM_getTransform(
                        &dctx->dtm, node->trpid, dctx->dfh.formatVersion));
        const size_t nbTrIns = wrappedTr->miGraphDesc.nbSOs + node->nbVOs;

        nodeInfo[transformIdx].inputStreamBaseIdx = (ZL_IDType)streamIdx;
        nodeInfo[transformIdx].numInputStreams    = (ZL_IDType)nbTrIns;
        nodeInfo[transformIdx].fusion             = NULL;

        // Run all validation that can be done at header decode time.
        ZL_ERR_IF_ERR(DCTX_validateNodeStatic(dctx, wrappedTr, node));

        ZL_DLOG(BLOCK,
                "node %i : transform %i needs %i processed inputs",
                (int)transformIdx,
                node->trpid.trid,
                nbTrIns);
        size_t const inputEndIdx = streamIdx + nbTrIns;

        for (size_t n = 0; n < node->nbRegens; n++) {
            size_t const outputStreamIdx =
                    inputEndIdx + node->regenDistances[n];
            ZL_ASSERT_GE(outputStreamIdx, inputEndIdx);

            // Set the output stream as regenerated
            ZL_ERR_IF_GE(outputStreamIdx, totalNbStreams, corruption);
            ZL_ERR_IF_NE(
                    dataInfo[outputStreamIdx].producerNodeIdx,
                    ZL_PRODUCER_STORE,
                    corruption,
                    "Graph inconsistency: regenerated stream is already assigned. \n");
            dataInfo[outputStreamIdx].producerNodeIdx = (ZL_IDType)transformIdx;
        }

        // Codec fusion
        const bool fused =
                DCtx_getAppliedGParam(dctx, ZL_DParam_enableCodecFusion)
                        == ZL_TernaryParam_enable
                && ZL_DecoderFusionState_maybeFuse(
                        &dctx->fusion, nodeInfo, dataInfo, transformIdx);

        // Append optimization (only when fusion is not applied)
        if (!fused && node->nbRegens == 1) {
            const size_t regenIdx = inputEndIdx + node->regenDistances[0];
            ZL_ASSERT_LT(regenIdx, totalNbStreams);
            if (regenIdx >= firstOutputIdx) {
                // Outputs are listed in reverse order in dataInfos.
                size_t const outputIdx    = totalNbStreams - (regenIdx + 1);
                ZL_Data* const outputData = dctx->outputs[outputIdx];
                ZL_TRY_LET_CONST(
                        size_t,
                        hasAppendOpt,
                        ZL_AppendToOutputOptimization_register(
                                dctx,
                                node,
                                dataInfo + streamIdx,
                                nbTrIns,
                                dataInfo + regenIdx,
                                outputData));
                if (hasAppendOpt) {
                    ZL_LOG(FRAME,
                           "AppendToOutputOptimization: Applied to output %zu: InputBegin=%zu, NumInputs=%zu, RegenIdx=%zu",
                           outputIdx,
                           streamIdx,
                           nbTrIns,
                           regenIdx);
                }
            }
        }

        streamIdx = inputEndIdx;
    }

    // Loop over each of the streams inputs, and insert stored
    // streams for each non-regenerated stream.
    for (streamIdx = 0; streamIdx < totalNbStreams; ++streamIdx) {
        // Skip over regenerated streams
        if (dataInfo[streamIdx].producerNodeIdx != ZL_PRODUCER_STORE) {
            continue;
        }

        // Check that we haven't run out ouf stored streams
        ZL_ERR_IF_EQ(
                storedStreamIdx,
                nbStoredStreams,
                corruption,
                "Inconsistency: frame contains more streams than expected. \n"
                "This is a corruption event: frame will not be decompressed. \n"
                "Plausible causes are: \n"
                " - Error introduced during transmission or storage (which could be verified by the compressed checksum) \n"
                " - Invalid Graph construction, which presumes a bug at compression time \n"
                " - Incorrect Transform definition, such as registering a different Transform with same ID as expected one \n");

        // Reference the stored stream at streamIdx
        size_t const storedSize =
                VECTOR_AT(dctx->dfh.storedStreamSizes, storedStreamIdx);
        ZL_DLOG(FRAME,
                "stored Stream %zu : size=%zu, streamNb=%zu",
                storedStreamIdx,
                storedSize,
                streamIdx);
        ZL_ERR_IF_NULL(
                DCTX_newStreamFromConstRef(
                        dctx,
                        (ZL_IDType)streamIdx,
                        ZL_Type_serial,
                        1,
                        storedSize,
                        (const char*)src + srcPos),
                allocation);
        srcPos += storedSize;
        ++storedStreamIdx;
    }

    ZL_ERR_IF_NE(
            storedStreamIdx,
            nbStoredStreams,
            corruption,
            "Inconsistency: frame does not contain as many streams as expected. \n"
            "This is a corruption event: frame will not be decompressed. \n"
            "Plausible causes are: \n"
            " - Error introduced during transmission or storage (which could be verified by the compressed checksum) \n"
            " - Invalid Graph construction, which presumes a bug at compression time \n"
            " - Incorrect Transform definition, such as registering a different Transform with same ID as expected one \n");

    ZL_ERR_IF_EQ(
            totalNbStreams, 0, corruption, "Frame doesn't contain any stream!");

    ZL_LOG(FRAME, "read so far from frame: %zu/%zu", srcPos, srcSize);
    ZL_ERR_IF_GT(srcPos, srcSize, srcSize_tooSmall);
    dctx->streamEndPos = srcPos;

    ZL_ASSERT_GE(srcPos, startPos);
    ZL_DLOG(SEQ, "Chunk's Streams size: %zu", srcPos - startPos);
    return ZL_returnValue(srcPos - startPos);
}

// decodeFrameHeader()
// @return : error, or success
// Note : in the future, this method will likely have its own unit
static ZL_Report decodeFrameHeader(
        ZL_DCtx* dctx,
        const void* src,
        size_t srcSize,
        size_t nbOutputs)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_DLOG(FRAME, "decodeFrameHeader (srcSize = %zu)", srcSize);
    ZL_TRY_LET(size_t, hSize, DFH_decodeFrameHeader(&dctx->dfh, src, srcSize));

    ZL_TRY_LET(size_t, nbOuts, ZL_FrameInfo_getNumOutputs(dctx->dfh.frameinfo));
    ZL_ERR_IF_NE(nbOuts, nbOutputs, userBuffers_invalidNum);
    dctx->nbOutputs = nbOutputs;
    return ZL_returnValue(hSize);
}

// -------------------------------------------
// Processing streams
// -------------------------------------------

ZL_Data* DCTX_newStream(
        ZL_DCtx* dctx,
        ZL_IDType streamID,
        ZL_Type stype,
        size_t eltWidth,
        size_t eltsCapacity)
{
    int finalStream = 0;
    /* presume that dctx->streams has been sized exactly
     * at frame header decoding time */
    size_t const totalNbStreams = dctx->dataInfo.size;
    ZL_DLOG(BLOCK,
            "DCTX_newStream: new buffer id=%i/%zu of (Width;Capacity)=(%zu;%zu)",
            streamID,
            totalNbStreams,
            eltWidth,
            eltsCapacity);
    ZL_ASSERT_NN(dctx);
    ZL_ASSERT_LT(streamID, totalNbStreams);
    if (streamID >= (unsigned)(totalNbStreams - dctx->nbOutputs)) {
        finalStream = 1;
    }
    ZL_DCtx_DataInfo* const info = &dctx->dataInfo.ptr[streamID];

    if (dctx->preserveStreams && info->data) {
        // Allow re-using an existing stream if preserveStreams is enabled.
        // This allows benchmarking a subset of decoding transforms.
        ZL_DLOG(BLOCK, "Re-using existing stream (preserve streams mode)");
        ZL_ASSERT_EQ(ZL_Data_type(info->data), stype);
        ZL_ASSERT_EQ(ZL_Data_numElts(info->data), eltsCapacity);
        ZL_ASSERT_EQ(ZL_Data_eltWidth(info->data), eltWidth);
        STREAM_clear(info->data);
        return info->data;
    }

    ZL_ASSERT_NULL(info->data, "destination stream already exists");
    /* Write last stream directly into output
     * if conditions are met */
    if (finalStream) {
        ZL_ASSERT_NULL(info->appendOpt);
        size_t const outputID = (totalNbStreams - streamID) - 1;
        ZL_ASSERT_NN(dctx->outputs);
        ZL_ASSERT_LT(outputID, dctx->nbOutputs);
        ZL_Data* const output = dctx->outputs[outputID];
        ZL_ASSERT_NN(output);
        ZL_DLOG(SEQ,
                "target %u is a final stream, of type %u",
                streamID,
                ZL_Data_type(output));
        if (!STREAM_hasBuffer(output)) {
            ZL_DLOG(SEQ,
                    "output Stream exists as an empty shell => Allocate buffer");
            if (ZL_isError(STREAM_reserve(
                        output, stype, eltWidth, eltsCapacity))) {
                ZL_DLOG(ERROR,
                        "allocation error: Failed to create final output stream!");
                return NULL;
            }
            info->data = output;
            return output;
        }
        // output stream references an external buffer
        if (stype == ZL_Type_string) {
            // this setup is not supported for String type
            goto _newStream;
        }
        size_t requestedDstCapacity;
        if (ZL_overflowMulST(eltsCapacity, eltWidth, &requestedDstCapacity)) {
            ZL_DLOG(BLOCK,
                    "DCTX_newStream: bytesCapacity overflow (%zu * %zu)",
                    eltsCapacity,
                    eltWidth);
            return NULL;
        }
        if (STREAM_byteCapacity(output) >= requestedDstCapacity) {
            ZL_DLOG(SEQ,
                    "Final stream %u (existing contentSize = %zu) has enough capacity (%zu >= %zu)",
                    streamID,
                    STREAM_byteSize(output),
                    STREAM_byteCapacity(output),
                    requestedDstCapacity);
            if (eltWidth == 0) {
                ZL_DLOG(BLOCK, "DCTX_newStream: EltWidth=0 is not allowed");
                return NULL;
            }
            if (ZL_isError(STREAM_typeAttachedBuffer(
                        output, stype, eltWidth, eltsCapacity))) {
                ZL_DLOG(ERROR,
                        "error initializing pre-allocated output stream!");
                return NULL;
            }
            info->data = output;
            return output;
        }
        // dctx->output is not large enough to ensure decoding success.
        // Let's create a new Stream for that.
    }

_newStream:
    /* create a new Stream, reserve requested area */
    info->data =
            STREAM_createInArena(dctx->streamArena, (ZL_DataID){ streamID });
    if (!info->data) {
        return NULL;
    }
    if (info->appendOpt != NULL) {
        ZL_Report report = ZL_AppendToOutputOptimization_newStreamHook(
                info, stype, eltWidth, eltsCapacity);
        if (ZL_isError(report)) {
            return NULL;
        }
        if (ZL_validResult(report) != 0) {
            return info->data;
        }
    }
    if (ZL_isError(STREAM_reserve(info->data, stype, eltWidth, eltsCapacity))) {
        ZL_DLOG(ERROR, "allocation error: Failed to create output stream!");
        return NULL;
    }
    return info->data;
}

/* for streams whose content is stored in the frame */
ZL_Data* DCTX_newStreamFromConstRef(
        ZL_DCtx* dctx,
        ZL_IDType streamID,
        ZL_Type st,
        size_t eltWidth,
        size_t numElts,
        const void* rPtr)
{
    ZL_DLOG(BLOCK,
            "DCTX_newStreamFromConstRef: new buffer id=%i/%zu of %zu elts",
            streamID,
            dctx->dataInfo.size,
            numElts);
    ZL_ASSERT_NN(dctx);
    ZL_ASSERT_LT(streamID, dctx->dataInfo.size);
    ZL_DCtx_DataInfo* const info = &dctx->dataInfo.ptr[streamID];

    ZL_ASSERT_NULL(info->data); // stream_id not used yet
    info->data =
            STREAM_createInArena(dctx->streamArena, (ZL_DataID){ streamID });
    if (!info->data) {
        return NULL;
    }
    ZL_Report const ret =
            STREAM_refConstBuffer(info->data, rPtr, st, eltWidth, numElts);
    if (ZL_isError(ret)) {
        return NULL;
    }

    return info->data;
}

ZL_Data* DCTX_newStreamFromStreamRef(
        ZL_DCtx* dctx,
        ZL_IDType streamID,
        ZL_Type st,
        size_t eltWidth,
        size_t numElts,
        const ZL_Data* ref,
        size_t offsetBytes)
{
    ZL_DLOG(BLOCK,
            "DCTX_newStreamFromStreamRef: new stream id=%i/%zu of %zu elts of width %zu",
            streamID,
            dctx->dataInfo.size,
            numElts,
            eltWidth);
    ZL_ASSERT_NN(dctx);
    ZL_ASSERT_LT(streamID, dctx->dataInfo.size);
    ZL_DCtx_DataInfo* const info = &dctx->dataInfo.ptr[streamID];

    // stream_id should not be used yet
    ZL_ASSERT_NULL(info->data);

    info->data =
            STREAM_createInArena(dctx->streamArena, (ZL_DataID){ streamID });
    if (!info->data) {
        return NULL;
    }
    ZL_Report const ret = STREAM_refStreamByteSlice(
            info->data, ref, st, offsetBytes, eltWidth, numElts);
    if (ZL_isError(ret)) {
        return NULL;
    }

    return info->data;
}

static const ZL_Data** DCTX_getNodeInputs(
        const ZL_DCtx* dctx,
        const DFH_NodeInfo* nodeInfo,
        Arena* workspaceArena)
{
    const ZL_Data** inputs = ALLOC_Arena_malloc(
            workspaceArena, nodeInfo->numInputStreams * sizeof(const ZL_Data*));
    if (inputs == NULL) {
        return NULL;
    }

    for (ZL_IDType n = 0; n < nodeInfo->numInputStreams; n++) {
        ZL_IDType const snb = nodeInfo->inputStreamBaseIdx + n;
        size_t const inb = nodeInfo->numInputStreams - 1 - n; // reverse order
        inputs[inb]      = dctx->dataInfo.ptr[snb].data;
    }

    return inputs;
}

static ZL_Report DCTX_validateNodeInputs(
        ZL_DCtx* dctx,
        const DTransform* dt,
        const DFH_NodeInfo* nodeInfo,
        const ZL_Data* const* inputs,
        size_t numInputs,
        bool allowNullInputs)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_EQ(nodeInfo->numInputStreams, numInputs);

    ZL_Type allowedVOTypes = 0;
    for (size_t i = 0; i < dt->miGraphDesc.nbVOs; i++) {
        allowedVOTypes |= dt->miGraphDesc.voTypes[i];
    }

    // Validate input types
    for (size_t i = 0; i < numInputs; ++i) {
        ZL_ASSERT(
                allowNullInputs || inputs[i] != NULL,
                "Already validated due to regen validation");

        // Skip NULL inputs (fused children in decoder fusion)
        if (inputs[i] == NULL) {
            continue;
        }

        // Validate the input type is correct
        // (only for the compulsory output streams)
        if (i < dt->miGraphDesc.nbSOs) {
            // Compulsory range
            if (ZL_Data_type(inputs[i]) != dt->miGraphDesc.soTypes[i]) {
                ZL_ERR(graph_invalid,
                       "Error processing stream transform %u: input stream %u has type %d, but we expected type %d",
                       dt->miGraphDesc.CTid,
                       (unsigned)i,
                       ZL_Data_type(inputs[i]),
                       dt->miGraphDesc.soTypes[i]);
            }
        } else {
            // Validate that variable streams match one of the allowed types
            // in the graph description. We can't validate more than that,
            // the transform is responsible for everything else.
            ZL_ERR_IF_NOT(
                    ZL_Data_type(inputs[i]) & allowedVOTypes,
                    graph_invalid,
                    "Error processing transform %u: Variable input stream %u has type 0x%x, but we expected a type that matches the mask 0x%x",
                    dt->miGraphDesc.CTid,
                    (unsigned)(i - dt->miGraphDesc.nbSOs),
                    ZL_Data_type(inputs[i]),
                    allowedVOTypes);
        }
    }

    return ZL_returnSuccess();
}

static const ZL_IDType* DCTX_getRegeneratedStreamIDs(
        const ZL_DCtx* dctx,
        const DFH_NodeInfo* nodeInfo,
        Arena* workspaceArena)
{
    ZL_IDType* const regensID = ALLOC_Arena_malloc(
            workspaceArena, sizeof(ZL_IDType) * nodeInfo->nbRegens);
    if (regensID == NULL) {
        return NULL;
    }

    const ZL_IDType inputEndIdx =
            nodeInfo->inputStreamBaseIdx + nodeInfo->numInputStreams;
    for (size_t n = 0; n < nodeInfo->nbRegens; n++) {
        regensID[n] = (ZL_IDType)(inputEndIdx + nodeInfo->regenDistances[n]);
        ZL_ASSERT_LT(regensID[n], dctx->dataInfo.size);
        // Preserved streams can be regenerated again, see DCTX_newStream()
        ZL_ASSERT(
                dctx->preserveStreams
                        || dctx->dataInfo.ptr[regensID[n]].data == NULL,
                "Validated in frame header decoding");
    }
    return regensID;
}

static ZL_Report DCTX_validateNodeOutputs(
        ZL_DCtx* dctx,
        const ZL_IDType* regenStreamIDs,
        size_t numRegens)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);

    for (size_t n = 0; n < numRegens; n++) {
        ZL_Data* outStream = dctx->dataInfo.ptr[regenStreamIDs[n]].data;
        ZL_ERR_IF_NULL(
                outStream,
                transform_executionFailure,
                "Node didn't create expected regenerated stream!");
        ZL_ASSERT(
                STREAM_isCommitted(outStream),
                "Decoding transform did not provide its output size");
    }

    return ZL_returnSuccess();
}

static void DCTX_freeDecoderInputStreams(
        ZL_DCtx* dctx,
        const DFH_NodeInfo* nodeInfo)
{
    if (!dctx->preserveStreams) {
        for (ZL_IDType n = 0; n < nodeInfo->numInputStreams; n++) {
            ZL_IDType const snb       = nodeInfo->inputStreamBaseIdx + n;
            ZL_Data** const streamPtr = &dctx->dataInfo.ptr[snb].data;
            STREAM_free(*streamPtr);
            *streamPtr = NULL;
        }
    }
}

static ZL_Report runFusedDecoder(
        ZL_DCtx* dctx,
        const DFH_NodeInfo* parentNodeInfo)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);

    ZL_ASSERT_EQ(
            DCtx_getAppliedGParam(dctx, ZL_DParam_enableCodecFusion),
            ZL_TernaryParam_enable);

    const ZL_DecoderFusionDesc* const fusion = parentNodeInfo->fusion;
    ZL_ASSERT_NN(fusion);
    ZL_ASSERT_EQ(parentNodeInfo->trpid.trt, trt_standard);

    // Skip child nodes, they will be fused with the parent
    if (!ZL_DecoderFusionDesc_isParent(fusion, parentNodeInfo)) {
        return ZL_returnSuccess();
    }

    const size_t numChildCodecs = fusion->pattern.numChildren;
    const size_t numFusedCodecs = numChildCodecs + 1;

    // Collate all the DFH_NodeInfo* and DTransform* for the children + parent
    ALLOC_ARENA_MALLOC_CHECKED(
            const DFH_NodeInfo*,
            nodeInfos,
            numFusedCodecs,
            dctx->fusionWkspArena);
    ALLOC_ARENA_MALLOC_CHECKED(
            const DTransform*, codecs, numFusedCodecs, dctx->fusionWkspArena);
    for (size_t childIdx = 0; childIdx < numChildCodecs; ++childIdx) {
        const ZL_IDType childNodeIdx = ZL_DecoderFusionDesc_getProducerNodeIdx(
                fusion, childIdx, parentNodeInfo, dctx->dataInfo.ptr);
        ZL_ASSERT_NE(childNodeIdx, ZL_PRODUCER_STORE);
        nodeInfos[childIdx] = &VECTOR_AT(dctx->dfh.nodes, childNodeIdx);
        ZL_TRY_SET(
                DTrPtr,
                codecs[childIdx],
                DTM_getTransform(
                        &dctx->dtm,
                        nodeInfos[childIdx]->trpid,
                        dctx->dfh.formatVersion));
    }
    nodeInfos[numChildCodecs] = parentNodeInfo;

    // Build & validate the inputs for each child
    ALLOC_ARENA_MALLOC_CHECKED(
            const ZL_Data**,
            codecInputs,
            numFusedCodecs,
            dctx->fusionWkspArena);
    for (size_t childIdx = 0; childIdx < numChildCodecs; ++childIdx) {
        const DFH_NodeInfo* childNodeInfo = nodeInfos[childIdx];
        codecInputs[childIdx] =
                DCTX_getNodeInputs(dctx, childNodeInfo, dctx->fusionWkspArena);
        ZL_ERR_IF_NULL(codecInputs[childIdx], allocation);
        ZL_ERR_IF_ERR(DCTX_validateNodeInputs(
                dctx,
                codecs[childIdx],
                childNodeInfo,
                codecInputs[childIdx],
                childNodeInfo->numInputStreams,
                /* allowNullInputs */ false));
    }

    // Build & validate the inputs for the parent
    // NOTE: Inputs from fused codecs are NULL
    ZL_TRY_SET(
            DTrPtr,
            codecs[numChildCodecs],
            DTM_getTransform(
                    &dctx->dtm,
                    parentNodeInfo->trpid,
                    dctx->dfh.formatVersion));
    codecInputs[numChildCodecs] =
            DCTX_getNodeInputs(dctx, parentNodeInfo, dctx->fusionWkspArena);
    ZL_ERR_IF_NULL(codecInputs[numChildCodecs], allocation);
    ZL_ERR_IF_ERR(DCTX_validateNodeInputs(
            dctx,
            codecs[numChildCodecs],
            parentNodeInfo,
            codecInputs[numChildCodecs],
            parentNodeInfo->numInputStreams,
            /* allowNullInputs */ true));

    // Build per-codec header table
    ALLOC_ARENA_MALLOC_CHECKED(
            ZL_RBuffer, codecHeaders, numFusedCodecs, dctx->fusionWkspArena);
    for (size_t i = 0; i < numChildCodecs; ++i) {
        ZL_TRY_SET(
                ZL_RBuffer,
                codecHeaders[i],
                ZL_RBuffer_slice(
                        dctx->thstream,
                        nodeInfos[i]->trhStart,
                        nodeInfos[i]->trhSize));
    }
    ZL_TRY_SET(
            ZL_RBuffer,
            codecHeaders[numChildCodecs],
            ZL_RBuffer_slice(
                    dctx->thstream,
                    parentNodeInfo->trhStart,
                    parentNodeInfo->trhSize));

    // Compute regen slots
    const ZL_IDType* regensID = DCTX_getRegeneratedStreamIDs(
            dctx, parentNodeInfo, dctx->fusionWkspArena);
    ZL_ERR_IF_NULL(regensID, allocation);

    // Call the fused decoder
    ZL_DecoderFusion state = {
        .dctx            = dctx,
        .workspaceArena  = dctx->fusionWkspArena,
        .nodeInfos       = nodeInfos,
        .codecs          = codecs,
        .codecHeaders    = codecHeaders,
        .codecInputs     = codecInputs,
        .numCodecs       = numFusedCodecs,
        .regenStreamIDs  = regensID,
        .numRegenStreams = parentNodeInfo->nbRegens,
    };
    ZL_ASSERT_NN(fusion->fusionFn);
    ZL_Report const report = fusion->fusionFn(&state);
    ZL_ERR_IF_ERR_COERCE(report);

    // Check transform's outcome
    ZL_ERR_IF_ERR(
            DCTX_validateNodeOutputs(dctx, regensID, parentNodeInfo->nbRegens));

    // Free consumed input streams (must happen before clearing the workspace
    // arena because nodeInfos is allocated on the workspace arena).
    for (size_t i = 0; i < numFusedCodecs; ++i) {
        DCTX_freeDecoderInputStreams(dctx, nodeInfos[i]);
    }

    // Clear the workspace arena
    ALLOC_Arena_freeAll(dctx->fusionWkspArena);

    return ZL_returnSuccess();
}

// @return : nb of streams processed
ZL_Report DCTX_runDecoder(
        ZL_DCtx* dctx,
        const DFH_NodeInfo* nodeInfo,
        bool withinFusedDecoder)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_NN(dctx);
    PublicTransformInfo const trid = nodeInfo->trpid;
    ZL_TRY_LET(
            DTrPtr,
            dt,
            DTM_getTransform(&dctx->dtm, trid, dctx->dfh.formatVersion));

    const char* const trName = DT_getTransformName(dt);
    ZL_RESULT_SCOPE_ADD_GRAPH_CONTEXT(
            (ZL_GraphContext){ .transformID = trid.trid, .name = trName });

    const ZL_IDType streamID = nodeInfo->inputStreamBaseIdx;
    const size_t nbInStreams = nodeInfo->numInputStreams;

    ZL_DLOG(BLOCK,
            "transformID = %u '%s' (type:%u)",
            trid.trid,
            DTM_getTransformName(&dctx->dtm, trid, dctx->dfh.formatVersion),
            trid.trt);
    ZL_DLOG(BLOCK,
            "runDecoder on input streams [%u-%u] with transform '%s'(%u)",
            streamID,
            streamID + (ZL_IDType)nbInStreams - 1,
            trName,
            nodeInfo->trpid.trid);

    // Run decoder fusion
    if (!withinFusedDecoder && nodeInfo->fusion != NULL) {
        ZL_ERR_IF_ERR(runFusedDecoder(dctx, nodeInfo));
        return ZL_returnValue(nbInStreams);
    }

    // Run append to output optimization
    if (nodeInfo->nbRegens == 1) {
        const size_t regenIdx =
                streamID + nbInStreams + nodeInfo->regenDistances[0];
        ZL_DCtx_DataInfo* info = &dctx->dataInfo.ptr[regenIdx];
        if (info->appendOpt) {
            ZL_TRY_LET(
                    size_t,
                    success,
                    ZL_AppendToOutputOptimization_preTransformHook(info));
            if (success) {
                ZL_DLOG(BLOCK,
                        "transform '%s' (id:%u) regenerated %zu streams (appended directly to output)",
                        trName,
                        dt->miGraphDesc.CTid,
                        nodeInfo->nbRegens);
                // We've replaced the transform execution by appending directly
                // to the output buffer.
                ZL_ASSERT(!dctx->preserveStreams);
                ZL_ASSERT(STREAM_isCommitted(info->data));
                for (size_t i = 0; i < nbInStreams; ++i) {
                    ZL_Data* input = dctx->dataInfo.ptr[streamID + i].data;
                    ZL_ASSERT_NULL(input, "Input must already be freed");
                }
                return ZL_returnValue(nbInStreams);
            }
        }
    }

    // Collect input streams
    const ZL_Data** inputs =
            DCTX_getNodeInputs(dctx, nodeInfo, dctx->decoderWkspArena);
    ZL_ERR_IF_NULL(inputs, allocation);

    // Validate input types
    ZL_ERR_IF_ERR(DCTX_validateNodeInputs(
            dctx,
            dt,
            nodeInfo,
            inputs,
            nbInStreams,
            /* allowNullInputs */ false));

    // Get the codec header
    ZL_TRY_LET(
            ZL_RBuffer,
            thContent,
            ZL_RBuffer_slice(
                    dctx->thstream, nodeInfo->trhStart, nodeInfo->trhSize));

    // Determine regenerated streams slots
    const ZL_IDType* regensID = DCTX_getRegeneratedStreamIDs(
            dctx, nodeInfo, dctx->decoderWkspArena);
    ZL_ERR_IF_NULL(regensID, allocation);

    // Run the transform
    ZL_DLOG(SEQ,
            "Running transform '%s' (id:%u), expected to regenerate %zu streams",
            trName,
            dt->miGraphDesc.CTid,
            nodeInfo->nbRegens);

    uint32_t const dictIdx = nodeInfo->dictIdx;
    const void* ddict      = NULL;
    /* Resolve dict from bundle if this transform uses a dictionary */
    if (dictIdx != ZL_DICT_INDEX_NONE) {
        ZL_ASSERT_NN(dctx->bundle);
        ZL_ERR_IF(
                dictIdx >= dctx->bundle->info.numDicts,
                corruption,
                "dictIdx %u out of range (bundle has %zu dicts)",
                (unsigned)dictIdx,
                dctx->bundle->info.numDicts);
        ZL_ERR_IF_NULL(
                dctx->bundle->dicts[dictIdx]->dictObj,
                dict_materialization,
                "Dict at index %u has not been materialized",
                (unsigned)dictIdx);
        ddict = dctx->bundle->dicts[dictIdx]->dictObj;
    }

    struct ZL_Decoder_s diState = {
        .dctx           = dctx,
        .dt             = dt,
        .statePtr       = DTM_getStatePtr(&dctx->dtm, nodeInfo->trpid),
        .workspaceArena = dctx->decoderWkspArena,
        .regensID       = regensID,
        .nbRegens       = nodeInfo->nbRegens,
        .thContent      = thContent,
        .ddict          = ddict,
    };
    ZL_ERR_IF_NULL(
            diState.statePtr,
            logicError,
            "Could not find state for transform %u",
            nodeInfo->trpid.trid);

    DWAYPOINT(on_codecDecode_start, &diState, inputs, nbInStreams);

    ZL_ASSERT_NN(dt->transformFn);
    ZL_Report const report = dt->transformFn(&diState, dt, inputs, nbInStreams);
    if (ZL_isError(report)) {
        DWAYPOINT(on_codecDecode_end, &diState, NULL, 0, report);
    }
    ZL_ERR_IF_ERR_COERCE(report);

    // Check transform's outcome
    ZL_ERR_IF_ERR(DCTX_validateNodeOutputs(dctx, regensID, nodeInfo->nbRegens));

    IF_DWAYPOINT_ENABLED(on_codecDecode_end, &diState)
    {
        VECTOR_CONST_POINTERS(ZL_Data) odata = { 0 };
        VECTOR_INIT(odata, nodeInfo->nbRegens);
        for (size_t n = 0; n < nodeInfo->nbRegens; n++) {
            const ZL_Data* d     = dctx->dataInfo.ptr[regensID[n]].data;
            bool pushbackSuccess = VECTOR_PUSHBACK(odata, d);
            if (!pushbackSuccess) {
                VECTOR_DESTROY(odata);
                ZL_ERR(allocation,
                       "Unable to append to the waypoint odata vector");
            }
        }
        DWAYPOINT(
                on_codecDecode_end,
                &diState,
                VECTOR_DATA(odata),
                VECTOR_SIZE(odata),
                ZL_returnSuccess());
        VECTOR_DESTROY(odata);
    }

    // Clear the workspace arena
    ALLOC_Arena_freeAll(dctx->decoderWkspArena);

    // Free the input streams
    DCTX_freeDecoderInputStreams(dctx, nodeInfo);

    ZL_DLOG(BLOCK,
            "decoder '%s' (id:%u) regenerated %zu streams",
            trName,
            dt->miGraphDesc.CTid,
            diState.nbRegens);

    return ZL_returnValue(nbInStreams);
}

static ZL_Report runDecoders(ZL_DCtx* dctx)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_DLOG(FRAME, "runDecoders (%zu stages)", dctx->dfh.nbDTransforms);
    ZL_ASSERT_NN(dctx);
    for (size_t stage = 0; stage < dctx->dfh.nbDTransforms; stage++) {
        ZL_DLOG(BLOCK, "decoding stage %zu", stage);
        DFH_NodeInfo const* nodeInfo = &VECTOR_AT(dctx->dfh.nodes, stage);
        ZL_ERR_IF_ERR(DCTX_runDecoder(
                dctx, nodeInfo, /* withinFusedDecoder */ false));
    }

    return ZL_returnSuccess();
}

/* Only used for specific benchmark scenarios */
ZL_Report DCTX_runTransformID(ZL_DCtx* dctx, ZL_IDType transformID)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT(dctx->preserveStreams);
    size_t totalOutputBytes = 0;
    for (size_t stage = 0, startingStream = 0; stage < dctx->dfh.nbDTransforms;
         stage++) {
        DFH_NodeInfo const* node       = &VECTOR_AT(dctx->dfh.nodes, stage);
        PublicTransformInfo const trid = node->trpid;
        if (trid.trid != transformID) {
            startingStream += node->numInputStreams;
            continue;
        }
        ZL_DLOG(BLOCK, "transformID = %u (type:%u)", trid.trid, trid.trt);
        ZL_TRY_LET(
                size_t,
                nbps,
                DCTX_runDecoder(dctx, node, /* inFusedDecoder */ false));
        startingStream += nbps;
        ZL_ERR_IF_NE(
                node->nbRegens,
                1,
                node_versionMismatch,
                "This method only supports Transforms regenerating a single stream");
        ZL_IDType const outStreamID =
                (ZL_IDType)(startingStream + node->regenDistances[0]);
        const ZL_Data* out = dctx->dataInfo.ptr[outStreamID].data;
        ZL_ASSERT_NN(out);
        ZL_ASSERT(
                STREAM_isCommitted(out),
                "Decoding transform did not provide its output size");
        size_t const outBytes = ZL_Data_numElts(out) * ZL_Data_eltWidth(out);
        totalOutputBytes += outBytes;
        ZL_DLOG(BLOCK, "produced %u bytes", (unsigned)outBytes);
    }
    return ZL_returnValue(totalOutputBytes);
}

static ZL_Report addChunksIntoFinalStreams(ZL_DCtx* dctx)
{
    ZL_ASSERT_NN(dctx);
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    size_t const nbStreams = dctx->dataInfo.size;
    ZL_ERR_IF_GT(
            dctx->nbOutputs,
            nbStreams,
            outputs_tooNumerous,
            "Frame header expected more streams than actually produced");
    ZL_ASSERT_GE(nbStreams, dctx->nbOutputs);
    for (size_t outputN = 0; outputN < dctx->nbOutputs; outputN++) {
        ZL_Data* const output      = dctx->outputs[outputN];
        size_t const lsid          = nbStreams - outputN - 1;
        const ZL_Data* chunkOutput = dctx->dataInfo.ptr[lsid].data;
        ZL_ERR_IF_NULL(
                chunkOutput, graph_invalid, "Final stream not produced!");
        ZL_Type const type    = ZL_Data_type(chunkOutput);
        size_t const eltWidth = ZL_Data_eltWidth(chunkOutput);
        if (type != ZL_Type_string)
            ZL_ASSERT_GT(eltWidth, 0);
        size_t const numElts         = ZL_Data_numElts(chunkOutput);
        size_t const chunkOutputSize = ZL_Data_contentSize(chunkOutput);

        ZL_TRY_LET(
                size_t,
                expectedType,
                ZL_FrameInfo_getOutputType(dctx->dfh.frameinfo, (int)outputN));
        ZL_ERR_IF_NE(
                type, expectedType, corruption, "Final stream type mismatch");

        if (chunkOutput == output) {
            ZL_DLOG(SEQ,
                    "final content already decompressed directly into output %zu (total size: %zu bytes)",
                    outputN,
                    ZL_Data_contentSize(output));
            continue;
        }

        ZL_DLOG(FRAME,
                "addChunksIntoFinalStreams %zu: %zu bytes",
                outputN,
                chunkOutputSize);

        ZL_ASSERT_NN(output);

        /* special case: output buffer not yet allocated
         * this can only happen for older frame version < ZL_CHUNK_VERSION_MIN
         * and for String type, since we have the size for other types */
        if (!STREAM_hasBuffer(output)) {
            ZL_ASSERT_EQ(type, ZL_Type_string);
            ZL_ASSERT_LT(dctx->dfh.formatVersion, ZL_CHUNK_VERSION_MIN);
            // @note (@cyan): works fine, because there is only one Chunk
            ZL_ERR_IF_ERR(STREAM_copyStringStream(output, chunkOutput));
            continue;
        }

        // All output buffers are expected to be pre-allocated and correctly
        // sized at this point.
        ZL_ASSERT(STREAM_hasBuffer(output));

        ZL_ERR_IF_GT(
                chunkOutputSize,
                STREAM_byteCapacity(output),
                dstCapacity_tooSmall, );
        // @note (@cyan): this could probably be checked only once, at the
        // beginning
        if (type == ZL_Type_numeric) {
            ZL_ERR_IF_NOT(
                    MEM_IS_ALIGNED_N(
                            ZL_Data_wPtr(output),
                            MEM_alignmentForNumericWidth(eltWidth)),
                    userBuffer_alignmentIncorrect,
                    "provided dst buffer is incorrectly aligned for numerics of width %zu bytes",
                    eltWidth);
        }

        if (type != ZL_Type_string) {
            // @note (@cyan): this step is only necessary to write eltWidth.
            // At this stage, stream is already sized for the entire output.
            // Note that @p numElts is only for current chunk.
            // But that's fine, STREAM_typeAttachedBuffer() only checks that
            // size is large enough. It will not size it down to numElts.
            ZL_ERR_IF_ERR(
                    STREAM_typeAttachedBuffer(output, type, eltWidth, numElts));
        }

        // Append chunk data into final output
        ZL_ERR_IF_ERR(STREAM_append(output, chunkOutput));
    }
    return ZL_returnValue(dctx->nbOutputs);
}

// Presumed successful
static void cleanChunkBuffers(ZL_DCtx* dctx)
{
    DCTX_freeStreams(dctx);
    ALLOC_Arena_freeAll(dctx->streamArena);
    ALLOC_Arena_freeAll(dctx->decoderWkspArena);
    ALLOC_Arena_freeAll(dctx->fusionWkspArena);
    ALLOC_Arena_freeAll(dctx->chunkArena);
    memset(&dctx->dataInfo, 0, sizeof(dctx->dataInfo));
}

static void cleanAllBuffers(ZL_DCtx* dctx)
{
    cleanChunkBuffers(dctx);
}

/**
 * Populates @p dctx to prepare for decoding the chunk beginning at @p
 * alreadyConsumed in
 * @p framePtr.
 *
 * Clears chunk-scoped buffers, decodes the chunk header and stored streams,
 * reads the optional content checksum into @p expectedContentHash, and
 * validates the optional compressed checksum before decoders run.
 *
 * @param dctx Decompression context to populate for the chunk.
 * @param framePtr Beginning of the frame buffer.
 * @param frameSize Size of @p framePtr in bytes.
 * @param alreadyConsumed Number of frame bytes consumed before this chunk.
 * @param expectedContentHash Output content checksum, or 0 when absent.
 * @return DCTX_FrameChunkInfo metadata for the prepared chunk.
 */
static ZL_RESULT_OF(DCTX_FrameChunkInfo) setupChunkDecode(
        ZL_DCtx* dctx,
        const void* framePtr,
        size_t frameSize,
        size_t alreadyConsumed,
        uint32_t* expectedContentHash)
{
    ZL_RESULT_DECLARE_SCOPE(DCTX_FrameChunkInfo, dctx);
    size_t consumedSize = alreadyConsumed;
    ZL_DLOG(BLOCK,
            "setupChunkDecode (frameSize=%zu, consumedSize=%zu)",
            frameSize,
            consumedSize);
    ZL_ASSERT_NN(dctx);

    // We clean at the beginning instead of the end
    // in case `DCTX_preserveStreams` is set,
    // requiring to preserve some results for StreamDump2
    cleanChunkBuffers(dctx);

    ZL_ASSERT_LE(consumedSize, frameSize);
    ZL_TRY_LET(
            size_t,
            chunkHeaderSize,
            DFH_decodeChunkHeader(
                    &dctx->dfh,
                    (const char*)framePtr + consumedSize,
                    frameSize - consumedSize));
    consumedSize += chunkHeaderSize;

    ZL_TRY_LET(
            size_t,
            chunkStreamsSize,
            fillStoredStreams(dctx, framePtr, frameSize, consumedSize));
    consumedSize += chunkStreamsSize;

    // If present, verify the compressed checksum before running decoders.
    // Assuming we aren't handling malicious inputs, this ensures that we
    // are running on valid data before we run the decoders.
    *expectedContentHash = 0;

    if (FrameInfo_hasContentChecksum(dctx->dfh.frameinfo)) {
        ZL_ERR_IF_LT(frameSize, consumedSize + 4, srcSize_tooSmall);
        *expectedContentHash =
                ZL_readCE32((const char*)framePtr + consumedSize);
        ZL_DLOG(SEQ, "stored contentHash: %08X", *expectedContentHash);
        consumedSize += 4;
    }

    if (FrameInfo_hasCompressedChecksum(dctx->dfh.frameinfo)) {
        ZL_ERR_IF_LT(frameSize, consumedSize + 4, srcSize_tooSmall);
#ifndef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
        if (DCtx_getAppliedGParam(dctx, ZL_DParam_checkCompressedChecksum)
            == ZL_TernaryParam_enable) {
            size_t startHash = dctx->dfh.formatVersion < ZL_CHUNK_VERSION_MIN
                    ? 0
                    : alreadyConsumed;
            uint32_t const expectedHash =
                    ZL_readCE32((const char*)framePtr + consumedSize);
            /* @note: versions <= 20 checksum the entire frame, 21+ only current
             * block */
            ZL_DLOG(SEQ,
                    "compressed checksum from pos %zu to %zu",
                    startHash,
                    consumedSize);
            ZL_ASSERT_LE(startHash, consumedSize);
            uint32_t const actualHash = (uint32_t)XXH3_64bits(
                    (const char*)framePtr + startHash,
                    consumedSize - startHash);

            ZL_DLOG(SEQ,
                    "actualCompressedHash:%08X vs %08X:expectedCompressedHash",
                    actualHash,
                    expectedHash);
            ZL_ERR_IF_NE(
                    actualHash,
                    expectedHash,
                    compressedChecksumWrong,
                    "Compressed checksum mismatch! This indicates data corruption after compression!");
        }
#endif
        consumedSize += 4;
    }

    DCTX_FrameChunkInfo const chunkInfo = {
        .chunkHeaderSize = chunkHeaderSize,
        .chunkSize       = consumedSize - alreadyConsumed,
    };
    return ZL_WRAP_VALUE(chunkInfo);
}

ZL_RESULT_OF(DCTX_FrameChunkInfo)
DCTX_prepareFrameChunk(
        ZL_DCtx* dctx,
        const void* framePtr,
        size_t frameSize,
        size_t chunkOffset)
{
    ZL_RESULT_DECLARE_SCOPE(DCTX_FrameChunkInfo, dctx);
    ZL_ASSERT_NN(dctx);

    // Parse-only setup has no output buffers. Temporarily hide final outputs
    // so fillStoredStreams skips output-backed append optimizations.
    size_t const nbOutputs = dctx->nbOutputs;
    dctx->nbOutputs        = 0;

    uint32_t dummyHash = 0;
    ZL_RESULT_OF(DCTX_FrameChunkInfo)
    const chunkResult = setupChunkDecode(
            dctx, framePtr, frameSize, chunkOffset, &dummyHash);

    dctx->nbOutputs = nbOutputs;
    ZL_ERR_IF_ERR(chunkResult);
    return chunkResult;
}

ZL_Report DCTX_prepareFrameChunkFromHeader(
        ZL_DCtx* dctx,
        const void* chunkHeader,
        size_t chunkHeaderSize,
        const void* chunkRef,
        size_t chunkSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_NN(dctx);
    ZL_ASSERT_NN(chunkHeader);
    ZL_ASSERT_NN(chunkRef);

    // We clean at the beginning instead of the end
    // in case `DCTX_preserveStreams` is set,
    // requiring to preserve some results for StreamDump2
    cleanChunkBuffers(dctx);

    ZL_TRY_LET(
            size_t,
            decodedChunkHeaderSize,
            DFH_decodeChunkHeader(&dctx->dfh, chunkHeader, chunkHeaderSize));

    size_t const nbOutputs        = dctx->nbOutputs;
    dctx->nbOutputs               = 0;
    ZL_Report const streamsResult = fillStoredStreams(
            dctx, chunkRef, chunkSize, decodedChunkHeaderSize);

    dctx->nbOutputs = nbOutputs;
    ZL_ERR_IF_ERR(streamsResult);
    return ZL_returnSuccess();
}
// -------------------------------------
// Main decompression functions
// -------------------------------------
/**
 * @return size of chunk, read from frame
 */
static ZL_Report ZL_DCtx_decompressChunk(
        ZL_DCtx* dctx,
        size_t nbOutputs,
        const void* framePtr,
        size_t frameSize,
        size_t alreadyConsumed)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_Data** outputs            = dctx->outputs;
    uint32_t expectedContentHash = 0;
    ZL_TRY_LET(
            DCTX_FrameChunkInfo,
            chunkInfo,
            setupChunkDecode(
                    dctx,
                    framePtr,
                    frameSize,
                    alreadyConsumed,
                    &expectedContentHash));
    size_t const chunkSize = chunkInfo.chunkSize;

    // start the decompression process.
    ZL_ERR_IF_ERR(runDecoders(dctx));

    // write result into user's buffer
    {
        ZL_TRY_LET(size_t, nbOuts, addChunksIntoFinalStreams(dctx));
        ZL_ERR_IF_NE(nbOuts, nbOutputs, corruption);
    }

    // verify block content checksum
    if (FrameInfo_hasContentChecksum(dctx->dfh.frameinfo)
        && DCtx_getAppliedGParam(dctx, ZL_DParam_checkContentChecksum)
                == ZL_TernaryParam_enable) {
#ifndef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
        // Generate checksum based on actual output
        for (size_t n = 0; n < nbOutputs; n++) {
            if (ZL_Data_type(outputs[n]) == ZL_Type_numeric) {
                ZL_ERR_IF_NOT(
                        ZL_isLittleEndian(),
                        temporaryLibraryLimitation,
                        "Cannot calculate hash of numeric output on non little-endian platforms");
            }
        }
        ZL_TRY_LET(
                size_t,
                actualHashT,
                STREAM_hashLastCommit_xxh3low32(
                        (const ZL_Data**)(void*)outputs,
                        nbOutputs,
                        dctx->dfh.formatVersion));
        uint32_t const actualContentHash = (uint32_t)actualHashT;
        ZL_DLOG(SEQ,
                "actualContentHash:%08X vs %08X:expectedContentHash",
                actualContentHash,
                expectedContentHash);
        const char* const errMsg =
                FrameInfo_hasCompressedChecksum(dctx->dfh.frameinfo)
                ? "Content checksum mismatch! This indicates that the data was corrupted during compression or decompression, because the compressed checksum matched! This can be caused by a Zstrong bug, other ASAN bugs in the process, or faulty hardware."
                : "Content checksum mismatch! This indicates that either data corruption after compression or that data was corrupted during compression or decompression!";
        ZL_ERR_IF_NE(
                actualContentHash,
                expectedContentHash,
                contentChecksumWrong,
                "%s",
                errMsg);
#else
        (void)expectedContentHash;
        (void)outputs;
#endif
    }

    return ZL_returnValue(chunkSize);
}

ZL_Report ZL_DCtx_decompressMultiTBuffer(
        ZL_DCtx* dctx,
        ZL_TypedBuffer* tbuffers[],
        size_t nbOutputs,
        const void* framePtr,
        size_t frameSize)
{
    ZL_DLOG(FRAME,
            "ZL_DCtx_decompressMultiTBuffer: decompress %zu bytes into %zu typed buffers",
            frameSize,
            nbOutputs);
    ZL_OC_startOperation(&dctx->opCtx, ZL_Operation_decompress);
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);

    DWAYPOINT(
            on_ZL_DCtx_decompressMultiTBuffer_start,
            dctx,
            nbOutputs,
            framePtr,
            frameSize);
    ZL_ERR_IF_ERR(DCtx_setAppliedParameters(dctx));

    // Clean up state - may be dirty if previous decompression failed
    cleanAllBuffers(dctx);

    // parameter sanitization
    ZL_Data** const outputs = ZL_codemodOutputsAsDatas(tbuffers);
    ZL_ASSERT_NN(outputs);
    for (size_t n = 0; n < nbOutputs; n++) {
        ZL_ASSERT_NN(outputs[n], "output %zu should not be NULL", n);
    }
    dctx->outputs = outputs;

    // read frame header
    ZL_TRY_LET(
            size_t,
            consumed,
            decodeFrameHeader(dctx, framePtr, frameSize, nbOutputs));
    ZL_DLOG(SEQ, "decoded frame header, of size %zu bytes", consumed);
    dctx->bundle = NULL;
    {
        const ZL_BundleID* const bundleID =
                ZL_FrameInfo_getBundleID(dctx->dfh.frameinfo);
        if (bundleID != NULL) {
            ZL_ERR_IF_NULL(
                    dctx->dictLoader,
                    dictNoRecord,
                    "Frame references a dict bundle but no dict loader is attached");
            ZL_RESULT_OF(ZL_DictBundleConstPtr)
            bundleRes =
                    ZL_DictLoader_fetchDictBundle(dctx->dictLoader, bundleID);
            ZL_ERR_IF_ERR(
                    bundleRes,
                    "Could not fetch dict bundle required by the frame");
            dctx->bundle = ZL_RES_value(bundleRes);
        }
    }

    // check buffers in outputs objects
    for (size_t n = 0; n < nbOutputs; n++) {
        if (STREAM_hasBuffer(outputs[n])) {
            // just check the buffer is appropriately sized
            ZL_TRY_LET(
                    size_t,
                    dSize,
                    ZL_FrameInfo_getDecompressedSize(
                            dctx->dfh.frameinfo, (int)n));
            ZL_ERR_IF_LT(
                    STREAM_byteCapacity(outputs[n]),
                    dSize,
                    dstCapacity_tooSmall,
                    "Buffer id%zu has insufficient capacity",
                    n);
            continue;
        }

        // here, output is just a shell: let's allocate its buffer(s).
        // Note: we would need eltWidth for `struct` and `numeric`,
        //       but we will only get that after the first chunk.
        // Note: we need nbStrings for `string`,
        //       which is only available for version >= ZL_CHUNK_VERSION_MIN
        ZL_TRY_LET(
                size_t,
                type_st,
                ZL_FrameInfo_getOutputType(dctx->dfh.frameinfo, (int)n));
        ZL_TRY_LET(
                size_t,
                dSize,
                ZL_FrameInfo_getDecompressedSize(dctx->dfh.frameinfo, (int)n));

        switch (type_st) {
            case ZL_Type_serial: {
                ZL_DLOG(SEQ,
                        "pre-allocating output %zu, type Serial, capacity %zu bytes",
                        n,
                        dSize);
                ZL_ERR_IF_ERR(
                        STREAM_reserve(outputs[n], ZL_Type_serial, 1, dSize));
            } break;

            case ZL_Type_struct:
            case ZL_Type_numeric: {
                /* only reserve the underlying buffer - typing will be added
                 * later, once eltWidth is discovered */
                ZL_DLOG(SEQ,
                        "pre-allocating output %zu, no type set, capacity %zu bytes",
                        n,
                        dSize);
                ZL_ERR_IF_ERR(STREAM_reserveRawBuffer(outputs[n], dSize));
            } break;

            case ZL_Type_string: {
                if (dctx->dfh.formatVersion < ZL_CHUNK_VERSION_MIN) {
                    // allocating output of type string is not possible:
                    // `numStrings` is not available.
                    break;
                }
                ZL_TRY_LET(
                        size_t,
                        numStrings,
                        ZL_FrameInfo_getNumElts(dctx->dfh.frameinfo, (int)n));
                ZL_ERR_IF_ERR(
                        STREAM_reserveStrings(outputs[n], numStrings, dSize));
            } break;
            default:
                ZL_ASSERT_FAIL("invalid type");
                break;
        }
    }

    // main decompression loop
    size_t chunkIndex = 0;
    while (1) {
        // Check end of frame marker
        if (dctx->dfh.formatVersion >= ZL_CHUNK_VERSION_MIN) {
            ZL_ERR_IF_LT(frameSize, consumed + 1, srcSize_tooSmall);
            uint8_t marker = ZL_read8((const char*)framePtr + consumed);
            ZL_DLOG(SEQ, "marker %u at pos %zu", marker, consumed);
            if (marker == 0) {
                ZL_DLOG(SEQ,
                        "End of frame detected at pos %zu",
                        marker,
                        consumed);
                consumed += 1;
                break;
            }
        }

        DWAYPOINT(on_decompressChunk_start, dctx, chunkIndex);
        ZL_TRY_LET(
                size_t,
                chunkSize,
                ZL_DCtx_decompressChunk(
                        dctx, nbOutputs, framePtr, frameSize, consumed));
        DWAYPOINT(on_decompressChunk_end, dctx, ZL_returnValue(chunkSize));
        ZL_DLOG(SEQ, "chunk size: %zu", chunkSize);
        consumed += chunkSize;
        chunkIndex++;

        if (dctx->dfh.formatVersion < ZL_CHUNK_VERSION_MIN)
            break;
    }

#if ZL_ENABLE_ASSERT
    {
        ZL_Report compressedSize = ZL_getCompressedSize(framePtr, frameSize);
#    ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
        if (ZL_isError(compressedSize)
            && ZL_errorCode(compressedSize) == ZL_ErrorCode_allocation) {
            // Allocation failure is allowed
        } else
#    endif
        {
            ZL_ASSERT_SUCCESS(
                    compressedSize,
                    "ZL_getCompressedSize() failed : %s",
                    ZL_ErrorCode_toString(ZL_errorCode(compressedSize)));
            ZL_ASSERT_EQ(
                    ZL_RES_value(compressedSize),
                    consumed,
                    "ZL_getCompressedSize() didn't return the same size as we used during decompression.");
        }
    }
#endif

#ifndef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    ZL_ERR_IF_NE(consumed, frameSize, srcSize_tooLarge);
#endif

    // check decompressed sizes
    for (size_t n = 0; n < nbOutputs; n++) {
        // just check the buffer is appropriately sized
        ZL_TRY_LET(
                size_t,
                dSize,
                ZL_FrameInfo_getDecompressedSize(dctx->dfh.frameinfo, (int)n));
        ZL_ERR_IF_NE(
                STREAM_byteSize(outputs[n]),
                dSize,
                corruption,
                "Regenerated size for output %zu is incorrect",
                n);
    }

    if (!dctx->preserveStreams) {
        // reclaim tmp memory
        cleanAllBuffers(dctx);
    }
    dctx->outputs = NULL;

    if (!DCtx_getAppliedGParam(dctx, ZL_DParam_stickyParameters)) {
        // If dctx parameters are not explicitly sticky, reset them
        ZL_ERR_IF_ERR(ZL_DCtx_resetParameters(dctx));
    }

    ZL_DLOG(BLOCK,
            "ZL_DCtx_decompressMultiTBuffer: success: decompressed %zu Typed Buffers",
            nbOutputs);
    DWAYPOINT(
            on_ZL_DCtx_decompressMultiTBuffer_end,
            dctx,
            ZL_returnValue(nbOutputs));
    return ZL_returnValue(nbOutputs);
}

ZL_Report ZL_DCtx_decompressTBuffer(
        ZL_DCtx* dctx,
        ZL_TypedBuffer* tbuffer,
        const void* compressed,
        size_t cSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_DLOG(FRAME,
            "ZL_DCtx_decompressTBuffer: decompressing a Typed Buffer of capacity %zu",
            STREAM_byteCapacity(ZL_codemodOutputAsData(tbuffer)));

    ZL_ERR_IF_ERR(ZL_DCtx_decompressMultiTBuffer(
            dctx, &tbuffer, 1, compressed, cSize));

    return ZL_returnValue(ZL_TypedBuffer_byteSize(tbuffer));
}

ZL_Report ZL_DCtx_decompressTyped(
        ZL_DCtx* dctx,
        ZL_OutputInfo* outputInfo,
        void* dst,
        size_t dstByteCapacity,
        const void* compressed,
        size_t cSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_DLOG(FRAME, "ZL_DCtx_decompressTyped");
    ZL_TypedBuffer* const tbuffer = ZL_TypedBuffer_create();
    ZL_ERR_IF_NULL(tbuffer, allocation);

    ZL_Report const tbir = STREAM_attachRawBuffer(
            ZL_codemodOutputAsData(tbuffer), dst, dstByteCapacity);
    if (ZL_isError(tbir)) {
        ZL_TypedBuffer_free(tbuffer);
        return tbir;
    }

    ZL_Report const dsr =
            ZL_DCtx_decompressTBuffer(dctx, tbuffer, compressed, cSize);

    if (ZL_isError(dsr)) {
        ZL_TypedBuffer_free(tbuffer);
        return dsr;
    }

    // final report on Success
    ZL_ASSERT_NN(outputInfo);
    outputInfo->type    = ZL_Output_type(tbuffer);
    ZL_Report const ewr = ZL_Output_eltWidth(tbuffer);
    if (ZL_isError(ewr)) {
        ZL_TypedBuffer_free(tbuffer);
        return ewr;
    }
    outputInfo->fixedWidth           = (uint32_t)ZL_validResult(ewr);
    outputInfo->decompressedByteSize = ZL_validResult(dsr);
    outputInfo->numElts = ZL_Data_numElts(ZL_codemodOutputAsData(tbuffer));

    ZL_TypedBuffer_free(tbuffer);
    return dsr;
}

ZL_Report ZL_DCtx_decompress(
        ZL_DCtx* dctx,
        void* const dst,
        const size_t dstCapacity,
        const void* const cSrc,
        const size_t cSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_DLOG(FRAME,
            "ZL_DCtx_decompress (cSrc=%zu, dstCapacity=%zu)",
            cSize,
            dstCapacity);
    // initialize @outInfo because analyzer does not know
    // that @outInfo is necessarily filled on success
    ZL_OutputInfo outInfo = { .type = ZL_Type_unassigned };
    ZL_Report const r     = ZL_DCtx_decompressTyped(
            dctx, &outInfo, dst, dstCapacity, cSrc, cSize);
    if (!ZL_isError(r)) {
        ZL_ERR_IF_NE(
                outInfo.type,
                ZL_Type_serial,
                GENERIC,
                "ZL_DCtx_decompress is only compatible with serialized output");
    }
    return r;
}

ZL_Report
ZL_decompress(void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    ZL_DCtx* const dctx = ZL_DCtx_create();
    ZL_ERR_IF_NULL(dctx, allocation);

    ZL_Report r = ZL_DCtx_decompress(dctx, dst, dstCapacity, src, srcSize);

    // Clear the info pointer because it points into the dctx
    ZL_RES_clearInfo(r);

    ZL_DCtx_free(dctx);
    return r;
}

ZL_CONST_FN
ZL_OperationContext* ZL_DCtx_getOperationContext(ZL_DCtx* dctx)
{
    if (dctx == NULL) {
        return NULL;
    }
    return &dctx->opCtx;
}

const char* ZL_DCtx_getErrorContextString(const ZL_DCtx* dctx, ZL_Report report)
{
    if (!ZL_isError(report)) {
        return NULL;
    }
    return ZL_OC_getErrorContextString(&dctx->opCtx, ZL_RES_error(report));
}

const char* ZL_DCtx_getErrorContextString_fromError(
        const ZL_DCtx* dctx,
        ZL_Error error)
{
    if (!ZL_E_isError(error)) {
        return NULL;
    }
    return ZL_OC_getErrorContextString(&dctx->opCtx, error);
}

ZL_Error_Array ZL_DCtx_getWarnings(ZL_DCtx const* dctx)
{
    return ZL_OC_getWarnings(&dctx->opCtx);
}

DFH_Struct const* DCtx_getFrameHeader(ZL_DCtx const* dctx)
{
    return &dctx->dfh;
}

/* Note : this is only used by streamdump2 currently */
ZL_Report DCtx_getNbInputStreams(ZL_DCtx const* dctx, ZL_IDType decoderIdx)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);

    ZL_ERR_IF_GE(decoderIdx, VECTOR_SIZE(dctx->dfh.nodes), GENERIC);
    return ZL_returnValue(
            VECTOR_AT(dctx->dfh.nodes, decoderIdx).numInputStreams);
}

/* Note : this is only used by streamdump2 currently */
const char* DCTX_getTrName(ZL_DCtx const* dctx, ZL_IDType decoderIdx)
{
    ZL_ASSERT_NN(dctx);
    if (decoderIdx >= VECTOR_SIZE(dctx->dfh.nodes))
        return NULL;
    const DFH_NodeInfo* const ni = &VECTOR_AT(dctx->dfh.nodes, decoderIdx);
    const PublicTransformInfo* const ptri = &(ni->trpid);
    return DTM_getTransformName(&dctx->dtm, *ptri, dctx->dfh.formatVersion);
}

size_t DCTX_streamMemory(ZL_DCtx const* dctx)
{
    return ALLOC_Arena_memAllocated(dctx->streamArena);
}

ZL_Report ZL_DCtx_attachDecompressIntrospectionHooks(
        ZL_DCtx* dctx,
        const ZL_DecompressIntrospectionHooks* hooks)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_NN(dctx);
    ZL_ERR_IF_NULL(hooks, allocation);
    ZL_OperationContext* oc = ZL_DCtx_getOperationContext(dctx);
    ZL_ASSERT_NN(oc);
    oc->decompressIntrospectionHooks = *hooks;
    oc->hasDecompressionHooks        = true;
    return ZL_returnSuccess();
}

ZL_Report ZL_DCtx_detachAllDecompressIntrospectionHooks(ZL_DCtx* dctx)
{
    ZL_ASSERT_NN(dctx);
    ZL_OperationContext* oc = ZL_DCtx_getOperationContext(dctx);
    ZL_ASSERT_NN(oc);
    ZL_zeroes(
            &oc->decompressIntrospectionHooks,
            sizeof(oc->decompressIntrospectionHooks));
    oc->hasDecompressionHooks = false;
    return ZL_returnSuccess();
}

ZL_Report DCTX_initFromFrameInfo(ZL_DCtx* dctx, const ZL_FrameInfo* frameInfo)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dctx);
    ZL_ASSERT_NN(dctx);
    ZL_TRY_LET(size_t, nbOutputs, ZL_FrameInfo_getNumOutputs(frameInfo));
    ZL_ERR_IF_ERR(DFH_setFrameInfo(
            &dctx->dfh, frameInfo, ZL_DCtx_getOperationContext(dctx)));
    dctx->nbOutputs = nbOutputs;
    return DCtx_setAppliedParameters(dctx);
}
