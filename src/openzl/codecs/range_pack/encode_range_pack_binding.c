// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/range_pack/encode_range_pack_binding.h"
#include "openzl/codecs/range_pack/encode_range_pack_kernel.h"

#include "openzl/codecs/zl_range_pack.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/estimate.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/numeric_operations.h" // NUMOP_*
#include "openzl/shared/utils.h"              // ZL_MIN
#include "openzl/zl_compressor.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"

ZL_Report EI_rangePack(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    ZL_ASSERT_EQ(nbIns, 1);
    ZL_ASSERT_NN(ins);
    const ZL_Input* in    = ins[0];
    void const* src       = ZL_Input_ptr(in);
    size_t const srcWidth = ZL_Input_eltWidth(in);
    size_t const nbElts   = ZL_Input_numElts(in);
    ZL_ASSERT(ZL_isLegalIntegerWidth(srcWidth));
    ZL_ElementRange const range =
            ZL_computeUnsignedRange(src, nbElts, srcWidth);
    // The decoder adds back any value up to the minimum.
    uint64_t subtracted = range.min;
    ZL_RefParam const bound =
            ZL_Encoder_getLocalParam(eictx, ZL_RANGE_PACK_MAX_SUBTRACTED_PID);
    if (bound.paramRef != NULL) {
        ZL_ERR_IF_NE(bound.paramSize, sizeof(uint64_t), graphParameter_invalid);
        subtracted = ZL_MIN(subtracted, *(const uint64_t*)bound.paramRef);
    }
    const size_t dstWidth = NUMOP_numericWidthForValue(range.max - subtracted);

    ZL_Output* const dstStream =
            ZL_Encoder_createTypedStream(eictx, 0, nbElts, dstWidth);
    ZL_ERR_IF(!dstStream, allocation);
    void* dst = ZL_Output_ptr(dstStream);
    rangePackEncode(dst, dstWidth, src, srcWidth, nbElts, subtracted);
    ZL_ERR_IF_ERR(ZL_Output_commit(dstStream, nbElts));

    /* Header has the source size and the min value if needed */
    uint8_t header[9];
    size_t header_size = 1;
    header[0]          = (uint8_t)srcWidth;
    if (subtracted != 0) {
        // Only the first srcWidth bytes are sent
        ZL_writeLE64(header + 1, subtracted);
        header_size += srcWidth;
    }
    ZL_Encoder_sendCodecHeader(eictx, &header, header_size);
    return ZL_returnValue(1);
}

ZL_NodeID ZL_Compressor_registerRangePackNode(
        ZL_Compressor* compressor,
        uint64_t maxSubtracted)
{
    const ZL_CopyParam copyParam        = { .paramPtr  = &maxSubtracted,
                                            .paramSize = sizeof(maxSubtracted),
                                            .paramId =
                                                    ZL_RANGE_PACK_MAX_SUBTRACTED_PID };
    const ZL_LocalCopyParams copyParams = { .copyParams   = &copyParam,
                                            .nbCopyParams = 1 };
    const ZL_LocalParams localParams    = { .copyParams = copyParams };
    return ZL_Compressor_registerParameterizedNode(
            compressor,
            &(const ZL_ParameterizedNodeDesc){
                    .node        = ZL_NODE_RANGE_PACK,
                    .localParams = &localParams,
            });
}
