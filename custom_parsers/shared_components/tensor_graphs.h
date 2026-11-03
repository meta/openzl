// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CUSTOM_PARSERS_SHARED_COMPONENTS_TENSOR_GRAPHS_H
#define CUSTOM_PARSERS_SHARED_COMPONENTS_TENSOR_GRAPHS_H

#include <stddef.h>
#include <stdint.h>

#include "openzl/shared/portability.h"
#include "openzl/zl_compressor.h"

ZL_BEGIN_C_DECLS

/// Classes of segments of a model weight file, each with its own successor.
/// Serialized compressors keep their list of successors, so new classes must
/// be appended: existing values never change.
typedef enum {
    ZL_TensorClass_header = 0,
    ZL_TensorClass_raw, // unknown content: gaps, unsupported types
    ZL_TensorClass_int8,
    ZL_TensorClass_int16,
    ZL_TensorClass_int32,
    ZL_TensorClass_int64,
    ZL_TensorClass_f16,
    ZL_TensorClass_bf16,
    ZL_TensorClass_f32,
    ZL_TensorClass_f64,
    ZL_TensorClass_atypical16, // low-cardinality 16-bit floats
    ZL_TensorClass_atypical32, // low-cardinality 32-bit floats
    ZL_TensorClass_scale16,  // float16 or bfloat16 scales of quantized weights
    ZL_TensorClass_fp8,      // 8-bit floats
    ZL_TensorClass_upcast32, // 32-bit floats converted from 16-bit floats
    ZL_TensorClass_count,
} ZL_TensorClass;

/// Number of elements of the scratch buffer of ZL_TensorGraphs_refineClass().
#define ZL_TENSOR_SAMPLE_SIZE 4096

/// @returns ZL_TensorClass_atypical16 or ZL_TensorClass_atypical32 for an f16,
/// bf16 or f32 tensor with few distinct values (masks, constants...),
/// ZL_TensorClass_upcast32 for an f32 tensor whose values mostly keep the 13
/// lowest bits of their fraction at zero, as values converted from f16 or bf16
/// do, else @p cls. @p sample is a scratch buffer of ZL_TENSOR_SAMPLE_SIZE
/// elements.
ZL_TensorClass ZL_TensorGraphs_refineClass(
        ZL_TensorClass cls,
        const uint8_t* data,
        size_t size,
        uint32_t* sample);

/// Registers the graph of the float16 or bfloat16 scales of quantized
/// weights, which chooses a codec per input. Later calls on the same
/// compressor return the same graph.
ZL_RESULT_OF(ZL_GraphID)
ZL_TensorGraphs_registerScale16Graph(ZL_Compressor* compressor);

/// Registers the graph of ZL_TensorClass_upcast32, which codes float32 values
/// converted from bfloat16 as bfloat16 values, and entropy codes the bytes of
/// the fraction of the others. Later calls on the same compressor return the
/// same graph.
ZL_RESULT_OF(ZL_GraphID)
ZL_TensorGraphs_registerUpcast32Graph(ZL_Compressor* compressor);

/// Fills @p successors with the default graph of each class. They are ACE
/// graphs, so that training can replace them. @p scale16 is the graph of
/// ZL_TensorGraphs_registerScale16Graph(), which parsers can also use inside
/// quantized blocks.
void ZL_TensorGraphs_buildSuccessors(
        ZL_Compressor* compressor,
        ZL_GraphID scale16,
        ZL_GraphID successors[ZL_TensorClass_count]);

ZL_END_C_DECLS

#endif
