// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CUSTOM_PARSERS_NUMPY_NUMPY_PARSER_H
#define CUSTOM_PARSERS_NUMPY_NUMPY_PARSER_H

#include "openzl/shared/portability.h"
#include "openzl/zl_compressor.h"

ZL_BEGIN_C_DECLS

#define ZL_NUMPY_DEFAULT_CHUNK_SIZE ((size_t)16 << 20)

/**
 * Registers a graph compressing NumPy .npy files and .npz archives, and the
 * NumPy arrays of pickles, such as those of pandas DataFrames.
 *
 * Each array becomes its own stream. Little-endian arrays of numbers go to the
 * Numeric Transformer, read as numbers of their element size: booleans,
 * integers, datetimes, timedeltas and opaque elements of 1, 2, 4 or 8 bytes,
 * and floats of 2, 4 or 8 bytes; complex numbers of 8 or 16 bytes are read as
 * pairs of floats. Other arrays (big-endian or extended precision numbers,
 * strings, objects, structured types), the headers and pickle opcodes, and
 * the members that `np.savez_compressed` deflates, are compressed apart.
 * Consecutive streams smaller than 16 KiB share a stream, sent to the raw
 * successor when their classes differ: the small arrays of a .npz archive are
 * compressed with the headers around them.
 * Successors are ACE graphs, so the resulting compressor can be trained.
 * Chunks are formed as with ZL_Safetensors_registerGraph().
 * Inputs that are neither .npy files, pickles of protocols 2 to 5, nor zip
 * archives, and format versions without chunks, are compressed with the
 * generic graph, and the reason is reported as a warning.
 */
ZL_RESULT_OF(ZL_GraphID)
ZL_NumPy_registerGraph(ZL_Compressor* compressor, size_t chunkSizeMax);

ZL_END_C_DECLS

#endif
