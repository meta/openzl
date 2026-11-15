// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_TOOLS_WASM_OPENZL_WASM_H
#define OPENZL_TOOLS_WASM_OPENZL_WASM_H

#include <stddef.h>
#include <stdint.h>

#include "openzl/zl_errors_types.h" // ZL_ErrorCode

#if defined(__cplusplus)
extern "C" {
#endif

void* openzl_wasm_malloc(size_t size);
void openzl_wasm_free(void* buf);

#define OPENZL_WASM_BENCHMARK_MAX_ITERATIONS 1000

/**
 * Default time budget for training. Some searches can ignore this, so training
 * can run longer.
 *
 * More time allows for a slightly better compressor with diminishing returns.
 *
 * Keep in sync with DEFAULT_TRAIN_TIME_SECS in the compression playground UI.
 */
#define OPENZL_WASM_TRAIN_DEFAULT_MAX_TIME_SECS 30

/**
 * Max threads the trainer will use.
 *
 * When building for wasm, CMake sets this to the number of workers created
 * at startup (OPENZL_WASM_PTHREAD_POOL_SIZE). Workers can't be added later
 * without freezing the page, so asking for more just gives you the max.
 */
#ifndef OPENZL_WASM_TRAIN_MAX_THREADS
#    define OPENZL_WASM_TRAIN_MAX_THREADS 4
#endif

/**
 * Returns the string descriptor for a ZL_ErrorCode.
 */
const char* openzl_wasm_errorString(ZL_ErrorCode code);

/**
 * Profile catalog, including profiles not supported by the current WASM build.
 *
 * These values index kProfiles in openzl_wasm.cpp, whose names match zli's and
 * key openzl::profiles::getProfileDescription().
 */
typedef enum {
    OPENZL_WASM_PROFILE_SERIAL                 = 0,
    OPENZL_WASM_PROFILE_U8                     = 1,
    OPENZL_WASM_PROFILE_I8                     = 2,
    OPENZL_WASM_PROFILE_LE_U16                 = 3,
    OPENZL_WASM_PROFILE_LE_I16                 = 4,
    OPENZL_WASM_PROFILE_LE_U32                 = 5,
    OPENZL_WASM_PROFILE_LE_I32                 = 6,
    OPENZL_WASM_PROFILE_LE_U64                 = 7,
    OPENZL_WASM_PROFILE_LE_I64                 = 8,
    OPENZL_WASM_PROFILE_BE_I16                 = 9,
    OPENZL_WASM_PROFILE_BE_I32                 = 10,
    OPENZL_WASM_PROFILE_BE_I64                 = 11,
    OPENZL_WASM_PROFILE_BE_U16                 = 12,
    OPENZL_WASM_PROFILE_BE_U32                 = 13,
    OPENZL_WASM_PROFILE_BE_U64                 = 14,
    OPENZL_WASM_PROFILE_CSV                    = 15,
    OPENZL_WASM_PROFILE_LZ                     = 16,
    OPENZL_WASM_PROFILE_NUMERIC_ML_SELECTOR_64 = 17,
    OPENZL_WASM_PROFILE_PARQUET                = 18,
    OPENZL_WASM_PROFILE_PYTORCH                = 19,
    OPENZL_WASM_PROFILE_SAO                    = 20,
    OPENZL_WASM_PROFILE_SDDL                   = 21,
    OPENZL_WASM_PROFILE_SDDL2                  = 22,
    OPENZL_WASM_PROFILE_ZSTD                   = 23,
    OPENZL_WASM_PROFILE_COUNT
} openzl_wasm_Profile;

/**
 * @returns The name of @p profile, e.g. "le-u32", or NULL if it is out of
 * range. Exposed so language bindings can derive their profile tables from
 * this.
 */
const char* openzl_wasm_profileName(openzl_wasm_Profile profile);

/** Returns profile help text, or NULL if it is out of range or undescribed. */
const char* openzl_wasm_profileDescription(openzl_wasm_Profile profile);

/** Returns 1 if the current WASM build supports the profile, or 0 otherwise. */
int openzl_wasm_profileIsSupported(openzl_wasm_Profile profile);

/** Returns the browser limitation, or NULL if supported or out of range. */
const char* openzl_wasm_profileUnsupportedReason(openzl_wasm_Profile profile);

/** Returns 1 if the profile supports training, or 0 otherwise. */
int openzl_wasm_profileIsTrainable(openzl_wasm_Profile profile);

/**
 * @returns The maximum iteration count accepted by the benchmark functions.
 */
int openzl_wasm_maxBenchmarkIterations(void);

/**
 * Serializes the graph for @p profile.
 *
 * The serialized compressor is pinned to ZL_MAX_FORMAT_VERSION, so it should
 * not be persisted across builds.
 *
 * @param profile           openzl_wasm_Profile that user wants to use.
 * @param compressionLevel  Compression level for compressor, if set to 0 will
 *                          use OpenZL's current default.
 * @param outBuf            On success, an owned buffer of serialized bytes,
 *                          release with openzl_wasm_free().
 * @param outSize           On success, the length of @p outBuf.
 * @returns                 ZL_ErrorCode_no_error on success.
 */
ZL_ErrorCode openzl_wasm_getSerializedCompressor(
        openzl_wasm_Profile profile,
        int compressionLevel,
        uint8_t** outBuf,
        size_t* outSize);

/**
 * Default number of compressors Pareto training asks the trainer for.
 *
 * This is also the smallest set the trainer's pruning can produce: it scores
 * candidates on three axes and pins the best and worst of each, so it refuses
 * to prune below 2 * 3.
 */
#define OPENZL_WASM_TRAIN_PARETO_CANDIDATES 6

/**
 * @returns OPENZL_WASM_TRAIN_PARETO_CANDIDATES. Exposed so language bindings
 * can size their out-parameter arrays from it rather than restating the value.
 */
int openzl_wasm_trainParetoCandidates(void);

typedef struct {
    /**
     * Size of the trainer's thread pool. OPENZL_WASM_TRAIN_MAX_THREADS is the
     * default when 0 is passed, and under Emscripten is also the ceiling,
     * because the worker pool is fixed at startup.
     */
    size_t threads;
    /**
     * Wall-clock budget, or 0 for OPENZL_WASM_TRAIN_DEFAULT_MAX_TIME_SECS.
     * The trainers spend whatever budget they are given, so this sets the
     * duration of the call rather than bounding it.
     */
    size_t maxTimeSecs;
    /**
     * Return the Pareto frontier rather than only the best-ratio
     * compressor. Non-zero enables it.
     */
    int paretoFrontier;
    /**
     * How many compressors to keep, defaulting to
     * OPENZL_WASM_TRAIN_PARETO_CANDIDATES. Ignored unless paretoFrontier is
     * set. Bounds the results, not the work: every candidate is searched and
     * benchmarked either way.
     *
     * Values below OPENZL_WASM_TRAIN_PARETO_CANDIDATES are raised to it, since
     * that is the floor of the trainer's pruning, and values above the
     * outCapacity passed to openzl_wasm_train() are lowered to it.
     */
    size_t maxNumCandidates;
    // Narrow the ACE search by not exploring successor graphs.
    int noAceSuccessors;
    // Skip the clustering trainer.
    int noClustering;
} openzl_wasm_TrainOptions;

/**
 * Trains @p compressor on @p src and serializes the result.
 *
 * Trains a compressor on data, resulting compressor can be used in
 * openzl_wasm_compress() or openzl_wasm_benchmarkCompress().
 *
 * Produces one compressor by default. With `options->paretoFrontier` can
 * produce a set of compressors trading compression ratio against speed.
 * The compressor must have a trainable graph. A compressor with no trainable
 * graph is an error, not a no-op.
 *
 * @param compressor      Serialized compressor to train.
 * @param compressorSize  Size of @p compressor.
 * @param src             Sample to train on. Must be non-NULL and non-empty.
 * @param srcSize         Size of @p src.
 * @param options         Training options, or NULL for the defaults.
 * @param outBufs         Array of at least @p outCapacity elements. Its first
 *                        @p *outCount entries are owned buffers of serialized
 *                        bytes, each released with openzl_wasm_free().
 * @param outSizes        Array of at least @p outCapacity elements. Its first
 *                        @p *outCount entries hold the length of the matching
 *                        @p outBufs entry.
 * @param outCapacity     How many entries @p outBufs and @p outSizes hold.
 *                        Must be non-zero. Size by
 *                        OPENZL_WASM_TRAIN_PARETO_CANDIDATES rather than by
 *                        `options->maxNumCandidates`: that is the floor of the
 *                        trainer's pruning, so training can return more than
 *                        was asked for. The excess is dropped.
 * @param outCount        On success, how many candidates were written.
 * @returns               ZL_ErrorCode_no_error on success.
 */
ZL_ErrorCode openzl_wasm_train(
        const uint8_t* compressor,
        size_t compressorSize,
        const uint8_t* src,
        size_t srcSize,
        const openzl_wasm_TrainOptions* options,
        uint8_t** outBufs,
        size_t* outSizes,
        size_t outCapacity,
        size_t* outCount);

/**
 * Compresses @p src with the compressor serialized in @p compressor, as
 * returned by openzl_wasm_getSerializedCompressor().
 *
 * @note Compression is permissive: a stage that rejects @p src falls back to a
 *       generic backend and the call still succeeds, at a worse ratio, rather
 *       than returning an error.
 *
 * @param compressor      Serialized compressor bytes.
 * @param compressorSize  Size of @p compressor.
 * @param src             Source bytes, may be NULL when @p srcSize is 0.
 * @param srcSize         Size of @p src.
 * @param outBuf          On success, an owned frame, release with
 *                        openzl_wasm_free().
 * @param outSize         On success, the length of @p outBuf.
 * @returns               ZL_ErrorCode_no_error on success.
 */
ZL_ErrorCode openzl_wasm_compress(
        const uint8_t* compressor,
        size_t compressorSize,
        const uint8_t* src,
        size_t srcSize,
        uint8_t** outBuf,
        size_t* outSize);

/**
 * Reads the decompressed size recorded in the frame header.
 *
 * It is exposed for callers that want the size before committing to the
 * decompression.
 *
 * @param outSize  On success, the recorded size. 0 is a valid answer.
 * @returns ZL_ErrorCode_no_error on success.
 */
ZL_ErrorCode openzl_wasm_getDecompressedSize(
        const uint8_t* src,
        size_t srcSize,
        size_t* outSize);

/**
 * Decompresses the frame in @p src.
 *
 * @param outBuf   On success, an owned buffer; release with
 *                 openzl_wasm_free(). Non-NULL even when @p outSize is 0.
 * @param outSize  On success, the length of @p outBuf.
 * @returns ZL_ErrorCode_no_error on success.
 */
ZL_ErrorCode openzl_wasm_decompress(
        const uint8_t* src,
        size_t srcSize,
        uint8_t** outBuf,
        size_t* outSize);

/**
 * Benchmarks compression reusing one deserialized compressor and one CCtx
 * throughout.
 *
 * @param iterations  Number of iterations to run. Must be between 1 and
 *                    OPENZL_WASM_BENCHMARK_MAX_ITERATIONS, anything outside
 *                    that returns ZL_ErrorCode_parameter_invalid.
 * @param outBuf      On success, an owned frame that needs to be released with
 *                    openzl_wasm_free().
 * @param outSize     Frame size.
 * @param outMs       Total milliseconds across all iterations.
 * @returns           ZL_ErrorCode_no_error on success.
 */
ZL_ErrorCode openzl_wasm_benchmarkCompress(
        const uint8_t* compressor,
        size_t compressorSize,
        const uint8_t* src,
        size_t srcSize,
        size_t iterations,
        uint8_t** outBuf,
        size_t* outSize,
        double* outMs);

/**
 * Benchmarks decompression reusing one DCtx.
 *
 * @param src         Compressed frame, as produced by openzl_wasm_compress().
 * @param srcSize     Size of @p src.
 * @param iterations  Number of iterations to run. Must be between 1 and
 *                    OPENZL_WASM_BENCHMARK_MAX_ITERATIONS, anything outside
 *                    that returns ZL_ErrorCode_parameter_invalid.
 * @param outMs       Total milliseconds across all iterations.
 * @returns           ZL_ErrorCode_no_error on success.
 */
ZL_ErrorCode openzl_wasm_benchmarkDecompress(
        const uint8_t* src,
        size_t srcSize,
        size_t iterations,
        double* outMs);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // OPENZL_TOOLS_WASM_OPENZL_WASM_H
