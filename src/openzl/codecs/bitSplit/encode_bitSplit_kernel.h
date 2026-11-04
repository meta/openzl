// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_BITSPLIT_ENCODE_BITSPLIT_KERNEL_H
#define OPENZL_CODECS_BITSPLIT_ENCODE_BITSPLIT_KERNEL_H

#include <stdbool.h> /* bool */
#include <stddef.h>  /* size_t */
#include <stdint.h>  /* uint8_t */

/*
 * Validation helpers for assert() precondition checks.
 * These verify that parameters are valid before the kernel operates.
 */

/**
 * Validates bitSplit parameters.
 *
 * Checks that:
 *   - Each bitWidths[i] > 0
 *   - sum(bitWidths) <= inputEltWidthBits
 *
 * Preconditions:
 *   - bitWidths != NULL
 *   - nbWidths > 0
 *
 * @param bitWidths Array of bit widths
 * @param nbWidths Number of bit widths (must be > 0)
 * @param inputEltWidthBits Input element width in bits
 * @param sumWidths Output: sum of all bit widths (may be NULL)
 * @return true if parameters are valid, false otherwise
 */
bool ZL_bitSplit_paramsAreValid(
        const uint8_t* bitWidths,
        size_t nbWidths,
        size_t inputEltWidthBits,
        size_t* sumWidths);

/**
 * Checks if top bits (beyond sum of widths) are zero for all elements.
 *
 * Preconditions:
 *   - srcEltWidth in {1, 2, 4, 8}
 *   - If nbElts > 0: src != NULL
 *
 * @param src Source array (may be NULL if nbElts == 0)
 * @param srcEltWidth Element width of source (1, 2, 4, or 8 bytes)
 * @param nbElts Number of elements to check
 * @param sumWidths Sum of bit widths used
 * @return true if all top bits are zero, false otherwise
 */
bool ZL_bitSplit_topBitsAreZero(
        const void* src,
        size_t srcEltWidth,
        size_t nbElts,
        size_t sumWidths);

/**
 * Encodes an array of elements by extracting bit ranges.
 *
 * Splits each source element into multiple destination streams by extracting
 * bit ranges. Bit ranges are extracted LSB-to-MSB: bitWidths[0] extracts the
 * lowest bits, bitWidths[1] the next higher bits, and so on.
 *
 * Preconditions:
 *   - srcEltWidth in {1, 2, 4, 8}
 *   - 0 < nbWidths <= 64
 *   - If nbElts > 0:
 *     - dstPtrs != NULL, and each dstPtrs[i] != NULL
 *     - dstEltWidths != NULL, each dstEltWidths[i] in {1, 2, 4, 8}
 *     - src != NULL
 *     - bitWidths != NULL, each bitWidths[i] > 0
 *     - sum(bitWidths) <= srcEltWidth * 8
 *     - Each bitWidths[i] <= dstEltWidths[i] * 8
 *     - Top bits of src elements (beyond sum(bitWidths)) must be zero
 *     - src must not overlap any dstPtrs[i] buffer
 *
 * @param dstPtrs Array of destination pointers (one per bit range)
 * @param dstEltWidths Array of destination element widths in bytes
 * @param nbElts Number of elements to encode
 * @param src Source array of input values
 * @param srcEltWidth Element width of source (1, 2, 4, or 8 bytes)
 * @param bitWidths Array of bit widths (LSB to MSB order)
 * @param nbWidths Number of bit widths (1-64)
 */
void ZL_bitSplitEncode(
        void* const dstPtrs[],
        const size_t* dstEltWidths,
        size_t nbElts,
        const void* src,
        size_t srcEltWidth,
        const uint8_t* bitWidths,
        size_t nbWidths);

#endif // OPENZL_CODECS_BITSPLIT_ENCODE_BITSPLIT_KERNEL_H
