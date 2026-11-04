// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_BITSPLIT_DECODE_BITSPLIT_KERNEL_H
#define OPENZL_CODECS_BITSPLIT_DECODE_BITSPLIT_KERNEL_H

#include <stddef.h> /* size_t */
#include <stdint.h> /* uint8_t */

/**
 * Decodes bit-split streams back into original integer elements.
 *
 * Reassembles values by concatenating bit ranges from multiple source streams.
 * Bit ranges are placed LSB-to-MSB: bitWidths[0] occupies the lowest bits,
 * bitWidths[1] the next higher bits, and so on.
 *
 * If sum(bitWidths) < dstEltWidth*8, upper bits of output are zero.
 *
 * Preconditions:
 *   - dstEltWidth in {1, 2, 4, 8}
 *   - 0 < nbWidths <= 64
 *   - If nbElts > 0:
 *     - dst != NULL
 *     - srcPtrs != NULL, and each srcPtrs[i] != NULL
 *     - srcEltWidths != NULL, each srcEltWidths[i] in {1, 2, 4, 8}
 *     - bitWidths != NULL, each bitWidths[i] > 0
 *     - Each bitWidths[i] <= srcEltWidths[i] * 8
 *     - sum(bitWidths) <= dstEltWidth * 8
 *     - dst must not overlap any srcPtrs[i] buffer
 *
 * @param dst Destination array for reconstructed values
 * @param dstEltWidth Element width of destination in bytes (1, 2, 4, or 8)
 * @param nbElts Number of elements to decode
 * @param srcPtrs Array of source pointers, one per bit range (none may be NULL)
 * @param srcEltWidths Array of source element widths in bytes
 * @param bitWidths Array of bit widths, ordered LSB to MSB
 * @param nbWidths Number of bit ranges / source streams (1-64)
 */
void ZL_bitSplitDecode(
        void* dst,
        size_t dstEltWidth,
        size_t nbElts,
        const void* const srcPtrs[],
        const size_t* srcEltWidths,
        const uint8_t* bitWidths,
        size_t nbWidths);

#endif // OPENZL_CODECS_BITSPLIT_DECODE_BITSPLIT_KERNEL_H
