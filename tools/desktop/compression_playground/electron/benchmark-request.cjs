// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

/** @import {DesktopRunConfig} from '../../../web/compression_playground/src/benchmarkTypes.ts' */

'use strict';

const {z} = require('zod');

const ITERATIONS = 'Iterations must be a positive integer';
const INVALID_ROW = 'Invalid compressor row';
const TRAINING = 'OpenZL training options are invalid';
const INVALID_PATH = 'Invalid input path';

/** @param {string} compressor */
function levelsSchema(compressor) {
  const message = `Invalid ${compressor} levels`;
  return z.array(z.int({error: message}), {error: message});
}

const trainingSchema = z
  .object(
    {
      paretoFrontier: z.literal(true, {error: TRAINING}),
      candidates: z.int({error: TRAINING}).min(1, TRAINING),
    },
    {error: TRAINING},
  )
  .nullable();

// The row's shape is checked first, so a malformed row is not reported as an
// unknown compressor.
const compressorSchema = z
  .object({rowId: z.int({error: INVALID_ROW})}, {error: INVALID_ROW})
  .loose()
  .pipe(
    z.discriminatedUnion(
      'compressor',
      [
        z.object({
          rowId: z.int(),
          compressor: z.literal('OpenZL'),
          levels: levelsSchema('OpenZL'),
          profile: z.string({error: 'OpenZL profile is missing'}).min(1, 'OpenZL profile is missing'),
          training: trainingSchema,
        }),
        z.object({rowId: z.int(), compressor: z.literal('zstd'), levels: levelsSchema('zstd')}),
        z.object({rowId: z.int(), compressor: z.literal('gzip'), levels: levelsSchema('gzip')}),
      ],
      {error: 'Unknown compressor'},
    ),
  );

/** Keys are in the order they are reported: the first invalid one names the error. */
const requestSchema = z.object(
  {
    iterations: z.int({error: ITERATIONS}).min(1, ITERATIONS),
    compressors: z.array(compressorSchema, {error: 'Invalid compressor list'}),
    input: z.object(
      {
        path: z
          .string({error: INVALID_PATH})
          .min(1, INVALID_PATH)
          .refine((path) => !path.includes('\0'), INVALID_PATH),
      },
      {error: 'Benchmark input is missing'},
    ),
  },
  {error: 'Invalid benchmark request'},
);

/**
 * Returns a copy of `request` holding only the fields `DesktopRunConfig` has,
 * so nothing else reaches the worker. Throws a TypeError naming the first
 * field that is not shaped as it should be. Only the shape is checked: which
 * levels, candidate counts and so on `zli` and zlib accept is theirs to say,
 * and a job they refuse is reported as a failure like any other.
 *
 * @param {unknown} request
 * @returns {DesktopRunConfig}
 */
function validateRequest(request) {
  const parsed = requestSchema.safeParse(request);
  if (parsed.success) {
    return parsed.data;
  }
  throw new TypeError(parsed.error.issues[0]?.message);
}

module.exports = {validateRequest};
