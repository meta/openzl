// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

/** @import {WorkerOutput} from './native-worker.cjs' */

'use strict';

const {z} = require('zod');

const count = z.int().min(0);
const duration = z.number().min(0);
// Throughput is Infinity when an operation measured 0 ms.
const throughput = z.number().min(0).or(z.literal(Infinity));

const candidateSchema = z
  .object({index: z.int().min(1), total: z.int().min(1)})
  .refine(({index, total}) => index <= total, {message: 'candidate index is past the total', path: ['index']})
  .nullable();

const trainingSchema = z.object({paretoFrontier: z.literal(true), candidates: z.int().min(1)}).nullable();

const jobBase = {id: z.string(), rowId: z.int(), level: z.int(), iterations: z.int().min(1)};

const jobSchema = z.discriminatedUnion('compressor', [
  z.object({...jobBase, compressor: z.literal('OpenZL'), profile: z.string(), training: trainingSchema}),
  z.object({...jobBase, compressor: z.literal('zstd')}),
  z.object({...jobBase, compressor: z.literal('gzip')}),
]);

const resultSchema = z.object({
  job: jobSchema,
  candidate: candidateSchema,
  iterations: z.int().min(1),
  srcSize: count,
  compressedSize: count,
  ratio: z.number().min(0),
  compressMs: duration,
  decompressMs: duration,
  compressMBps: throughput,
  decompressMBps: throughput,
});

/** The browser worker's `RunMessage`s, and what only the desktop worker sends. */
const workerMessageSchema = z.discriminatedUnion('type', [
  z.object({type: z.literal('child'), pid: z.int().min(1), state: z.enum(['spawned', 'exited'])}),
  z.object({type: z.literal('loading')}),
  z.object({
    type: z.literal('started'),
    totalJobs: count,
    rejected: z.array(z.object({rowId: z.int(), message: z.string()})),
  }),
  z.object({type: z.literal('step'), fraction: z.number().min(0).max(1)}),
  z.object({type: z.literal('result'), result: resultSchema}),
  z.object({
    type: z.literal('failure'),
    failure: z.object({job: jobSchema, candidate: candidateSchema, message: z.string()}),
  }),
  z.object({type: z.literal('finished')}),
  z.object({type: z.literal('failed'), message: z.string()}),
  z.object({type: z.literal('cancelled')}),
]);

/**
 * Returns a validated copy of a message from the benchmark worker, or throws
 * naming the first invalid field. Only validated copies reach the renderer.
 *
 * @param {unknown} message
 * @returns {WorkerOutput}
 */
function parseWorkerMessage(message) {
  const parsed = workerMessageSchema.safeParse(message);
  if (parsed.success) {
    return parsed.data;
  }
  const [issue] = parsed.error.issues;
  const field = issue !== undefined && issue.path.length > 0 ? `${issue.path.join('.')}: ` : '';
  throw new Error(`Native benchmark process sent an invalid message (${field}${issue?.message})`);
}

module.exports = {parseWorkerMessage};
