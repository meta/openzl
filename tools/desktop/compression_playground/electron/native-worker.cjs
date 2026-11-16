// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/**
 * @import {BenchmarkJob, DesktopRunConfig, RejectedCompressor, RunMessage} from '../../../web/compression_playground/src/benchmarkTypes.ts'
 * @import {BenchmarkResult as BenchmarkMetrics} from '../../../wasm/js/wasm_api.js'
 * @import {ZliMeasurement} from './zli-client.cjs'
 */

/** @typedef {Extract<RunMessage, {type: 'result' | 'failure'}>} Measurement */

const fs = require('node:fs/promises');
const {performance} = require('node:perf_hooks');
const zlib = require('node:zlib');
const {measureOpenZL} = require('./zli-client.cjs');

/**
 * @param {Buffer} input
 * @param {number} iterations
 * @param {(value: Buffer) => Buffer} compress
 * @param {(value: Buffer) => Buffer} decompress
 * @returns {ZliMeasurement}
 */
function nativeCodecBenchmark(input, iterations, compress, decompress) {
  let compressed = compress(input);
  const compressStart = performance.now();
  for (let index = 0; index < iterations; index += 1) {
    compressed = compress(input);
  }
  const compressMs = performance.now() - compressStart;

  let decompressed = decompress(compressed);
  const decompressStart = performance.now();
  for (let index = 0; index < iterations; index += 1) {
    decompressed = decompress(compressed);
  }
  const decompressMs = performance.now() - decompressStart;
  if (!decompressed.equals(input)) {
    throw new Error('Native codec did not reproduce the input');
  }
  return {iterations, srcSize: input.length, compressedSize: compressed.length, compressMs, decompressMs};
}

/**
 * @param {Buffer} input
 * @param {number} level
 * @param {number} iterations
 */
function benchmarkZstd(input, level, iterations) {
  if (typeof zlib.zstdCompressSync !== 'function' || typeof zlib.zstdDecompressSync !== 'function') {
    throw new Error('This Electron runtime does not include native zstd support');
  }
  const params = {[zlib.constants.ZSTD_c_compressionLevel]: level};
  return nativeCodecBenchmark(
    input,
    iterations,
    (value) => zlib.zstdCompressSync(value, {params}),
    (value) => zlib.zstdDecompressSync(value),
  );
}

/**
 * @param {Buffer} input
 * @param {number} level
 * @param {number} iterations
 */
function benchmarkGzip(input, level, iterations) {
  return nativeCodecBenchmark(
    input,
    iterations,
    (value) => zlib.gzipSync(value, {level}),
    (value) => zlib.gunzipSync(value),
  );
}

/**
 * The browser's metrics for a measurement: its sizes and times, with the ratio
 * and throughput worked out from them.
 *
 * @param {ZliMeasurement} measurement
 * @returns {BenchmarkMetrics}
 */
function withThroughput(measurement) {
  const bytesMeasured = measurement.srcSize * measurement.iterations;
  return {
    ...measurement,
    ratio: measurement.compressedSize === 0 ? 0 : measurement.srcSize / measurement.compressedSize,
    compressMBps: measurement.compressMs === 0 ? Infinity : bytesMeasured / measurement.compressMs / 1000,
    decompressMBps: measurement.decompressMs === 0 ? Infinity : bytesMeasured / measurement.decompressMs / 1000,
  };
}

/** @param {unknown} error */
function messageOf(error) {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Splits a run into one job per compressor per level, as the browser's
 * `buildJobs` does, so job ids and `totalJobs` mean the same in both builds. A
 * row with no levels is rejected rather than dropped. Which profiles can run is
 * the page's to decide: it only offers those this build's `zli` can run, and
 * one that cannot is reported by `zli` itself.
 *
 * @param {DesktopRunConfig} config
 * @returns {{jobs: BenchmarkJob[]; rejected: RejectedCompressor[]}}
 */
function planJobs(config) {
  /** @type {BenchmarkJob[]} */
  const jobs = [];
  /** @type {RejectedCompressor[]} */
  const rejected = [];
  for (const compressor of config.compressors) {
    if (compressor.levels.length === 0) {
      rejected.push({rowId: compressor.rowId, message: 'No levels selected'});
      continue;
    }
    for (const level of compressor.levels) {
      const base = {
        id: `${compressor.rowId}-${compressor.compressor}-${level}`,
        rowId: compressor.rowId,
        level,
        iterations: config.iterations,
      };
      jobs.push(
        compressor.compressor === 'OpenZL'
          ? {...base, compressor: 'OpenZL', profile: compressor.profile, training: compressor.training}
          : {...base, compressor: compressor.compressor},
      );
    }
  }
  return {jobs, rejected};
}

/**
 * Every measurement a job produces, as `measureRun.ts` yields them in the
 * browser: one, or one per trained candidate as soon as it is measured. A
 * candidate that cannot be measured is a `failure` naming it, and the rest of
 * its frontier is still measured.
 *
 * @param {BenchmarkJob} job
 * @param {{zliPath: string; inputPath: string; readInput: () => Promise<Buffer>}} context
 * @returns {AsyncGenerator<Measurement>}
 */
async function* measure(job, {zliPath, inputPath, readInput}) {
  switch (job.compressor) {
    case 'OpenZL':
      for await (const outcome of measureOpenZL(zliPath, inputPath, job)) {
        yield 'error' in outcome
          ? {type: 'failure', failure: {job, candidate: outcome.candidate, message: messageOf(outcome.error)}}
          : {type: 'result', result: {...withThroughput(outcome.measurement), job, candidate: outcome.candidate}};
      }
      return;
    case 'zstd': {
      const measurement = benchmarkZstd(await readInput(), job.level, job.iterations);
      yield {type: 'result', result: {...withThroughput(measurement), job, candidate: null}};
      return;
    }
    case 'gzip': {
      const measurement = benchmarkGzip(await readInput(), job.level, job.iterations);
      yield {type: 'result', result: {...withThroughput(measurement), job, candidate: null}};
    }
  }
}

/**
 * Runs `config`, posting the browser worker's messages in its order:
 * `loading`, `started` with the job count, a `result` per measurement or a
 * `failure` for a job that could not be measured, then `finished`. One job at a
 * time, since these are wall-clock timings. A failed job does not stop the run.
 *
 * @param {{zliPath: string; config: DesktopRunConfig}} payload
 * @param {(message: RunMessage) => void} post
 */
async function run({zliPath, config}, post) {
  post({type: 'loading'});
  const {jobs, rejected} = planJobs(config);
  post({type: 'started', totalJobs: jobs.length, rejected});
  /** @type {Promise<Buffer> | undefined} */
  let input;
  // Only zstd and gzip need the bytes; zli reads the file itself.
  const readInput = () => (input ??= fs.readFile(config.input.path));
  for (const job of jobs) {
    try {
      for await (const measurement of measure(job, {zliPath, inputPath: config.input.path, readInput})) {
        post(measurement);
      }
    } catch (error) {
      // Reported and stepped over, as in the browser: one compressor that
      // cannot be measured should not cost the others their results.
      post({type: 'failure', failure: {job, candidate: null, message: messageOf(error)}});
    }
  }
  post({type: 'finished'});
}

/**
 * Worker protocol. From the main process: {zliPath, config}. To it: run()'s
 * messages, ending in exactly one of {type: 'finished'} or {type: 'failed',
 * message}.
 *
 * @param {{zliPath: string; config: DesktopRunConfig}} payload
 * @param {(message: RunMessage) => void} post
 */
async function execute(payload, post) {
  try {
    await run(payload, post);
  } catch (error) {
    post({type: 'failed', message: messageOf(error)});
  }
}

const {parentPort} = process;
if (parentPort !== undefined) {
  parentPort.once('message', (event) => {
    void execute(event.data, (message) => parentPort.postMessage(message));
  });
}

module.exports = {planJobs, run};
