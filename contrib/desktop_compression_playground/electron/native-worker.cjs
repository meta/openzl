// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/**
 * @import {BenchmarkJob, DesktopRunConfig, RejectedCompressor, RunMessage} from '../../../tools/web/compression_playground/src/benchmarkTypes.ts'
 * @import {BenchmarkResult as BenchmarkMetrics} from '../../../tools/wasm/js/wasm_api.js'
 * @import {ZliChildEvent, ZliMeasurement} from './zli-client.cjs'
 */

/** @typedef {Extract<RunMessage, {type: 'result' | 'failure'}>} Measurement */

/**
 * The run the worker is given: the main process has resolved the bundled
 * sample, if that was asked for, to the path it is installed at.
 *
 * @typedef {Omit<DesktopRunConfig, 'input'> & {input: {path: string}}} ResolvedRunConfig
 */

/**
 * What the worker sends the main process: the run's messages, each zli process
 * it starts and ends, and `cancelled`, which only the desktop app has.
 *
 * @typedef {RunMessage | {type: 'cancelled'} | ({type: 'child'} & ZliChildEvent)} WorkerOutput
 */

/**
 * @typedef {object} RunOptions
 * @property {AbortSignal} [signal] Aborting it stops the run, and run() rejects with its reason.
 * @property {(event: ZliChildEvent) => void} [onChild]
 */

/**
 * @typedef {object} MeasureContext
 * @property {string} zliPath
 * @property {string} zstdBenchmarkPath
 * @property {string} inputPath
 * @property {() => Promise<Buffer>} readInput
 * @property {string} [tempRoot]
 * @property {(fraction: number) => void} onStep How far the job is through its training, as zli reports it.
 */

const fs = require('node:fs/promises');
const {performance} = require('node:perf_hooks');
const zlib = require('node:zlib');
const {checkInputPath, measureOpenZL, measureZstd} = require('./zli-client.cjs');

/**
 * zlib is synchronous, so the worker yields between timed iterations to receive
 * a cancel message. Only the codec calls themselves are timed.
 *
 * @param {number} iterations
 * @param {AbortSignal | undefined} signal
 * @param {() => Buffer} operation
 */
async function timedIterations(iterations, signal, operation) {
  let elapsedMs = 0;
  let output = Buffer.alloc(0);
  for (let index = 0; index < iterations; index += 1) {
    await new Promise((resolve) => setImmediate(resolve));
    signal?.throwIfAborted();
    const start = performance.now();
    output = operation();
    elapsedMs += performance.now() - start;
  }
  return {elapsedMs, output};
}

/**
 * gzip stays on Electron's zlib, since OpenZL does not vendor zlib.
 *
 * @param {Buffer} input
 * @param {number} level
 * @param {number} iterations
 * @param {AbortSignal} [signal]
 * @returns {Promise<ZliMeasurement>}
 */
async function benchmarkGzip(input, level, iterations, signal) {
  const compress = () => zlib.gzipSync(input, {level});
  // The untimed warm-up calls cannot be interrupted either, so check first.
  signal?.throwIfAborted();
  compress();
  const {elapsedMs: compressMs, output: compressed} = await timedIterations(iterations, signal, compress);

  const decompress = () => zlib.gunzipSync(compressed);
  signal?.throwIfAborted();
  decompress();
  const {elapsedMs: decompressMs, output: decompressed} = await timedIterations(iterations, signal, decompress);
  if (!decompressed.equals(input)) {
    throw new Error('gzip did not reproduce the input');
  }
  return {iterations, srcSize: input.length, compressedSize: compressed.length, compressMs, decompressMs};
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

/**
 * gzip compresses the whole input in memory, and Node reads at most 2 GiB less
 * a byte into one buffer.
 */
const GZIP_MAX_INPUT_BYTES = 2 ** 31 - 1;

/** @param {unknown} error */
function messageOf(error) {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Splits a run into one job per compressor per level, as the browser's
 * `buildJobs` does, so job ids and `totalJobs` mean the same in both builds. A
 * row with no levels is rejected rather than dropped, as is a gzip row for an
 * input too large for gzip to read. Which profiles can run is
 * the page's to decide: it only offers those this build's `zli` can run, and
 * one that cannot is reported by `zli` itself.
 *
 * @param {ResolvedRunConfig} config
 * @param {number} inputSize The input's size in bytes.
 * @returns {{jobs: BenchmarkJob[]; rejected: RejectedCompressor[]}}
 */
function planJobs(config, inputSize) {
  /** @type {BenchmarkJob[]} */
  const jobs = [];
  /** @type {RejectedCompressor[]} */
  const rejected = [];
  for (const compressor of config.compressors) {
    if (compressor.levels.length === 0) {
      rejected.push({rowId: compressor.rowId, message: 'No levels selected'});
      continue;
    }
    if (compressor.compressor === 'gzip' && inputSize > GZIP_MAX_INPUT_BYTES) {
      rejected.push({rowId: compressor.rowId, message: 'gzip can only benchmark an input smaller than 2 GiB'});
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
          ? {
              ...base,
              compressor: 'OpenZL',
              profile: compressor.profile,
              profileArg: compressor.profileArg,
              training: compressor.training,
            }
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
 * @param {MeasureContext & RunOptions} context
 * @returns {AsyncGenerator<Measurement>}
 */
async function* measure(job, {zliPath, zstdBenchmarkPath, inputPath, readInput, tempRoot, onStep, signal, onChild}) {
  switch (job.compressor) {
    case 'OpenZL':
      for await (const outcome of measureOpenZL(zliPath, inputPath, job, {
        signal,
        onChild,
        tempRoot,
        onProgress: onStep,
      })) {
        yield 'error' in outcome
          ? {type: 'failure', failure: {job, candidate: outcome.candidate, message: messageOf(outcome.error)}}
          : {type: 'result', result: {...withThroughput(outcome.measurement), job, candidate: outcome.candidate}};
      }
      return;
    case 'zstd': {
      const measurement = await measureZstd(zstdBenchmarkPath, inputPath, job, {signal, onChild, tempRoot});
      yield {type: 'result', result: {...withThroughput(measurement), job, candidate: null}};
      return;
    }
    case 'gzip': {
      const measurement = await benchmarkGzip(await readInput(), job.level, job.iterations, signal);
      yield {type: 'result', result: {...withThroughput(measurement), job, candidate: null}};
    }
  }
}

/**
 * Runs `config`, posting the browser worker's messages in its order:
 * `loading`, `started` with the job count, a `result` per measurement as soon
 * as it is made or a `failure` for a job that could not be measured, then
 * `finished`. While a job trains, `step` carries how far through training it
 * is, as the browser forwards its trainer's progress; the page places that
 * within the job's share of the run. One job at a
 * time, since these are wall-clock timings. A failed job does not stop the run;
 * only cancelling does, and then run() rejects with the signal's reason.
 *
 * @param {{zliPath: string; zstdBenchmarkPath: string; config: ResolvedRunConfig; tempRoot?: string}} payload
 * @param {(message: RunMessage) => void} post
 * @param {RunOptions} [options]
 */
async function run({zliPath, zstdBenchmarkPath, config, tempRoot}, post, {signal, onChild} = {}) {
  post({type: 'loading'});
  // Once for the whole run, so a path zli cannot read fails the run before any
  // job starts rather than failing every job.
  const inputPath = await checkInputPath(config.input.path, {tempRoot});
  const {size} = await fs.stat(inputPath);
  const {jobs, rejected} = planJobs(config, size);
  post({type: 'started', totalJobs: jobs.length, rejected});
  /** @type {Promise<Buffer> | undefined} */
  let input;
  // Only gzip needs the bytes; zli and zstd-benchmark read the file themselves.
  const readInput = () => (input ??= fs.readFile(inputPath));
  /** @param {number} fraction */
  const onStep = (fraction) => post({type: 'step', fraction});
  const context = {zliPath, zstdBenchmarkPath, inputPath, readInput, tempRoot, onStep, signal, onChild};
  for (const job of jobs) {
    signal?.throwIfAborted();
    try {
      for await (const measurement of measure(job, context)) {
        post(measurement);
      }
    } catch (error) {
      if (signal?.aborted) {
        throw error;
      }
      // Reported and stepped over, as in the browser: one compressor that
      // cannot be measured should not cost the others their results.
      post({type: 'failure', failure: {job, candidate: null, message: messageOf(error)}});
    }
  }
  post({type: 'finished'});
}

/**
 * Worker protocol. From the main process: {type: 'start', zliPath,
 * zstdBenchmarkPath, config, tempRoot}, then optionally {type: 'cancel'}. To
 * it: {type: 'child', pid, state} for every native process, run()'s messages, ending in exactly one of
 * {type: 'finished'}, {type: 'failed', message}, or {type: 'cancelled'}.
 *
 * @param {Electron.ParentPort} parentPort
 */
function serve(parentPort) {
  const controller = new AbortController();
  /** @param {WorkerOutput} message */
  const postMessage = (message) => parentPort.postMessage(message);
  parentPort.on('message', ({data}) => {
    if (data?.type === 'cancel') {
      controller.abort(new Error('Benchmark cancelled'));
    } else if (data?.type === 'start') {
      const {signal} = controller;
      const onChild = (/** @type {ZliChildEvent} */ event) => postMessage({type: 'child', ...event});
      run(data, postMessage, {signal, onChild}).catch((error) =>
        postMessage(
          // Only a rejection the abort caused: a run can fail on its own just
          // as a cancel arrives, and its error should still be seen.
          signal.aborted && error === signal.reason ? {type: 'cancelled'} : {type: 'failed', message: messageOf(error)},
        ),
      );
    }
  });
}

if (process.parentPort !== undefined) {
  serve(process.parentPort);
}

module.exports = {planJobs, run};
