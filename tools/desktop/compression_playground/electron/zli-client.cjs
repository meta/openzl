// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/** @import {Candidate, OpenZlBenchmarkJob} from '../../../web/compression_playground/src/benchmarkTypes.ts' */

const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {spawn} = require('node:child_process');

const TRAINING_THREADS = Math.min(4, Math.max(1, os.availableParallelism()));
// The leading columns of zli benchmark --output-csv. The path column after them
// is not read, so a quoted path cannot shift the numbers.
const CSV_COLUMNS = ['srcSize', 'compressedSize', 'compressionRatio', 'ctimeMs', 'dtimeMs', 'iters'];

/**
 * What one `zli benchmark` measured, before throughput is worked out from it.
 *
 * @typedef {object} ZliMeasurement
 * @property {number} iterations
 * @property {number} srcSize
 * @property {number} compressedSize
 * @property {number} compressMs
 * @property {number} decompressMs
 */

/**
 * One measurement of an OpenZL job, or why one of its trained candidates could
 * not be measured.
 *
 * @typedef {{candidate: Candidate | null; measurement: ZliMeasurement} | {candidate: Candidate; error: unknown}} OpenZLOutcome
 */

/**
 * @param {string} zliPath
 * @param {string[]} args
 * @returns {Promise<void>}
 */
function runZli(zliPath, args) {
  return new Promise((resolve, reject) => {
    // --verbose 1 limits stderr to errors, so a failure message can be shown as-is.
    const child = spawn(zliPath, [...args, '--verbose', '1'], {
      stdio: ['ignore', 'ignore', 'pipe'],
      windowsHide: true,
    });
    child.stderr.setEncoding('utf8');
    let stderr = '';
    let settled = false;
    /** @param {unknown} error */
    const fail = (error) => {
      if (!settled) {
        settled = true;
        reject(error);
      }
    };
    child.stderr.on('data', (/** @type {string} */ chunk) => {
      process.stderr.write(chunk);
      stderr += chunk;
    });
    child.once('error', (error) => {
      fail(error);
    });
    child.once('close', (code, signal) => {
      if (settled) {
        return;
      }
      if (code !== 0) {
        const detail = stderr.trim();
        fail(
          new Error(
            detail !== ''
              ? detail
              : `zli exited unexpectedly (${signal === null ? `code ${code}` : `signal ${signal}`})`,
          ),
        );
        return;
      }
      settled = true;
      resolve();
    });
  });
}

/**
 * Reads the single row zli benchmark --output-csv writes for one input file.
 *
 * @param {string} csvPath
 * @returns {Promise<ZliMeasurement>}
 */
async function readBenchmarkCsv(csvPath) {
  const [header = '', row = ''] = (await fs.readFile(csvPath, 'utf8')).split(/\r?\n/);
  const names = header.split(',').slice(0, CSV_COLUMNS.length);
  const values = row.split(',').slice(0, CSV_COLUMNS.length);
  if (
    names.join() !== CSV_COLUMNS.join() ||
    values.length !== CSV_COLUMNS.length ||
    values.some((value) => value.trim() === '' || !(Number(value) >= 0))
  ) {
    throw new Error('zli returned an unexpected result');
  }
  const field = Object.fromEntries(CSV_COLUMNS.map((name, index) => [name, Number(values[index])]));
  return {
    iterations: field.iters,
    srcSize: field.srcSize,
    compressedSize: field.compressedSize,
    compressMs: field.ctimeMs,
    decompressMs: field.dtimeMs,
  };
}

/**
 * @param {string} zliPath
 * @param {string} inputPath
 * @param {string[]} compressorArgs
 * @param {OpenZlBenchmarkJob} job
 * @param {string} csvPath
 */
async function benchmark(zliPath, inputPath, compressorArgs, job, csvPath) {
  await runZli(zliPath, [
    'benchmark',
    inputPath,
    ...compressorArgs,
    '--level',
    String(job.level),
    '--num-iters',
    String(job.iterations),
    '--output-csv',
    csvPath,
  ]);
  return readBenchmarkCsv(csvPath);
}

/**
 * @param {string} zliPath
 * @param {string} inputPath
 * @param {OpenZlBenchmarkJob} job
 * @param {number} candidates
 * @param {string} outputDirectory
 */
async function trainedCandidatePaths(zliPath, inputPath, job, candidates, outputDirectory) {
  await runZli(zliPath, [
    'train',
    inputPath,
    '--profile',
    job.profile,
    '--output',
    outputDirectory,
    '--pareto-frontier',
    '--max-num-candidates',
    String(candidates),
    '--threads',
    String(TRAINING_THREADS),
    '--use-all-samples',
  ]);
  // zli train --pareto-frontier writes candidate i to <output>/<i>.zc.
  const indices = (await fs.readdir(outputDirectory))
    .flatMap((name) => /^(\d+)\.zc$/.exec(name)?.[1] ?? [])
    .map(Number)
    .sort((a, b) => a - b);
  if (indices.length === 0) {
    throw new Error('OpenZL training produced no candidates');
  }
  return indices.map((index) => path.join(outputDirectory, `${index}.zc`));
}

/**
 * Removes a scratch directory. A failure is ignored: on Windows a zli that was
 * just killed can still hold a file open, and throwing here would replace the
 * run's own result or error with this one.
 *
 * @param {string} directory
 */
async function removeScratch(directory) {
  try {
    await fs.rm(directory, {recursive: true, force: true});
  } catch {
    // Left behind; see above.
  }
}

/**
 * Measures an OpenZL job with zli, yielding each measurement as it is made: the
 * one an untrained job produces, or one per trained candidate, best ratio
 * first. zli train benchmarks candidates at a fixed iteration count and has no
 * --level, so each candidate is benchmarked again with the job's settings. As
 * in the browser, a candidate that cannot be benchmarked is yielded as an error
 * and its siblings are still measured; anything else, training included,
 * throws. zli reads `inputPath` where it is; only its CSV and candidate files go
 * in a scratch directory.
 *
 * @param {string} zliPath
 * @param {string} inputPath
 * @param {OpenZlBenchmarkJob} job
 * @returns {AsyncGenerator<OpenZLOutcome>}
 */
async function* measureOpenZL(zliPath, inputPath, job) {
  // Absolute paths never start with '-', so zli cannot parse them as flags.
  const input = path.resolve(inputPath);
  const workDirectory = await fs.mkdtemp(path.join(os.tmpdir(), 'openzl-playground-'));
  try {
    if (job.training === null) {
      const csvPath = path.join(workDirectory, 'benchmark.csv');
      yield {candidate: null, measurement: await benchmark(zliPath, input, ['--profile', job.profile], job, csvPath)};
      return;
    }
    const candidatePaths = await trainedCandidatePaths(
      zliPath,
      input,
      job,
      job.training.candidates,
      path.join(workDirectory, 'candidates'),
    );
    for (const [index, candidatePath] of candidatePaths.entries()) {
      const candidate = {index: index + 1, total: candidatePaths.length};
      const csvPath = path.join(workDirectory, `candidate-${index}.csv`);
      let measurement;
      try {
        measurement = await benchmark(zliPath, input, ['--compressor', candidatePath], job, csvPath);
      } catch (error) {
        yield {candidate, error};
        continue;
      }
      yield {candidate, measurement};
    }
  } finally {
    await removeScratch(workDirectory);
  }
}

module.exports = {measureOpenZL};
