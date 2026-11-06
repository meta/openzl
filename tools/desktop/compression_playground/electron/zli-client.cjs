// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/** @import {Candidate, OpenZlBenchmarkJob} from '../../../web/compression_playground/src/benchmarkTypes.ts' */

const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {z} = require('zod');

const TRAINING_THREADS = Math.min(4, Math.max(1, os.availableParallelism()));
const csvCount = z.string().trim().regex(/^\d+$/, 'expected a whole number').transform(Number);
const csvNumber = z.string().trim().min(1, 'expected a number').transform(Number).pipe(z.number().finite().min(0));
// The columns of a zli benchmark --output-csv row this client reads. The path
// column comes last and is not read, so a quoted path cannot shift the numbers.
const benchmarkRowSchema = z.object({
  srcSize: csvCount,
  compressedSize: csvCount,
  ctimeMs: csvNumber,
  dtimeMs: csvNumber,
  iters: csvCount.pipe(z.int().min(1)),
});
const KILL_GRACE_MS = 2000;

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
 * A zli process starting or ending, so the main process can kill any still
 * running if the worker dies.
 *
 * @typedef {object} ZliChildEvent
 * @property {number} pid
 * @property {'spawned' | 'exited'} state
 */

/**
 * @typedef {object} ZliContext
 * @property {AbortSignal} [signal] Aborting it stops the running zli and rejects with its reason.
 * @property {(event: ZliChildEvent) => void} [onChild]
 */

/**
 * SIGTERM first, then SIGKILL if zli has not exited. Windows terminates on either.
 *
 * @param {import('node:child_process').ChildProcess} child
 */
function terminate(child) {
  child.kill('SIGTERM');
  const escalate = setTimeout(() => child.kill('SIGKILL'), KILL_GRACE_MS);
  // 'exit' rather than 'close', which waits for stdio and need not follow.
  child.once('exit', () => clearTimeout(escalate));
}

// At --verbose 3, zli prints a progress bar on stderr for each step of
// training, one line per redraw on a pipe:
// `[=====---------------------------------------------] Training ACE graph 1 / 2: ACE progress`.
const PROGRESS_LINE = /^\[([=-]+)\] /;
const GRAPH_COUNTER = /\bgraph (\d+) \/ (\d+)\b/;

/**
 * How far through training a zli progress line says it is, or undefined for
 * any other line. Each ACE graph gets a bar of its own that starts again from
 * empty, so a bar's fill is placed within its graph's share of the whole.
 *
 * @param {string} line
 * @returns {number | undefined}
 */
function progressFraction(line) {
  const bar = PROGRESS_LINE.exec(line)?.[1];
  if (bar === undefined) {
    return undefined;
  }
  const fill = bar.replaceAll('-', '').length / bar.length;
  const [, graph = 0, graphs = 0] = GRAPH_COUNTER.exec(line)?.map(Number) ?? [];
  return graph >= 1 && graphs >= graph ? (graph - 1 + fill) / graphs : fill;
}

/**
 * What zli printed about why it failed. Its progress and other INFO lines come
 * first; a fatal error comes last, usually as a heading and its tab-indented
 * detail (`I/O Exception:` then `\tFailed to open input file ...`), otherwise
 * as one line. Empty when zli was killed by a signal (a crash, or out of
 * memory) and printed no heading: whatever INFO line came last says nothing
 * about that, so the signal is reported instead.
 *
 * @param {string} stderr
 * @param {boolean} killed
 */
function failureFromVerboseStderr(stderr, killed) {
  const lines = stderr
    .trim()
    .split(/\r?\n/)
    .filter((line) => progressFraction(line) === undefined);
  const heading = lines.findLastIndex((line, index) => !line.startsWith('\t') && lines[index + 1]?.startsWith('\t'));
  if (heading !== -1) {
    return lines.slice(heading).join('\n');
  }
  return killed ? '' : (lines.at(-1) ?? '');
}

/**
 * Runs zli, which reports failures on stderr. With `onProgress`, zli runs at
 * --verbose 3: each progress line is reported as a fraction, and only the
 * fatal error is kept as the failure message.
 *
 * @param {string} zliPath
 * @param {string[]} args
 * @param {ZliContext & {onProgress?: (fraction: number) => void}} context
 * @returns {Promise<void>}
 */
function runZli(zliPath, args, {signal, onChild, onProgress}) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    // --verbose 1 limits stderr to errors, so a failure message can be shown
    // as-is; 3 adds the progress lines `onProgress` reads.
    const child = spawn(zliPath, [...args, '--verbose', onProgress === undefined ? '1' : '3'], {
      stdio: ['ignore', 'ignore', 'pipe'],
      windowsHide: true,
    });
    child.stderr.setEncoding('utf8');
    let stderr = '';
    /**
     * Why zli was stopped; reported once it has actually exited.
     *
     * @type {unknown}
     */
    let stopReason;
    let settled = false;
    /** @param {unknown} error */
    const fail = (error) => {
      if (!settled) {
        settled = true;
        reject(error);
      }
    };
    const onAbort = () => {
      if (stopReason === undefined) {
        stopReason = signal?.reason;
        terminate(child);
      }
    };
    signal?.addEventListener('abort', onAbort, {once: true});
    const cleanup = () => {
      signal?.removeEventListener('abort', onAbort);
    };
    if (child.pid !== undefined) {
      onChild?.({pid: child.pid, state: 'spawned'});
    }

    // The end of the last chunk, until the rest of its line arrives.
    let partialLine = '';
    child.stderr.on('data', (/** @type {string} */ chunk) => {
      stderr += chunk;
      if (onProgress === undefined) {
        process.stderr.write(chunk);
        return;
      }
      const lines = (partialLine + chunk).split(/\r?\n/);
      partialLine = lines.pop() ?? '';
      for (const line of lines) {
        const fraction = progressFraction(line);
        if (fraction === undefined) {
          process.stderr.write(`${line}\n`);
        } else {
          onProgress(fraction);
        }
      }
    });
    child.once('error', (error) => {
      cleanup();
      fail(error);
    });
    child.once('close', (code, exitSignal) => {
      cleanup();
      // A last line with no newline after it, which the 'data' handler held back.
      if (partialLine !== '' && progressFraction(partialLine) === undefined) {
        process.stderr.write(`${partialLine}\n`);
      }
      if (child.pid !== undefined) {
        onChild?.({pid: child.pid, state: 'exited'});
      }
      if (settled) {
        return;
      }
      if (stopReason !== undefined) {
        fail(stopReason);
        return;
      }
      if (code !== 0) {
        const detail = onProgress === undefined ? stderr.trim() : failureFromVerboseStderr(stderr, exitSignal !== null);
        fail(
          new Error(
            detail !== ''
              ? detail
              : `zli exited unexpectedly (${exitSignal === null ? `code ${code}` : `signal ${exitSignal}`})`,
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
 * A value that is not the number it should be fails naming its column.
 *
 * @param {string} csvPath
 * @returns {Promise<ZliMeasurement>}
 */
async function readBenchmarkCsv(csvPath) {
  const [header = '', row = ''] = (await fs.readFile(csvPath, 'utf8')).split(/\r?\n/);
  const values = row.split(',');
  const names = header.split(',').map((name) => name.trim());
  // Read by name, so a repeated column would silently replace the first.
  const repeated = names.find((name, index) => names.indexOf(name) !== index);
  if (repeated !== undefined) {
    throw new Error(`zli returned an unexpected result (column ${repeated} appears twice)`);
  }
  const parsed = benchmarkRowSchema.safeParse(Object.fromEntries(names.map((name, index) => [name, values[index]])));
  if (!parsed.success) {
    const [issue] = parsed.error.issues;
    const field = issue !== undefined && issue.path.length > 0 ? `${issue.path.join('.')}: ` : '';
    throw new Error(`zli returned an unexpected result (${field}${issue?.message})`);
  }
  const field = parsed.data;
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
 * @param {ZliContext} context
 */
async function benchmark(zliPath, inputPath, compressorArgs, job, csvPath, context) {
  await runZli(
    zliPath,
    [
      'benchmark',
      inputPath,
      ...compressorArgs,
      '--level',
      String(job.level),
      '--num-iters',
      String(job.iterations),
      '--output-csv',
      csvPath,
    ],
    context,
  );
  return readBenchmarkCsv(csvPath);
}

/**
 * @param {string} zliPath
 * @param {string} inputPath
 * @param {OpenZlBenchmarkJob} job
 * @param {number} candidates
 * @param {string} outputDirectory
 * @param {ZliContext & {onProgress?: (fraction: number) => void}} context
 */
async function trainedCandidatePaths(zliPath, inputPath, job, candidates, outputDirectory, context) {
  await runZli(
    zliPath,
    [
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
    ],
    context,
  );
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
 * throws, and so does cancelling, which stops the running zli. While training,
 * `onProgress` hears each fraction zli's progress bar shows. zli reads
 * `inputPath` where it is; only its CSV and candidate files go in a scratch
 * directory under `tempRoot`, which defaults to the OS temp directory.
 *
 * @param {string} zliPath
 * @param {string} inputPath
 * @param {OpenZlBenchmarkJob} job
 * @param {ZliContext & {tempRoot?: string; onProgress?: (fraction: number) => void}} [options]
 * @returns {AsyncGenerator<OpenZLOutcome>}
 */
async function* measureOpenZL(zliPath, inputPath, job, {signal, onChild, onProgress, tempRoot = os.tmpdir()} = {}) {
  const context = {signal, onChild};
  // Absolute paths never start with '-', so zli cannot parse them as flags.
  const input = path.resolve(inputPath);
  const workDirectory = await fs.mkdtemp(path.join(tempRoot, 'openzl-playground-'));
  try {
    if (job.training === null) {
      const csvPath = path.join(workDirectory, 'benchmark.csv');
      const args = ['--profile', job.profile];
      yield {candidate: null, measurement: await benchmark(zliPath, input, args, job, csvPath, context)};
      return;
    }
    const candidatePaths = await trainedCandidatePaths(
      zliPath,
      input,
      job,
      job.training.candidates,
      path.join(workDirectory, 'candidates'),
      {...context, onProgress},
    );
    for (const [index, candidatePath] of candidatePaths.entries()) {
      const candidate = {index: index + 1, total: candidatePaths.length};
      const csvPath = path.join(workDirectory, `candidate-${index}.csv`);
      let measurement;
      try {
        measurement = await benchmark(zliPath, input, ['--compressor', candidatePath], job, csvPath, context);
      } catch (error) {
        // A cancel is not this candidate failing: it ends the job.
        if (signal?.aborted) {
          throw error;
        }
        yield {candidate, error};
        continue;
      }
      yield {candidate, measurement};
    }
  } finally {
    await removeScratch(workDirectory);
  }
}

module.exports = {measureOpenZL, progressFraction};
