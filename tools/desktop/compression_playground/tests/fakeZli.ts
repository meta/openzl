// Copyright (c) Meta Platforms, Inc. and affiliates.

import {chmodSync, mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import type {OpenZlBenchmarkJob, TrainingConfig} from '../../../web/compression_playground/src/benchmarkTypes.ts';

/**
 * Script for a zli stand-in. Training writes the given candidates beside a
 * benchmark.csv, as zli train --pareto-frontier does, and each candidate's
 * contents become its compressedSize. A benchmark writes zli's CSV header and
 * one row whose quoted path contains a comma, and fails if its --compressor
 * path does not exist or is the one FAKE_ZLI_FAIL names. Each call's input
 * path is appended to FAKE_ZLI_INPUTS when it is set.
 */
function fakeZliScript(candidateIndices: number[]): string {
  return `
const fs = require('node:fs');
const path = require('node:path');
const args = process.argv.slice(2);
// A test sets this to stand in for a zli that writes CRLF, as on Windows.
const eol = process.env.FAKE_ZLI_EOL ?? '\\n';
// Without the trailing path column, a stray CR would land on the last name read.
const pathColumn = process.env.FAKE_ZLI_PATH_COLUMN !== '0';
const flag = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
if (process.env.FAKE_ZLI_INPUTS !== undefined) {
  fs.appendFileSync(process.env.FAKE_ZLI_INPUTS, args[1] + '\\n');
}
if (args[0] === 'train') {
  // As zli does at --verbose 3 on a pipe: INFO lines and one line per redraw.
  if (flag('--verbose') === '3') {
    process.stderr.write('Using all provided training samples, total size 3\\n');
    for (const bar of ['[----]', '[==--]', '[====]']) {
      process.stderr.write(bar + ' Training ACE graph 1 / 1: ACE progress\\n');
    }
  }
  const output = flag('--output');
  fs.mkdirSync(output, {recursive: true});
  for (const index of ${JSON.stringify(candidateIndices)}) {
    fs.writeFileSync(output + '/' + index + '.zc', String(index + 1));
  }
  fs.writeFileSync(output + '/benchmark.csv', '');
} else {
  const compressor = flag('--compressor');
  if (compressor !== undefined && (!fs.existsSync(compressor) || path.basename(compressor) === process.env.FAKE_ZLI_FAIL)) {
    process.stderr.write('Cannot load compressor: ' + path.basename(compressor) + '\\n');
    process.exit(1);
  }
  const compressedSize = compressor === undefined ? 2 : fs.readFileSync(compressor, 'utf8');
  fs.writeFileSync(
    flag('--output-csv'),
    'srcSize,compressedSize,compressionRatio,ctimeMs,dtimeMs,iters' + (pathColumn ? ',path' : '') + eol +
      '3,' + compressedSize + ',1.5,1,1,' + flag('--num-iters') + (pathColumn ? ',"/tmp/a,b"' : '') + eol,
  );
}
`;
}

export const FAKE_ZLI = fakeZliScript([0]);

/** Like FAKE_ZLI, but training writes candidates 10, 2, 0 and 1, in that order. */
export const UNORDERED_CANDIDATES_ZLI = fakeZliScript([10, 2, 0, 1]);

/**
 * Stands in for a zli that ignores SIGTERM and never finishes. It writes a
 * `ready` file beside itself once SIGTERM is ignored, so a test can wait for
 * that before signalling it.
 */
export const HANGING_ZLI = `
process.on('SIGTERM', () => {});
require('node:fs').writeFileSync(require('node:path').join(__dirname, 'ready'), '');
setInterval(() => {}, 1000);
`;

/** An OpenZL job at level 6 and 4 iterations, untrained unless `training` says otherwise. */
export function openZLJob(training: TrainingConfig | null = null): OpenZlBenchmarkJob {
  return {id: '1-OpenZL-6', rowId: 1, level: 6, iterations: 4, compressor: 'OpenZL', profile: 'serial', training};
}

const directories: string[] = [];

function scratchDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), 'zli-client-test-'));
  directories.push(directory);
  return directory;
}

/** Removes every directory fakeZli and inputFile created; call from afterEach. */
export function removeFakeZlis(): void {
  for (const directory of directories.splice(0)) {
    rmSync(directory, {recursive: true, force: true});
  }
}

/** Writes `script` as an executable zli stand-in, in a directory of its own, and returns its path. */
export function fakeZli(script = FAKE_ZLI): string {
  const zliPath = join(scratchDirectory(), 'zli');
  writeFileSync(zliPath, `#!${process.execPath}\n${script}`);
  chmodSync(zliPath, 0o755);
  return zliPath;
}

/** Writes `contents` to a file and returns its path. */
export function inputFile(contents: string | Uint8Array = new Uint8Array([1, 2, 3])): string {
  const inputPath = join(scratchDirectory(), 'input.bin');
  writeFileSync(inputPath, contents);
  return inputPath;
}
