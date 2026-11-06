// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment node

import {chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import fsPromises from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, describe, expect, it, vi} from 'vitest';
import type {OpenZlBenchmarkJob, TrainingConfig} from '../../../web/compression_playground/src/benchmarkTypes.ts';
import {measureOpenZL} from '../electron/zli-client.cjs';

/**
 * Stands in for zli. Training writes candidates 0, 1, 2 and 10 beside a
 * benchmark.csv, as zli train --pareto-frontier does, and each candidate's
 * contents become its compressedSize. A benchmark writes zli's CSV header and
 * one row whose quoted path contains a comma, and fails if its --compressor
 * path does not exist or is the one FAKE_ZLI_FAIL names. Each call's input path
 * is appended to FAKE_ZLI_INPUTS when it is set.
 */
const FAKE_ZLI = `
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
  const output = flag('--output');
  fs.mkdirSync(output, {recursive: true});
  for (const index of [10, 2, 0, 1]) {
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

let workDirectory: string | undefined;

afterEach(() => {
  delete process.env.FAKE_ZLI_EOL;
  delete process.env.FAKE_ZLI_PATH_COLUMN;
  delete process.env.FAKE_ZLI_FAIL;
  delete process.env.FAKE_ZLI_INPUTS;
  if (workDirectory !== undefined) {
    rmSync(workDirectory, {recursive: true, force: true});
    workDirectory = undefined;
  }
});

function scratch(): string {
  workDirectory ??= mkdtempSync(join(tmpdir(), 'zli-client-test-'));
  return workDirectory;
}

function fakeZli(): string {
  const zliPath = join(scratch(), 'zli');
  writeFileSync(zliPath, `#!${process.execPath}\n${FAKE_ZLI}`);
  chmodSync(zliPath, 0o755);
  return zliPath;
}

function inputFile(): string {
  const inputPath = join(scratch(), 'input.bin');
  writeFileSync(inputPath, new Uint8Array([1, 2, 3]));
  return inputPath;
}

function openZLJob(training: TrainingConfig | null = null): OpenZlBenchmarkJob {
  return {id: '1-OpenZL-6', rowId: 1, level: 6, iterations: 4, compressor: 'OpenZL', profile: 'serial', training};
}

async function collect<T>(outcomes: AsyncIterable<T>): Promise<T[]> {
  const collected = [];
  for await (const outcome of outcomes) {
    collected.push(outcome);
  }
  return collected;
}

const MEASUREMENT = {iterations: 4, srcSize: 3, compressMs: 1, decompressMs: 1};

describe('measureOpenZL', () => {
  it('reads a CSV written with CRLF line endings and no path column', async () => {
    // On Windows zli writes CRLF, which would otherwise leave a CR on `iters`.
    process.env.FAKE_ZLI_EOL = '\r\n';
    process.env.FAKE_ZLI_PATH_COLUMN = '0';

    const outcomes = await collect(measureOpenZL(fakeZli(), inputFile(), openZLJob()));

    expect(outcomes).toEqual([{candidate: null, measurement: {...MEASUREMENT, compressedSize: 2}}]);
  });

  it('has zli read the input where it is, rather than a copy', async () => {
    const inputs = join(scratch(), 'inputs.txt');
    process.env.FAKE_ZLI_INPUTS = inputs;
    const inputPath = inputFile();

    await collect(measureOpenZL(fakeZli(), inputPath, openZLJob({paretoFrontier: true, candidates: 10})));

    // One training run, then four candidate benchmarks, all of the same file.
    expect(readFileSync(inputs, 'utf8').trim().split('\n')).toEqual(Array(5).fill(inputPath));
  });

  it('reports why zli failed even when its scratch directory cannot be removed', async () => {
    // As on Windows, where a zli that was just killed can still hold a file open.
    const rm = vi.spyOn(fsPromises, 'rm').mockRejectedValue(new Error('EBUSY: resource busy or locked'));
    try {
      await expect(collect(measureOpenZL(join(tmpdir(), 'no-such-zli'), inputFile(), openZLJob()))).rejects.toThrow(
        /ENOENT/,
      );
      expect(rm).toHaveBeenCalled();
    } finally {
      rm.mockRestore();
      for (const [directory] of rm.mock.calls) {
        rmSync(directory, {recursive: true, force: true});
      }
    }
  });

  it('benchmarks every trained candidate in index order, counted from 1', async () => {
    const outcomes = await collect(
      measureOpenZL(fakeZli(), inputFile(), openZLJob({paretoFrontier: true, candidates: 10})),
    );

    expect(outcomes).toEqual(
      [1, 2, 3, 11].map((compressedSize, index) => ({
        candidate: {index: index + 1, total: 4},
        measurement: {...MEASUREMENT, compressedSize},
      })),
    );
  });

  it('still benchmarks the other candidates when one cannot be', async () => {
    process.env.FAKE_ZLI_FAIL = '1.zc';

    const outcomes = await collect(
      measureOpenZL(fakeZli(), inputFile(), openZLJob({paretoFrontier: true, candidates: 10})),
    );

    expect(outcomes.map((outcome) => outcome.candidate?.index)).toEqual([1, 2, 3, 4]);
    expect(outcomes[1]).toEqual({candidate: {index: 2, total: 4}, error: new Error('Cannot load compressor: 1.zc')});
    expect(outcomes.filter((outcome) => 'measurement' in outcome)).toHaveLength(3);
  });
});
