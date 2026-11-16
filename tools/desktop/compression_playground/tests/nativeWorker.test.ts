// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment node

import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import type {DesktopRunConfig, RunMessage} from '../../../web/compression_playground/src/benchmarkTypes.ts';
import {planJobs, run} from '../electron/native-worker.cjs';

let workDirectory: string | undefined;

afterEach(() => {
  if (workDirectory !== undefined) {
    rmSync(workDirectory, {recursive: true, force: true});
    workDirectory = undefined;
  }
});

function inputFile(): string {
  workDirectory = mkdtempSync(join(tmpdir(), 'native-worker-test-'));
  const inputPath = join(workDirectory, 'input.bin');
  writeFileSync(inputPath, 'OpenZL compression playground sample data\n'.repeat(100));
  return inputPath;
}

function config(compressors: DesktopRunConfig['compressors'], path = '/tmp/sample.bin'): DesktopRunConfig {
  return {input: {path}, iterations: 2, compressors};
}

describe('planJobs', () => {
  it('splits each row into one job per level, with the browser job ids', () => {
    const {jobs, rejected} = planJobs(
      config([
        {
          rowId: 1,
          compressor: 'OpenZL',
          levels: [6],
          profile: 'serial',
          training: {paretoFrontier: true, candidates: 10},
        },
        {rowId: 2, compressor: 'zstd', levels: [1, 3]},
      ]),
    );

    expect(rejected).toEqual([]);
    expect(jobs).toEqual([
      {
        id: '1-OpenZL-6',
        rowId: 1,
        level: 6,
        iterations: 2,
        compressor: 'OpenZL',
        profile: 'serial',
        training: {paretoFrontier: true, candidates: 10},
      },
      {id: '2-zstd-1', rowId: 2, level: 1, iterations: 2, compressor: 'zstd'},
      {id: '2-zstd-3', rowId: 2, level: 3, iterations: 2, compressor: 'zstd'},
    ]);
  });

  it('rejects a row with no levels rather than dropping it', () => {
    const {jobs, rejected} = planJobs(config([{rowId: 4, compressor: 'gzip', levels: []}]));

    expect(jobs).toEqual([]);
    expect(rejected).toEqual([{rowId: 4, message: 'No levels selected'}]);
  });
});

describe('run', () => {
  it("posts the browser worker's messages in its order, and goes on past a job that fails", async () => {
    const messages: RunMessage[] = [];

    await run(
      {
        zliPath: join(tmpdir(), 'no-such-zli'),
        config: config(
          [
            {rowId: 1, compressor: 'OpenZL', levels: [6], profile: 'serial', training: null},
            {rowId: 2, compressor: 'zstd', levels: [3]},
            {rowId: 3, compressor: 'gzip', levels: [6]},
          ],
          inputFile(),
        ),
      },
      (message) => messages.push(message),
    );

    expect(messages.map((message) => message.type)).toEqual([
      'loading',
      'started',
      'failure',
      'result',
      'result',
      'finished',
    ]);
    expect(messages[1]).toEqual({type: 'started', totalJobs: 3, rejected: []});
    const failure = messages[2];
    expect(failure.type === 'failure' && failure.failure).toMatchObject({
      job: {id: '1-OpenZL-6'},
      candidate: null,
      message: expect.stringContaining('ENOENT'),
    });
    for (const message of messages.slice(3, 5)) {
      expect(message.type === 'result' && message.result).toMatchObject({
        candidate: null,
        iterations: 2,
        srcSize: 4200,
        ratio: expect.any(Number),
        compressMBps: expect.any(Number),
        decompressMBps: expect.any(Number),
      });
    }
  });
});
