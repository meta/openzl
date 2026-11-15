// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it} from 'vitest';
import {buildJobs, isRunInProgress, toCompressorConfig, type RunConfig, type RunState} from '../src/benchmarkTypes.ts';
import {createCompressorRow} from '../src/compressors.ts';

const FILE = new File(['x'], 'sample.bin');
const runWith = (compressors: RunConfig['compressors'], iterations = 5): RunConfig => ({
  input: FILE,
  compressors,
  iterations,
});

describe('toCompressorConfig', () => {
  it('removes UI state from OpenZL rows and carries the training settings', () => {
    const row = createCompressorRow(7, 'OpenZL');
    row.optionsOpen = false;
    row.trainRequested = true;
    // Not the default 6, so a `toCompressorConfig` that quietly emitted the
    // default would fail here, and inside the 6-25 the binding accepts.
    row.candidates = 7;

    expect(toCompressorConfig(row)).toEqual({
      rowId: 7,
      compressor: 'OpenZL',
      levels: [6],
      profile: 'serial',
      training: {paretoFrontier: true, candidates: 7, maxTimeSecs: 30},
    });
  });

  it('preserves training requests and validates capability when building jobs', () => {
    const row = createCompressorRow(7, 'OpenZL');
    row.profile = 'fixed_profile';
    row.trainRequested = true;

    const config = toCompressorConfig(row);
    expect(config).toMatchObject({training: {paretoFrontier: true, candidates: 6}});
    expect(buildJobs(runWith([config])).jobs).toEqual([
      expect.objectContaining({profile: 'fixed_profile', training: null}),
    ]);
  });

  it('drops the candidate count when the row is not training', () => {
    const row = createCompressorRow(7, 'OpenZL');
    row.trainRequested = false;
    // Not the default 6, so a `toCompressorConfig` that quietly emitted the
    // default would fail here, and inside the 6-25 the binding accepts.
    row.candidates = 7;

    expect(toCompressorConfig(row)).toMatchObject({training: null});
  });
});

describe('isRunInProgress', () => {
  const cases: readonly [RunState, boolean][] = [
    [{status: 'idle'}, false],
    [{status: 'loading'}, true],
    [{status: 'running', completedJobs: 0, totalJobs: 1, results: [], failures: [], step: null}, true],
    [{status: 'completed', results: [], failures: []}, false],
    [{status: 'error', message: 'failed', results: [], failures: []}, false],
  ];

  it.each(cases)('classifies the %s state', (runState, expected) => {
    expect(isRunInProgress(runState)).toBe(expected);
  });
});

describe('buildJobs', () => {
  it('emits one job per level so totalJobs counts measurements, not rows', () => {
    const {jobs} = buildJobs(
      runWith([
        {rowId: 1, compressor: 'zstd', levels: [1, 5, 19]},
        {rowId: 2, compressor: 'gzip', levels: [1, 9]},
      ]),
    );

    expect(jobs).toHaveLength(5);
    expect(jobs.map((job) => job.level)).toEqual([1, 5, 19, 1, 9]);
    expect(jobs.every((job) => job.iterations === 5)).toBe(true);
  });

  it('gives every job a distinct id once a row expands', () => {
    const {jobs} = buildJobs(runWith([{rowId: 1, compressor: 'zstd', levels: [1, 5, 19]}]));
    expect(new Set(jobs.map((job) => job.id)).size).toBe(3);
  });

  it('carries profile names unchanged into jobs', () => {
    const {
      jobs: [job],
    } = buildJobs(runWith([{rowId: 1, compressor: 'OpenZL', levels: [6], profile: 'le-u32', training: null}]));
    expect(job).toMatchObject({compressor: 'OpenZL', profile: 'le-u32', level: 6});
  });

  it('reports OpenZL profiles the browser cannot build instead of skipping them', () => {
    const {jobs, rejected} = buildJobs(
      runWith([
        {rowId: 1, compressor: 'OpenZL', levels: [6], profile: 'parquet', training: null},
        {rowId: 2, compressor: 'gzip', levels: [6]},
      ]),
    );

    expect(jobs.map((job) => job.rowId)).toEqual([2]);
    expect(rejected).toEqual([{rowId: 1, message: 'The parquet profile has no browser build'}]);
  });

  it('reports a row with no levels rather than letting it vanish from the run', () => {
    // The picker allows an empty row, so this arrives from the UI: without it
    // the row would contribute no job and nothing would say why.
    const {jobs, rejected} = buildJobs(
      runWith([
        {rowId: 1, compressor: 'zstd', levels: []},
        {rowId: 2, compressor: 'gzip', levels: [6]},
      ]),
    );

    expect(jobs.map((job) => job.rowId)).toEqual([2]);
    expect(rejected).toEqual([{rowId: 1, message: 'No levels selected'}]);
  });

  it('rejects a row once however many levels it would have expanded into', () => {
    const {jobs, rejected} = buildJobs(
      runWith([{rowId: 1, compressor: 'OpenZL', levels: [1, 6, 9], profile: 'csv', training: null}]),
    );

    expect(jobs).toEqual([]);
    expect(rejected).toHaveLength(1);
  });

  it('carries the training settings onto every job the row expands into', () => {
    const {jobs} = buildJobs(
      runWith([
        {
          rowId: 1,
          compressor: 'OpenZL',
          levels: [1, 6],
          profile: 'serial',
          training: {paretoFrontier: true, candidates: 12},
        },
      ]),
    );

    expect(jobs).toHaveLength(2);
    expect(jobs.every((job) => job.compressor === 'OpenZL' && job.training?.candidates === 12)).toBe(true);
  });
});
