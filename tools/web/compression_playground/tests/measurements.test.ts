// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it} from 'vitest';
import {
  barFraction,
  formatBytes,
  formatRatio,
  frontierTag,
  peaksOf,
  rowDetail,
  runSummary,
  sortMeasurements,
} from '../src/measurements.ts';
import type {BenchmarkJob, JobResult} from '../src/benchmarkTypes.ts';

function result(
  job: Partial<BenchmarkJob> & Pick<BenchmarkJob, 'compressor'>,
  metrics: Partial<JobResult> = {},
): JobResult {
  return {
    iterations: 1,
    srcSize: 1000,
    compressedSize: 100,
    ratio: 10,
    compressMs: 1,
    decompressMs: 1,
    compressMBps: 100,
    decompressMBps: 200,
    candidate: null,
    ...metrics,
    job: {
      id: 'j',
      rowId: 1,
      level: 1,
      iterations: 1,
      profile: 'serial',
      profileArg: '',
      training: null,
      ...job,
    } as BenchmarkJob,
  };
}

describe('frontierTag', () => {
  it('names only the ends and the middle of a frontier', () => {
    expect([1, 2, 3, 4, 5].map((i) => frontierTag(i, 5))).toEqual(['max ratio', null, 'balanced', null, 'max speed']);
    expect([1, 2, 3, 4, 5, 6].map((i) => frontierTag(i, 6))).toEqual([
      'max ratio',
      null,
      'balanced',
      null,
      null,
      'max speed',
    ]);
  });

  it('leaves a single measurement unlabelled and a pair without a middle', () => {
    // One point has nothing to be the extreme of, and two are both extremes.
    expect(frontierTag(1, 1)).toBeNull();
    expect([1, 2].map((i) => frontierTag(i, 2))).toEqual(['max ratio', 'max speed']);
  });
});

describe('sortMeasurements', () => {
  const mixed = [
    result({compressor: 'zstd', rowId: 2, level: 1}, {ratio: 3}),
    result({compressor: 'gzip', rowId: 3, level: 6}, {ratio: 5}),
    result({compressor: 'zstd', rowId: 2, level: 9}, {ratio: 9}),
  ];

  it('orders every row together rather than within a codec', () => {
    const ratios = (sorted: ReturnType<typeof sortMeasurements>) => sorted.map((row) => row.ratio);
    expect(ratios(sortMeasurements(mixed, {key: 'ratio', direction: 'desc'}))).toEqual([9, 5, 3]);
    expect(ratios(sortMeasurements(mixed, {key: 'ratio', direction: 'asc'}))).toEqual([3, 5, 9]);
  });

  it('keeps the order the run produced when sorted by codec', () => {
    expect(sortMeasurements(mixed, {key: 'codec', direction: 'asc'})).toBe(mixed);
  });

  it('reverses only the configuration order when the codec sort is flipped', () => {
    // A row's own measurements keep their order: zstd stays level 1 then 9.
    const levels = sortMeasurements(mixed, {key: 'codec', direction: 'desc'}).map(
      (row) => `${row.job.compressor} ${String(row.job.level)}`,
    );
    expect(levels).toEqual(['gzip 6', 'zstd 1', 'zstd 9']);
  });
});

describe('formatting', () => {
  it('scales bytes the way the design writes them', () => {
    expect(formatBytes(361)).toBe('361 B');
    expect(formatBytes(1900)).toBe('1.9 KB');
    expect(formatBytes(1_000_000)).toBe('1.0 MB');
  });

  it('moves to the next unit when rounding would print a thousand', () => {
    expect(formatBytes(999_960)).toBe('1.0 MB');
    expect(formatBytes(999_949)).toBe('999.9 KB');
  });

  it('keeps two decimals on a ratio however large it is', () => {
    expect(formatRatio(15235.46)).toBe('15235.46×');
    expect(formatRatio(3.204)).toBe('3.20×');
  });
});

describe('rowDetail', () => {
  it('says on every row what produced it, trained candidates included', () => {
    // Sorting can put any two rows side by side, so none can lean on another.
    expect(rowDetail(result({compressor: 'OpenZL', profile: 'le-u32'}, {candidate: {index: 1, total: 3}}))).toBe(
      'trained · le-u32',
    );
    expect(rowDetail(result({compressor: 'OpenZL', profile: 'le-u32'}))).toBe('le-u32');
    expect(rowDetail(result({compressor: 'zstd', level: 9}))).toBe('Level 9');
  });
});

describe('runSummary', () => {
  const job = (rowId: number, level: number) => result({compressor: 'zstd', rowId, level}, {srcSize: 5_000_000}).job;

  it('has nothing to say until the run finishes', () => {
    expect(runSummary({status: 'idle'})).toBeNull();
    expect(
      runSummary({status: 'running', completedJobs: 1, totalJobs: 2, results: [], failures: [], step: null}),
    ).toBeNull();
  });

  it('counts compressor rows, not the jobs they expanded into', () => {
    const summary = runSummary({
      status: 'completed',
      results: [
        result({compressor: 'zstd', rowId: 2, level: 1}, {srcSize: 5_000_000}),
        result({compressor: 'zstd', rowId: 2, level: 9}, {srcSize: 5_000_000}),
        result({compressor: 'gzip', rowId: 3, level: 6}, {srcSize: 5_000_000}),
      ],
      failures: [],
    });
    expect(summary).toEqual({srcSize: 5_000_000, succeeded: 2, failed: 0});
  });

  it('keeps a row that lost one level out of the failed count', () => {
    // The failure list under the table names it; the row still produced.
    const summary = runSummary({
      status: 'completed',
      results: [result({compressor: 'zstd', rowId: 2, level: 1}, {srcSize: 100})],
      failures: [{job: job(2, 19), candidate: null, message: 'out of memory'}],
    });
    expect(summary).toMatchObject({succeeded: 1, failed: 0});
  });

  it('counts a row that produced nothing as failed, once', () => {
    const summary = runSummary({
      status: 'completed',
      results: [result({compressor: 'gzip', rowId: 3, level: 6}, {srcSize: 100})],
      failures: [
        {job: job(2, 1), candidate: null, message: 'boom'},
        {job: job(2, 9), candidate: null, message: 'boom'},
      ],
    });
    expect(summary).toMatchObject({succeeded: 1, failed: 1});
  });

  it('has no size to report when nothing was measured', () => {
    const summary = runSummary({
      status: 'completed',
      results: [],
      failures: [{job: job(2, 1), candidate: null, message: 'boom'}],
    });
    expect(summary).toEqual({srcSize: null, succeeded: 0, failed: 1});
  });
});

describe('barFraction', () => {
  it('measures against the whole run, so codecs stay comparable', () => {
    const peaks = peaksOf([result({compressor: 'zstd'}, {ratio: 10}), result({compressor: 'gzip'}, {ratio: 5})]);
    expect(peaks.ratio).toBe(10);
    expect(barFraction(10, peaks.ratio)).toBe(1);
    expect(barFraction(5, peaks.ratio)).toBe(0.5);
  });

  it('keeps a sliver visible and never overflows the track', () => {
    // A bar of zero width reads as a missing value rather than a small one.
    expect(barFraction(0.0001, 1000)).toBe(0.02);
    expect(barFraction(5, 1)).toBe(1);
    expect(barFraction(1, 0)).toBe(0);
  });
});
