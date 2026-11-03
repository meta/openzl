// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it} from 'vitest';
import {buildSeries, formatRatioTick, formatSpeedTick, logDomain, logTicks} from '../src/charts.ts';
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
      id: `${job.rowId ?? 1}-${job.compressor}-${job.level ?? 1}`,
      rowId: 1,
      level: 1,
      iterations: 1,
      profile: 'serial',
      training: null,
      ...job,
    } as BenchmarkJob,
  };
}

describe('buildSeries', () => {
  it.each(['compressMBps', 'decompressMBps'] as const)('leaves untrained OpenZL jobs as points on %s', (axis) => {
    const series = buildSeries(
      [
        result({id: 'untrained-1', compressor: 'OpenZL', rowId: 1}),
        result({id: 'untrained-4', compressor: 'OpenZL', rowId: 4}),
      ],
      axis,
    );

    expect(series.size).toBe(2);
    for (const one of series.values()) {
      expect(one.compressor).toBe('OpenZL');
      expect(one.points).toHaveLength(1);
    }
  });

  it('keeps trained OpenZL frontiers separate from other jobs', () => {
    const first = {
      id: 'trained-1',
      rowId: 1,
      compressor: 'OpenZL',
      training: {paretoFrontier: true, candidates: 6},
    } as const;
    const second = {...first, id: 'trained-4', rowId: 4};
    const series = buildSeries(
      [
        result(first, {candidate: {index: 1, total: 2}, compressMBps: 90}),
        result({id: 'untrained-2', rowId: 2, compressor: 'OpenZL'}, {compressMBps: 200}),
        result(second, {candidate: {index: 1, total: 2}, compressMBps: 150}),
        result(first, {candidate: {index: 2, total: 2}, compressMBps: 500}),
        result(second, {candidate: {index: 2, total: 2}, compressMBps: 400}),
      ],
      'compressMBps',
    );

    expect([...series.keys()]).toEqual(['trained-1', 'untrained-2', 'trained-4']);
    const frontiers = [...series.values()];
    expect(frontiers.map((one) => one.points.map((point) => point.x))).toEqual([[90, 500], [200], [150, 400]]);
  });

  it.each(['zstd', 'gzip'] as const)('joins %s levels across compressor rows', (compressor) => {
    const [series] = buildSeries(
      [result({compressor, rowId: 1, level: 1}), result({compressor, rowId: 4, level: 9})],
      'compressMBps',
    ).values();
    expect(series.points).toHaveLength(2);
  });

  it('orders points by speed so the line does not double back', () => {
    const [series] = buildSeries(
      [
        result({compressor: 'zstd', level: 1}, {compressMBps: 500, ratio: 3}),
        result({compressor: 'zstd', level: 9}, {compressMBps: 90, ratio: 7}),
      ],
      'compressMBps',
    ).values();
    expect(series.points.map((point) => point.x)).toEqual([90, 500]);
  });

  it.each(['compressMBps', 'decompressMBps'] as const)('orders OpenZL candidates by speed on %s', (axis) => {
    const job = {compressor: 'OpenZL', training: {paretoFrontier: true, candidates: 3}} as const;
    const [series] = buildSeries(
      [
        result(job, {candidate: {index: 1, total: 3}, ratio: 7, compressMBps: 200, decompressMBps: 300}),
        result(job, {candidate: {index: 2, total: 3}, ratio: 5, compressMBps: 300, decompressMBps: 100}),
        result(job, {candidate: {index: 3, total: 3}, ratio: 3, compressMBps: 100, decompressMBps: 200}),
      ],
      axis,
    ).values();
    const speeds = series.points.map((point) => point.x);
    expect(speeds).toEqual([...speeds].sort((a, b) => a - b));
  });

  it('plots the axis it was asked for', () => {
    const [byDecompress] = buildSeries(
      [result({compressor: 'gzip'}, {decompressMBps: 400})],
      'decompressMBps',
    ).values();
    expect(byDecompress.points[0].x).toBe(400);
  });

  it('drops what a log axis cannot place, as the domain does', () => {
    // An empty input measures as 0 MB/s at a ratio of 0. `logDomain` already
    // leaves it out of the range, so a point at log(0) would have nowhere
    // on the frame to go.
    const [series] = buildSeries(
      [
        result({compressor: 'zstd', level: 1}, {compressMBps: 0, ratio: 0}),
        result({compressor: 'zstd', level: 9}, {compressMBps: 90, ratio: 7}),
      ],
      'compressMBps',
    ).values();
    expect(series.points.map((point) => point.x)).toEqual([90]);
  });

  it('names a point with its codec and OpenZL profile or zstd/gzip level, since a tooltip shows it alone', () => {
    const [trained] = buildSeries(
      [result({compressor: 'OpenZL', profile: 'csv'}, {candidate: {index: 2, total: 5}})],
      'compressMBps',
    ).values();
    const [untrained] = buildSeries([result({compressor: 'OpenZL', level: 3})], 'compressMBps').values();
    const [levelled] = buildSeries([result({compressor: 'zstd', level: 9})], 'compressMBps').values();
    expect(trained.points[0].label).toBe('OpenZL csv #2');
    expect(untrained.points[0].label).toBe('OpenZL serial');
    expect(levelled.points[0].label).toBe('zstd Level 9');
  });

  it('drops a point the log axis has no place for, infinite or not a number', () => {
    const [series] = buildSeries(
      [
        result({compressor: 'zstd', level: 1}, {compressMBps: Number.POSITIVE_INFINITY}),
        result({compressor: 'zstd', level: 3}, {ratio: Number.NaN}),
        result({compressor: 'zstd', level: 9}, {compressMBps: 90, ratio: 7}),
      ],
      'compressMBps',
    ).values();
    expect(series.points.map((point) => point.x)).toEqual([90]);
  });
});

describe('logDomain', () => {
  it('leaves margin either side so points are not on the frame', () => {
    const [low, high] = logDomain([100, 500]);
    expect(low).toBeLessThan(100);
    expect(high).toBeGreaterThan(500);
  });

  it('gives a single point a domain it can sit inside', () => {
    // Identical ends make a zero-width scale, which cannot be inverted.
    const [low, high] = logDomain([42, 42]);
    expect(low).toBeLessThan(42);
    expect(high).toBeGreaterThan(42);
  });

  it('ignores values a log axis cannot place', () => {
    expect(logDomain([0, -5, 200])).toEqual(logDomain([200]));
    expect(logDomain([])).toEqual([1, 10]);
  });
});

describe('logTicks', () => {
  it('labels the powers of ten inside the domain', () => {
    expect(logTicks([224 / 1.3, 15235 * 1.3])).toEqual([1000, 10000]);
    // The design's speed axis carries exactly one label, so one is enough.
    expect(logTicks([38 / 1.3, 520 * 1.3])).toEqual([100]);
  });

  it('picks round numbers when less than a decade is on screen', () => {
    // Real zstd and gzip ratios land here, and the domain's padded ends make
    // labels like `3.46×` and `6.99×` that nobody reads off an axis.
    expect(logTicks([3.46, 6.99])).toEqual([4, 5, 6]);
    expect(logTicks([3.5, 9.6])).toEqual([4, 6, 8]);
  });

  it('labels something however narrow the range, without float drift', () => {
    // A step accumulated as a float would land on 5.029999999999999 here.
    expect(logTicks([5.02, 5.04])).toEqual([5.02, 5.03, 5.04]);
  });
});

describe('tick formatting', () => {
  it('shortens a ratio to what fits beside the axis', () => {
    expect(formatRatioTick(1000)).toBe('1.0k×');
    expect(formatRatioTick(10000)).toBe('10k×');
    expect(formatRatioTick(342)).toBe('342×');
    expect(formatRatioTick(5.4)).toBe('5.4×');
  });

  it('leaves the unit to the axis title', () => {
    expect(formatSpeedTick(100)).toBe('100');
    expect(formatSpeedTick(38.25)).toBe('38.3');
  });
});
