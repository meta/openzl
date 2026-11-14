// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {JobResult, RunState} from './benchmarkTypes.ts';

/**
 * Columns the table can be ordered by. `codec` ascending is the order the run
 * produced. Not the level: zstd's 5 and gzip's 5 are different settings, so
 * one order across codecs would mean nothing.
 */
export type SortKey = 'codec' | 'compressedSize' | 'ratio' | 'compressMBps' | 'decompressMBps';
export type SortDirection = 'asc' | 'desc';
export interface Sort {
  readonly key: SortKey;
  readonly direction: SortDirection;
}
export const DEFAULT_SORT: Sort = {key: 'codec', direction: 'asc'};

/**
 * The direction a column sorts in when first chosen: its best value first.
 * The smallest compressed size is the best one, unlike the largest ratio and
 * speeds, and the codec starts in the order the run produced.
 */
export const FIRST_DIRECTION: Record<SortKey, SortDirection> = {
  codec: 'asc',
  compressedSize: 'asc',
  ratio: 'desc',
  compressMBps: 'desc',
  decompressMBps: 'desc',
};

/** Named ends of a trained frontier, which is ordered best ratio first. */
export type FrontierTag = 'max ratio' | 'balanced' | 'max speed';

/**
 * A single measurement needs no rank to read, and a frontier of two has no
 * middle. Three or more take all three names, the middle by rank rather than
 * by any property of the numbers, which matches how the frontier is ordered.
 */
export function frontierTag(index: number, total: number): FrontierTag | null {
  if (total < 2) {
    return null;
  }
  if (index === 1) {
    return 'max ratio';
  }
  if (index === total) {
    return 'max speed';
  }
  return total >= 3 && index === Math.ceil(total / 2) ? 'balanced' : null;
}

/**
 * What one measurement ran at, for the `LEVEL / PROFILE` column: the profile
 * for OpenZL, marked when it was trained, and the level otherwise. Every row
 * says this for itself, since sorting can put any two rows next to each other.
 */
export function rowDetail(result: JobResult): string {
  if (result.job.compressor !== 'OpenZL') {
    return `Level ${String(result.job.level)}`;
  }
  const name = result.job.profile;
  return result.candidate === null ? name : `trained · ${name}`;
}

/**
 * Across every row rather than within a codec: sorting is for seeing which
 * measurement comes out ahead, whatever produced it. By codec it is the order
 * the run produced -- configuration order, then ascending level or the
 * frontier's own order -- with descending reversing the configuration order
 * only. `Array.sort` is stable, so ties keep the run's order too.
 */
export function sortMeasurements(results: readonly JobResult[], sort: Sort): readonly JobResult[] {
  if (sort.key === 'codec') {
    if (sort.direction === 'asc') {
      return results;
    }
    const rank = new Map<number, number>();
    for (const result of results) {
      if (!rank.has(result.job.rowId)) {
        rank.set(result.job.rowId, rank.size);
      }
    }
    const rankOf = (result: JobResult) => rank.get(result.job.rowId) ?? 0;
    return [...results].sort((a, b) => rankOf(b) - rankOf(a));
  }
  const {key} = sort;
  const sign = sort.direction === 'asc' ? 1 : -1;
  return [...results].sort((a, b) => (a[key] - b[key]) * sign);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1000) {
    return `${String(bytes)} B`;
  }
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1000;
  let unit = 0;
  // Compared as it will print: 999.96 KB rounds to `1000.0 KB` otherwise,
  // where the next unit says `1.0 MB`.
  while (Math.round(value * 10) / 10 >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

export const formatRatio = (ratio: number): string => `${ratio.toFixed(2)}×`;
export const formatSpeed = (mbps: number): string => `${String(Math.round(mbps))} MB/s`;

/**
 * How much of its track a bar fills, against the largest value anywhere in the
 * run rather than per codec, which would give every codec's best a full bar
 * and read as a tie between things the tool exists to tell apart.
 */
export function barFraction(value: number, peak: number): number {
  return peak <= 0 ? 0 : Math.max(0.02, Math.min(1, value / peak));
}

export interface Peaks {
  readonly ratio: number;
  readonly compressMBps: number;
  readonly decompressMBps: number;
}

export function peaksOf(results: readonly JobResult[]): Peaks {
  return {
    ratio: Math.max(0, ...results.map((result) => result.ratio)),
    compressMBps: Math.max(0, ...results.map((result) => result.compressMBps)),
    decompressMBps: Math.max(0, ...results.map((result) => result.decompressMBps)),
  };
}

export interface RunSummary {
  /** Absent when nothing was measured, which is the only source for it. */
  readonly srcSize: number | null;
  readonly succeeded: number;
  readonly failed: number;
}

/**
 * What a finished run amounts to, counted in compressor rows rather than jobs.
 * A row is what the reader added, and the design words it that way; jobs would
 * double-count a row that was expanded across six levels.
 *
 * A row counts as failed only when it produced nothing at all. One level of six
 * failing leaves the row succeeded, and the failure list under the table names
 * it. Rows rejected before becoming jobs are in neither count -- nothing
 * carries them this far.
 */
export function runSummary(runState: RunState): RunSummary | null {
  if (runState.status !== 'completed') {
    return null;
  }
  const succeeded = new Set(runState.results.map((result) => result.job.rowId));
  const failed = new Set(
    runState.failures.map((failure) => failure.job.rowId).filter((rowId) => !succeeded.has(rowId)),
  );
  return {
    srcSize: runState.results[0]?.srcSize ?? null,
    succeeded: succeeded.size,
    failed: failed.size,
  };
}

/**
 * Tells apart every measurement of a run. The job id alone does not: a trained
 * job posts one result per candidate, all carrying its id.
 */
export function measurementKey(measurement: Pick<JobResult, 'job' | 'candidate'>): string {
  return `${measurement.job.id}-${String(measurement.candidate?.index ?? 0)}`;
}

/** `idle` and `loading` carry no outcome yet. */
export function resultsOf(runState: RunState): readonly JobResult[] {
  return runState.status === 'idle' || runState.status === 'loading' ? [] : runState.results;
}
