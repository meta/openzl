// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {CompressorName} from './compressors.ts';
import type {JobResult} from './benchmarkTypes.ts';
import {measurementKey} from './measurements.ts';

/** A codec's levels or one OpenZL job's measurements on a ratio-against-speed chart. */
export interface Series {
  readonly compressor: CompressorName;
  readonly points: readonly ChartPoint[];
}

export interface ChartPoint {
  readonly x: number;
  readonly y: number;
  readonly label: string;
  /** The measurement's `measurementKey`, so a table row can find its point. */
  readonly key: string;
}

/** Which throughput a chart plots; the ratio is always the other axis. */
export type SpeedAxis = 'compressMBps' | 'decompressMBps';

/**
 * Each OpenZL training job has its own curve. zstd and gzip each have one curve
 * connecting their selected levels.
 */
export function buildSeries(results: readonly JobResult[], axis: SpeedAxis): ReadonlyMap<string, Series> {
  const bySeries = new Map<string, {compressor: CompressorName; points: ChartPoint[]}>();
  for (const result of results) {
    if (!plottable(result[axis]) || !plottable(result.ratio)) {
      continue;
    }
    const point = {x: result[axis], y: result.ratio, label: pointLabel(result), key: measurementKey(result)};
    const compressor = result.job.compressor;
    const key = compressor === 'OpenZL' ? result.job.id : compressor;
    const series = bySeries.get(key);
    if (series === undefined) {
      bySeries.set(key, {
        compressor,
        points: [point],
      });
    } else {
      series.points.push(point);
    }
  }
  // Ascending speed, so the line joining the points reads left to right
  // instead of doubling back on itself.
  for (const series of bySeries.values()) {
    series.points.sort((a, b) => a.x - b.x);
  }
  return bySeries;
}

/** Named with its codec and its OpenZL profile or zstd/gzip level: a tooltip shows one point. */
function pointLabel(result: JobResult): string {
  const job = result.job;
  if (job.compressor !== 'OpenZL') {
    return `${job.compressor} Level ${String(job.level)}`;
  }
  const name = `${job.compressor} ${job.profile}`;
  if (result.candidate !== null) {
    return `${name} #${String(result.candidate.index)}`;
  }
  return name;
}

/**
 * A log axis has no position for zero, a negative or a non-finite value, so
 * neither the domain nor a point may carry one. An empty input measures as
 * 0 MB/s at a ratio of 0, and both sides drop it rather than asking the scale
 * for log(0).
 */
function plottable(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * A log axis cannot start at zero and recharts will not pick a domain for one,
 * so both ends come from the data with a decade's worth of margin either side.
 */
export function logDomain(values: readonly number[]): readonly [number, number] {
  const positive = values.filter(plottable);
  if (positive.length === 0) {
    return [1, 10];
  }
  const low = Math.min(...positive);
  const high = Math.max(...positive);
  // A single point, or several identical ones, would give a zero-width domain
  // that the scale cannot invert.
  return low === high ? [low / 2, high * 2] : [low / 1.3, high * 1.3];
}

/**
 * Powers of ten inside the domain, which is what the design labels. Real zstd
 * and gzip ratios span less than a decade, where that comes back empty and the
 * axis needs round numbers picked inside the range instead -- the domain's own
 * ends are padded, so labelling those gives `3.5×` and `9.6×`.
 */
export function logTicks(domain: readonly [number, number]): readonly number[] {
  const [low, high] = domain;
  const ticks: number[] = [];
  for (let power = Math.ceil(Math.log10(low)); Math.pow(10, power) <= high; power += 1) {
    ticks.push(Math.pow(10, power));
  }
  // One is enough -- the design's speed axis labels a single `100`.
  return ticks.length > 0 ? ticks : niceTicks(low, high);
}

/**
 * Multiples of a 1, 2 or 5 step, which are the numbers people read off axes.
 * The step is always smaller than the range it divides, so this cannot come
 * back empty -- checked over 400k random sub-decade ranges, where the fewest it produced
 * was one.
 */
function niceTicks(low: number, high: number): readonly number[] {
  const step = niceStep((high - low) / 4);
  const ticks: number[] = [];
  for (let tick = Math.ceil(low / step) * step; tick <= high; tick += step) {
    // Accumulating a float step drifts, and the drift shows up in the label.
    ticks.push(Number(tick.toPrecision(12)));
  }
  return ticks;
}

function niceStep(rough: number): number {
  const power = Math.pow(10, Math.floor(Math.log10(rough)));
  const scaled = rough / power;
  const step = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return step * power;
}

/** `1.0k×`, `10k×`, `342×`: the axis has room for four characters, not seven. */
export function formatRatioTick(ratio: number): string {
  if (ratio < 1000) {
    return `${String(Number(ratio.toFixed(1)))}×`;
  }
  const thousands = ratio / 1000;
  return `${thousands < 10 ? thousands.toFixed(1) : String(Math.round(thousands))}k×`;
}

/** The unit lives in the axis title, so a tick is just the number. */
export function formatSpeedTick(mbps: number): string {
  return String(Number(mbps.toFixed(1)));
}
