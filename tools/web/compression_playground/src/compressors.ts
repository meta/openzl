// Copyright (c) Meta Platforms, Inc. and affiliates.

export const COMPRESSOR_NAMES = ['OpenZL', 'zstd', 'gzip'] as const;
export type CompressorName = (typeof COMPRESSOR_NAMES)[number];

export type OpenZlProfile = string;

export {getOpenZlProfiles, getOpenZlProfileDescriptions, isTrainableProfile} from './wasmProfiles.ts';

function range(min: number, max: number): readonly number[] {
  return Array.from({length: max - min + 1}, (_, index) => min + index);
}

export const OPENZL_LEVELS = range(1, 9);
export const ZSTD_LEVELS = range(1, 19);
export const GZIP_LEVELS = range(1, 9);
/**
 * The window the WASM trainer honours: it silently raises anything below
 * `OPENZL_WASM_TRAIN_PARETO_CANDIDATES` (6, the floor of its pruning) and
 * throws above `MAX_TRAIN_CANDIDATES` (25, which bounds its allocations).
 */
export const TRAINED_CANDIDATE_COUNTS = range(6, 25);

export const OPENZL_DEFAULT_LEVEL = 6;

/**
 * How many levels a new zstd or gzip row starts with. Its own number rather
 * than the trained candidate count: each compressor's settings are independent
 * of the others'.
 */
export const DEFAULT_LEVEL_COUNT = 6;

/**
 * `count` levels spread across the compressor's range, so a row starts as a
 * curve rather than a point.
 *
 * Rounding lands twice on the same level when the range is tight, so the set
 * deduplicates and the fill tops it back up. That also clamps: gzip has nine
 * levels and zstd nineteen, so asking for more than the range holds returns
 * the whole range.
 *
 * Even spacing by level number is not even spacing on a log-log chart, so this
 * is worth revisiting once there are real curves to look at.
 */
export function evenlySpacedLevels(compressor: 'zstd' | 'gzip', count: number): readonly number[] {
  const all = levelsFor(compressor);
  // The step below divides by `count - 1`, which is the gap count rather than
  // the point count, so one point has no gap to divide by.
  if (count <= 1) {
    return all.slice(0, 1);
  }
  const picked = new Set<number>();
  for (let index = 0; index < count; index += 1) {
    picked.add(all[Math.round((index * (all.length - 1)) / (count - 1))]);
  }
  for (const level of all) {
    if (picked.size >= count) {
      break;
    }
    picked.add(level);
  }
  return [...picked].sort((a, b) => a - b);
}
export const DEFAULT_OPENZL_PROFILE: OpenZlProfile = 'serial';
export const DEFAULT_TRAINED_CANDIDATES = 6;

export function levelsFor(compressor: CompressorName): readonly number[] {
  switch (compressor) {
    case 'OpenZL':
      return OPENZL_LEVELS;
    case 'zstd':
      return ZSTD_LEVELS;
    case 'gzip':
      return GZIP_LEVELS;
  }
}

export function defaultLevelsFor(compressor: 'zstd' | 'gzip'): readonly number[] {
  return evenlySpacedLevels(compressor, DEFAULT_LEVEL_COUNT);
}

export interface CompressorRowBase {
  id: number;
}

/**
 * One measurement per selected level, which is what gives zstd and gzip a
 * curve on the charts rather than a point. OpenZL is not one of these: its
 * several measurements come from training, so it carries a single level.
 */
export interface LeveledRowBase extends CompressorRowBase {
  levels: readonly number[];
}

/**
 * Profile, training and the collapsible options section are OpenZL-only, so
 * they live on this variant rather than on every row. `compressor` is the
 * discriminant: narrowing on it is what stops a gzip row from carrying a
 * training flag the run seam would then act on.
 */
export interface OpenZlRow extends CompressorRowBase {
  compressor: 'OpenZL';
  /** Range depends on the compressor; see `levelsFor`. */
  level: number;
  profile: OpenZlProfile;
  /** What the user asked for, not what will happen: see `isTrainableProfile`. */
  trainRequested: boolean;
  candidates: number;
  optionsOpen: boolean;
}

/** Kept apart from `GzipRow` despite matching today, so either can gain options. */
export interface ZstdRow extends LeveledRowBase {
  compressor: 'zstd';
}

export interface GzipRow extends LeveledRowBase {
  compressor: 'gzip';
}

export type CompressorRow = OpenZlRow | ZstdRow | GzipRow;

export function createCompressorRow(id: number, compressor: 'OpenZL', candidates?: number): OpenZlRow;
export function createCompressorRow(id: number, compressor: CompressorName, candidates?: number): CompressorRow;
export function createCompressorRow(
  id: number,
  compressor: CompressorName,
  candidates = DEFAULT_TRAINED_CANDIDATES,
): CompressorRow {
  if (compressor === 'OpenZL') {
    return {
      id,
      compressor,
      level: OPENZL_DEFAULT_LEVEL,
      profile: DEFAULT_OPENZL_PROFILE,
      trainRequested: true,
      candidates,
      optionsOpen: true,
    };
  }
  return {id, compressor, levels: defaultLevelsFor(compressor)};
}

export const ITERATIONS_MIN = 1;
export const ITERATIONS_MAX = 15;
export const ITERATIONS_DEFAULT = 5;
export const ITERATION_TICKS = [1, 5, 10, 15] as const;

/**
 * Half the iterations thumb width in pixels. With the slider's default
 * "contain" alignment the thumb center travels between this inset and
 * (width - inset), so anything aligning to the dot — tick labels, in
 * particular — must use the same inset. Keep in sync with the thumb's
 * `boxSize`, which derives from this.
 */
export const ITERATIONS_THUMB_HALF_PX = 8;

/**
 * CSS `left` for a tick label so its center sits exactly under the dot when
 * the slider holds that value: the proportional position within the thumb's
 * inset travel, paired with `translateX(-50%)` at the call site.
 */
export function tickLabelLeft(tick: number): string {
  const fraction = Math.round(((tick - ITERATIONS_MIN) / (ITERATIONS_MAX - ITERATIONS_MIN)) * 10000) / 10000;
  return `calc(${ITERATIONS_THUMB_HALF_PX}px + ${fraction} * (100% - ${ITERATIONS_THUMB_HALF_PX * 2}px))`;
}
