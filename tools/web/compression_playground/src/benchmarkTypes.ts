// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {BenchmarkResult as BenchmarkMetrics, ProfileMetadata} from '../../../wasm/js/wasm_api.js';
import {getOpenZlProfiles, isTrainableProfile, type CompressorRow, type OpenZlProfile} from './compressors.ts';
import {isBrowserSupportedProfile} from './wasmProfiles.ts';

interface CompressorConfigBase {
  readonly rowId: number;
  /**
   * One measurement per level: several for a zstd or gzip row, so it
   * contributes a curve rather than a point, and one for an OpenZL row, whose
   * several measurements come from training instead.
   */
  readonly levels: readonly number[];
}

/**
 * Present only when the row asks for training, so an untrained run cannot carry
 * a candidate count the worker would then act on. Training also takes a time
 * budget, which belongs here once the UI exposes it.
 */
export interface TrainingConfig {
  /**
   * Stated rather than assumed: the binding ignores `maxNumCandidates` unless
   * `paretoFrontier` is set, and without it training returns a single
   * best-ratio compressor. A candidate count only means anything alongside the
   * frontier, so the two travel together.
   */
  readonly paretoFrontier: true;
  /**
   * How many of the trained compressors to keep, not how many are built --
   * `wasm_api.d.ts` is explicit that this bounds the results, not the work.
   * The binding raises anything under `OPENZL_WASM_TRAIN_PARETO_CANDIDATES`
   * (6) and throws over 25, the window `TRAINED_CANDIDATE_COUNTS` offers.
   */
  readonly candidates: number;
  /**
   * Wall-clock budget in seconds. The trainers spend whatever they are given
   * rather than finishing early, so this sets how long a run takes rather than
   * capping it. The UI does not offer it yet; the trained smoke test sets it,
   * which is the only reason that test is seconds rather than minutes.
   */
  readonly maxTimeSecs?: number;
}

export interface OpenZlCompressorConfig extends CompressorConfigBase {
  readonly compressor: 'OpenZL';
  readonly profile: OpenZlProfile;
  readonly training: TrainingConfig | null;
}

export interface ZstdCompressorConfig extends CompressorConfigBase {
  readonly compressor: 'zstd';
}

export interface GzipCompressorConfig extends CompressorConfigBase {
  readonly compressor: 'gzip';
}

export type CompressorConfig = OpenZlCompressorConfig | ZstdCompressorConfig | GzipCompressorConfig;

export interface RunConfig {
  readonly input: File;
  readonly compressors: readonly CompressorConfig[];
  readonly iterations: number;
}

interface BenchmarkJobBase {
  readonly id: string;
  readonly rowId: number;
  readonly level: number;
  readonly iterations: number;
}

export interface OpenZlBenchmarkJob extends BenchmarkJobBase {
  readonly compressor: 'OpenZL';
  readonly profile: OpenZlProfile;
  readonly training: TrainingConfig | null;
}

export interface ZstdBenchmarkJob extends BenchmarkJobBase {
  readonly compressor: 'zstd';
}

export interface GzipBenchmarkJob extends BenchmarkJobBase {
  readonly compressor: 'gzip';
}

export type BenchmarkJob = OpenZlBenchmarkJob | ZstdBenchmarkJob | GzipBenchmarkJob;

/**
 * Which compressor out of a trained frontier produced this measurement, best
 * ratio first, so the table can rank the rows a single job expands into and
 * label the ends. Null when the job produced one measurement, which is every
 * job that is not training.
 */
export interface Candidate {
  readonly index: number;
  readonly total: number;
}

export interface JobResult extends BenchmarkMetrics {
  readonly job: BenchmarkJob;
  readonly candidate: Candidate | null;
}

export interface JobFailure {
  readonly job: BenchmarkJob;
  /**
   * The one candidate that could not be benchmarked, whose siblings are still
   * measured. Null when the whole job failed, training included.
   */
  readonly candidate: Candidate | null;
  readonly message: string;
}

interface RunOutcome {
  readonly results: readonly JobResult[];
  readonly failures: readonly JobFailure[];
}

export interface IdleRunState {
  readonly status: 'idle';
}

export interface LoadingRunState {
  readonly status: 'loading';
}

export interface RunningRunState extends RunOutcome {
  readonly status: 'running';
  readonly completedJobs: number;
  readonly totalJobs: number;
  /**
   * How far the current job is through a step of its own work, 0 to 1, when
   * it can say. Only training reports: it is one job that runs for minutes,
   * so without this the job counter sits on the same number for the whole of
   * it. A trained job takes more than one step and each reports from 0, so
   * this can fall back before the job is done.
   */
  readonly step: number | null;
}

export interface CompletedRunState extends RunOutcome {
  readonly status: 'completed';
}

export interface ErrorRunState extends RunOutcome {
  readonly status: 'error';
  readonly message: string;
}

export type RunState = IdleRunState | LoadingRunState | RunningRunState | CompletedRunState | ErrorRunState;

/**
 * What the worker sends back. One `result` per measurement rather than per job,
 * because a training job yields one per trained candidate; `totalJobs` counts
 * jobs, so it is the denominator for progress, not for rows.
 *
 * Failures arrive per job and do not stop the run: one compressor that cannot
 * be measured should not cost the others their results.
 */
export type WorkerMessage =
  | {readonly type: 'loading'}
  | {readonly type: 'started'; readonly totalJobs: number; readonly rejected: readonly RejectedCompressor[]}
  | {readonly type: 'step'; readonly fraction: number}
  | {readonly type: 'result'; readonly result: JobResult}
  | {readonly type: 'failure'; readonly failure: JobFailure}
  | {readonly type: 'finished'}
  | {readonly type: 'failed'; readonly message: string}
  | {readonly type: 'profiles'; readonly profiles: readonly ProfileMetadata[]}
  | {readonly type: 'profilesFailed'; readonly message: string};

/** The messages that belong to a run, as opposed to the profile lookup. */
export type RunMessage = Exclude<WorkerMessage, {type: 'profiles' | 'profilesFailed'}>;

/**
 * A run as the desktop app's backend takes it. The backend runs the native CLI,
 * which reads the input where it is on disk, so the input is a path rather than
 * the file itself; everything else is the browser's `RunConfig`.
 */
export interface DesktopRunConfig extends Omit<RunConfig, 'input'> {
  readonly input: {readonly path: string};
}

/** What the page sends in. The `File` rides along, since it clones. */
export type WorkerRequest = {readonly config: RunConfig; type: 'benchmarkRequest'} | {readonly type: 'profilesRequest'};

function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${String(value)}`);
}

export function toCompressorConfig(row: CompressorRow): CompressorConfig {
  switch (row.compressor) {
    case 'OpenZL':
      return {
        rowId: row.id,
        compressor: row.compressor,
        levels: [row.level],
        profile: row.profile,
        training: row.trainRequested ? {paretoFrontier: true, candidates: row.candidates} : null,
      };
    case 'zstd':
    case 'gzip':
      // Already the set the user picked, so it crosses unchanged. Sorted, since
      // the charts join consecutive points into a curve and the table lists
      // them in order, and the picker cannot be relied on for that.
      return {rowId: row.id, compressor: row.compressor, levels: [...row.levels].sort((a, b) => a - b)};
    default:
      return assertNever(row);
  }
}

/**
 * A compressor the run cannot measure, reported rather than skipped: a row that
 * contributes nothing and says nothing reads as a benchmark that lost it.
 */
export interface RejectedCompressor {
  readonly rowId: number;
  readonly message: string;
}

export interface RunPlan {
  readonly jobs: readonly BenchmarkJob[];
  readonly rejected: readonly RejectedCompressor[];
}

/**
 * Flattens the run into one job per compressor per level, which is the unit the
 * worker measures and the unit `totalJobs` counts.
 */
export function buildJobs(config: RunConfig): RunPlan {
  const jobs: BenchmarkJob[] = [];
  const rejected: RejectedCompressor[] = [];

  for (const compressor of config.compressors) {
    // Reported rather than dropped, the same as a profile with no browser
    // build: a row that contributes nothing and says nothing reads as a
    // benchmark that lost it. The picker lets a row be emptied, so this is
    // reachable from the UI rather than only from a config built elsewhere.
    if (compressor.levels.length === 0) {
      rejected.push({rowId: compressor.rowId, message: 'No levels selected'});
      continue;
    }

    const base = (level: number) => ({
      id: `${compressor.rowId}-${compressor.compressor}-${level}`,
      rowId: compressor.rowId,
      level,
      iterations: config.iterations,
    });

    switch (compressor.compressor) {
      case 'OpenZL': {
        const profile = compressor.profile;
        // With no profiles loaded the jobs still go out, so each reports why
        // the module failed rather than being rejected here.
        if (getOpenZlProfiles().length > 0 && !isBrowserSupportedProfile(profile)) {
          // The picker refuses these profiles, so this is a backstop for a
          // config assembled anywhere else.
          rejected.push({
            rowId: compressor.rowId,
            message: `The ${compressor.profile} profile has no browser build`,
          });
          break;
        }
        for (const level of compressor.levels) {
          // Apply training capability here without discarding the user's request,
          // so switching back to a trainable profile restores their setting.
          const training = isTrainableProfile(compressor.profile) ? compressor.training : null;
          jobs.push({...base(level), compressor: 'OpenZL', profile, training});
        }
        break;
      }
      case 'zstd':
        for (const level of compressor.levels) {
          jobs.push({...base(level), compressor: 'zstd'});
        }
        break;
      case 'gzip':
        for (const level of compressor.levels) {
          jobs.push({...base(level), compressor: 'gzip'});
        }
        break;
      default:
        assertNever(compressor);
    }
  }

  return {jobs, rejected};
}

export function isRunInProgress(runState: RunState): boolean {
  switch (runState.status) {
    case 'loading':
    case 'running':
      return true;
    case 'idle':
    case 'completed':
    case 'error':
      return false;
    default:
      return assertNever(runState);
  }
}
