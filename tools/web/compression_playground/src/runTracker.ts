// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {Candidate, ErrorRunState, JobFailure, JobResult, RunMessage, RunState} from './benchmarkTypes.ts';

/**
 * Folds a run's `WorkerMessage`s into `RunState`. Both executors feed it: the
 * browser's from its worker, the desktop app's from `zli` once each row is
 * matched back to its job.
 */
export interface RunTracker {
  /** The state the run is in once `message` has landed. */
  apply(message: RunMessage): RunState;
  /** Ends the run as an error, keeping whatever it had already measured. */
  error(message: string): ErrorRunState;
}

/** The last measurement a job posts: its only one, or its frontier's last candidate. */
function endsJob(candidate: Candidate | null): boolean {
  return candidate === null || candidate.index === candidate.total;
}

export function trackRun(): RunTracker {
  const results: JobResult[] = [];
  const failures: JobFailure[] = [];
  let totalJobs = 0;
  let step: number | null = null;

  /**
   * Jobs, not measurements: a trained job arrives as one result or failure per
   * candidate, and it is done when all of them have -- `candidate.total` says
   * how many that is. Anything without a candidate settles its job outright:
   * the one measurement of an untrained job, or a failure of the whole job,
   * which still counts once if some candidates landed before it.
   */
  const completedJobs = (): number => {
    const done = new Set<string>();
    const seen = new Map<string, number>();
    for (const {job, candidate} of [...results, ...failures]) {
      if (candidate === null) {
        done.add(job.id);
        continue;
      }
      const count = (seen.get(job.id) ?? 0) + 1;
      seen.set(job.id, count);
      if (count >= candidate.total) {
        done.add(job.id);
      }
    }
    return Math.min(done.size, totalJobs);
  };

  const progress = (): RunState => ({
    status: 'running',
    completedJobs: completedJobs(),
    totalJobs,
    results: [...results],
    failures: [...failures],
    step,
  });

  const error = (message: string): ErrorRunState => ({
    status: 'error',
    message,
    results: [...results],
    failures: [...failures],
  });

  return {
    apply(message) {
      switch (message.type) {
        case 'loading':
          return {status: 'loading'};
        case 'started':
          totalJobs = message.totalJobs;
          for (const rejected of message.rejected) {
            // `RunState` has nowhere to put a row that never became a job, so
            // for now this only reaches the console.
            console.warn(`OpenZL: row ${String(rejected.rowId)} produced no jobs -- ${rejected.message}`);
          }
          return progress();
        case 'step':
          // Kept inside 0 to 1 so it can only move the bar across this job's
          // share, never into the next one's, and never below where it already
          // is: a step that went back would move the bar backwards.
          if (Number.isFinite(message.fraction)) {
            step = Math.max(step ?? 0, Math.min(1, Math.max(0, message.fraction)));
          }
          return progress();
        case 'result':
          // Done reporting once its last measurement lands. A trained job posts
          // one per candidate after training, and dropping the step at the
          // first would take back the training the bar has already shown.
          if (endsJob(message.result.candidate)) {
            step = null;
          }
          results.push(message.result);
          return progress();
        case 'failure':
          // Done reporting too, and it is already counted. Left standing, a
          // training job that died at 90% would lend that 90% to the next
          // job's share of the bar. A candidate that failed with siblings
          // still to come leaves its job reporting.
          if (endsJob(message.failure.candidate)) {
            step = null;
          }
          failures.push(message.failure);
          return progress();
        case 'finished':
          return {status: 'completed', results: [...results], failures: [...failures]};
        case 'failed':
          return error(message.message);
      }
    },
    error,
  };
}
