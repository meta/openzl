// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {RunState} from './benchmarkTypes.ts';

/**
 * How far through the run, and what it is doing. A training job runs for
 * minutes and counts as one job, so the job counter alone would sit still for
 * most of a trained run; the trainer's own progress fills that stretch.
 */
export function progressOf(runState: RunState): {fraction: number; label: string} | null {
  if (runState.status === 'loading') {
    return {fraction: 0, label: 'Preparing for benchmarks…'};
  }
  if (runState.status !== 'running') {
    return null;
  }
  const {completedJobs, totalJobs, step} = runState;
  const done = totalJobs === 0 ? 0 : completedJobs / totalJobs;
  // The last job's result and `finished` are two messages, so there is a frame
  // between them where every job is counted and the run is still `running`.
  const running = Math.min(completedJobs + 1, totalJobs);
  const label = `Running ${String(running)} out of ${String(totalJobs)}…`;
  if (step === null) {
    return {fraction: done, label};
  }
  // The step belongs to the job after the ones already counted, so its own
  // fraction advances the bar across that job's share and no further.
  const share = totalJobs === 0 ? 0 : 1 / totalJobs;
  return {fraction: Math.min(1, done + step * share), label};
}
