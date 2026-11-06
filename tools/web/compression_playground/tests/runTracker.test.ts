// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it} from 'vitest';
import {buildJobs, type BenchmarkJob, type RunConfig, type RunState, type RunMessage} from '../src/benchmarkTypes.ts';
import {progressOf} from '../src/progress.ts';
import {trackRun} from '../src/runTracker.ts';

function jobsFor(config: Partial<RunConfig>): readonly BenchmarkJob[] {
  return buildJobs({
    input: new File([new Uint8Array(10)], 'input.bin'),
    iterations: 1,
    compressors: [
      {
        rowId: 1,
        compressor: 'OpenZL',
        levels: [6],
        profile: 'serial',
        profileArg: '',
        training: {paretoFrontier: true, candidates: 10},
      },
      {rowId: 2, compressor: 'gzip', levels: [5]},
    ],
    ...config,
  } as RunConfig).jobs;
}

const metrics = {
  iterations: 1,
  srcSize: 10,
  compressedSize: 5,
  ratio: 2,
  compressMs: 1,
  decompressMs: 1,
  compressMBps: 1,
  decompressMBps: 1,
};

describe('trackRun', () => {
  it('moves the progress bar from 0 to 100% and never backwards, whatever order steps arrive in', () => {
    const [trained, gzip] = jobsFor({});
    const tracker = trackRun();
    const messages: RunMessage[] = [
      {type: 'started', totalJobs: 2, rejected: []},
      {type: 'step', fraction: 0},
      {type: 'step', fraction: 0.4},
      {type: 'step', fraction: 0.8},
      // A step that goes back, as the desktop backend's did when a second ACE
      // graph or the candidates started again from 0.
      {type: 'step', fraction: 0},
      {type: 'step', fraction: 0.9},
      {type: 'result', result: {...metrics, job: trained, candidate: {index: 1, total: 2}}},
      {type: 'result', result: {...metrics, job: trained, candidate: {index: 2, total: 2}}},
      {type: 'result', result: {...metrics, job: gzip, candidate: null}},
    ];
    const fractions = messages.map((message) => progressOf(tracker.apply(message))?.fraction);

    expect(fractions[0]).toBe(0);
    expect(fractions.at(-1)).toBe(1);
    for (let index = 1; index < fractions.length; index += 1) {
      expect(fractions[index]).toBeGreaterThanOrEqual(fractions[index - 1] ?? 0);
    }
    const finished: RunState = tracker.apply({type: 'finished'});
    expect(finished.status).toBe('completed');
  });

  it('starts the next job’s share of the bar from its start', () => {
    const [trained, gzip] = jobsFor({});
    const tracker = trackRun();
    tracker.apply({type: 'started', totalJobs: 2, rejected: []});
    tracker.apply({type: 'step', fraction: 0.9});
    tracker.apply({type: 'result', result: {...metrics, job: trained, candidate: {index: 1, total: 1}}});
    // The first job is done and its step dropped, so the bar is at exactly half.
    expect(progressOf(tracker.apply({type: 'step', fraction: 0}))?.fraction).toBe(0.5);
    expect(
      progressOf(tracker.apply({type: 'result', result: {...metrics, job: gzip, candidate: null}}))?.fraction,
    ).toBe(1);
  });
});
