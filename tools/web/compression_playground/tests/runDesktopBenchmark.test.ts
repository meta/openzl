// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment node

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import type {BenchmarkJob, CompressorConfig, JobResult, RunMessage, RunState} from '../src/benchmarkTypes.ts';
import type {DesktopRunConfig, DesktopRunRequest, OpenZLDesktopBridge} from '../src/desktop.ts';
import {BUNDLED_SAMPLE} from '../src/sample.ts';

/**
 * A stand-in for the preload bridge. The test decides when the backend says
 * what, and when the run ends.
 */
function fakeBridge(path = '/tmp/data.bin') {
  const started: {
    config: DesktopRunConfig;
    send: (message: RunMessage) => void;
    resolve: () => void;
    reject: (error: Error) => void;
    cancel: ReturnType<typeof vi.fn>;
  }[] = [];
  const bridge: OpenZLDesktopBridge = {
    getPathForFile: () => path,
    listProfiles: () => Promise.resolve([]),
    startBenchmark: (config, onMessage) => {
      const cancel = vi.fn();
      const done = new Promise<void>((resolve, reject) => {
        started.push({config, send: (message) => onMessage?.(message), resolve, reject, cancel});
      });
      return {done, cancel};
    },
  };
  return {bridge, started};
}

const OPENZL: CompressorConfig = {
  rowId: 1,
  compressor: 'OpenZL',
  levels: [6],
  profile: 'serial',
  training: null,
};

const OPENZL_JOB: BenchmarkJob = {
  id: '1-OpenZL-6',
  rowId: 1,
  level: 6,
  iterations: 3,
  compressor: 'OpenZL',
  profile: 'serial',
  training: null,
};

function configWith(
  compressors: readonly CompressorConfig[],
  input: DesktopRunRequest['input'] = new File(['x'], 'data.bin'),
): DesktopRunRequest {
  return {input, compressors, iterations: 3};
}

function result(overrides: Partial<JobResult> = {}): JobResult {
  return {
    job: OPENZL_JOB,
    candidate: null,
    iterations: 3,
    srcSize: 1000,
    compressedSize: 250,
    ratio: 4,
    compressMs: 10,
    decompressMs: 5,
    compressMBps: 100,
    decompressMBps: 200,
    ...overrides,
  };
}

/** The run the backend was asked for first, which a test expects to have started. */
function first<T>(started: T[]): T {
  const [run] = started;
  if (run === undefined) {
    throw new Error('No run was started');
  }
  return run;
}

/** Lets the input resolve and the run start, which takes a few microtasks. */
const settle = () => vi.advanceTimersByTimeAsync(0);

// Fresh each time: the executor refuses a second run while one is in flight,
// and that is module state a test must not inherit from the one before.
let runDesktopBenchmark: typeof import('../src/runDesktopBenchmark.ts').runDesktopBenchmark;

function run(config: DesktopRunRequest, bridge: OpenZLDesktopBridge) {
  const states: RunState[] = [];
  runDesktopBenchmark(config, bridge, (state) => states.push(state));
  return {states, last: () => states[states.length - 1]};
}

beforeEach(async () => {
  vi.useFakeTimers();
  vi.resetModules();
  ({runDesktopBenchmark} = await import('../src/runDesktopBenchmark.ts'));
});

afterEach(() => {
  vi.useRealTimers();
});

describe('runDesktopBenchmark', () => {
  it('reports itself busy before the backend has even started', () => {
    const {bridge, started} = fakeBridge();
    const {states} = run(configWith([OPENZL]), bridge);

    expect(states).toEqual([{status: 'loading'}]);
    expect(started).toHaveLength(0);
  });

  it("sends the browser's RunConfig, with the input named by its path", async () => {
    const {bridge, started} = fakeBridge('/data/input.bin');
    const zstd: CompressorConfig = {rowId: 2, compressor: 'zstd', levels: [1, 3]};
    run(configWith([OPENZL, zstd]), bridge);
    await settle();

    expect(started[0]?.config).toEqual({input: {path: '/data/input.bin'}, compressors: [OPENZL, zstd], iterations: 3});
  });

  it('sends a profile that cannot train untrained, as buildJobs plans it in the browser', async () => {
    const {setWasmProfiles} = await import('../src/wasmProfiles.ts');
    setWasmProfiles([{name: 'parquet', description: 'Parquet', isWasmSupported: true, isTrainable: false}]);
    const {bridge, started} = fakeBridge();
    const training = {paretoFrontier: true, candidates: 10} as const;
    run(configWith([{...OPENZL, profile: 'parquet', training}]), bridge);
    await settle();

    expect(started[0]?.config.compressors).toEqual([{...OPENZL, profile: 'parquet', training: null}]);
  });

  it("folds the backend's messages into RunState, as the browser's runner does", async () => {
    const {bridge, started} = fakeBridge();
    const {last} = run(configWith([{...OPENZL, training: {paretoFrontier: true, candidates: 10}}]), bridge);
    await settle();
    const backend = first(started);

    backend.send({type: 'loading'});
    backend.send({type: 'started', totalJobs: 1, rejected: []});
    backend.send({type: 'step', fraction: 0.5});
    expect(last()).toMatchObject({status: 'running', completedJobs: 0, totalJobs: 1, step: 0.5});

    backend.send({type: 'result', result: result({candidate: {index: 1, total: 2}})});
    expect(last()).toMatchObject({completedJobs: 0, results: [{candidate: {index: 1, total: 2}}]});
    backend.send({type: 'result', result: result({candidate: {index: 2, total: 2}})});
    expect(last()).toMatchObject({completedJobs: 1, step: null});

    backend.send({type: 'finished'});
    backend.resolve();
    await settle();
    const completed = last();
    expect(completed).toMatchObject({status: 'completed', failures: []});
    expect(completed?.status === 'completed' && completed.results).toHaveLength(2);
  });

  it('records a job the backend could not measure as a failure, and runs on', async () => {
    const {bridge, started} = fakeBridge();
    const {last} = run(configWith([OPENZL]), bridge);
    await settle();
    const backend = first(started);

    backend.send({type: 'started', totalJobs: 2, rejected: []});
    backend.send({type: 'failure', failure: {job: OPENZL_JOB, candidate: null, message: 'Profile not found'}});
    expect(last()).toMatchObject({status: 'running', completedJobs: 1, failures: [{message: 'Profile not found'}]});
  });

  it('ends in an error that keeps what was already measured when the run fails', async () => {
    const {bridge, started} = fakeBridge();
    const {last} = run(configWith([OPENZL]), bridge);
    await settle();
    const backend = first(started);

    backend.send({type: 'started', totalJobs: 2, rejected: []});
    backend.send({type: 'result', result: result()});
    backend.send({type: 'failed', message: 'Native benchmark process exited unexpectedly (code 1)'});
    backend.reject(new Error('Native benchmark process exited unexpectedly (code 1)'));
    await settle();

    expect(last()).toMatchObject({
      status: 'error',
      message: 'Native benchmark process exited unexpectedly (code 1)',
      results: [result()],
    });
  });

  it('ends in an error when the run is cancelled', async () => {
    const {bridge, started} = fakeBridge();
    const {last} = run(configWith([OPENZL]), bridge);
    await settle();

    started[0]?.reject(new Error('Benchmark cancelled'));
    await settle();

    expect(last()).toMatchObject({status: 'error', message: 'Benchmark cancelled'});
  });

  it('ends in an error, before starting the backend, and stays usable, when the file has no path', async () => {
    const {bridge, started} = fakeBridge('');
    const {last} = run(configWith([OPENZL]), bridge);
    await settle();

    expect(started).toHaveLength(0);
    expect(last()).toMatchObject({status: 'error', message: 'data.bin is not a file on disk'});

    const next = fakeBridge();
    run(configWith([OPENZL]), next.bridge);
    await settle();
    expect(next.started).toHaveLength(1);
  });

  it('asks the backend for the bundled sample, leaving its path to the main process', async () => {
    const {bridge, started} = fakeBridge();
    run(configWith([OPENZL], BUNDLED_SAMPLE), bridge);
    await settle();

    expect(started[0]?.config.input).toEqual(BUNDLED_SAMPLE);
  });

  it('starts no second run while one is in flight, and frees the next once it ends', async () => {
    const {bridge, started} = fakeBridge();
    run(configWith([OPENZL]), bridge);
    await settle();
    const second = run(configWith([OPENZL]), bridge);
    await settle();

    expect(started).toHaveLength(1);
    expect(second.states).toEqual([]);

    started[0]?.send({type: 'finished'});
    started[0]?.resolve();
    await settle();
    run(configWith([OPENZL]), bridge);
    await settle();
    expect(started).toHaveLength(2);
  });
});
