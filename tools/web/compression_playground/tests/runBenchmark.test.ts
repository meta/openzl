// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment jsdom
import {afterEach, describe, expect, it, vi} from 'vitest';
import type {RunConfig, RunState} from '../src/benchmarkTypes.ts';

/**
 * A stand-in for the real worker: jsdom has no `Worker`, and these cases are
 * about what `runBenchmark` does around it rather than what it computes.
 */
class FakeWorker {
  static instances: FakeWorker[] = [];
  readonly listeners = new Map<string, ((event: unknown) => void)[]>();
  readonly posted: unknown[] = [];
  terminated = false;

  constructor() {
    FakeWorker.instances.push(this);
  }
  addEventListener(type: string, handler: (event: unknown) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), handler]);
  }
  removeEventListener(type: string, handler: (event: unknown) => void) {
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter((h) => h !== handler),
    );
  }
  postMessage(message: unknown) {
    this.posted.push(message);
  }
  terminate() {
    this.terminated = true;
  }
  emit(type: string, event: unknown) {
    for (const handler of [...(this.listeners.get(type) ?? [])]) {
      handler(event);
    }
  }
}

const config = {input: new File(['x'], 'x.bin'), compressors: [], iterations: 1} as unknown as RunConfig;

async function freshModule() {
  vi.resetModules();
  FakeWorker.instances = [];
  vi.stubGlobal('Worker', FakeWorker);
  return vi.importActual<typeof import('../src/runBenchmark.ts')>('../src/runBenchmark.ts');
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('runBenchmark', () => {
  it('requests profiles once and caches them on the UI side', async () => {
    const {getBenchmarkProfiles, runBenchmark, cancelRun} = await freshModule();
    const first = getBenchmarkProfiles();
    const second = getBenchmarkProfiles();
    const profiles = [
      {name: 'future_profile', description: 'From the loaded build', isWasmSupported: true, isTrainable: true},
    ];
    const [worker] = FakeWorker.instances;
    expect(worker.posted).toEqual([{type: 'profilesRequest'}]);
    worker.emit('message', {data: {type: 'profiles', profiles}});
    expect(await first).toEqual(profiles);
    const {isBrowserSupportedProfile} = await import('../src/wasmProfiles.ts');
    expect(isBrowserSupportedProfile('future_profile')).toBe(true);
    expect(await second).toBe(await first);
    expect(await getBenchmarkProfiles()).toBe(await first);
    expect(worker.listeners.get('message')).toHaveLength(0);
    expect(worker.listeners.get('error')).toHaveLength(0);

    runBenchmark(config, () => undefined);
    expect(FakeWorker.instances).toHaveLength(1);
    expect(worker.posted).toEqual([{type: 'profilesRequest'}, {type: 'benchmarkRequest', config}]);
    cancelRun();
    expect(await getBenchmarkProfiles()).toBe(await first);
    expect(FakeWorker.instances).toHaveLength(1);
  });

  it('retries failed profile discovery without discarding the benchmark worker', async () => {
    const {getBenchmarkProfiles} = await freshModule();
    const first = getBenchmarkProfiles();
    const rejected = expect(first).rejects.toThrow('WASM unavailable');
    const [worker] = FakeWorker.instances;
    worker.emit('message', {data: {type: 'profilesFailed', message: 'WASM unavailable'}});
    await rejected;
    const retry = getBenchmarkProfiles();
    worker.emit('message', {data: {type: 'profiles', profiles: []}});
    expect(await retry).toEqual([]);
    expect(FakeWorker.instances).toHaveLength(1);
    expect(worker.posted).toEqual([{type: 'profilesRequest'}, {type: 'profilesRequest'}]);
  });

  it('keeps an active benchmark running when posting a profile request fails', async () => {
    const {getBenchmarkProfiles, runBenchmark} = await freshModule();
    const states: RunState[] = [];
    runBenchmark(config, (state) => states.push(state));
    const [worker] = FakeWorker.instances;
    vi.spyOn(worker, 'postMessage').mockImplementationOnce(() => {
      throw new Error('Unable to send profile request');
    });

    await expect(getBenchmarkProfiles()).rejects.toThrow('Unable to send profile request');
    expect(worker.terminated).toBe(false);
    expect(states.at(-1)).toMatchObject({status: 'loading'});

    const retry = getBenchmarkProfiles();
    worker.emit('message', {data: {type: 'profiles', profiles: []}});
    expect(await retry).toEqual([]);
    expect(FakeWorker.instances).toHaveLength(1);

    worker.emit('message', {data: {type: 'finished'}});
    expect(states.at(-1)).toMatchObject({status: 'completed'});
  });

  it('rejects pending profile discovery on cancellation and reloads on a new worker', async () => {
    const {getBenchmarkProfiles, cancelRun} = await freshModule();
    const first = getBenchmarkProfiles();
    const rejected = expect(first).rejects.toThrow('the run was cancelled');
    cancelRun();
    await rejected;
    expect(FakeWorker.instances[0].terminated).toBe(true);
    const retry = getBenchmarkProfiles();
    FakeWorker.instances[1].emit('message', {data: {type: 'profiles', profiles: []}});
    expect(await retry).toEqual([]);
    expect(FakeWorker.instances).toHaveLength(2);
  });

  it('discards a worker that crashes during profile discovery', async () => {
    const {getBenchmarkProfiles} = await freshModule();
    const first = getBenchmarkProfiles();
    const rejected = expect(first).rejects.toThrow('worker exploded');
    FakeWorker.instances[0].emit('error', {message: 'worker exploded'});
    await rejected;
    expect(FakeWorker.instances[0].terminated).toBe(true);
    const retry = getBenchmarkProfiles();
    FakeWorker.instances[1].emit('message', {data: {type: 'profiles', profiles: []}});
    expect(await retry).toEqual([]);
  });

  it('reports itself busy before the worker has said anything', async () => {
    // The run button disables off this state. Messages carry no run id, so a
    // second press in the gap would put two runs on one worker and each
    // listener would read both runs' messages.
    const {runBenchmark} = await freshModule();
    const states: RunState[] = [];

    runBenchmark(config, (state) => states.push(state));

    expect(states).toHaveLength(1);
    expect(states[0].status).toBe('loading');
  });

  it('discards a worker that died rather than handing it to the next run', async () => {
    const {runBenchmark} = await freshModule();
    const states: RunState[] = [];

    runBenchmark(config, (state) => states.push(state));
    const [first] = FakeWorker.instances;
    first.emit('error', {message: 'worker exploded'});

    expect(states.at(-1)).toMatchObject({status: 'error', message: 'worker exploded'});
    expect(first.terminated).toBe(true);

    // A dead worker never sends another message, so reusing it would leave the
    // next run waiting rather than failing.
    runBenchmark(config, (state) => states.push(state));
    expect(FakeWorker.instances).toHaveLength(2);
  });

  it('counts a trained job once, and only once its last candidate lands', async () => {
    const {runBenchmark} = await freshModule();
    const states: RunState[] = [];
    const candidate = (index: number) => ({
      type: 'result',
      result: {job: {id: 'openzl-6'}, candidate: {index, total: 3}},
    });

    runBenchmark(config, (state) => states.push(state));
    const [worker] = FakeWorker.instances;
    worker.emit('message', {data: {type: 'started', totalJobs: 2, rejected: []}});

    worker.emit('message', {data: candidate(1)});
    worker.emit('message', {data: candidate(2)});
    // Two of three: the job is still running, and saying otherwise would run
    // the bar ahead of the work.
    expect(states.at(-1)).toMatchObject({completedJobs: 0, totalJobs: 2});

    worker.emit('message', {data: candidate(3)});
    expect(states.at(-1)).toMatchObject({completedJobs: 1});

    // Measured some, then failed: still the one job.
    worker.emit('message', {
      data: {type: 'failure', failure: {job: {id: 'openzl-6'}, candidate: null, message: 'stopped'}},
    });
    expect(states.at(-1)).toMatchObject({completedJobs: 1});
  });

  it('counts a candidate that failed toward its job, not as the whole job', async () => {
    const {runBenchmark} = await freshModule();
    const states: RunState[] = [];
    const job = {id: 'openzl-6'};

    runBenchmark(config, (state) => states.push(state));
    const [worker] = FakeWorker.instances;
    worker.emit('message', {data: {type: 'started', totalJobs: 1, rejected: []}});

    worker.emit('message', {data: {type: 'result', result: {job, candidate: {index: 1, total: 3}}}});
    worker.emit('message', {
      data: {type: 'failure', failure: {job, candidate: {index: 2, total: 3}, message: 'candidate 2 threw'}},
    });
    // The third candidate is still to come, so the job is not done yet.
    expect(states.at(-1)).toMatchObject({completedJobs: 0});

    worker.emit('message', {data: {type: 'result', result: {job, candidate: {index: 3, total: 3}}}});
    expect(states.at(-1)).toMatchObject({completedJobs: 1});
  });

  it('ignores a second run while one is in flight', async () => {
    // Messages carry no run id, so a second run on the same worker would have
    // each listener read both runs' messages.
    const {runBenchmark} = await freshModule();
    const second: RunState[] = [];

    runBenchmark(config, () => undefined);
    runBenchmark(config, (state) => second.push(state));

    expect(second).toHaveLength(0);
    expect(FakeWorker.instances[0].posted).toHaveLength(1);
  });

  it('ends a cancelled run with what it measured, and frees the next one', async () => {
    const {runBenchmark, cancelRun} = await freshModule();
    const states: RunState[] = [];
    const result = {job: {id: 'zstd-3'}, candidate: null};

    runBenchmark(config, (state) => states.push(state));
    const [first] = FakeWorker.instances;
    first.emit('message', {data: {type: 'started', totalJobs: 2, rejected: []}});
    first.emit('message', {data: {type: 'result', result}});

    cancelRun();

    // A terminated worker sends nothing more, so without this the page would
    // stay on `running`.
    expect(first.terminated).toBe(true);
    expect(states.at(-1)).toMatchObject({status: 'error', message: 'the run was cancelled', results: [result]});

    runBenchmark(config, (state) => states.push(state));
    expect(FakeWorker.instances).toHaveLength(2);
    expect(FakeWorker.instances[1].posted).toHaveLength(1);
  });

  it("drops a job's step once it stops reporting, however it stopped", async () => {
    const {runBenchmark} = await freshModule();
    const states: RunState[] = [];

    runBenchmark(config, (state) => states.push(state));
    const [worker] = FakeWorker.instances;
    worker.emit('message', {data: {type: 'started', totalJobs: 2, rejected: []}});

    worker.emit('message', {data: {type: 'step', fraction: 0.9}});
    expect(states.at(-1)).toMatchObject({step: 0.9});

    worker.emit('message', {data: {type: 'result', result: {job: {id: 'openzl-6'}, candidate: null}}});
    expect(states.at(-1)).toMatchObject({step: null});

    // A job that died partway through reporting has stopped too. Left
    // standing, its 90% would lend itself to the next job's share of the bar.
    worker.emit('message', {data: {type: 'step', fraction: 0.9}});
    worker.emit('message', {
      data: {type: 'failure', failure: {job: {id: 'zstd-1'}, candidate: null, message: 'stopped'}},
    });
    expect(states.at(-1)).toMatchObject({step: null});
  });

  it("keeps a trained job's step until its last candidate is in", async () => {
    // Training comes first and its candidates are measured after, so dropping
    // the step at the first one would take back what the bar already showed.
    const {runBenchmark} = await freshModule();
    const states: RunState[] = [];
    const job = {id: 'openzl-6'};

    runBenchmark(config, (state) => states.push(state));
    const [worker] = FakeWorker.instances;
    worker.emit('message', {data: {type: 'started', totalJobs: 2, rejected: []}});
    worker.emit('message', {data: {type: 'step', fraction: 0.9}});

    worker.emit('message', {data: {type: 'result', result: {job, candidate: {index: 1, total: 3}}}});
    expect(states.at(-1)).toMatchObject({step: 0.9});
    worker.emit('message', {
      data: {type: 'failure', failure: {job, candidate: {index: 2, total: 3}, message: 'candidate 2 threw'}},
    });
    expect(states.at(-1)).toMatchObject({step: 0.9});

    worker.emit('message', {data: {type: 'result', result: {job, candidate: {index: 3, total: 3}}}});
    expect(states.at(-1)).toMatchObject({step: null});
  });

  it('keeps a step inside its own job and never moves it back, whatever the trainer reports', async () => {
    // Past 1 it would run the bar into the next job's share.
    const {runBenchmark} = await freshModule();
    const states: RunState[] = [];

    runBenchmark(config, (state) => states.push(state));
    const [worker] = FakeWorker.instances;
    worker.emit('message', {data: {type: 'started', totalJobs: 2, rejected: []}});

    worker.emit('message', {data: {type: 'step', fraction: -0.2}});
    expect(states.at(-1)).toMatchObject({step: 0});
    worker.emit('message', {data: {type: 'step', fraction: 0.4}});
    worker.emit('message', {data: {type: 'step', fraction: Number.NaN}});
    expect(states.at(-1)).toMatchObject({step: 0.4});
    // Nor back: a trainer whose next phase starts again from 0 would move the
    // bar backwards.
    worker.emit('message', {data: {type: 'step', fraction: 0.1}});
    expect(states.at(-1)).toMatchObject({step: 0.4});
    worker.emit('message', {data: {type: 'step', fraction: 1.7}});
    expect(states.at(-1)).toMatchObject({step: 1});
  });

  it('keeps one worker across runs that end normally', async () => {
    // Instantiating the module costs megabytes, so it outlives a single run.
    const {runBenchmark} = await freshModule();
    const states: RunState[] = [];

    runBenchmark(config, (state) => states.push(state));
    FakeWorker.instances[0].emit('message', {data: {type: 'finished'}});
    runBenchmark(config, (state) => states.push(state));

    expect(FakeWorker.instances).toHaveLength(1);
    expect(FakeWorker.instances[0].posted).toHaveLength(2);
  });
});
