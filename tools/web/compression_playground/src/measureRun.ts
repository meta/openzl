// Copyright (c) Meta Platforms, Inc. and affiliates.

import {createOpenZL, type OpenZL} from '../../../wasm/js/wasm_api.js';
import {benchmarkGzip, benchmarkOpenZL, benchmarkZstd} from './benchmark.ts';
import {buildJobs, type BenchmarkJob, type RunConfig, type WorkerMessage} from './benchmarkTypes.ts';
import {setWasmProfiles} from './wasmProfiles.ts';

/** Where a message goes. The worker posts; a test collects. */
type Emit = (message: WorkerMessage) => void;

type Measurement = Extract<WorkerMessage, {type: 'result' | 'failure'}>;

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * One module for the worker's life. Instantiating costs several megabytes and
 * about 70ms, and nothing about it is per-run.
 */
let modulePromise: Promise<OpenZL> | undefined;

/**
 * Where the module's training progress goes. The module outlives a run and the
 * callback is fixed at construction, so it reports to whoever is running now
 * rather than to the run that happened to load it.
 */
let reportStep: ((fraction: number) => void) | null = null;

export function loadOpenZL(): Promise<OpenZL> {
  modulePromise ??= createOpenZL({
    // The callback's message is the trainer's own, and says things like
    // `Training ACE graph 1 / 1`, so only the fraction is carried out.
    onTrainingProgress: (fraction: number) => {
      reportStep?.(fraction);
    },
  })
    .then((openzl) => {
      setWasmProfiles(openzl.profiles);
      return openzl;
    })
    .catch((error: unknown) => {
      modulePromise = undefined;
      throw error;
    });
  return modulePromise;
}

/**
 * Every measurement a job produces. One for anything but training, which
 * returns a compressor per point on its Pareto frontier.
 *
 * Yielded rather than collected: a trained job runs for minutes, so holding its
 * measurements back would leave the page still until the last candidate is
 * benchmarked, and anything thrown part way would take the candidates already
 * measured with it.
 */
async function* measure(job: BenchmarkJob, data: Uint8Array, openzl: OpenZL | null): AsyncGenerator<Measurement> {
  switch (job.compressor) {
    case 'OpenZL': {
      if (openzl === null) {
        throw new Error('the OpenZL module is not loaded');
      }
      const base = openzl.getSerializedCompressor(job.profile, job.level);
      if (job.training === null) {
        yield {
          type: 'result',
          result: {...(await benchmarkOpenZL(openzl, data, base, job.iterations)), job, candidate: null},
        };
        return;
      }
      const frontier = openzl.train(data, base, {
        paretoFrontier: job.training.paretoFrontier,
        maxNumCandidates: job.training.candidates,
        ...(job.training.maxTimeSecs === undefined ? {} : {maxTimeSecs: job.training.maxTimeSecs}),
      });
      for (const [index, compressor] of frontier.entries()) {
        // The binding returns the frontier best ratio first, which is the
        // order the table ranks and labels by.
        const candidate = {index: index + 1, total: frontier.length};
        // Caught per candidate: each is benchmarked by its own call, so one
        // that throws says nothing about the rest of the frontier.
        let metrics;
        try {
          metrics = await benchmarkOpenZL(openzl, data, compressor, job.iterations);
        } catch (error) {
          yield {type: 'failure', failure: {job, candidate, message: messageOf(error)}};
          continue;
        }
        yield {type: 'result', result: {...metrics, job, candidate}};
      }
      return;
    }
    case 'zstd':
      yield {type: 'result', result: {...(await benchmarkZstd(data, job.level, job.iterations)), job, candidate: null}};
      return;
    case 'gzip':
      yield {type: 'result', result: {...(await benchmarkGzip(data, job.level, job.iterations)), job, candidate: null}};
      return;
  }
}

export async function measureRun(config: RunConfig, post: Emit): Promise<void> {
  post({type: 'loading'});
  const data = new Uint8Array(await config.input.arrayBuffer());
  // Awaited here so the cost is paid once, but a failure is carried into the
  // loop rather than thrown out of the run: zstd and gzip need nothing from
  // this module, and the whole point of the per-job catch below is that one
  // compressor going down does not take the others with it.
  let openzl: OpenZL | null = null;
  let openzlError: unknown;
  if (config.compressors.some((compressor) => compressor.compressor === 'OpenZL' && compressor.levels.length > 0)) {
    try {
      openzl = await loadOpenZL();
    } catch (error) {
      openzlError = error;
    }
  }
  // buildJobs depends on the profile metadata loaded from wasm
  const {jobs, rejected} = buildJobs(config);
  post({type: 'started', totalJobs: jobs.length, rejected});

  // `train()` is synchronous, so these are posted from inside the call that
  // blocks this thread. They reach the page because `postMessage` queues on
  // the receiving side rather than needing this one to yield.
  reportStep = (fraction) => {
    post({type: 'step', fraction});
  };
  try {
    await runJobs(jobs, data, openzl, openzlError, post);
  } finally {
    reportStep = null;
  }

  post({type: 'finished'});
}

/**
 * One job at a time, deliberately. These are speed measurements taken with a
 * wall clock, so two of them running at once contend for cores and memory
 * bandwidth and both come back low -- by an amount that depends on what else
 * happened to be running. Compression is bandwidth-bound, so this is not a
 * small effect, and the numbers stop being reproducible.
 *
 * What it costs is bounded: on a 5 MB input the twelve level jobs come to
 * about 11 seconds in total, against minutes for one training job, which is
 * not something running them side by side would help with.
 */
async function runJobs(
  jobs: readonly BenchmarkJob[],
  data: Uint8Array,
  openzl: OpenZL | null,
  openzlError: unknown,
  post: Emit,
): Promise<void> {
  for (const job of jobs) {
    try {
      if (job.compressor === 'OpenZL' && openzlError !== undefined) {
        throw openzlError;
      }
      for await (const measurement of measure(job, data, openzl)) {
        post(measurement);
      }
    } catch (error) {
      // Reported and stepped over: one compressor that cannot be measured
      // should not cost the others their results.
      post({type: 'failure', failure: {job, candidate: null, message: messageOf(error)}});
    }
  }
}
