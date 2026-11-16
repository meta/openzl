// Copyright (c) Meta Platforms, Inc. and affiliates.

import {
  isRunInProgress,
  type RunConfig,
  type RunState,
  type WorkerMessage,
  type WorkerRequest,
} from './benchmarkTypes.ts';
import type {ProfileMetadata} from '../../../wasm/js/wasm_api.js';
import {setWasmProfiles} from './wasmProfiles.ts';
import {trackRun} from './runTracker.ts';

/**
 * Drives a run on a worker and folds what comes back into `RunState`.
 *
 * The worker outlives a run so the WASM module is loaded once rather than per
 * press of Run. `cancelRun` is the exception: a synchronous train() cannot be
 * interrupted from JS, so stopping one means discarding the thread it is on,
 * and the next run pays for the module again.
 */
let worker: Worker | undefined;
let profilesPromise: Promise<readonly ProfileMetadata[]> | undefined;
let cancelProfiles: ((message: string) => void) | undefined;

function getWorker(): Worker {
  worker ??= new Worker(new URL('./benchmarkWorker.ts', import.meta.url), {type: 'module'});
  return worker;
}

/** Reads profiles once and keeps the metadata on the UI side. */
export async function getBenchmarkProfiles(): Promise<readonly ProfileMetadata[]> {
  if (profilesPromise !== undefined) {
    return profilesPromise;
  }
  const current = getWorker();
  const promise = new Promise<readonly ProfileMetadata[]>((resolveProfiles, rejectProfiles) => {
    function cleanup() {
      current.removeEventListener('message', onMessage);
      current.removeEventListener('error', onError);
      cancelProfiles = undefined;
    }
    function onMessage(event: MessageEvent<WorkerMessage>) {
      if (event.data.type === 'profiles') {
        cleanup();
        setWasmProfiles(event.data.profiles);
        resolveProfiles(event.data.profiles);
      } else if (event.data.type === 'profilesFailed') {
        cancelProfiles?.(event.data.message);
      }
    }
    function onError(event: ErrorEvent) {
      cancelRun(event.message || 'the benchmark worker stopped');
    }
    cancelProfiles = (message) => {
      cleanup();
      profilesPromise = undefined;
      rejectProfiles(new Error(message));
    };
    current.addEventListener('message', onMessage);
    current.addEventListener('error', onError);
  });
  profilesPromise = promise;
  try {
    const request: WorkerRequest = {type: 'profilesRequest'};
    current.postMessage(request);
  } catch (error) {
    cancelProfiles?.(error instanceof Error ? error.message : String(error));
  }
  return promise;
}

/**
 * The run in flight, as one value rather than a worker here and a handful of
 * closures there: whoever ends a run -- itself, an error, or a cancel button
 * one day -- needs the same four things to end it cleanly.
 */
interface ActiveRun {
  readonly worker: Worker;
  readonly onMessage: (event: MessageEvent<WorkerMessage>) => void;
  readonly onError: (event: ErrorEvent) => void;
  readonly onState: (state: RunState) => void;
  /** Ends the run as an error, keeping whatever it had already measured. */
  readonly fail: (message: string) => void;
}

let activeRun: ActiveRun | undefined;

function detach(run: ActiveRun): void {
  run.worker.removeEventListener('message', run.onMessage);
  run.worker.removeEventListener('error', run.onError);
  // Only if it is still the one in flight: a run that ends after another has
  // started owns its own handlers, not the pointer.
  if (activeRun === run) {
    activeRun = undefined;
  }
}

/** Discards the worker, which is the only way to stop a run in progress. */
export function cancelRun(message = 'the run was cancelled'): void {
  cancelProfiles?.(message);
  worker?.terminate();
  worker = undefined;
  // A terminated worker sends nothing more, so the run has to be ended here or
  // the page sits on `running` for good.
  activeRun?.fail(message);
}

export function runBenchmark(config: RunConfig, onState: (state: RunState) => void): void {
  // The run button disabling off `loading` below normally keeps a second run
  // from getting here; this keeps any caller that does from putting two runs
  // on one worker.
  if (activeRun !== undefined) {
    return;
  }

  // Before anything is posted, not when the worker gets round to saying
  // `loading`. The run button disables off this state, and messages carry no
  // run id -- a second press in that gap would put two runs on one worker and
  // both listeners would read both runs' messages.
  onState({status: 'loading'});

  const tracker = trackRun();
  const runWorker = getWorker();
  // `onMessage` and `onError` are function declarations below, so they are
  // already bound here.
  const thisRun: ActiveRun = {
    worker: runWorker,
    onMessage,
    onError,
    onState,
    fail: (message) => {
      finish(tracker.error(message));
    },
  };
  const finish = (state: RunState) => {
    detach(thisRun);
    onState(state);
  };

  function onMessage(event: MessageEvent<WorkerMessage>) {
    const message = event.data;
    // The profile lookup shares the worker; its answers are not part of the run.
    if (message.type === 'profiles' || message.type === 'profilesFailed') {
      return;
    }
    const state = tracker.apply(message);
    if (isRunInProgress(state)) {
      onState(state);
    } else {
      finish(state);
    }
  }

  function onError(event: ErrorEvent) {
    // The worker died rather than reporting: the module failed to start, or
    // something threw where nothing was catching. Discard it -- a dead worker
    // never sends another message, so reusing it would leave the next run
    // waiting forever instead of failing.
    cancelRun(event.message || 'the benchmark worker stopped');
  }

  activeRun = thisRun;
  runWorker.addEventListener('message', onMessage);
  runWorker.addEventListener('error', onError);
  const request: WorkerRequest = {type: 'benchmarkRequest', config};
  runWorker.postMessage(request);
}
