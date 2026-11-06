// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {CompressorConfig, DesktopRunConfig, RunConfig, RunMessage, RunState} from './benchmarkTypes.ts';
import {inputForFile, type OpenZLDesktopBridge} from './desktop.ts';
import {trackRun} from './runTracker.ts';
import {isTrainableProfile} from './wasmProfiles.ts';

/**
 * Runs a benchmark through the desktop app's `zli`. The backend takes the
 * browser's `RunConfig`, with the input named by its path, and sends back the
 * browser worker's own messages, so the same `RunTracker` folds them into
 * `RunState`, and everything past the Run button -- progress, table, charts --
 * is one code path.
 */

let inFlight = false;

/**
 * As `buildJobs` does for the browser: a profile that cannot train runs
 * untrained, without discarding the user's request, so switching back to a
 * trainable profile restores their setting.
 */
function forDesktop(compressor: CompressorConfig): CompressorConfig {
  if (compressor.compressor !== 'OpenZL' || isTrainableProfile(compressor.profile)) {
    return compressor;
  }
  return {...compressor, training: null};
}

export function runDesktopBenchmark(
  config: RunConfig,
  bridge: OpenZLDesktopBridge,
  onState: (state: RunState) => void,
): void {
  // The Run button disables off the state below, which normally keeps a second
  // run from getting here; this keeps any caller that does from starting two
  // `zli` runs that would skew each other's timings.
  if (inFlight) {
    return;
  }
  inFlight = true;

  // Before anything else: the button has to disable now, not when the backend
  // gets round to starting.
  onState({status: 'loading'});

  const tracker = trackRun();
  let finished = false;

  // Once only: `failed` arrives as a message and then as `done` rejecting, and
  // the second must not replace the first.
  const finish = (state: RunState) => {
    if (finished) {
      return;
    }
    finished = true;
    inFlight = false;
    onState(state);
  };

  function onMessage(message: RunMessage) {
    if (finished) {
      return;
    }
    const state = tracker.apply(message);
    if (message.type === 'finished' || message.type === 'failed') {
      finish(state);
    } else {
      onState(state);
    }
  }

  // Everything that can throw is inside the chain rather than called here: a
  // throw out of this function would leave the page on `loading` with every
  // later run turned away. `getPathForFile` throws for anything it cannot name
  // a path for.
  Promise.resolve()
    .then(() => {
      const desktopConfig: DesktopRunConfig = {
        ...config,
        input: inputForFile(config.input, bridge),
        compressors: config.compressors.map(forDesktop),
      };
      return bridge.startBenchmark(desktopConfig, onMessage).done;
    })
    .catch((error: unknown) => {
      // Cancelled, or the backend could not start: the run's own `failed`, if
      // it sent one, has already ended it.
      finish(tracker.error(error instanceof Error ? error.message : String(error)));
    });
}
