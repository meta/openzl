// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {RunConfig, RunState} from './benchmarkTypes.ts';
import {fetchSample} from './fetchSample.ts';
import {runBenchmark} from './runBenchmark.ts';

/** A run's setup without its input, which the sample brings with it. */
export type RunSetup = Omit<RunConfig, 'input'>;

/**
 * What a build changes about the page. Every field is optional: the browser
 * leaves them all out and gets the defaults, and the desktop app fills in its
 * own.
 */
export interface BackendSettings {
  /** The candidate counts a trained row can pick. */
  readonly candidateCounts?: readonly number[];
  /** How many candidates a new trained row starts with. */
  readonly defaultCandidates?: number;
  /** What the profile picker says of a profile this build cannot run. */
  readonly unavailableHeading?: string;
  /** Whether to say that the measurements come from WASM. Defaults to true. */
  readonly showWasmNotice?: boolean;
  /** Whether to check that the browser can run the WASM module. Defaults to true. */
  readonly checkEngineSupport?: boolean;
  /** Where the header's site links point, when they cannot be relative. */
  readonly headerLinks?: {
    readonly homeHref: string;
    readonly navigation: readonly {readonly label: string; readonly href: string}[];
  };
}

/**
 * Where a run happens: the WASM worker in the browser, or `zli` in the desktop
 * app. Both feed the same `RunState`, so the page is written against this once
 * and never asks which it got.
 */
export interface Backend {
  readonly settings: BackendSettings;
  /** Where the profiles come from when it is not the WASM worker. */
  readonly loadProfiles: (() => Promise<unknown>) | undefined;
  run(config: RunConfig, onState: (state: RunState) => void): void;
  /** Gets the sample, and resolves to what runs it. */
  loadSample(): Promise<(setup: RunSetup, onState: (state: RunState) => void) => void>;
}

export const webBackend: Backend = {
  settings: {},
  loadProfiles: undefined,
  run: (config, onState) => {
    runBenchmark(config, onState);
  },
  loadSample: async () => {
    const input = await fetchSample();
    return (setup, onState) => {
      runBenchmark({input, ...setup}, onState);
    };
  },
};

/** Whether this is the desktop app, known without loading any of its code. */
export function isDesktop(): boolean {
  return typeof window !== 'undefined' && window.openzlDesktop !== undefined;
}

/**
 * The desktop app's backend. Its code is imported here, and only here, so the
 * website never loads it.
 */
export async function loadDesktopBackend(): Promise<Backend> {
  const bridge = typeof window === 'undefined' ? undefined : window.openzlDesktop;
  if (bridge === undefined) {
    throw new Error('The desktop bridge is missing');
  }
  const {desktopBackend} = await import('./desktopBackend.ts');
  return desktopBackend(bridge);
}
