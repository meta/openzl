// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {ProfileMetadata} from '../../../wasm/js/wasm_api.js';
import type {RunConfig, RunMessage} from './benchmarkTypes.ts';
import {setWasmProfiles} from './wasmProfiles.ts';

/**
 * A profile the app's `zli list-profiles` lists, with what the app can do with
 * it, from `electron/profiles.cjs`.
 */
export interface DesktopProfile {
  readonly name: string;
  readonly description: string;
  readonly runnable: boolean;
  readonly trainable: boolean;
  /** Why the app cannot run the profile, when it cannot. */
  readonly unavailableReason?: string;
}

export interface BenchmarkRun {
  /** Settles once the run ends: resolves when it finished, rejects when it failed or was cancelled. */
  done: Promise<void>;
  /** Stops the run and its native processes; `done` then rejects. */
  cancel(): void;
}

/** What `electron/preload.cjs` exposes on `window`. */
export interface OpenZLDesktopBridge {
  /**
   * Runs `config` with the app's `zli`, passing `onMessage` each of the run's
   * messages as it arrives: the browser worker's own, from `loading` to
   * `finished` or `failed`.
   */
  startBenchmark(config: DesktopRunConfig, onMessage?: (message: RunMessage) => void): BenchmarkRun;
  getPathForFile(file: File): string;
  /** The profiles the app's `zli list-profiles` reports, and what the app can do with each. */
  listProfiles(): Promise<readonly DesktopProfile[]>;
}

declare global {
  interface Window {
    openzlDesktop?: OpenZLDesktopBridge;
  }
}

let desktopProfiles: Promise<readonly ProfileMetadata[]> | undefined;

/**
 * The app's profiles in the form the page's profile store takes, which names
 * what a build can run after the browser's WASM build.
 */
export function desktopProfileMetadata(profiles: readonly DesktopProfile[]): readonly ProfileMetadata[] {
  return profiles.map(({name, description, runnable, trainable, unavailableReason}) => ({
    name,
    description,
    isWasmSupported: runnable,
    isTrainable: trainable,
    ...(unavailableReason === undefined ? {} : {wasmUnsupportedReason: unavailableReason}),
  }));
}

/**
 * The app's profiles, from its `zli list-profiles`, loaded into the profile
 * store once. A failed lookup is tried again next time.
 */
export function loadDesktopProfiles(bridge: OpenZLDesktopBridge): Promise<readonly ProfileMetadata[]> {
  desktopProfiles ??= bridge.listProfiles().then(
    (listed) => {
      const profiles = desktopProfileMetadata(listed);
      setWasmProfiles(profiles);
      return profiles;
    },
    (error: unknown) => {
      desktopProfiles = undefined;
      throw error;
    },
  );
  return desktopProfiles;
}

/**
 * The sample the desktop app ships with. It stays on disk for the backend to
 * read, so the page holds only its name and size.
 */
export interface BundledSample {
  readonly kind: 'bundled-sample';
  readonly name: string;
  readonly size: number;
}

/**
 * A run as the desktop app's backend takes it. The backend runs the native CLI,
 * which reads the input where it is on disk, so the input is a path rather than
 * the file itself; everything else is the browser's `RunConfig`.
 */
export interface DesktopRunConfig extends Omit<RunConfig, 'input'> {
  readonly input: {readonly path: string} | BundledSample;
}

/** The desktop app's window of trained candidates: `zli train` rejects fewer than 10. */
export const DESKTOP_TRAINED_CANDIDATE_COUNTS: readonly number[] = Array.from({length: 16}, (_, index) => 10 + index);
export const DESKTOP_DEFAULT_TRAINED_CANDIDATES = 10;

/**
 * A `RunConfig` for the desktop backend, which can also run the bundled
 * sample. Only desktop code sees the sample as an input; the browser's
 * `RunConfig` is always a `File`.
 */
export interface DesktopRunRequest extends Omit<RunConfig, 'input'> {
  readonly input: File | BundledSample;
}

/**
 * Names the input for the backend, which reads it from disk. A file the user
 * picked or dropped always has a path in Electron; one made in the page does
 * not, and the backend has no way to read it. The bundled sample goes as it
 * is, and the main process supplies its path.
 */
export function inputForFile(
  input: DesktopRunRequest['input'],
  bridge: OpenZLDesktopBridge,
): DesktopRunConfig['input'] {
  if ('kind' in input) {
    return input;
  }
  const path = bridge.getPathForFile(input);
  if (path === '') {
    throw new Error(`${input.name} is not a file on disk`);
  }
  return {path};
}
