// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {ProfileMetadata} from '../../../wasm/js/wasm_api.js';
import type {DesktopRunConfig, RunInput, RunMessage} from './benchmarkTypes.ts';
import {desktopProfileMetadata, setWasmProfiles, type ListedProfile} from './wasmProfiles.ts';

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
  /** The profiles the app's `zli list-profiles` reports, with its descriptions of them. */
  listProfiles(): Promise<readonly ListedProfile[]>;
}

declare global {
  interface Window {
    openzlDesktop?: OpenZLDesktopBridge;
  }
}

export function desktopBridge(): OpenZLDesktopBridge | undefined {
  return typeof window === 'undefined' ? undefined : window.openzlDesktop;
}

let desktopProfiles: Promise<readonly ProfileMetadata[]> | undefined;

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
 * Names the input for the backend, which reads it from disk. A file the user
 * picked or dropped always has a path in Electron; one made in the page does
 * not, and the backend has no way to read it. The bundled sample goes as it
 * is, and the main process supplies its path.
 */
export function inputForFile(input: RunInput, bridge: OpenZLDesktopBridge): DesktopRunConfig['input'] {
  if ('kind' in input) {
    return input;
  }
  const path = bridge.getPathForFile(input);
  if (path === '') {
    throw new Error(`${input.name} is not a file on disk`);
  }
  return {path};
}
