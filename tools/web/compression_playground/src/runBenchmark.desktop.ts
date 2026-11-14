// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {ProfileMetadata} from '../../../wasm/js/wasm_api.js';
import type {RunConfig, RunState} from './benchmarkTypes.ts';
import {desktopBridge, loadDesktopProfiles} from './desktop.ts';

/**
 * Replaces `runBenchmark.ts` in the desktop build (see vite.config.ts). The
 * desktop app runs `zli` and never loads the WASM module, so its bundle leaves
 * out the worker and builds without the Emscripten output.
 */
const UNAVAILABLE = 'The desktop backend is unavailable, and this build has no browser backend';

export function runBenchmark(_config: RunConfig, onState: (state: RunState) => void): void {
  onState({status: 'error', message: UNAVAILABLE, results: [], failures: []});
}

/** The app's profiles, from its `zli`; this build has no WASM module to ask. */
export function getBenchmarkProfiles(): Promise<readonly ProfileMetadata[]> {
  const bridge = desktopBridge();
  return bridge === undefined ? Promise.reject(new Error(UNAVAILABLE)) : loadDesktopProfiles(bridge);
}
