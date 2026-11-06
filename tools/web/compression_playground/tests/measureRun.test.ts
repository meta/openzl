// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it, vi} from 'vitest';
import {toCompressorConfig} from '../src/benchmarkTypes.ts';
import {createCompressorRow} from '../src/compressors.ts';
import type {WorkerMessage} from '../src/benchmarkTypes.ts';

const LOAD_FAILURE = 'the OpenZL module could not be instantiated';

// Both, because both load the same module: a build where `createOpenZL` fails
// and the iteration ceiling still answers does not exist.
vi.mock('../../../wasm/js/wasm_api.js', () => ({
  createOpenZL: () => Promise.reject(new Error(LOAD_FAILURE)),
  getOpenZLMaxIterations: () => Promise.reject(new Error(LOAD_FAILURE)),
}));

describe('measureRun when the OpenZL module will not load', () => {
  it('still measures the codecs that do not need it', async () => {
    // The module is loaded once for the whole run rather than per job, so a
    // failure there is easy to let escape the loop and lose gzip and zstd
    // along with OpenZL.
    const {measureRun} = await import('../src/measureRun.ts');
    const openzl = createCompressorRow(1, 'OpenZL');
    openzl.trainRequested = false;
    const messages: WorkerMessage[] = [];

    await measureRun(
      {
        input: new File([new TextEncoder().encode('a'.repeat(4000))], 'sample.txt'),
        compressors: [openzl, createCompressorRow(2, 'gzip')].map(toCompressorConfig),
        iterations: 1,
      },
      (message) => messages.push(message),
    );

    const results = messages.flatMap((m) => (m.type === 'result' ? [m.result] : []));
    const failures = messages.flatMap((m) => (m.type === 'failure' ? [m.failure] : []));
    expect(results.length).toBeGreaterThan(0);
    expect(new Set(results.map((r) => r.job.compressor))).toEqual(new Set(['gzip']));
    expect(new Set(failures.map((f) => f.job.compressor))).toEqual(new Set(['OpenZL']));
    // Carrying the load error keeps the row's message actionable; without it
    // the row reports only that no module was passed to the measurement.
    expect(new Set(failures.map((f) => f.message))).toEqual(new Set([LOAD_FAILURE]));
    expect(messages.at(-1)?.type).toBe('finished');
  });
});

describe('measureRun with the bundled sample', () => {
  it('refuses it, since only the desktop app ships the file', async () => {
    const {measureRun} = await import('../src/measureRun.ts');
    const {BUNDLED_SAMPLE} = await import('../src/sample.ts');

    await expect(
      measureRun(
        {input: BUNDLED_SAMPLE, compressors: [toCompressorConfig(createCompressorRow(1, 'gzip'))], iterations: 1},
        vi.fn(),
      ),
    ).rejects.toThrow('The bundled sample is only available in the desktop app');
  });
});
