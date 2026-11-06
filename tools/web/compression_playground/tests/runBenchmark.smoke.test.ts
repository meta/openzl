// Copyright (c) Meta Platforms, Inc. and affiliates.
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';
import {measureRun} from '../src/measureRun.ts';
import {toCompressorConfig} from '../src/benchmarkTypes.ts';
import {createCompressorRow} from '../src/compressors.ts';
import type {WorkerMessage} from '../src/benchmarkTypes.ts';

/**
 * The only cases here that need the real module, and `openzl.js` / `openzl.wasm`
 * are build artifacts rather than checked-in files, so CI has neither. Building
 * them is in this diff's test plan; without them these skip and the rest of the
 * suite, which runs against stubs, still means something.
 */
const hasModule = ['openzl.js', 'openzl.wasm'].every((name) =>
  existsSync(fileURLToPath(new URL(`../../../wasm/js/${name}`, import.meta.url))),
);

describe.skipIf(!hasModule)('runBenchmark end to end', () => {
  it('measures every job against a real module', async () => {
    const bytes = new TextEncoder().encode('int main(void) { return 0; }\n'.repeat(4000));
    const openzl = createCompressorRow(1, 'OpenZL');
    openzl.trainRequested = false;
    const messages: WorkerMessage[] = [];
    await measureRun(
      {
        input: new File([bytes], 'sample.c'),
        compressors: [openzl, createCompressorRow(2, 'zstd'), createCompressorRow(3, 'gzip')].map(toCompressorConfig),
        iterations: 3,
      },
      (message) => messages.push(message),
    );
    const results = messages.flatMap((m) => (m.type === 'result' ? [m.result] : []));
    const failures = messages.flatMap((m) => (m.type === 'failure' ? [m.failure] : []));
    expect(messages[messages.length - 1].type).toBe('finished');
    // Asserted rather than printed: a failure here is the thing that would
    // make the count below wrong, so it says so itself.
    expect(failures).toEqual([]);
    expect(results).toHaveLength(13);
  }, 60000);
});

describe.skipIf(!hasModule)('measureRun with training', () => {
  it('expands one trained job into a ranked candidate per result', async () => {
    // Generated rather than read from disk: the samples in public/ are local
    // scratch, and the shape is what matters -- climbing timestamps with a few
    // sensor columns, which is what the numeric profiles exist for.
    const values = new Uint32Array(100_000);
    let stamp = 1_700_000_000;
    for (let index = 0; index < values.length; index += 4) {
      stamp += 1 + (index % 3);
      values[index] = stamp;
      values[index + 1] = 20 + (index % 5);
      values[index + 2] = 1000 + (index % 7);
      values[index + 3] = index % 4 === 0 ? 1 : 0;
    }
    const bytes = new Uint8Array(values.buffer);
    const openzl = createCompressorRow(1, 'OpenZL');
    openzl.profile = 'le-u32';
    openzl.trainRequested = true;
    openzl.candidates = 6;

    const messages: WorkerMessage[] = [];
    await measureRun(
      {
        input: new File([bytes], 'sample.u32'),
        // Built here rather than through `toCompressorConfig` so the budget can
        // be small: the trainer spends whatever it is given, so the default
        // would make this a minute-long test.
        compressors: [
          {
            rowId: 1,
            compressor: 'OpenZL',
            levels: [6],
            profile: 'le-u32',
            profileArg: '',
            training: {paretoFrontier: true, candidates: 6, maxTimeSecs: 3},
          },
        ],
        iterations: 1,
      },
      (message) => messages.push(message),
    );

    const started = messages.find((m) => m.type === 'started');
    const results = messages.flatMap((m) => (m.type === 'result' ? [m.result] : []));
    // One job, many measurements: the thing the protocol had to allow for.
    expect(started).toMatchObject({totalJobs: 1});
    expect(results.length).toBeGreaterThan(1);
    expect(results.map((r) => r.candidate?.index)).toEqual(results.map((_, i) => i + 1));
    expect(new Set(results.map((r) => r.candidate?.total))).toEqual(new Set([results.length]));
    // The binding hands the frontier back best ratio first, which is the order
    // the table ranks by.
    const ratios = results.map((r) => r.ratio);
    expect([...ratios].sort((a, b) => b - a)).toEqual(ratios);
  }, 300000);
});
