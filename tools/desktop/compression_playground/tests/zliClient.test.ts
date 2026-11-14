// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment node

import {existsSync, readdirSync, readFileSync, rmSync} from 'node:fs';
import fsPromises from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {measureOpenZL, type ZliChildEvent} from '../electron/zli-client.cjs';
import {fakeZli, HANGING_ZLI, inputFile, openZLJob, removeFakeZlis, UNORDERED_CANDIDATES_ZLI} from './fakeZli.ts';

const TRAINED = {paretoFrontier: true, candidates: 10} as const;
const MEASUREMENT = {iterations: 4, srcSize: 3, compressMs: 1, decompressMs: 1};

afterEach(() => {
  delete process.env.FAKE_ZLI_EOL;
  delete process.env.FAKE_ZLI_PATH_COLUMN;
  delete process.env.FAKE_ZLI_FAIL;
  delete process.env.FAKE_ZLI_INPUTS;
  removeFakeZlis();
});

async function collect<T>(outcomes: AsyncIterable<T>): Promise<T[]> {
  const collected = [];
  for await (const outcome of outcomes) {
    collected.push(outcome);
  }
  return collected;
}

/** A zli whose training prints `stderr` and exits with `exit`, or is killed by a signal. */
function failingTraining(stderr: string, exit: string): string {
  return fakeZli(`process.stderr.write(${JSON.stringify(stderr)});\n${exit};`);
}

describe('measureOpenZL', () => {
  it('reads a CSV written with CRLF line endings and no path column', async () => {
    // On Windows zli writes CRLF, which would otherwise leave a CR on `iters`.
    process.env.FAKE_ZLI_EOL = '\r\n';
    process.env.FAKE_ZLI_PATH_COLUMN = '0';

    const outcomes = await collect(measureOpenZL(fakeZli(), inputFile(), openZLJob()));

    expect(outcomes).toEqual([{candidate: null, measurement: {...MEASUREMENT, compressedSize: 2}}]);
  });

  it('has zli read the input where it is, rather than a copy', async () => {
    const inputPath = inputFile();
    const inputs = join(dirname(inputPath), 'inputs.txt');
    process.env.FAKE_ZLI_INPUTS = inputs;

    await collect(measureOpenZL(fakeZli(UNORDERED_CANDIDATES_ZLI), inputPath, openZLJob(TRAINED)));

    // One training run, then four candidate benchmarks, all of the same file.
    expect(readFileSync(inputs, 'utf8').trim().split('\n')).toEqual(Array(5).fill(inputPath));
  });

  it('reports why zli failed even when its scratch directory cannot be removed', async () => {
    // As on Windows, where a zli that was just killed can still hold a file open.
    const rm = vi.spyOn(fsPromises, 'rm').mockRejectedValue(new Error('EBUSY: resource busy or locked'));
    try {
      await expect(collect(measureOpenZL(join(tmpdir(), 'no-such-zli'), inputFile(), openZLJob()))).rejects.toThrow(
        /ENOENT/,
      );
      expect(rm).toHaveBeenCalled();
    } finally {
      rm.mockRestore();
      for (const [directory] of rm.mock.calls) {
        rmSync(directory, {recursive: true, force: true});
      }
    }
  });

  it('benchmarks every trained candidate in index order, counted from 1, each as soon as it is measured', async () => {
    const outcomes = await collect(measureOpenZL(fakeZli(UNORDERED_CANDIDATES_ZLI), inputFile(), openZLJob(TRAINED)));

    expect(outcomes).toEqual(
      [1, 2, 3, 11].map((compressedSize, index) => ({
        candidate: {index: index + 1, total: 4},
        measurement: {...MEASUREMENT, compressedSize},
      })),
    );
  });

  it('still benchmarks the other candidates when one cannot be', async () => {
    process.env.FAKE_ZLI_FAIL = '1.zc';

    const outcomes = await collect(measureOpenZL(fakeZli(UNORDERED_CANDIDATES_ZLI), inputFile(), openZLJob(TRAINED)));

    expect(outcomes.map((outcome) => outcome.candidate?.index)).toEqual([1, 2, 3, 4]);
    expect(outcomes[1]).toEqual({candidate: {index: 2, total: 4}, error: new Error('Cannot load compressor: 1.zc')});
    expect(outcomes.filter((outcome) => 'measurement' in outcome)).toHaveLength(3);
  });

  it("reports training's progress as zli's progress bar shows it, and nothing else", async () => {
    const trained: number[] = [];
    await collect(
      measureOpenZL(fakeZli(), inputFile(), openZLJob(TRAINED), {onProgress: (fraction) => trained.push(fraction)}),
    );
    expect(trained).toEqual([0, 0.5, 1]);

    const untrained: number[] = [];
    await collect(
      measureOpenZL(fakeZli(), inputFile(), openZLJob(), {onProgress: (fraction) => untrained.push(fraction)}),
    );
    expect(untrained).toEqual([]);
  });

  it('reads training progress across ACE graphs as one fraction', async () => {
    const twoGraphs = fakeZli(`
const fs = require('node:fs');
const args = process.argv.slice(2);
const flag = (name) => args[args.indexOf(name) + 1];
if (args[0] === 'train') {
  for (const line of ['[----] Training ACE graph 1 / 2: ACE progress', '[====] Training ACE graph 1 / 2: ACE progress',
                      '[----] Training ACE graph 2 / 2: ACE progress', '[==--] Training ACE graph 2 / 2: ACE progress']) {
    process.stderr.write(line + '\\n');
  }
  fs.mkdirSync(flag('--output'), {recursive: true});
  fs.writeFileSync(flag('--output') + '/0.zc', '1');
} else {
  fs.writeFileSync(flag('--output-csv'), 'srcSize,compressedSize,compressionRatio,ctimeMs,dtimeMs,iters\\n3,1,3,1,1,1\\n');
}
`);
    const fractions: number[] = [];
    await collect(
      measureOpenZL(twoGraphs, inputFile(), openZLJob(TRAINED), {onProgress: (fraction) => fractions.push(fraction)}),
    );
    // Each graph's bar starts again from empty; the whole only moves forward.
    expect(fractions).toEqual([0, 0.5, 0.5, 0.75]);
  });

  it('reports only the fatal error when training fails, not its progress', async () => {
    const failing = failingTraining(
      'Using all provided training samples, total size 1\n[==--] Training ACE graph 1 / 1: ACE progress\nI/O Exception:\n\tFailed to open input file\n',
      'process.exit(1)',
    );

    await expect(
      collect(measureOpenZL(failing, inputFile(), openZLJob(TRAINED), {onProgress: vi.fn()})),
    ).rejects.toThrow(/^I\/O Exception:\n\tFailed to open input file$/);
  });

  it('reports the signal, not its last INFO line, when zli is killed mid-training', async () => {
    // A crash or an out-of-memory kill leaves no error heading.
    const killed = failingTraining(
      'Using all provided training samples, total size 1\n[==--] Training ACE graph 1 / 1: ACE progress\n',
      "process.kill(process.pid, 'SIGKILL')",
    );

    await expect(
      collect(measureOpenZL(killed, inputFile(), openZLJob(TRAINED), {onProgress: vi.fn()})),
    ).rejects.toThrow('zli exited unexpectedly (signal SIGKILL)');
  });

  it('reports a one-line error zli prints without detail', async () => {
    const failing = failingTraining(
      '[==--] Training ACE graph 1 / 1: ACE progress\nNo command specified\n',
      'process.exit(1)',
    );

    await expect(
      collect(measureOpenZL(failing, inputFile(), openZLJob(TRAINED), {onProgress: vi.fn()})),
    ).rejects.toThrow(/^No command specified$/);
  });

  it('kills a zli that ignores SIGTERM when the run is aborted, and cleans up', async () => {
    const zliPath = fakeZli(HANGING_ZLI);
    const inputPath = inputFile();
    const runDirectories = () => readdirSync(tmpdir()).filter((entry) => entry.startsWith('openzl-playground-'));
    const before = runDirectories();
    const controller = new AbortController();
    const events: ZliChildEvent[] = [];
    const run = collect(
      measureOpenZL(zliPath, inputPath, openZLJob(), {
        signal: controller.signal,
        onChild: (event) => events.push(event),
      }),
    );
    await vi.waitFor(() => expect(events).toHaveLength(1));
    // Aborting before the stand-in ignores SIGTERM would kill it at once.
    await vi.waitFor(() => expect(existsSync(join(dirname(zliPath), 'ready'))).toBe(true));
    const [{pid}] = events;

    const abortedAt = Date.now();
    controller.abort(new Error('Benchmark cancelled'));
    await expect(run).rejects.toThrow('Benchmark cancelled');

    // SIGTERM was ignored, so only the SIGKILL after the grace period ended it.
    expect(Date.now() - abortedAt).toBeGreaterThanOrEqual(1900);
    expect(() => process.kill(pid, 0)).toThrow();
    expect(events).toEqual([
      {pid, state: 'spawned'},
      {pid, state: 'exited'},
    ]);
    expect(runDirectories()).toEqual(before);
  }, 10_000);

  it('does not start zli once the run is aborted', async () => {
    const controller = new AbortController();
    controller.abort(new Error('Benchmark cancelled'));
    const onChild = vi.fn();

    await expect(
      collect(measureOpenZL(fakeZli(HANGING_ZLI), inputFile(), openZLJob(), {signal: controller.signal, onChild})),
    ).rejects.toThrow('Benchmark cancelled');
    expect(onChild).not.toHaveBeenCalled();
  });

  it('does not report a cancel during a candidate as that candidate failing', async () => {
    const controller = new AbortController();
    const outcomes: unknown[] = [];
    const run = (async () => {
      for await (const outcome of measureOpenZL(fakeZli(UNORDERED_CANDIDATES_ZLI), inputFile(), openZLJob(TRAINED), {
        signal: controller.signal,
      })) {
        outcomes.push(outcome);
        controller.abort(new Error('Benchmark cancelled'));
      }
    })();

    await expect(run).rejects.toThrow('Benchmark cancelled');
    expect(outcomes).toHaveLength(1);
  });
});
