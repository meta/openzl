// Copyright (c) Meta Platforms, Inc. and affiliates.

import {createRequire} from 'node:module';
import {runInNewContext} from 'node:vm';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

// Importing the CLI harness must not launch Electron or require a native build.
const {entryPoint, runWithTimeout, waitForStarted} = createRequire(import.meta.url)(
  '../electron/tests/cancellation-smoke.cjs',
) as {
  entryPoint(isElectron: boolean, isMainModule: boolean): 'harness' | 'driver' | 'none';
  runWithTimeout(
    command: string,
    args: string[],
    timeoutMs: number,
  ): Promise<{code: number | null; stdout: string; stderr: string; timedOut: boolean}>;
  waitForStarted(started: Promise<void>, done: Promise<void>, timeoutMs?: number): Promise<void>;
};

function deferred() {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<void>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return {promise, resolve, reject};
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('cancellation smoke renderer-start acknowledgement', () => {
  it('waits for renderer delivery even after the native process is observable', async () => {
    const started = deferred();
    const done = deferred();
    let acknowledged = false;
    const waiting = waitForStarted(started.promise, done.promise).then(() => {
      acknowledged = true;
    });

    await vi.advanceTimersByTimeAsync(100);
    expect(acknowledged).toBe(false);
    started.resolve();
    await waiting;
    expect(acknowledged).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('preserves an early run failure instead of waiting for the timeout', async () => {
    const started = deferred();
    const done = deferred();
    const error = new Error('The benchmark process exited unexpectedly');
    const waiting = expect(waitForStarted(started.promise, done.promise)).rejects.toBe(error);

    done.reject(error);
    await waiting;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('rejects a run that finishes before acknowledging its start', async () => {
    const started = deferred();
    const done = deferred();
    const waiting = expect(waitForStarted(started.promise, done.promise)).rejects.toThrow(
      'The benchmark finished before the renderer received started',
    );

    done.resolve();
    await waiting;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('bounds a missing acknowledgement and clears its timeout', async () => {
    const started = deferred();
    const done = deferred();
    const waiting = expect(waitForStarted(started.promise, done.promise, 250)).rejects.toThrow(
      'The renderer never received benchmark started',
    );

    await vi.advanceTimersByTimeAsync(250);
    await waiting;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('runs the serialized helper with only browser globals', async () => {
    const rendererWait = runInNewContext(`(${waitForStarted.toString()})`, {
      setTimeout,
      clearTimeout,
    }) as typeof waitForStarted;
    const started = deferred();
    const done = deferred();
    const waiting = rendererWait(started.promise, done.promise, 250);

    await vi.advanceTimersByTimeAsync(100);
    started.resolve();
    await waiting;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('accepts an acknowledgement that has already arrived', async () => {
    await expect(waitForStarted(Promise.resolve(), deferred().promise)).resolves.toBeUndefined();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('cancellation smoke entry point', () => {
  it('runs the harness under Electron even when this file is not the main module', () => {
    expect(entryPoint(true, true)).toBe('harness');
    expect(entryPoint(true, false)).toBe('harness');
  });

  it('starts the driver only when run directly under Node', () => {
    expect(entryPoint(false, true)).toBe('driver');
    expect(entryPoint(false, false)).toBe('none');
  });
});

describe('cancellation smoke process timeout', () => {
  beforeEach(() => vi.useRealTimers());

  it('kills a process that never exits instead of waiting for it', async () => {
    const started = Date.now();
    const outcome = await runWithTimeout(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], 300);

    expect(outcome.timedOut).toBe(true);
    expect(outcome.code).toBeNull();
    expect(Date.now() - started).toBeLessThan(10_000);
  });

  it('returns the output of a process that exits in time', async () => {
    const outcome = await runWithTimeout(
      process.execPath,
      ['-e', 'process.stdout.write("out"); process.stderr.write("err"); process.exitCode = 3'],
      30_000,
    );

    expect(outcome).toEqual({code: 3, stdout: 'out', stderr: 'err', timedOut: false});
  });
});
