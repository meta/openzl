// Copyright (c) Meta Platforms, Inc. and affiliates.

import {createRequire} from 'node:module';
import {runInNewContext} from 'node:vm';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

// Importing the CLI harness must not launch Electron or require a native build.
const {waitForStarted} = createRequire(import.meta.url)('../electron/tests/cancellation-smoke.cjs') as {
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
