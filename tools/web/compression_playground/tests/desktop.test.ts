// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment node

import {describe, expect, it, vi} from 'vitest';
import {inputForFile, type OpenZLDesktopBridge} from '../src/desktop.ts';
import {BUNDLED_SAMPLE} from '../src/sample.ts';

function bridgeWithPath(path: string): OpenZLDesktopBridge {
  return {getPathForFile: () => path, startBenchmark: vi.fn(), listProfiles: vi.fn()};
}

describe('desktop input', () => {
  it('passes a path through when the file has one', () => {
    expect(inputForFile(new File(['x'], 'a.bin'), bridgeWithPath('/tmp/a.bin'))).toEqual({path: '/tmp/a.bin'});
  });

  it('refuses a file with no path, which the backend could not read', () => {
    expect(() => inputForFile(new File(['abc'], 'made-up.bin'), bridgeWithPath(''))).toThrow(
      'made-up.bin is not a file on disk',
    );
  });

  it('takes a path for a file of any size, without reading it', () => {
    const file = {size: 10 * 1024 ** 3} as unknown as File;

    expect(inputForFile(file, bridgeWithPath('/tmp/big.bin'))).toEqual({path: '/tmp/big.bin'});
  });

  it('asks for the bundled sample by name, leaving its path to the main process', () => {
    const getPathForFile = vi.fn(() => '/never');

    expect(inputForFile(BUNDLED_SAMPLE, {getPathForFile, startBenchmark: vi.fn(), listProfiles: vi.fn()})).toEqual(
      BUNDLED_SAMPLE,
    );
    expect(getPathForFile).not.toHaveBeenCalled();
  });
});
