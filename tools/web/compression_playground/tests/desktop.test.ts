// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment node

import {describe, expect, it, vi} from 'vitest';
import {
  DESKTOP_DEFAULT_TRAINED_CANDIDATES,
  DESKTOP_TRAINED_CANDIDATE_COUNTS,
  desktopProfileMetadata,
  inputForFile,
  type OpenZLDesktopBridge,
} from '../src/desktop.ts';
import {profileDescription, setWasmProfiles} from '../src/wasmProfiles.ts';
import {BUNDLED_SAMPLE} from '../src/sample.ts';
import {TEST_PROFILES} from './setup.ts';

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

describe('desktopProfileMetadata', () => {
  const profiles = [
    {name: 'serial', description: 'Serial data (aka raw bytes)', runnable: true, trainable: true},
    {name: 'pytorch', description: 'Pytorch model', runnable: true, trainable: false},
    {
      name: 'sddl',
      description: 'SDDL data',
      runnable: false,
      trainable: false,
      unavailableReason: 'Cannot run in the desktop app yet: needs a description file.',
    },
  ];

  it('takes what the app can run and train from the main process', () => {
    expect(desktopProfileMetadata(profiles)).toEqual([
      {name: 'serial', description: 'Serial data (aka raw bytes)', isWasmSupported: true, isTrainable: true},
      {name: 'pytorch', description: 'Pytorch model', isWasmSupported: true, isTrainable: false},
      {
        name: 'sddl',
        description: 'SDDL data',
        isWasmSupported: false,
        isTrainable: false,
        wasmUnsupportedReason: 'Cannot run in the desktop app yet: needs a description file.',
      },
    ]);
  });

  it('explains the ones it cannot run in desktop terms', () => {
    setWasmProfiles(desktopProfileMetadata(profiles));
    try {
      expect(profileDescription('sddl')).toBe(
        'SDDL data. Cannot run in the desktop app yet: needs a description file.',
      );
      expect(profileDescription('serial')).toBe('Serial data (aka raw bytes)');
    } finally {
      setWasmProfiles(TEST_PROFILES);
    }
  });
});

describe('the desktop candidate window', () => {
  it('starts at the 10 candidates zli train takes at least, with the default among them', () => {
    expect(DESKTOP_TRAINED_CANDIDATE_COUNTS).toContain(DESKTOP_DEFAULT_TRAINED_CANDIDATES);
    expect(Math.min(...DESKTOP_TRAINED_CANDIDATE_COUNTS)).toBe(10);
    expect(Math.max(...DESKTOP_TRAINED_CANDIDATE_COUNTS)).toBe(25);
  });
});
