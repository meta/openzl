// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it} from 'vitest';
import {getOpenZlProfiles} from '../src/compressors.ts';
import {
  desktopProfileMetadata,
  isBrowserSupportedProfile,
  isTrainableProfile,
  profileDescription,
  setWasmProfiles,
} from '../src/wasmProfiles.ts';
import {TEST_PROFILES} from './setup.ts';

describe('isBrowserSupportedProfile', () => {
  it('maps exactly the profiles the binding builds', () => {
    // The fixture includes a profile beyond the binding's convenience constants.
    expect(getOpenZlProfiles().filter(isBrowserSupportedProfile)).toEqual([
      'fixed_profile',
      'future_profile',
      'le-u16',
      'le-u32',
      'serial',
    ]);
    expect(isTrainableProfile('fixed_profile')).toBe(false);
  });

  it('lists every profile supplied by the loaded build', () => {
    expect(getOpenZlProfiles()).toHaveLength(11);
    expect(getOpenZlProfiles().filter((p) => !isBrowserSupportedProfile(p))).toHaveLength(6);
  });

  it('says why for every profile that supplies a reason', () => {
    for (const profile of TEST_PROFILES) {
      if ('wasmUnsupportedReason' in profile) {
        expect(profileDescription(profile.name)).toContain(profile.wasmUnsupportedReason);
      }
    }
  });
});

describe('profileDescription', () => {
  it('handles an unknown profile', () => {
    expect(profileDescription('unknown_profile')).toBe('');
  });

  it('handles profiles before metadata has loaded', () => {
    setWasmProfiles([]);
    try {
      expect(profileDescription('serial')).toBe('');
    } finally {
      setWasmProfiles(TEST_PROFILES);
    }
  });

  it.each([
    ['parquet', 'Cannot run in the browser: needs a custom parser graph.'],
    ['be-u32', 'Cannot run in the browser: the binding reads numbers little-endian only.'],
    ['sddl2', 'Cannot run in the browser: needs a description file, which this page cannot supply.'],
  ])('explains why %s is unpickable', (profile, reason) => {
    expect(profileDescription(profile)).toContain(`. ${reason}`);
  });

  it('leaves the description unchanged when no reason is supplied', () => {
    expect(profileDescription('unsupported_profile')).toBe('Profile unavailable in this WASM build');
  });

  it('leaves a usable profile to its own help text', () => {
    expect(profileDescription('serial')).toBe('Serial data (aka raw bytes)');
  });
});

describe('desktopProfileMetadata', () => {
  const listed = [
    {name: 'serial', description: 'Serial data (aka raw bytes)'},
    // A profile added to the CLI after this page was written.
    {name: 'brand-new', description: 'Some format the CLI learned yesterday'},
    {name: 'pytorch', description: 'Pytorch model generated from torch.save(). Training is not supported.'},
    {name: 'sddl3', description: 'Pass a path to the description file with --profile-arg.'},
    {name: 'future-ml', description: 'Numeric data using a learned selector (Placeholder)'},
  ];

  it('runs every profile zli lists but the ones the app cannot supply an input for', () => {
    expect(desktopProfileMetadata(listed).map(({name, isWasmSupported}) => [name, isWasmSupported])).toEqual([
      ['serial', true],
      ['brand-new', true],
      ['pytorch', true],
      ['sddl3', false],
      ['future-ml', false],
    ]);
  });

  it('reads trainability off the CLI’s own description', () => {
    const trainable = Object.fromEntries(
      desktopProfileMetadata(listed).map(({name, isTrainable}) => [name, isTrainable]),
    );
    expect(trainable).toMatchObject({serial: true, 'brand-new': true, pytorch: false});
  });

  it('explains the ones it cannot run in desktop terms', () => {
    setWasmProfiles(desktopProfileMetadata(listed));
    try {
      expect(profileDescription('sddl3')).toMatch(/Cannot run in the desktop app yet: needs a description file/);
      expect(profileDescription('future-ml')).toMatch(
        /Cannot run in the desktop app yet: the CLI lists it as a placeholder/,
      );
      expect(profileDescription('brand-new')).toBe('Some format the CLI learned yesterday');
    } finally {
      setWasmProfiles(TEST_PROFILES);
    }
  });
});
