// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it} from 'vitest';
import {desktopProfiles} from '../electron/profiles.cjs';

describe('desktopProfiles', () => {
  const listed = [
    {name: 'serial', description: 'Serial data (aka raw bytes)'},
    {name: 'pytorch', description: 'Pytorch model generated from torch.save(). Training is not supported.'},
    {name: 'sddl', description: 'Pass a path to the data description file with --profile-arg.'},
    {name: 'numeric-ml-selector-64', description: '64 bit numeric data using ml selectors (Placeholder)'},
    // A profile added to zli after the table was written.
    {name: 'brand-new', description: 'Some format the CLI learned yesterday'},
  ];

  it('runs and trains the profiles zli lists as the table says, keeping zli’s descriptions', () => {
    const profiles = desktopProfiles(listed);

    expect(profiles.map(({name, runnable, trainable}) => [name, runnable, trainable])).toEqual([
      ['serial', true, true],
      ['pytorch', true, false],
      ['sddl', false, false],
      ['numeric-ml-selector-64', false, false],
      ['brand-new', false, false],
    ]);
    expect(profiles.map(({description}) => description)).toEqual(listed.map(({description}) => description));
  });

  it('says why it cannot run a profile, and only then', () => {
    const reasons = Object.fromEntries(
      desktopProfiles(listed).map(({name, unavailableReason}) => [name, unavailableReason]),
    );

    expect(reasons).toEqual({
      serial: undefined,
      pytorch: undefined,
      sddl: 'Cannot run in the desktop app yet: needs a description file, which this app cannot pick.',
      'numeric-ml-selector-64': 'Cannot run in the desktop app yet: zli lists it as a placeholder.',
      'brand-new': 'Cannot run in the desktop app yet: this app does not know the profile.',
    });
  });

  it('does not read a profile named after an object property as a known one', () => {
    expect(desktopProfiles([{name: 'constructor', description: ''}])[0]).toMatchObject({runnable: false});
  });
});
