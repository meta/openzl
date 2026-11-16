// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment node

import fs from 'node:fs';
import {describe, expect, it} from 'vitest';
import packageJson from '../package.json';
import {validateRequest} from '../electron/benchmark-request.cjs';
import {SAMPLE_NAME, SAMPLE_SIZE, SAMPLE_SOURCE_PATH, asksForSample} from '../electron/sample.cjs';

describe('the bundled sample', () => {
  it('is the file the shared constants name, at the size they give', () => {
    // The page shows the name and size without reading the file, so they must not drift.
    expect(fs.statSync(SAMPLE_SOURCE_PATH).size).toBe(SAMPLE_SIZE);
  });

  it('is installed under the name the main process looks for', () => {
    const resource = packageJson.build.extraResources.find((entry) => entry.to === SAMPLE_NAME);
    expect(resource).toBeDefined();
    expect(fs.realpathSync(new URL(`../${resource?.from}`, import.meta.url))).toBe(fs.realpathSync(SAMPLE_SOURCE_PATH));
  });
});

describe('asksForSample', () => {
  const request = (input: object) =>
    validateRequest({input, iterations: 1, compressors: [{rowId: 1, compressor: 'gzip', levels: [1]}]}).input;

  it('is true only for the bundled sample', () => {
    expect(asksForSample(request({kind: 'bundled-sample', name: SAMPLE_NAME, size: SAMPLE_SIZE}))).toBe(true);
    expect(asksForSample(request({path: '/data/input.bin'}))).toBe(false);
  });

  it('is false for a path that also carries a kind, which validation strips', () => {
    expect(asksForSample(request({path: '/data/input.bin', kind: 'bundled-sample'}))).toBe(false);
  });
});
