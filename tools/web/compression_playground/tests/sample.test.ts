// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it} from 'vitest';
import sample from '../src/bundledSample.json';
import {SAMPLE_NAME} from '../src/fetchSample.ts';
import {BUNDLED_SAMPLE} from '../src/sample.ts';

describe('the bundled sample', () => {
  it('is the file the shared constants name, at the size they give', () => {
    expect(BUNDLED_SAMPLE).toEqual({kind: 'bundled-sample', name: sample.name, size: sample.size});
  });

  it('is the file the browser downloads as its sample, under the same name', () => {
    expect(SAMPLE_NAME).toBe(BUNDLED_SAMPLE.name);
  });
});
