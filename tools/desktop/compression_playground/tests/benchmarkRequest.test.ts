// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it} from 'vitest';
import {validateRequest} from '../electron/benchmark-request.cjs';

function openZLRow(overrides: Record<string, unknown> = {}) {
  return {
    rowId: 1,
    compressor: 'OpenZL',
    levels: [6],
    profile: 'serial',
    training: {paretoFrontier: true, candidates: 10},
    ...overrides,
  };
}

function request(overrides: Record<string, unknown> = {}) {
  return {
    input: {path: '/tmp/sample.bin'},
    iterations: 5,
    compressors: [
      openZLRow(),
      {rowId: 2, compressor: 'zstd', levels: [1, 5]},
      {rowId: 3, compressor: 'gzip', levels: [5]},
    ],
    ...overrides,
  };
}

function rejects(value: unknown, message: string) {
  const call = () => validateRequest(value);
  expect(call).toThrow(TypeError);
  expect(call).toThrow(message);
}

describe('validateRequest', () => {
  it('returns an equal copy of a valid request', () => {
    const valid = request();
    const validated = validateRequest(valid);
    expect(validated).toEqual(valid);
    expect(validated).not.toBe(valid);
  });

  it('keeps only the fields DesktopRunConfig has', () => {
    const validated = validateRequest(
      request({
        extra: true,
        input: {path: '/tmp/sample.bin', data: [1]},
        compressors: [openZLRow({extra: 1, training: {paretoFrontier: true, candidates: 10, extra: 2}})],
      }),
    );
    expect(validated).toEqual(request({compressors: [openZLRow()]}));
  });

  it('sets no limits of its own on levels, candidates, iterations or the number of rows', () => {
    // What zli and zlib accept is theirs to say; they report what they refuse.
    const unusual = [
      request({iterations: 1_000_000}),
      request({compressors: []}),
      request({compressors: Array.from({length: 200}, (_, rowId) => ({rowId, compressor: 'gzip', levels: [1]}))}),
      request({compressors: [openZLRow({levels: [0, 22], training: {paretoFrontier: true, candidates: 500}})]}),
      request({compressors: [{rowId: 1, compressor: 'zstd', levels: [-5, 40]}]}),
      request({compressors: [{rowId: 1, compressor: 'gzip', levels: []}]}),
      request({compressors: [openZLRow({training: null})]}),
    ];
    for (const valid of unusual) {
      expect(validateRequest(valid)).toEqual(valid);
    }
  });

  it('rejects a request that is not an object', () => {
    rejects(null, 'Invalid benchmark request');
    rejects('run', 'Invalid benchmark request');
    rejects([], 'Invalid benchmark request');
  });

  it('rejects iterations that are not a positive integer', () => {
    for (const iterations of [0, -1, 1.5, '5', undefined]) {
      rejects(request({iterations}), 'Iterations must be a positive integer');
    }
  });

  it('rejects a compressor list that is not a list', () => {
    for (const compressors of [undefined, {}, 'zstd']) {
      rejects(request({compressors}), 'Invalid compressor list');
    }
  });

  it('rejects malformed rows and unknown compressors', () => {
    rejects(request({compressors: [null]}), 'Invalid compressor row');
    rejects(request({compressors: [{rowId: '1', compressor: 'gzip', levels: [5]}]}), 'Invalid compressor row');
    rejects(request({compressors: [{rowId: 1, compressor: 'brotli', levels: [5]}]}), 'Unknown compressor');
    rejects(request({compressors: [{rowId: 1, compressor: 'gzip', levels: 5}]}), 'Invalid gzip levels');
    rejects(request({compressors: [{rowId: 1, compressor: 'zstd', levels: [2.5]}]}), 'Invalid zstd levels');
  });

  it('accepts the bundled sample as input, leaving its path to the main process', () => {
    const sample = {kind: 'bundled-sample', name: 'sra0.bin', size: 2071976};
    expect(validateRequest(request({input: sample})).input).toEqual(sample);
    // Whatever else it carries, a request for the sample is not one for a path.
    expect(validateRequest(request({input: {...sample, path: 'examples/sra0.bin'}})).input).toEqual(sample);
  });

  it('rejects invalid OpenZL options', () => {
    rejects(request({compressors: [openZLRow({profile: ''})]}), 'OpenZL profile is missing');
    for (const training of [undefined, true, {candidates: 10}, {paretoFrontier: true, candidates: 0}]) {
      rejects(request({compressors: [openZLRow({training})]}), 'OpenZL training options are invalid');
    }
  });

  it('rejects missing or malformed input', () => {
    const NOT_INPUT = 'Benchmark input must be a file path or the bundled sample';
    for (const input of [null, {}, {path: 42}, {data: [1]}, {kind: 'other-sample', name: 'x', size: 1}]) {
      rejects(request({input}), NOT_INPUT);
    }
    for (const path of ['', 'sample\0.bin']) {
      rejects(request({input: {path}}), 'Invalid input path');
    }
  });
});
