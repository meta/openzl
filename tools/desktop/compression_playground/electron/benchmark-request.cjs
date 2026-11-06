// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * @import {CompressorConfig, DesktopRunConfig, TrainingConfig} from '../../../web/compression_playground/src/benchmarkTypes.ts'
 */

'use strict';

/**
 * @param {unknown} value
 * @returns {value is Record<string, unknown>}
 */
function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * @param {unknown} value
 * @returns {value is number}
 */
function isPositiveInteger(value) {
  return Number.isInteger(value) && Number(value) >= 1;
}

/**
 * @param {unknown} training
 * @returns {TrainingConfig | null}
 */
function trainingConfig(training) {
  if (training === null) {
    return null;
  }
  if (!isObject(training) || training.paretoFrontier !== true || !isPositiveInteger(training.candidates)) {
    throw new TypeError('OpenZL training options are invalid');
  }
  return {paretoFrontier: true, candidates: training.candidates};
}

/**
 * @param {unknown} row
 * @returns {CompressorConfig}
 */
function compressorConfig(row) {
  if (!isObject(row) || !Number.isInteger(row.rowId)) {
    throw new TypeError('Invalid compressor row');
  }
  const rowId = Number(row.rowId);
  const {compressor, levels} = row;
  if (compressor !== 'OpenZL' && compressor !== 'zstd' && compressor !== 'gzip') {
    throw new TypeError('Unknown compressor');
  }
  if (!Array.isArray(levels) || !levels.every((level) => Number.isInteger(level))) {
    throw new TypeError(`Invalid ${compressor} levels`);
  }
  /** @type {number[]} */
  const copied = [...levels];
  if (compressor !== 'OpenZL') {
    return {rowId, compressor, levels: copied};
  }
  if (typeof row.profile !== 'string' || row.profile === '') {
    throw new TypeError('OpenZL profile is missing');
  }
  return {rowId, compressor, levels: copied, profile: row.profile, training: trainingConfig(row.training)};
}

/**
 * Returns a copy of `request` holding only the fields `DesktopRunConfig` has,
 * so nothing else reaches the worker. Throws a TypeError naming the first
 * field that is not shaped as it should be. Only the shape is checked: which
 * levels, candidate counts and so on `zli` and zlib accept is theirs to say,
 * and a job they refuse is reported as a failure like any other.
 *
 * @param {unknown} request
 * @returns {DesktopRunConfig}
 */
function validateRequest(request) {
  if (!isObject(request)) {
    throw new TypeError('Invalid benchmark request');
  }
  if (!isPositiveInteger(request.iterations)) {
    throw new TypeError('Iterations must be a positive integer');
  }
  if (!Array.isArray(request.compressors)) {
    throw new TypeError('Invalid compressor list');
  }
  const compressors = request.compressors.map(compressorConfig);
  const {input} = request;
  if (!isObject(input)) {
    throw new TypeError('Benchmark input is missing');
  }
  if (typeof input.path !== 'string' || input.path === '' || input.path.includes('\0')) {
    throw new TypeError('Invalid input path');
  }
  return {input: {path: input.path}, compressors, iterations: request.iterations};
}

module.exports = {validateRequest};
