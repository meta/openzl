// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

/** @import {DesktopRunConfig} from '../../../web/compression_playground/src/desktop.ts' */

'use strict';

const path = require('node:path');
// Shared with the page, which shows the sample's name and size without reading it.
const {webConstants} = require('./web-constants.cjs');
const sample = /** @type {{path: string; name: string; size: number}} */ (webConstants('bundledSample.json'));

const SAMPLE_NAME = sample.name;
const SAMPLE_SIZE = sample.size;

/**
 * The sample in the source tree, which electron-builder copies into the app's
 * resources. `sample.path` is relative to the OpenZL repository root.
 */
const SAMPLE_SOURCE_PATH = path.resolve(__dirname, '..', '..', '..', '..', sample.path);

/**
 * Whether a validated request's input asks for the bundled sample.
 *
 * @param {DesktopRunConfig['input']} input
 * @returns {input is Extract<DesktopRunConfig['input'], {kind: 'bundled-sample'}>}
 */
function asksForSample(input) {
  return 'kind' in input && input.kind === 'bundled-sample';
}

module.exports = {SAMPLE_NAME, SAMPLE_SIZE, SAMPLE_SOURCE_PATH, asksForSample};
