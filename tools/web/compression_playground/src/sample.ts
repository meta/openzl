// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {BundledSample} from './desktop.ts';
import sample from './bundledSample.json';

/**
 * The OpenZL example file the desktop app installs. `bundledSample.json` names
 * it for both the page and the main process; the page only needs its name and
 * size, to name it to the backend.
 */
export const BUNDLED_SAMPLE: BundledSample = {kind: 'bundled-sample', name: sample.name, size: sample.size};
