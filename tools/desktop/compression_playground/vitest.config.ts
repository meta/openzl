// Copyright (c) Meta Platforms, Inc. and affiliates.

import {configDefaults, defineConfig} from 'vitest/config';

// These run stand-in binaries written as `#!node` scripts, which Windows
// cannot execute. On Windows, CI's smoke tests drive the real zli and
// zstd-benchmark through the same code instead.
const SPAWNS_SCRIPTS = ['tests/zliClient.test.ts', 'tests/nativeWorker.test.ts'];

// The Electron code runs under Node, so the tests do too.
export default defineConfig({
  test: {
    environment: 'node',
    exclude: [...configDefaults.exclude, ...(process.platform === 'win32' ? SPAWNS_SCRIPTS : [])],
  },
});
