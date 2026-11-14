// Copyright (c) Meta Platforms, Inc. and affiliates.

import {defineConfig} from 'vitest/config';

// The Electron code runs under Node, so the tests do too.
export default defineConfig({
  test: {environment: 'node'},
});
