// Copyright (c) Meta Platforms, Inc. and affiliates.

import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';
import pluginPrettier from 'eslint-plugin-prettier/recommended';
import {defineConfig} from 'eslint/config';

export default defineConfig([
  {
    ignores: ['**/node_modules/**', '**/build/**', '**/dist/**', '**/release/**'],
  },
  {
    files: ['**/*.{js,mjs,cjs,ts,mts,cts}'],
    plugins: {js},
    extends: ['js/recommended'],
    languageOptions: {globals: {...globals.browser, ...globals.node}},
  },
  tseslint.configs.strict,
  tseslint.configs.stylistic,
  // prettier needs to be at the bottom
  pluginPrettier,
]);
