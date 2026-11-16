// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// Builds what the smoke tests need, once, and runs them, the same way on every
// platform: `yarn test:smoke` runs them all, `yarn test:smoke cancellation`
// just that one. On Linux with no display, the ones that open a window run
// under xvfb-run.

const fs = require('node:fs');
const path = require('node:path');
const {ensureChromiumSandbox, getPlatform, run, yarn} = require('../../setup.cjs');

const ROOT = path.join(__dirname, '..', '..');

/**
 * Each test, what it needs built first, and whether it opens a window.
 * window-security and ui run inside Electron; the others run under Node and
 * launch Electron, or the packaged app, themselves.
 */
const SMOKE_TESTS = {
  'native-training': {needs: ['native'], window: false, electron: false},
  'window-security': {needs: ['native', 'renderer'], window: true, electron: true},
  ui: {needs: ['native', 'renderer'], window: true, electron: true},
  cancellation: {needs: ['native', 'renderer'], window: true, electron: false},
  'packaged-app': {needs: ['native', 'renderer', 'package'], window: true, electron: false},
};

const BUILDS = {
  native: () => yarn(['build:native']),
  renderer: () => {
    yarn(['build:renderer']);
    // The desktop build leaves the browser's WASM worker out; the window
    // security and UI tests then check the page works without it.
    const bundled = fs.readdirSync(path.join(ROOT, 'dist'), {recursive: true, encoding: 'utf8'});
    const wasm = bundled.filter((file) => file.endsWith('.wasm') || path.basename(file) === 'openzl.js');
    if (wasm.length > 0) {
      throw new Error(`The desktop renderer bundles the WASM module: ${wasm.join(', ')}`);
    }
  },
  package: () => {
    // Unsigned: a test build has no signing identity to use, and an ad hoc
    // signature is enough for macOS to launch it.
    yarn(['electron-builder', '--dir', '--publish', 'never', '-c.mac.identity=null']);
    if (getPlatform() === 'linux') {
      ensureChromiumSandbox(path.join(ROOT, 'release', 'linux-unpacked'));
    }
  },
};

/** The command that gives `command` a display, if it needs one and there is none. */
function withDisplay(command) {
  if (getPlatform() !== 'linux' || process.env.DISPLAY || process.env.WAYLAND_DISPLAY) {
    return command;
  }
  return ['xvfb-run', '--auto-servernum', ...command];
}

function main(names) {
  const unknown = names.filter((name) => !(name in SMOKE_TESTS));
  if (unknown.length > 0) {
    throw new Error(`Unknown smoke tests: ${unknown.join(', ')}. Choose from ${Object.keys(SMOKE_TESTS).join(', ')}.`);
  }
  const selected = names.length > 0 ? names : Object.keys(SMOKE_TESTS);
  // In the order BUILDS lists them, since packaging needs the other two.
  const needed = new Set(selected.flatMap((name) => SMOKE_TESTS[name].needs));
  for (const [build, step] of Object.entries(BUILDS)) {
    if (needed.has(build)) {
      step();
    }
  }
  for (const name of selected) {
    const test = SMOKE_TESTS[name];
    // Under Node, the electron package exports the path to its binary.
    const runner = test.electron ? require('electron') : process.execPath;
    const command = [runner, path.join(__dirname, `${name}-smoke.cjs`)];
    const [executable, ...args] = test.window ? withDisplay(command) : command;
    run(executable, args);
  }
}

try {
  main(process.argv.slice(2));
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
