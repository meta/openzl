// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// Run with Electron after `yarn build:renderer`: loads the real main process
// and inspects the window it creates.

const assert = require('node:assert/strict');
const {app, BrowserWindow} = require('electron');

const loadViolations = [];
app.on('web-contents-created', (_event, contents) => {
  contents.on('console-message', ({message}) => {
    if (message.includes('Content Security Policy')) {
      loadViolations.push(message);
    }
  });
});

require('../main.cjs');

const TIMEOUT_MS = 30_000;

async function mainWindow() {
  for (;;) {
    const [window] = BrowserWindow.getAllWindows();
    if (window !== undefined) {
      if (window.webContents.isLoading()) {
        await new Promise((resolve) => window.webContents.once('did-finish-load', resolve));
      }
      return window;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

function rendererBenchmark(window, request) {
  return window.webContents.executeJavaScript(
    `window.openzlDesktop.startBenchmark(${JSON.stringify(request)}).done.then(() => 'resolved', (error) => error.message)`,
  );
}

async function main() {
  const window = await mainWindow();

  const preferences = window.webContents.getLastWebPreferences();
  assert.equal(preferences.contextIsolation, true);
  assert.equal(preferences.nodeIntegration, false);
  assert.equal(preferences.nodeIntegrationInSubFrames, false);
  assert.equal(preferences.sandbox, true);
  assert.equal(preferences.webSecurity, true);

  const renderer = await window.webContents.executeJavaScript(`(async () => ({
    require: typeof require,
    process: typeof process,
    module: typeof module,
    bridge: Object.keys(window.openzlDesktop).sort(),
    bridgeFrozen: Object.isFrozen(window.openzlDesktop),
    origin: location.origin,
    rendered: document.getElementById('root').childElementCount > 0,
    permissions: {
      notifications: await Notification.requestPermission(),
      geolocation: (await navigator.permissions.query({name: 'geolocation'})).state,
      camera: (await navigator.permissions.query({name: 'camera'})).state,
      microphone: (await navigator.permissions.query({name: 'microphone'})).state,
    },
  }))()`);
  assert.deepEqual(renderer, {
    require: 'undefined',
    process: 'undefined',
    module: 'undefined',
    bridge: ['getPathForFile', 'startBenchmark'],
    bridgeFrozen: true,
    origin: 'app://bundle',
    rendered: true,
    permissions: {notifications: 'denied', geolocation: 'denied', camera: 'denied', microphone: 'denied'},
  });

  // Stubbed after startup: main.cjs calls shell.openExternal through the shared module object.
  const {shell} = require('electron');
  const opened = [];
  shell.openExternal = async (url) => {
    opened.push(url);
  };
  await window.webContents.executeJavaScript(
    `window.open('https://example.com/');
    window.open('http://openzl.org/');
    window.open('https://openzl.org.example.com/');
    window.open('https://openzl.org/getting-started/quick-start/');
    location.assign('https://openzl.org/sddl/');`,
    true,
  );
  await new Promise((resolve) => setTimeout(resolve, 500));
  assert.deepEqual(opened, ['https://openzl.org/getting-started/quick-start/', 'https://openzl.org/sddl/']);

  const input = {path: '/nonexistent/sample.bin'};
  const row = {rowId: 1, compressor: 'OpenZL', levels: [6], profile: 'serial'};
  const training = {paretoFrontier: true, candidates: 10};
  const rejections = {
    iterations: await rendererBenchmark(window, {input, iterations: 0, compressors: [{...row, training}]}),
    training: await rendererBenchmark(window, {
      input,
      iterations: 1,
      compressors: [{...row, training: {candidates: 10}}],
    }),
    compressor: await rendererBenchmark(window, {
      input,
      iterations: 1,
      compressors: [{rowId: 1, compressor: 'lz4', levels: [1]}],
    }),
  };
  assert.match(rejections.iterations, /Iterations must be a positive integer/);
  assert.match(rejections.training, /OpenZL training options are invalid/);
  assert.match(rejections.compressor, /Unknown compressor/);

  // Runs after page load, so loadViolations only holds violations from the
  // app itself; the probes below are expected to add their own.
  assert.deepEqual(loadViolations, []);
  const csp = await window.webContents.executeJavaScript(`(async () => {
    const violations = [];
    document.addEventListener('securitypolicyviolation', (event) => violations.push(event.effectiveDirective));
    let evalRan = false;
    try {
      eval('evalRan = true');
    } catch {}
    const inline = document.createElement('script');
    inline.textContent = 'window.inlineRan = true';
    document.head.append(inline);
    await fetch('https://example.invalid/').catch(() => {});
    const wasm = await WebAssembly.compile(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])).then(() => true, () => false);
    const fonts = {
      inter: (await document.fonts.load('600 16px Inter')).length > 0,
      gabarito: (await document.fonts.load('600 16px Gabarito')).length > 0,
    };
    await new Promise((resolve) => setTimeout(resolve, 100));
    return {evalRan, inlineRan: window.inlineRan === true, wasm, fonts, violations: [...new Set(violations)].sort()};
  })()`);
  assert.deepEqual(csp, {
    evalRan: false,
    inlineRan: false,
    wasm: true,
    fonts: {inter: true, gabarito: true},
    violations: ['connect-src', 'script-src-elem', 'script-src'].sort(),
  });

  process.stdout.write(`${JSON.stringify({ok: true, preferences, renderer, rejections, csp}, null, 2)}\n`);
}

app.whenReady().then(() => {
  const timeout = setTimeout(() => {
    process.stderr.write(`Timed out after ${TIMEOUT_MS} ms\n`);
    app.exit(2);
  }, TIMEOUT_MS);
  main().then(
    () => {
      clearTimeout(timeout);
      app.exit(0);
    },
    (error) => {
      clearTimeout(timeout);
      process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
      app.exit(1);
    },
  );
});
