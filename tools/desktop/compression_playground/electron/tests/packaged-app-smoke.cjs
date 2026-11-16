// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// Run with Node after `desktop:pack`: launches the packaged app and checks the
// fuses, the app:// renderer, and a real benchmark through the bundled zli.

const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');

const RELEASE = path.resolve(__dirname, '..', '..', 'release');
const PRODUCT = 'OpenZL Compression Playground';
const TIMEOUT_MS = 60_000;
const EVALUATE_TIMEOUT_MS = 5 * 60_000;

function packagedExecutable() {
  if (process.platform === 'win32') {
    return path.join(RELEASE, 'win-unpacked', `${PRODUCT}.exe`);
  }
  if (process.platform === 'darwin') {
    const dir = fs.readdirSync(RELEASE).find((entry) => entry.startsWith('mac'));
    assert.ok(dir !== undefined, `No macOS build under ${RELEASE}`);
    return path.join(RELEASE, dir, `${PRODUCT}.app`, 'Contents', 'MacOS', PRODUCT);
  }
  return path.join(RELEASE, 'linux-unpacked', 'openzl-compression-playground');
}

function launch(executable, args, env) {
  const child = spawn(executable, args, {env: {...process.env, ...env}, stdio: ['ignore', 'pipe', 'pipe']});
  const output = {stdout: '', stderr: ''};
  child.stdout.setEncoding('utf8').on('data', (chunk) => (output.stdout += chunk));
  child.stderr.setEncoding('utf8').on('data', (chunk) => (output.stderr += chunk));
  const exited = new Promise((resolve) => child.once('exit', resolve));
  return {child, output, exited};
}

async function waitFor(check, what) {
  const deadline = Date.now() + TIMEOUT_MS;
  for (;;) {
    const value = check();
    if (value) {
      return value;
    }
    assert.ok(Date.now() < deadline, `Timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

function evaluateOnce(page, expression) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(page.webSocketDebuggerUrl);
    // The expression runs real benchmarks, so the limit is generous; without
    // one a reply that never comes would hang the test until CI kills it.
    const timer = setTimeout(() => {
      socket.close();
      reject(new Error(`Runtime.evaluate did not reply within ${EVALUATE_TIMEOUT_MS} ms`));
    }, EVALUATE_TIMEOUT_MS);
    socket.onerror = (event) => {
      clearTimeout(timer);
      reject(event);
    };
    // A no-op once the reply has resolved the promise.
    socket.onclose = () => {
      clearTimeout(timer);
      reject(new Error('The DevTools socket closed before Runtime.evaluate replied'));
    };
    socket.onopen = () =>
      socket.send(
        JSON.stringify({
          id: 1,
          method: 'Runtime.evaluate',
          params: {expression, awaitPromise: true, returnByValue: true},
        }),
      );
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.id === 1) {
        clearTimeout(timer);
        resolve(message);
        socket.close();
      }
    };
  });
}

async function evaluateInPage(port, expression) {
  const page = await waitFor(() => pageTargets.get(port), 'the app page');
  const deadline = Date.now() + TIMEOUT_MS;
  for (;;) {
    const {result, error} = await evaluateOnce(page, expression);
    // The first document is replaced while the app finishes loading.
    if (error?.message === 'Execution context was destroyed.' && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 250));
      continue;
    }
    assert.equal(error, undefined, JSON.stringify(error));
    assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
}

const pageTargets = new Map();

async function pollPageTarget(port) {
  const deadline = Date.now() + TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = targets.find((target) => target.type === 'page' && target.url.startsWith('app://'));
      if (page !== undefined) {
        pageTargets.set(port, page);
        return;
      }
    } catch {
      // The DevTools endpoint is not listening yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  assert.fail(`No app:// page target appeared on DevTools port ${port}`);
}

/**
 * A port the OS says is free. Still a race between this closing and Electron
 * binding it, but a narrow one, unlike a random pick that may already be taken.
 */
function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const {port} = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function main() {
  const executable = packagedExecutable();
  const {getCurrentFuseWire, FuseV1Options} = require('@electron/fuses');
  // The 1.x entry point does not re-export FuseState.
  const {FuseState} = require('@electron/fuses/dist/constants');
  const wire = await getCurrentFuseWire(executable);
  const fuses = Object.fromEntries(
    ['RunAsNode', 'EnableCookieEncryption', 'EnableNodeOptionsEnvironmentVariable', 'EnableNodeCliInspectArguments']
      .concat(['OnlyLoadAppFromAsar', 'GrantFileProtocolExtraPrivileges'])
      .map((name) => [name, wire[FuseV1Options[name]] === FuseState.ENABLE]),
  );
  assert.deepEqual(fuses, {
    RunAsNode: false,
    EnableCookieEncryption: true,
    EnableNodeOptionsEnvironmentVariable: false,
    EnableNodeCliInspectArguments: false,
    OnlyLoadAppFromAsar: true,
    GrantFileProtocolExtraPrivileges: false,
  });

  const workDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'openzl-packaged-'));
  const sample = path.join(workDirectory, 'sample input.txt');
  fs.writeFileSync(sample, 'OpenZL compression playground sample data\n'.repeat(20_000));

  try {
    // With the RunAsNode fuse enabled, this would run as Node, print, and exit.
    const asNode = launch(executable, ['-e', "process.stdout.write('ran-as-node')"], {ELECTRON_RUN_AS_NODE: '1'});
    await new Promise((resolve) => setTimeout(resolve, 3000));
    asNode.child.kill();
    await asNode.exited;
    assert.ok(!asNode.output.stdout.includes('ran-as-node'), 'ELECTRON_RUN_AS_NODE was honoured');

    const port = await freePort();
    const app = launch(executable, [`--remote-debugging-port=${port}`, '--inspect=0'], {});
    try {
      await pollPageTarget(port);
      const page = await evaluateInPage(
        port,
        `(async () => {
          const deadline = Date.now() + 20000;
          while (!(document.getElementById('root')?.childElementCount > 0)) {
            if (Date.now() > deadline) {
              return {notRendered: location.href, html: document.documentElement.outerHTML.slice(0, 500)};
            }
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
          const messages = await window.openzlDesktop.benchmark({
            input: {path: ${JSON.stringify(sample)}},
            iterations: 1,
            compressors: [
              {rowId: 1, compressor: 'OpenZL', levels: [1], profile: 'serial', training: null},
              {rowId: 2, compressor: 'zstd', levels: [3]},
            ],
          });
          const results = messages.filter((message) => message.type === 'result');
          return {
            origin: location.origin,
            events: messages.map((message) => message.type + (message.result ? ' ' + message.result.job.id : '')),
            srcSizes: results.map((message) => message.result.srcSize),
            candidates: results.map((message) => message.result.candidate),
          };
        })()`,
      );
      const size = fs.statSync(sample).size;
      // Neither row trains: one result per job, in order, between the browser's
      // start and finish messages.
      assert.deepEqual(page, {
        origin: 'app://bundle',
        events: ['loading', 'started', 'result 1-OpenZL-1', 'result 2-zstd-3', 'finished'],
        srcSizes: [size, size],
        candidates: [null, null],
      });
      assert.ok(!app.output.stderr.includes('Debugger listening'), '--inspect opened a debugger');
      process.stdout.write(
        `${JSON.stringify({ok: true, executable, fuses, runAsNodeHonoured: false, inspectOpened: false, page}, null, 2)}\n`,
      );
    } finally {
      app.child.kill();
      await app.exited;
    }
  } finally {
    fs.rmSync(workDirectory, {recursive: true, force: true});
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
