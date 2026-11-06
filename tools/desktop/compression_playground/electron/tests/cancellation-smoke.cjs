// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// Checks that no zli process or scratch directory outlives a run that ends
// early. Run with Node after `desktop:build:renderer` and `desktop:build:native`:
// it relaunches this file under Electron once per scenario. Needs a display and
// `ps`, so Linux or macOS.

const assert = require('node:assert/strict');
const {execFileSync, spawn} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const ZLI = path.resolve(__dirname, '..', 'native', 'build', 'Release', 'zli');
const SCENARIOS = ['cancel', 'busy', 'reload', 'kill-worker', 'quit'];
const GONE_WITHIN_MS = 4000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitUntil(check, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (!check()) {
    if (Date.now() > deadline) {
      return false;
    }
    await sleep(100);
  }
  return true;
}

function zliTrainPids() {
  // Far past the default 1 MB: a busy machine's process list, long command
  // lines included, can outgrow it.
  return execFileSync('ps', ['-eo', 'pid=,args='], {encoding: 'utf8', maxBuffer: 64 * 1024 * 1024})
    .split('\n')
    .map((line) => line.trim().match(/^(\d+)\s+(.*)$/))
    .filter((match) => match !== null && match[2].startsWith(`${ZLI} train`))
    .map((match) => Number(match[1]));
}

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function runDirectories() {
  return fs
    .readdirSync(os.tmpdir())
    .filter((entry) => entry.startsWith('openzl-run-'))
    .sort();
}

async function harness(scenario, sample) {
  const {app, BrowserWindow} = require('electron');
  require('../main.cjs');
  await app.whenReady();
  let window;
  assert.ok(
    await waitUntil(() => (window = BrowserWindow.getAllWindows()[0]) !== undefined, 20_000),
    'The app window never appeared',
  );
  if (window.webContents.isLoading()) {
    await new Promise((resolve) => window.webContents.once('did-finish-load', resolve));
  }

  const request = {
    input: {path: sample},
    iterations: 1,
    compressors: [
      {
        rowId: 1,
        compressor: 'OpenZL',
        levels: [1],
        profile: 'serial',
        training: {paretoFrontier: true, candidates: 10},
      },
    ],
  };
  await window.webContents.executeJavaScript(`
    window.testEvents = [];
    window.testRun = window.openzlDesktop.startBenchmark(${JSON.stringify(request)}, (event) => window.testEvents.push(event));
    window.testOutcome = window.testRun.done.then(() => 'resolved', (error) => error.message);
    undefined;
  `);
  assert.ok(await waitUntil(() => zliTrainPids().length > 0, 30_000), 'zli train never started');
  const pids = zliTrainPids();
  const result = {scenario, pids};

  const endedAt = Date.now();
  if (scenario === 'cancel') {
    // Still training: no candidate has been measured, so only the run's start
    // and training's steps have been reported.
    result.eventsBeforeCancel = await window.webContents.executeJavaScript(
      'window.testEvents.map((event) => event.type)',
    );
    assert.deepEqual(result.eventsBeforeCancel.slice(0, 2), ['loading', 'started']);
    assert.ok(
      result.eventsBeforeCancel.slice(2).every((type) => type === 'step'),
      `reported ${result.eventsBeforeCancel} before any candidate`,
    );
    result.outcome = await window.webContents.executeJavaScript('window.testRun.cancel(); window.testOutcome');
    assert.equal(result.outcome, 'Benchmark cancelled');
  } else if (scenario === 'busy') {
    const outcome = (js) => window.webContents.executeJavaScript(js);
    result.second = await outcome(
      `window.openzlDesktop.startBenchmark(${JSON.stringify(request)}).done.then(() => 'resolved', (error) => error.message)`,
    );
    assert.equal(result.second, 'A benchmark is already running');
    result.outcome = await outcome('window.testRun.cancel(); window.testOutcome');
    assert.equal(result.outcome, 'Benchmark cancelled');
    const gzip = {input: {path: sample}, iterations: 1, compressors: [{rowId: 9, compressor: 'gzip', levels: [1]}]};
    result.afterCancel = await outcome(`(() => {
      const messages = [];
      return window.openzlDesktop
        .startBenchmark(${JSON.stringify(gzip)}, (message) =>
          messages.push(message.type + (message.result ? ' ' + message.result.job.id : '')),
        )
        .done.then(() => messages);
    })()`);
    assert.deepEqual(result.afterCancel, ['loading', 'started', 'result 9-gzip-1', 'finished']);
  } else if (scenario === 'reload') {
    window.webContents.reload();
  } else if (scenario === 'kill-worker') {
    const worker = app.getAppMetrics().find((metric) => metric.name === 'OpenZL benchmark');
    assert.ok(worker !== undefined, 'benchmark utility process not found');
    process.kill(worker.pid, 'SIGKILL');
    result.outcome = await window.webContents.executeJavaScript('window.testOutcome');
    assert.match(result.outcome, /Native benchmark process exited unexpectedly/);
  } else {
    // The Node side checks for survivors once Electron has exited.
    process.stdout.write(`RESULT ${JSON.stringify(result)}\n`);
    app.quit();
    return;
  }
  assert.ok(
    await waitUntil(() => pids.every((pid) => !alive(pid)), GONE_WITHIN_MS),
    `zli ${pids} still running after ${scenario}`,
  );
  result.zliGoneAfterMs = Date.now() - endedAt;
  process.stdout.write(`RESULT ${JSON.stringify(result)}\n`);
  app.exit(0);
}

function runScenario(electron, scenario, sample) {
  return new Promise((resolve) => {
    const child = spawn(electron, [__filename, `--scenario=${scenario}`, `--sample=${sample}`], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8').on('data', (chunk) => (stdout += chunk));
    child.stderr.setEncoding('utf8').on('data', (chunk) => (stderr += chunk));
    child.once('exit', (code) => {
      const line = stdout.split('\n').find((candidate) => candidate.startsWith('RESULT '));
      resolve({code, result: line === undefined ? undefined : JSON.parse(line.slice('RESULT '.length)), stderr});
    });
  });
}

async function main() {
  assert.ok(fs.existsSync(ZLI), `Build zli first (desktop:build:native): ${ZLI}`);
  // Under Node, the electron package exports the path to its binary.
  const electron = require('electron');
  const workDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'openzl-cancel-test-'));
  const sample = path.join(workDirectory, 'sample.bin');
  // Repetitive but not trivial data keeps training busy long enough to interrupt.
  fs.writeFileSync(sample, Buffer.from(Array.from({length: 8_000_000}, (_, index) => (index * 2654435761) % 251)));
  const before = runDirectories();
  const report = [];
  try {
    for (const scenario of SCENARIOS) {
      const {code, result, stderr} = await runScenario(electron, scenario, sample);
      assert.ok(result !== undefined && code === 0, `${scenario} failed (exit ${code}):\n${stderr.slice(-2000)}`);
      assert.ok(
        await waitUntil(() => result.pids.every((pid) => !alive(pid)), GONE_WITHIN_MS),
        `${scenario}: zli ${result.pids} outlived the app`,
      );
      assert.deepEqual(runDirectories(), before, `${scenario}: left a run directory behind`);
      report.push(result);
    }
  } finally {
    fs.rmSync(workDirectory, {recursive: true, force: true});
  }
  process.stdout.write(`${JSON.stringify({ok: true, report}, null, 2)}\n`);
}

if (process.versions.electron !== undefined) {
  const argument = (name) => process.argv.find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3);
  harness(argument('scenario'), argument('sample')).catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
    require('electron').app.exit(1);
  });
} else {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
    process.exitCode = 1;
  });
}
