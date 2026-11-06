// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// Run with Electron after `yarn build:renderer` and `yarn build:native`:
// drives the real page the way a user would. It turns training off, clicks the
// sample button, and waits for the run to fill in the results table and the
// charts, so the whole path from the page through the backend to `zli` and
// back is exercised together. Then it runs every profile the app offers on
// its own, each on a file it can parse.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {app, BrowserWindow} = require('electron');

// The page's errors, printed as they happen, so a failure says what broke.
app.on('web-contents-created', (_event, contents) => {
  contents.on('console-message', ({level, message, lineNumber, sourceId}) => {
    if (level === 'error') {
      process.stderr.write(`Renderer error: ${message} (${sourceId}:${lineNumber})\n`);
    }
  });
  contents.on('preload-error', (_preloadEvent, preloadPath, error) => {
    process.stderr.write(`Preload error in ${preloadPath}: ${error.stack ?? error.message}\n`);
  });
  contents.on('render-process-gone', (_goneEvent, {reason, exitCode}) => {
    process.stderr.write(`Renderer process gone: ${reason} (exit code ${exitCode})\n`);
  });
});

require('../main.cjs');
const {profileInput} = require('./profile-inputs.cjs');

const TIMEOUT_MS = 5 * 60_000;
// For the whole test, which a hung page call would otherwise never end.
const TEST_TIMEOUT_MS = 15 * 60_000;

// The finished run's summary line, or null before there is one. It gives the
// input's size only when something measured it.
const DONE_LINE = `(() => {
  const match = /Done — (?:(.+) · )?(\\d+) succeeded, (\\d+) failed/.exec(document.body.innerText);
  return match === null ? null : {size: match[1], succeeded: Number(match[2]), failed: Number(match[3])};
})()`;

const SAMPLE_BUTTON = `[...document.querySelectorAll('button')]
  .find((button) => button.textContent.trim() === 'Try a sample')`;

async function mainWindow() {
  const deadline = Date.now() + TIMEOUT_MS;
  for (;;) {
    const [window] = BrowserWindow.getAllWindows();
    if (window !== undefined) {
      if (window.webContents.isLoading()) {
        await new Promise((resolve) => window.webContents.once('did-finish-load', resolve));
      }
      return window;
    }
    assert.ok(Date.now() < deadline, 'The app window never appeared');
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

async function waitFor(window, expression, what) {
  const deadline = Date.now() + TIMEOUT_MS;
  for (;;) {
    const value = await window.webContents.executeJavaScript(expression);
    if (value) {
      return value;
    }
    assert.ok(Date.now() < deadline, `Timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

/**
 * Narrows the zstd or gzip row at `position` to level 1 through its levels
 * picker, as a user would: open it, clear every level, tick one, close it.
 */
async function pickOnlyLevelOne(window, position, compressor) {
  const page = (expression) => window.webContents.executeJavaScript(expression);
  const trigger = `[...document.querySelectorAll('button')]
    .find((button) => button.getAttribute('aria-label')?.endsWith(', row ${position}'))`;
  const group = `document.querySelector('[role="group"][aria-label="${compressor} levels"]')`;
  await page(`${trigger}.click()`);
  await waitFor(window, `${group} !== null`, `the ${compressor} levels picker`);
  await page(
    `[...${group}.parentElement.querySelectorAll('button')].find((button) => button.textContent.trim() === 'None').click()`,
  );
  await waitFor(
    window,
    `${trigger}?.getAttribute('aria-label') === 'No levels, row ${position}'`,
    `${compressor} levels to clear`,
  );
  await page(`[...${group}.querySelectorAll('label')].find((label) => label.textContent.trim() === 'Level 1').click()`);
  await waitFor(window, `${trigger}?.getAttribute('aria-label') === '1, row ${position}'`, `${compressor} level 1`);
  // Clicking the trigger again closes it, as it would for a user.
  await page(`${trigger}.click()`);
  await waitFor(
    window,
    `${trigger}?.getAttribute('aria-expanded') === 'false'`,
    `the ${compressor} levels picker to close`,
  );
}

/** Turns training off; training the sample takes minutes, and native-training-smoke.cjs covers it. */
async function turnOffTraining(window) {
  const trainCheckbox = `[...document.querySelectorAll('label')]
    .find((element) => element.textContent.trim() === 'TRAIN')
    ?.querySelector('input[type=checkbox]')`;
  // Disabled until the app's profiles load, when it shows unchecked even
  // though the row still asks to train; turning it off before then does nothing.
  await waitFor(window, `${trainCheckbox}?.disabled === false`, 'the TRAIN checkbox to be enabled');
  const found = await window.webContents.executeJavaScript(`(() => {
    const checkbox = ${trainCheckbox};
    if (checkbox === undefined || checkbox === null) {
      return false;
    }
    if (checkbox.checked) {
      checkbox.click();
    }
    return true;
  })()`);
  assert.ok(found, 'No TRAIN checkbox');
  // The checkbox is controlled, so its state lands once the page re-renders.
  await waitFor(window, `${trainCheckbox}?.checked === false`, 'training to turn off');
}

/**
 * The finished run's table rows, each as its two row headers (compressor, and
 * level or profile), and how many points each chart draws. The charts measure
 * their container before drawing, so they can fill in after the table; this
 * gives them a moment to reach `points` rather than reading them once.
 */
async function readResults(window, points) {
  let results;
  const chartsDeadline = Date.now() + 30_000;
  for (;;) {
    results = await window.webContents.executeJavaScript(`(() => {
      const rows = [...document.querySelectorAll('table[aria-labelledby="measurements-title"] tbody tr')];
      // Each chart's points on their own: counted together, one chart with
      // every point would pass for two.
      const pointsIn = (title) => {
        const heading = [...document.querySelectorAll('p')].find((element) => element.textContent.trim() === title);
        const chart = heading?.parentElement?.parentElement;
        return chart === undefined || chart === null ? -1 : chart.querySelectorAll('.recharts-scatter-symbol').length;
      };
      return {
        chosen: document.querySelector('[data-testid="upload-dropzone"]')?.innerText ?? '',
        rows: rows.map((row) => [...row.querySelectorAll('th')].map((header) => header.innerText.trim())),
        compressionPoints: pointsIn('Compression'),
        decompressionPoints: pointsIn('Decompression'),
      };
    })()`);
    if ((results.compressionPoints >= points && results.decompressionPoints >= points) || Date.now() > chartsDeadline) {
      return results;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

async function main() {
  const window = await mainWindow();
  const page = (expression) => window.webContents.executeJavaScript(expression);

  await waitFor(window, `${SAMPLE_BUTTON} !== undefined`, 'the sample button');
  await turnOffTraining(window);

  // zstd and gzip default to six levels each. One each makes the expected
  // result exact: one measurement per compressor, so a compressor that
  // measured nothing cannot hide behind the others' extra rows.
  await pickOnlyLevelOne(window, 2, 'zstd');
  await pickOnlyLevelOne(window, 3, 'gzip');

  await page(`${SAMPLE_BUTTON}.click()`);

  const done = await waitFor(window, DONE_LINE, 'the run to finish');
  const {rows, ...results} = await readResults(window, 3);
  // The first row header names the compressor.
  results.compressors = rows.map(([compressor]) => compressor).sort();

  // Exactly one measurement each for OpenZL, zstd and gzip, in the table and
  // in both charts.
  assert.deepEqual({succeeded: done.succeeded, failed: done.failed}, {succeeded: 3, failed: 0});
  assert.match(results.chosen, /sra0\.bin/);
  assert.deepEqual(results.compressors, ['OpenZL', 'gzip', 'zstd']);
  assert.equal(results.compressionPoints, 3, 'compression chart points');
  assert.equal(results.decompressionPoints, 3, 'decompression chart points');

  const profiles = await runEveryProfile(window);

  process.stdout.write(`${JSON.stringify({ok: true, ...done, ...results, profiles}, null, 2)}\n`);
}

/**
 * Chooses `file` in the page's file input, as picking it in the system dialog
 * would. It goes through the DevTools protocol, which the page cannot do
 * itself, so the page gets a real `File` that `getPathForFile` can name.
 */
async function chooseFile(window, file) {
  const devtools = window.webContents.debugger;
  if (!devtools.isAttached()) {
    devtools.attach('1.3');
  }
  const {root} = await devtools.sendCommand('DOM.getDocument');
  const {nodeId} = await devtools.sendCommand('DOM.querySelector', {
    nodeId: root.nodeId,
    selector: '[data-testid="upload-dropzone"] input[type=file]',
  });
  assert.notEqual(nodeId, 0, 'No file input');
  await devtools.sendCommand('DOM.setFileInputFiles', {nodeId, files: [file]});
}

/**
 * The profiles the app's picker lets a user run: the ones its `zli` lists,
 * less those the CLI says need a file path or marks as placeholders, by the
 * rule in the page's `desktopUnavailableReason`.
 */
async function runnableProfiles(window) {
  const listed = await window.webContents.executeJavaScript('window.openzlDesktop.listProfiles()');
  return listed
    .filter(({description}) => !/\bPass a path to\b/i.test(description) && !/\(Placeholder\)/i.test(description))
    .map(({name}) => name);
}

/**
 * A test that runs `profile` on `input` from a freshly loaded page, with only
 * the OpenZL row left, and checks the run measured it once: in the table and
 * in both charts.
 */
function profileSmokeTest(profile, input) {
  return async (window) => {
    const page = (expression) => window.webContents.executeJavaScript(expression);
    const loaded = new Promise((resolve) => window.webContents.once('did-finish-load', resolve));
    window.webContents.reload();
    await loaded;
    await waitFor(window, `${SAMPLE_BUTTON} !== undefined`, 'the sample button');
    await turnOffTraining(window);

    // The last row first, so the one left to remove is still row 2.
    for (const position of [3, 2]) {
      await page(`document.querySelector('button[aria-label="Remove row ${position}"]').click()`);
      await waitFor(
        window,
        `document.querySelector('button[aria-label="Remove row ${position}"]') === null`,
        `row ${position} to go`,
      );
    }

    const trigger = `document.querySelector('[data-scope="select"][data-part="trigger"]')`;
    const option = `document.querySelector('[role="option"][data-value="${profile}"]')`;
    await page(`${trigger}.click()`);
    // Until the app's zli answers, the picker lists the profiles the page
    // knows, which can lack one the CLI has since added.
    await waitFor(window, `${option} !== null`, `the ${profile} option`);
    assert.equal(await page(`${option}.hasAttribute('data-disabled')`), false, `${profile} is disabled in the picker`);
    await page(`${option}.click()`);
    await waitFor(window, `${trigger}.textContent.trim() === '${profile}'`, `${profile} to be picked`);

    await chooseFile(window, input);
    await waitFor(
      window,
      `document.querySelector('[data-testid="upload-dropzone"]').innerText.includes(${JSON.stringify(path.basename(input))})`,
      `${path.basename(input)} to be chosen`,
    );
    await page(
      `[...document.querySelectorAll('button')].find((button) => button.textContent.trim() === 'Run benchmark').click()`,
    );

    // A run whose every measurement failed ends in an error, not a done line.
    const done = await waitFor(
      window,
      `${DONE_LINE} ?? /Run failed: .*/.exec(document.body.innerText)?.[0]`,
      `the ${profile} run to finish`,
    );
    assert.equal(typeof done, 'object', done);
    // A failed measurement is listed by name with zli's message, which is what to read.
    const failures = await page(`[...document.querySelectorAll('li')]
      .map((item) => item.innerText)
      .filter((text) => text.startsWith('OpenZL '))
      .join('; ')`);
    assert.deepEqual({succeeded: done.succeeded, failed: done.failed}, {succeeded: 1, failed: 0}, failures);
    const {rows, compressionPoints, decompressionPoints} = await readResults(window, 1);
    assert.deepEqual(rows, [['OpenZL', profile]]);
    assert.equal(compressionPoints, 1, 'compression chart points');
    assert.equal(decompressionPoints, 1, 'decompression chart points');
    return {profile, input: path.basename(input), size: done.size};
  };
}

/**
 * Runs every profile the app offers through `profileSmokeTest`, and fails
 * once all have run with every profile that did not pass.
 */
async function runEveryProfile(window) {
  const profiles = await runnableProfiles(window);
  assert.ok(profiles.includes('serial'), `No serial profile in ${profiles.join(', ')}`);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'openzl-ui-profiles-'));
  try {
    const passed = [];
    const failed = [];
    for (const profile of profiles) {
      try {
        passed.push(await profileSmokeTest(profile, profileInput(profile, directory))(window));
      } catch (error) {
        failed.push(`${profile}: ${error.message}`);
      }
    }
    assert.deepEqual(failed, [], `Profiles that failed:\n${failed.join('\n')}`);
    return passed;
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
}

app.whenReady().then(() => {
  const timeout = setTimeout(() => {
    process.stderr.write(`Timed out after ${TEST_TIMEOUT_MS} ms\n`);
    app.exit(2);
  }, TEST_TIMEOUT_MS);
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
