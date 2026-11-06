// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// Run with Electron after `yarn build:renderer` and `yarn build:native`:
// drives the real page the way a user would. It turns training off, clicks the
// sample button, and waits for the run to fill in the results table and the
// charts, so the whole path from the page through the backend to `zli` and
// back is exercised together.

const assert = require('node:assert/strict');
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

const TIMEOUT_MS = 5 * 60_000;
// For the whole test, which a hung page call would otherwise never end.
const TEST_TIMEOUT_MS = 15 * 60_000;

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

async function main() {
  const window = await mainWindow();
  const page = (expression) => window.webContents.executeJavaScript(expression);

  await waitFor(
    window,
    `[...document.querySelectorAll('button')].some((button) => button.textContent.trim() === 'Try a sample')`,
    'the sample button',
  );

  // Training the sample takes minutes; native-training-smoke.cjs covers it.
  const trainCheckbox = `[...document.querySelectorAll('label')]
    .find((element) => element.textContent.trim() === 'TRAIN')
    ?.querySelector('input[type=checkbox]')`;
  // Disabled until the app's profiles load, when it shows unchecked even
  // though the row still asks to train; turning it off before then does nothing.
  await waitFor(window, `${trainCheckbox}?.disabled === false`, 'the TRAIN checkbox to be enabled');
  const found = await page(`(() => {
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

  // zstd and gzip default to six levels each. One each makes the expected
  // result exact: one measurement per compressor, so a compressor that
  // measured nothing cannot hide behind the others' extra rows.
  await pickOnlyLevelOne(window, 2, 'zstd');
  await pickOnlyLevelOne(window, 3, 'gzip');

  await page(`[...document.querySelectorAll('button')]
    .find((button) => button.textContent.trim() === 'Try a sample')
    .click()`);

  const done = await waitFor(
    window,
    `(() => {
      const match = /Done — (.+) · (\\d+) succeeded, (\\d+) failed/.exec(document.body.innerText);
      return match === null ? null : {size: match[1], succeeded: Number(match[2]), failed: Number(match[3])};
    })()`,
    'the run to finish',
  );
  // The charts measure their container before drawing, so they can fill in
  // after the table; give them a moment rather than reading them once.
  let results;
  const chartsDeadline = Date.now() + 30_000;
  for (;;) {
    results = await page(`(() => {
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
        // The first row header names the compressor.
        compressors: rows.map((row) => row.querySelector('th')?.innerText.trim() ?? '').sort(),
        compressionPoints: pointsIn('Compression'),
        decompressionPoints: pointsIn('Decompression'),
      };
    })()`);
    if ((results.compressionPoints >= 3 && results.decompressionPoints >= 3) || Date.now() > chartsDeadline) {
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }

  // Exactly one measurement each for OpenZL, zstd and gzip, in the table and
  // in both charts.
  assert.deepEqual({succeeded: done.succeeded, failed: done.failed}, {succeeded: 3, failed: 0});
  assert.match(results.chosen, /sra0\.bin/);
  assert.deepEqual(results.compressors, ['OpenZL', 'gzip', 'zstd']);
  assert.equal(results.compressionPoints, 3, 'compression chart points');
  assert.equal(results.decompressionPoints, 3, 'decompression chart points');

  process.stdout.write(`${JSON.stringify({ok: true, ...done, ...results}, null, 2)}\n`);
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
