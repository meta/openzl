// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

const childProcess = require('node:child_process');
const path = require('node:path');
const iconv = require('iconv-lite');

const UTF8 = 65001;

/**
 * On Windows `zli` and `zstd-benchmark` read their arguments in the ANSI code
 * page, not as UTF-16 or UTF-8, so a path with a character outside that code
 * page reaches them mangled (or best-fit mapped to a different file). These
 * are the characters of `text` that `codePage` cannot encode, by the Windows
 * tables iconv-lite ships. Empty for UTF-8, which encodes everything, and for
 * a code page iconv-lite does not know: refusing what might work would be
 * worse than letting the tool report its own error.
 *
 * @param {string} text
 * @param {number} codePage
 * @returns {string[]}
 */
function unrepresentableCharacters(text, codePage) {
  const encoding = `cp${codePage}`;
  if (codePage === UTF8 || !iconv.encodingExists(encoding)) {
    return [];
  }
  // iconv-lite writes '?' for a character it cannot encode, so it does not
  // come back out.
  return [
    ...new Set(
      [...text].filter((character) => iconv.decode(iconv.encode(character, encoding), encoding) !== character),
    ),
  ];
}

/**
 * Throws a message the page can show when `text` has characters `codePage` cannot encode.
 *
 * @param {string} text
 * @param {string} what
 * @param {string} fix
 * @param {number} codePage
 */
function assertReadableByNativeTools(text, what, fix, codePage) {
  const bad = unrepresentableCharacters(text, codePage);
  if (bad.length === 0) {
    return;
  }
  const shown = bad
    .slice(0, 3)
    .map((character) => `"${character}"`)
    .join(', ');
  throw new Error(
    `${what} contains ${shown}${bad.length > 3 ? ' and more' : ''}, which this PC's ANSI code page (${codePage}) cannot represent, and the native benchmark tools (zli, zstd-benchmark) read paths in that code page. ${fix}`,
  );
}

/**
 * The ACP value in `reg query` output, or undefined.
 *
 * @param {string} output
 * @returns {number | undefined}
 */
function parseCodePage(output) {
  // Anchored to the value name: the key also holds OEMCP and MACCP, whose
  // names end in ACP.
  const match = /^\s*ACP\s+REG_SZ\s+(\d+)\s*$/m.exec(output);
  return match === null ? undefined : Number(match[1]);
}

/** @type {Promise<number | undefined> | undefined} */
let detected;

/**
 * The ACP value from the registry, or undefined if it cannot be read.
 *
 * @returns {Promise<number | undefined>}
 */
function readCodePage() {
  return new Promise((resolve) => {
    childProcess.execFile(
      // Absolute: Windows resolves a bare name through the current directory
      // before PATH, so a stray reg.exe there would run instead.
      path.win32.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'reg.exe'),
      ['query', 'HKLM\\SYSTEM\\CurrentControlSet\\Control\\Nls\\CodePage', '/v', 'ACP'],
      {windowsHide: true},
      (error, stdout) => {
        resolve(error === null ? parseCodePage(stdout) : undefined);
      },
    );
  });
}

/**
 * This PC's ANSI code page; undefined if it cannot be read. Callers share one
 * read, and only a successful one is kept, so a failed read is retried.
 *
 * @returns {Promise<number | undefined>}
 */
function windowsAnsiCodePage() {
  detected ??= readCodePage().then((codePage) => {
    if (codePage === undefined) {
      detected = undefined;
    }
    return codePage;
  });
  return detected;
}

module.exports = {
  assertReadableByNativeTools,
  parseCodePage,
  unrepresentableCharacters,
  windowsAnsiCodePage,
};
