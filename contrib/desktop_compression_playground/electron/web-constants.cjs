// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

const fs = require('node:fs');
const path = require('node:path');

/**
 * One of the JSON files the page and the desktop app share, which live in the
 * web playground's `src/`. The packaged app carries copies in
 * `resources/shared/` (`extraResources` in package.json); everywhere else they
 * are read where they are.
 *
 * @param {string} name
 * @returns {unknown}
 */
function webConstants(name) {
  const packaged = process.resourcesPath === undefined ? undefined : path.join(process.resourcesPath, 'shared', name);
  return require(
    packaged !== undefined && fs.existsSync(packaged)
      ? packaged
      : path.join(__dirname, '..', '..', '..', 'tools', 'web', 'compression_playground', 'src', name),
  );
}

module.exports = {webConstants};
