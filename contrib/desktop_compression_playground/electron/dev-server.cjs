// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/** @import {ViteDevServer} from 'vite' */

const path = require('node:path');
const {pathToFileURL} = require('node:url');

const WEB_PLAYGROUND = path.join(__dirname, '..', '..', '..', 'tools', 'web', 'compression_playground');

/**
 * Serves the web playground with the web workspace's Vite, which takes the next
 * free port when `port` (by default 5173) is in use, and resolves with the
 * server and the URL it chose.
 *
 * @param {{port?: number}} [options]
 * @returns {Promise<{server: ViteDevServer, url: string}>}
 */
async function startDevServer({port} = {}) {
  // Vite is an ES module, so it is imported, from where the web workspace has it.
  const vitePath = require.resolve('vite', {paths: [WEB_PLAYGROUND]});
  /** @type {typeof import('vite')} */
  const vite = await import(pathToFileURL(vitePath).href);
  // The desktop mode benchmarks with zli rather than the WASM module.
  const server = await vite.createServer({root: WEB_PLAYGROUND, base: './', mode: 'desktop', server: {port}});
  await server.listen();
  const url = server.resolvedUrls?.local[0];
  if (url === undefined) {
    await server.close();
    throw new Error('Vite did not report the URL it serves the playground on');
  }
  return {server, url};
}

module.exports = {startDevServer};
