// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/** @import {DesktopRunConfig, RunMessage} from '../../../web/compression_playground/src/benchmarkTypes.ts' */

const {contextBridge, ipcRenderer, webUtils} = require('electron');

/**
 * Starts a run on a port of its own, so the page never handles IPC or ports
 * directly. `messages` resolves with every message the run sent, in order, once
 * it finishes, and rejects if it fails or is cancelled.
 *
 * @param {DesktopRunConfig} config
 */
function startBenchmark(config) {
  const {port1, port2} = new MessageChannel();
  let settled = false;
  /** @type {RunMessage[]} */
  const received = [];
  /** @type {Promise<RunMessage[]>} */
  const messages = new Promise((resolve, reject) => {
    port1.onmessage = ({data}) => {
      if (data?.type === 'failed' || data?.type === 'cancelled') {
        settled = true;
        port1.close();
        reject(new Error(data.type === 'cancelled' ? 'Benchmark cancelled' : data.message));
        return;
      }
      received.push(data);
      if (data?.type === 'finished') {
        settled = true;
        port1.close();
        resolve(received);
      }
    };
  });
  ipcRenderer.postMessage('openzl:benchmark', config, [port2]);
  // Cancelling a run that has ended does nothing, rather than posting to a closed port.
  const cancel = () => {
    if (!settled) {
      port1.postMessage({type: 'cancel'});
    }
  };
  return Object.freeze({messages, cancel});
}

contextBridge.exposeInMainWorld(
  'openzlDesktop',
  Object.freeze({
    startBenchmark,
    getPathForFile: (/** @type {File} */ file) => webUtils.getPathForFile(file),
  }),
);
