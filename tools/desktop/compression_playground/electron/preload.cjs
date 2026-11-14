// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/** @import {DesktopRunConfig, RunMessage} from '../../../web/compression_playground/src/benchmarkTypes.ts' */

const {contextBridge, ipcRenderer, webUtils} = require('electron');

/**
 * Starts a run on a port of its own, so the page never handles IPC or ports
 * directly. `onMessage` hears each of the run's messages as it arrives, the
 * browser worker's `WorkerMessage`s from `loading` to `finished` or `failed`;
 * `done` then settles, rejecting if the run failed or was cancelled.
 *
 * @param {DesktopRunConfig} config
 * @param {(message: RunMessage) => void} [onMessage]
 */
function startBenchmark(config, onMessage) {
  const {port1, port2} = new MessageChannel();
  let settled = false;
  /** @type {Promise<void>} */
  const done = new Promise((resolve, reject) => {
    port1.onmessage = ({data}) => {
      if (data?.type !== 'cancelled') {
        onMessage?.(data);
      }
      if (data?.type !== 'finished' && data?.type !== 'failed' && data?.type !== 'cancelled') {
        return;
      }
      settled = true;
      port1.close();
      if (data.type === 'finished') {
        resolve();
      } else {
        reject(new Error(data.type === 'cancelled' ? 'Benchmark cancelled' : data.message));
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
  return Object.freeze({done, cancel});
}

contextBridge.exposeInMainWorld(
  'openzlDesktop',
  Object.freeze({
    startBenchmark,
    getPathForFile: (/** @type {File} */ file) => webUtils.getPathForFile(file),
  }),
);
