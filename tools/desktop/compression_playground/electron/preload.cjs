// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/** @import {DesktopRunConfig} from '../../../web/compression_playground/src/benchmarkTypes.ts' */

const {contextBridge, ipcRenderer, webUtils} = require('electron');

contextBridge.exposeInMainWorld(
  'openzlDesktop',
  Object.freeze({
    benchmark: (/** @type {DesktopRunConfig} */ config) => ipcRenderer.invoke('openzl:benchmark', config),
    getPathForFile: (/** @type {File} */ file) => webUtils.getPathForFile(file),
  }),
);
