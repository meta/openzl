// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/** @import {DesktopRunConfig, RunMessage} from '../../../web/compression_playground/src/benchmarkTypes.ts' */

const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {app, BrowserWindow, dialog, ipcMain, Menu, net, protocol, session, shell, utilityProcess} = require('electron');
const {APP_ORIGIN, APP_SCHEME, originOf, resolveBundlePath} = require('./app-protocol.cjs');
const {validateRequest} = require('./benchmark-request.cjs');
const {startDevServer} = require('./dev-server.cjs');
const {isAllowedExternalUrl} = require('./external-links.cjs');

const BENCHMARK_CHANNEL = 'openzl:benchmark';

/** @type {BrowserWindow | undefined} */
let mainWindow;
/** @type {string | undefined} */
let rendererOrigin;
/** @type {Promise<string | undefined> | undefined} */
let devServerPromise;

// Must run before the app is ready. A standard, secure scheme gives the bundle a
// real origin, so CSP 'self', fetch, and module scripts behave as on the web.
protocol.registerSchemesAsPrivileged([
  {scheme: APP_SCHEME, privileges: {standard: true, secure: true, supportFetchAPI: true}},
]);

function zliPath() {
  const executable = process.platform === 'win32' ? 'zli.exe' : 'zli';
  return app.isPackaged
    ? path.join(process.resourcesPath, 'native', executable)
    : path.join(__dirname, 'native', 'build', 'Release', executable);
}

function nativeHostPath() {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'native', 'native-worker.cjs')
    : path.join(__dirname, 'native-worker.cjs');
}

/** @param {Electron.IpcMainInvokeEvent} event */
function isTrustedSender(event) {
  if (
    mainWindow === undefined ||
    event.sender !== mainWindow.webContents ||
    event.senderFrame !== event.sender.mainFrame
  ) {
    return false;
  }
  // Both sides can be undefined, and undefined === undefined must not pass.
  const origin = originOf(event.senderFrame.url);
  return origin !== undefined && origin === rendererOrigin;
}

function serveRenderer() {
  const root = path.join(__dirname, '..', 'dist');
  protocol.handle(APP_SCHEME, async (request) => {
    const filePath = resolveBundlePath(root, request.url);
    if (filePath === undefined) {
      return new Response('Not found', {status: 404});
    }
    try {
      // fs reads into the asar archive, so this works packaged too.
      await fs.promises.access(filePath);
      return await net.fetch(pathToFileURL(filePath).toString());
    } catch (caught) {
      const error = /** @type {NodeJS.ErrnoException} */ (caught);
      // A path inside the bundle can still name a file that is not there. Any
      // other failure is a real read error, logged so a page that will not
      // load can be diagnosed rather than looking like a missing file.
      if (error.code === 'ENOENT' || error.code === 'ENOTDIR') {
        return new Response('Not found', {status: 404});
      }
      console.error(`Could not serve ${request.url}: ${error.message}`);
      return new Response('Could not read the file', {status: 500});
    }
  });
}

/**
 * Runs `config` in its own utility process. Resolves with every message the run
 * sent, in order, once it reports `finished`; rejects if it reports `failed` or
 * dies.
 *
 * @param {DesktopRunConfig} config
 * @returns {Promise<RunMessage[]>}
 */
function runBenchmark(config) {
  return new Promise((resolve, reject) => {
    const worker = utilityProcess.fork(nativeHostPath(), [], {
      serviceName: 'OpenZL benchmark',
      stdio: 'inherit',
    });
    /** @type {RunMessage[]} */
    const messages = [];
    let settled = false;
    worker.on('message', (/** @type {RunMessage} */ message) => {
      if (settled) {
        return;
      }
      messages.push(message);
      if (message.type !== 'finished' && message.type !== 'failed') {
        return;
      }
      settled = true;
      worker.kill();
      if (message.type === 'finished') {
        resolve(messages);
      } else {
        reject(new Error(message.message));
      }
    });
    worker.once('error', (type, location) => {
      settled = true;
      worker.kill();
      reject(new Error(`Native benchmark process failed (${type} at ${location})`));
    });
    worker.once('exit', (code) => {
      if (!settled) {
        reject(new Error(`Native benchmark process exited unexpectedly (code ${code})`));
      }
    });
    worker.once('spawn', () => {
      try {
        worker.postMessage({zliPath: zliPath(), config});
      } catch (error) {
        settled = true;
        worker.kill();
        reject(error);
      }
    });
  });
}

/**
 * The URL to load the page from in development, if any. `--dev-server=auto`
 * serves the web playground with Vite for as long as the app runs, once however
 * many windows open; `--dev-server=URL` loads URL.
 *
 * @returns {Promise<string | undefined>}
 */
function devServerUrl() {
  devServerPromise ??= (async () => {
    const value = process.argv.find((argument) => argument.startsWith('--dev-server='))?.slice('--dev-server='.length);
    if (app.isPackaged || value !== 'auto') {
      return value;
    }
    const {server, url} = await startDevServer();
    server.printUrls();
    app.once('will-quit', () => void server.close());
    return url;
  })();
  return devServerPromise;
}

// The UI needs no browser permissions, and Electron grants requests by default.
function denyPermissions() {
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
  session.defaultSession.setPermissionCheckHandler(() => false);
  session.defaultSession.setDevicePermissionHandler(() => false);
}

function registerIpc() {
  ipcMain.handle(BENCHMARK_CHANNEL, (event, rawRequest) => {
    if (!isTrustedSender(event)) {
      throw new Error('Untrusted benchmark request');
    }
    return runBenchmark(validateRequest(rawRequest));
  });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 900,
    minWidth: 900,
    minHeight: 640,
    title: 'OpenZL Compression Playground',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({url}) => {
    if (isAllowedExternalUrl(url)) {
      void shell.openExternal(url);
    }
    return {action: 'deny'};
  });
  mainWindow.webContents.on('will-navigate', (event, url) => {
    event.preventDefault();
    if (isAllowedExternalUrl(url)) {
      void shell.openExternal(url);
    }
  });

  const devServer = await devServerUrl();
  if (!app.isPackaged && devServer !== undefined) {
    rendererOrigin = originOf(devServer);
    await mainWindow.loadURL(devServer);
    if (process.argv.includes('--devtools')) {
      mainWindow.webContents.openDevTools({mode: 'detach'});
    }
  } else {
    rendererOrigin = APP_ORIGIN;
    await mainWindow.loadURL(`${APP_ORIGIN}/index.html`);
  }
}

const started = app.whenReady().then(async () => {
  Menu.setApplicationMenu(null);
  denyPermissions();
  serveRenderer();
  registerIpc();
  await createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createWindow();
    }
  });
});

// Otherwise a failed start is a blank window, or no window and a process left
// running, with the reason only in a console no one is watching.
started.catch((error) => {
  dialog.showErrorBox(
    'OpenZL Compression Playground could not start',
    error instanceof Error ? error.message : String(error),
  );
  app.exit(1);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
