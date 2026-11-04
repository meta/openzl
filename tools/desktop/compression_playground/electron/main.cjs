// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

/**
 * @import {DesktopRunConfig} from '../../../web/compression_playground/src/benchmarkTypes.ts'
 * @import {WorkerOutput} from './native-worker.cjs'
 */

/**
 * One benchmark run: its worker, the zli PIDs it has reported, and its
 * scratch directory, which the main process owns so a killed worker cannot
 * leave it behind.
 *
 * @typedef {object} Run
 * @property {Set<number>} childPids
 * @property {string} tempRoot
 * @property {Electron.UtilityProcess} worker
 * @property {Electron.MessagePortMain} port
 * @property {boolean} finished
 * @property {NodeJS.Timeout | undefined} watchdog
 * @property {() => void} cancel
 */

/** What ends a run, as the renderer hears it. */
/** @typedef {Extract<WorkerOutput, {type: 'finished' | 'failed' | 'cancelled'}>} RunEnd */

const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {app, BrowserWindow, dialog, ipcMain, Menu, net, protocol, session, shell, utilityProcess} = require('electron');
const {APP_ORIGIN, APP_SCHEME, originOf, resolveBundlePath} = require('./app-protocol.cjs');
const {validateRequest} = require('./benchmark-request.cjs');
const {isAllowedExternalUrl} = require('./external-links.cjs');

const BENCHMARK_CHANNEL = 'openzl:benchmark';
const CANCEL_GRACE_MS = 5000;

/** @type {BrowserWindow | undefined} */
let mainWindow;
/** @type {string | undefined} */
let rendererOrigin;
/** @type {Set<Run>} */
const activeRuns = new Set();

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

/** @param {Electron.IpcMainEvent} event */
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
 * Starts one benchmark run in its own utility process, reporting to the
 * renderer over `port`. The main process records every zli PID the worker
 * reports, so it can kill them if the worker dies or ignores a cancel: children
 * do not exit with their parent on their own.
 *
 * @param {DesktopRunConfig} config
 * @param {Electron.MessagePortMain} port
 */
function startRun(config, port) {
  // Before the fork: a worker started first would be leaked if this threw, since
  // nothing has recorded it yet for `will-quit` to clean up.
  let tempRoot;
  try {
    tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'openzl-run-'));
  } catch (error) {
    port.postMessage({type: 'failed', message: `Could not create a scratch directory: ${messageOf(error)}`});
    port.close();
    return;
  }
  let worker;
  try {
    worker = utilityProcess.fork(nativeHostPath(), [], {
      serviceName: 'OpenZL benchmark',
      stdio: 'inherit',
    });
  } catch (error) {
    fs.rmSync(tempRoot, {recursive: true, force: true});
    port.postMessage({type: 'failed', message: `Could not start the benchmark process: ${messageOf(error)}`});
    port.close();
    return;
  }
  /** @type {Run} */
  const run = {
    childPids: new Set(),
    tempRoot,
    worker,
    port,
    finished: false,
    watchdog: undefined,
    cancel: () => {
      if (!run.finished && run.watchdog === undefined) {
        try {
          worker.postMessage({type: 'cancel'});
        } catch {
          // The worker has exited; its 'exit' event, or the watchdog, ends the run.
        }
        run.watchdog = setTimeout(() => disposeRun(run, {type: 'cancelled'}), CANCEL_GRACE_MS);
      }
    },
  };

  worker.on('message', (/** @type {WorkerOutput} */ message) => {
    switch (message.type) {
      case 'child':
        if (message.state === 'exited') {
          run.childPids.delete(message.pid);
        } else if (run.finished) {
          // Reported after the run's children were killed, so nothing else will.
          killPid(message.pid);
        } else {
          run.childPids.add(message.pid);
        }
        return;
      case 'finished':
      case 'failed':
      case 'cancelled':
        // The worker only reports these once its zli has closed.
        disposeRun(run, message, {childrenExited: true});
        return;
      default:
        if (!run.finished) {
          port.postMessage(message);
        }
    }
  });
  worker.once('error', (type, location) =>
    disposeRun(run, {type: 'failed', message: `Native benchmark process failed (${type} at ${location})`}),
  );
  // A worker that exits cleanly while cancelling may not have reported it
  // first. One that exits with an error code crashed, cancel or not.
  worker.once('exit', (code) =>
    disposeRun(
      run,
      run.watchdog !== undefined && code === 0
        ? {type: 'cancelled'}
        : {type: 'failed', message: `Native benchmark process exited unexpectedly (code ${code})`},
    ),
  );
  worker.once('spawn', () => {
    try {
      worker.postMessage({type: 'start', zliPath: zliPath(), config, tempRoot});
    } catch (error) {
      // The worker died between spawning and receiving its start message.
      disposeRun(run, {type: 'failed', message: `Could not start the benchmark process: ${messageOf(error)}`});
    }
  });

  port.on('message', ({data}) => {
    if (data?.type === 'cancel') {
      run.cancel();
    }
  });
  // The renderer closed its end: it reloaded, crashed, or its window closed.
  port.on('close', () => run.cancel());
  port.start();
  activeRuns.add(run);
}

/** @param {unknown} error */
function messageOf(error) {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Ends `run` and sends the renderer `message`, once: later calls do nothing.
 * Kills the run's zli processes and its worker, and removes its scratch
 * directory. `childrenExited` says the worker reported the end itself, which it
 * only does once its zli has closed, so its children need not be looked up.
 *
 * @param {Run} run
 * @param {RunEnd} message
 * @param {{childrenExited?: boolean}} [options]
 */
function disposeRun(run, message, {childrenExited = false} = {}) {
  if (run.finished) {
    return;
  }
  run.finished = true;
  clearTimeout(run.watchdog);
  activeRuns.delete(run);
  killChildren(run);
  // Before the directory goes: a worker still running could be writing into it.
  killWorkerTree(run.worker, {lookUpChildren: !childrenExited});
  removeTempRoot(run);
  run.port.postMessage(message);
  run.port.close();
}

/** @param {number} pid */
function killPid(pid) {
  try {
    process.kill(pid, 'SIGKILL');
  } catch {
    // Already exited.
  }
}

/** @param {Run} run */
function killChildren(run) {
  for (const pid of run.childPids) {
    killPid(pid);
  }
  run.childPids.clear();
}

/**
 * The PIDs of `parent`'s child processes, from `ps` (Linux and macOS).
 *
 * @param {number} parent
 * @returns {number[]}
 */
function childPidsOf(parent) {
  try {
    return execFileSync('ps', ['-A', '-o', 'pid=,ppid='], {encoding: 'utf8'})
      .split('\n')
      .map((line) => line.trim().split(/\s+/).map(Number))
      .filter(([pid, ppid]) => ppid === parent && Number.isInteger(pid))
      .map(([pid]) => Number(pid));
  } catch {
    return [];
  }
}

/**
 * Kills the worker and any zli it started. A zli the worker spawned just
 * before being killed may not have been reported yet, since `child` messages
 * are asynchronous, so unless `lookUpChildren` is false the worker's children
 * are looked up rather than trusted to have been reported. On Linux and macOS
 * the worker is stopped first, so it cannot start another between the lookup
 * and the kill, and then SIGKILLed; on Windows `taskkill /T` kills the tree in
 * one step.
 *
 * @param {Electron.UtilityProcess} worker
 * @param {{lookUpChildren: boolean}} options
 */
function killWorkerTree(worker, {lookUpChildren}) {
  const {pid} = worker;
  if (pid !== undefined) {
    if (process.platform === 'win32') {
      try {
        execFileSync('taskkill', ['/PID', String(pid), ...(lookUpChildren ? ['/T'] : []), '/F'], {stdio: 'ignore'});
      } catch {
        // Already exited.
      }
    } else {
      if (lookUpChildren) {
        try {
          process.kill(pid, 'SIGSTOP');
        } catch {
          // Already exited.
        }
        for (const child of childPidsOf(pid)) {
          killPid(child);
        }
      }
      // A stopped process never handles the SIGTERM worker.kill() sends.
      killPid(pid);
    }
  }
  worker.kill();
}

/** @param {Run} run */
function removeTempRoot(run) {
  try {
    // The processes just killed may not have released their files yet, which
    // blocks removal on Windows; the retries wait them out.
    fs.rmSync(run.tempRoot, {recursive: true, force: true, maxRetries: 3});
  } catch (error) {
    // Cleanup must not stop the run from reporting its outcome.
    console.warn(`Could not remove ${run.tempRoot}: ${messageOf(error)}`);
  }
}

// The UI needs no browser permissions, and Electron grants requests by default.
function denyPermissions() {
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
  session.defaultSession.setPermissionCheckHandler(() => false);
  session.defaultSession.setDevicePermissionHandler(() => false);
}

function registerIpc() {
  ipcMain.on(BENCHMARK_CHANNEL, (event, rawRequest) => {
    const [port] = event.ports;
    if (port === undefined) {
      return;
    }
    let config;
    try {
      if (!isTrustedSender(event)) {
        throw new Error('Untrusted benchmark request');
      }
      config = validateRequest(rawRequest);
    } catch (error) {
      port.postMessage({
        type: 'failed',
        message: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
      });
      port.close();
      return;
    }
    startRun(config, port);
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

  const devServer = process.argv
    .find((argument) => argument.startsWith('--dev-server='))
    ?.slice('--dev-server='.length);
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

app.on('will-quit', () => {
  for (const run of activeRuns) {
    disposeRun(run, {type: 'cancelled'});
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
