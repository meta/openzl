// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// Gets a checkout ready to run the app, whatever platform it is on: `yarn
// setup`, which `yarn dev` runs first. `node setup.cjs native` only builds the
// native binaries.

const {spawnSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

/** The renderer's Vite and React come from the web tools' Yarn workspace. */
const WEB_WORKSPACE = path.join(__dirname, '..', '..', 'tools', 'web');
const NATIVE_SOURCE = path.join(__dirname, 'electron', 'native');
const NATIVE_BUILD = path.join(NATIVE_SOURCE, 'build');
const NATIVE_TARGETS = ['zli', 'zstd-benchmark'];
const ELECTRON_DIST = path.join(__dirname, 'node_modules', 'electron', 'dist');

/** @typedef {'linux' | 'mac' | 'windows'} Platform */

/** @returns {Platform} */
function getPlatform() {
  switch (process.platform) {
    case 'linux':
      return 'linux';
    case 'darwin':
      return 'mac';
    case 'win32':
      return 'windows';
    default:
      throw new Error(`The desktop app does not build on ${process.platform}`);
  }
}

/**
 * Runs a command with its output shown, and throws if it fails.
 *
 * @param {string} command
 * @param {string[]} args
 */
function run(command, args) {
  console.log(`> ${[command, ...args].join(' ')}`);
  const result = spawnSync(command, args, {stdio: 'inherit'});
  if (result.error !== undefined) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(
      `${command} exited with ${result.status === null ? `signal ${result.signal}` : `code ${result.status}`}`,
    );
  }
}

/**
 * Whether a command can be run, judged by running it with `args` quietly.
 *
 * @param {string} command
 * @param {string[]} args
 */
function succeeds(command, args) {
  const result = spawnSync(command, args, {stdio: 'ignore'});
  return result.error === undefined && result.status === 0;
}

/**
 * Runs the Yarn running this script, as a script under Node, since on Windows
 * `yarn` is a `.cmd`, which Node does not start without a shell. Yarn names
 * itself in the user agent it sets; its script need not end in `.js`, as the
 * copy fbsource uses does not.
 *
 * @param {string[]} args
 */
function yarn(args) {
  const yarnScript = process.env.npm_execpath;
  if (yarnScript === undefined || !process.env.npm_config_user_agent?.startsWith('yarn/')) {
    throw new Error('Run this through Yarn: yarn setup');
  }
  run(process.execPath, [yarnScript, ...args]);
}

/**
 * Whether the dynamic linker knows a shared library. `ldconfig` is usually in
 * /sbin, which is not always on PATH for a normal user.
 *
 * @param {string} name
 */
function hasSharedLibrary(name) {
  for (const ldconfig of ['ldconfig', '/sbin/ldconfig']) {
    const result = spawnSync(ldconfig, ['-p'], {encoding: 'utf8'});
    if (result.error === undefined && result.status === 0) {
      return result.stdout.includes(name);
    }
  }
  return false;
}

/**
 * Electron's Chromium loads ALSA's library at startup, and minimal installs and
 * WSL may not have it. Installed with apt where there is apt; anywhere else the
 * error says what to install.
 */
function ensureAlsa() {
  if (hasSharedLibrary('libasound.so.2')) {
    return;
  }
  if (!succeeds('apt-get', ['--version'])) {
    throw new Error("Electron needs ALSA's shared library, libasound.so.2. Install it with your package manager.");
  }
  // Ubuntu 24.04 renamed the package; before it, only the old name exists.
  const alsa = succeeds('apt-cache', ['show', 'libasound2t64']) ? 'libasound2t64' : 'libasound2';
  /** @param {string[]} args */
  const aptGet = (args) => (process.getuid?.() === 0 ? run('apt-get', args) : run('sudo', ['apt-get', ...args]));
  aptGet(['update']);
  aptGet(['install', '-y', alsa]);
}

/**
 * Lets Electron's Chromium sandbox its renderers on Linux distributions that
 * stop unprivileged processes making user namespaces, as Ubuntu 24.04 does.
 * Chromium then needs its setuid helper, `chrome-sandbox` beside the Electron
 * binary in `electronDirectory`, to be owned by root, which neither Yarn nor
 * electron-builder can arrange, so Electron would otherwise refuse to start.
 *
 * @param {string} electronDirectory
 */
function ensureChromiumSandbox(electronDirectory) {
  const setting = '/proc/sys/kernel/apparmor_restrict_unprivileged_userns';
  // Without the setting, the kernel does not restrict them.
  if (!fs.existsSync(setting) || fs.readFileSync(setting, 'utf8').trim() !== '1') {
    return;
  }
  const helper = path.join(electronDirectory, 'chrome-sandbox');
  const stat = fs.statSync(helper);
  if (stat.uid === 0 && (stat.mode & 0o4777) === 0o4755) {
    return;
  }
  console.log(`Making ${helper} setuid root, which Chromium's sandbox needs here`);
  /** @param {string[]} args */
  const asRoot = (args) => (process.getuid?.() === 0 ? run(args[0] ?? '', args.slice(1)) : run('sudo', args));
  asRoot(['chown', 'root:root', helper]);
  asRoot(['chmod', '4755', helper]);
}

/**
 * The CMake arguments the platform needs. On Windows OpenZL builds with
 * clang-cl, as windows-ci.yml does, not MSVC's cl, which needs Visual Studio's
 * "C++ Clang tools for Windows" component.
 *
 * @param {Platform} platform
 */
function cmakeArguments(platform) {
  return platform === 'windows' ? ['-T', 'ClangCL'] : [];
}

/**
 * Whether the native build directory was configured for this source directory
 * and toolset. One that was not -- configured for MSVC before clang-cl was
 * needed, or for where the app lived before it moved -- cannot be reconfigured
 * in place and is built again from scratch.
 *
 * @param {Platform} platform
 */
function buildDirectoryIsCurrent(platform) {
  const cachePath = path.join(NATIVE_BUILD, 'CMakeCache.txt');
  if (!fs.existsSync(cachePath)) {
    return true;
  }
  const cache = fs.readFileSync(cachePath, 'utf8');
  /** @param {string} name */
  const entry = (name) => new RegExp(`^${name}:\\w+=(.*)$`, 'm').exec(cache)?.[1]?.trim() ?? '';
  // Real paths, since a checkout reached through a symlink names the same
  // directory differently.
  /** @param {string} directory */
  const realPath = (directory) => (fs.existsSync(directory) ? fs.realpathSync(directory) : path.resolve(directory));
  return (
    realPath(entry('CMAKE_HOME_DIRECTORY')) === realPath(NATIVE_SOURCE) &&
    (platform !== 'windows' || entry('CMAKE_GENERATOR_TOOLSET') === 'ClangCL')
  );
}

/** @param {Platform} platform */
function buildNative(platform) {
  if (!buildDirectoryIsCurrent(platform)) {
    console.log(`Removing ${NATIVE_BUILD}, which was configured for another source directory or toolset`);
    fs.rmSync(NATIVE_BUILD, {recursive: true, force: true});
  }
  run('cmake', ['-S', NATIVE_SOURCE, '-B', NATIVE_BUILD, '-DCMAKE_BUILD_TYPE=Release', ...cmakeArguments(platform)]);
  // Make, CMake's default generator on Linux and macOS, builds one file at a time otherwise.
  const parallel = ['--parallel', process.env.CMAKE_BUILD_PARALLEL_LEVEL ?? String(os.availableParallelism())];
  run('cmake', ['--build', NATIVE_BUILD, '--config', 'Release', ...parallel, '--target', ...NATIVE_TARGETS]);
}

/** @param {string[]} args */
function main(args) {
  const platform = getPlatform();
  if (args[0] === 'native') {
    buildNative(platform);
    return;
  }
  if (args.length > 0) {
    throw new Error(`Unknown arguments: ${args.join(' ')}. Run with none, or with native.`);
  }
  // In CI, as for `yarn install` there, the lockfile must already be right.
  yarn(['--cwd', WEB_WORKSPACE, 'install', ...(process.env.CI === undefined ? [] : ['--frozen-lockfile'])]);
  if (platform === 'linux') {
    ensureAlsa();
    ensureChromiumSandbox(ELECTRON_DIST);
  }
  buildNative(platform);
}

if (require.main === module) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

module.exports = {ensureChromiumSandbox, getPlatform, run, yarn};
