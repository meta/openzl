// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// Gets a checkout ready to run the app, whatever platform it is on: `yarn
// setup`, which `yarn dev` runs first. `node setup.cjs native` only builds the
// native binaries.

const {spawnSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

/** The renderer's Vite and React come from the web tools' Yarn workspace. */
const WEB_WORKSPACE = path.join(__dirname, '..', '..', 'web');
const NATIVE_SOURCE = path.join(__dirname, 'electron', 'native');
const NATIVE_BUILD = path.join(NATIVE_SOURCE, 'build');
const NATIVE_TARGETS = ['zli'];

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
 * Runs Yarn: the one running this script when there is one, since on Windows
 * `yarn` is a `.cmd` that cannot be spawned without a shell.
 *
 * @param {string[]} args
 */
function yarn(args) {
  const yarnScript = process.env.npm_execpath;
  if (yarnScript !== undefined && yarnScript.endsWith('.js')) {
    run(process.execPath, [yarnScript, ...args]);
  } else {
    run(process.platform === 'win32' ? 'yarn.cmd' : 'yarn', args);
  }
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
  run('cmake', ['--build', NATIVE_BUILD, '--config', 'Release', '--target', ...NATIVE_TARGETS]);
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
  yarn(['--cwd', WEB_WORKSPACE, 'install']);
  if (platform === 'linux') {
    ensureAlsa();
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

module.exports = {getPlatform};
