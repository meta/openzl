// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

// electron-builder afterPack hook. It runs before code signing, so flipping
// fuses here does not invalidate a signature.

/** @import {AfterPackContext} from 'electron-builder' */

const fs = require('node:fs/promises');
const path = require('node:path');
const {flipFuses, FuseV1Options, FuseVersion} = require('@electron/fuses');
const {Arch} = require('electron-builder');

/** @param {AfterPackContext} context */
function electronBinaryPath({appOutDir, electronPlatformName, packager}) {
  switch (electronPlatformName) {
    case 'darwin':
      return path.join(appOutDir, `${packager.appInfo.productFilename}.app`);
    case 'win32':
      return path.join(appOutDir, `${packager.appInfo.productFilename}.exe`);
    default: {
      // Linux's packager is the one with an executable name of its own.
      const linuxPackager = /** @type {{executableName: string}} */ (/** @type {unknown} */ (packager));
      return path.join(appOutDir, linuxPackager.executableName);
    }
  }
}

/**
 * The app excludes node_modules, but main.cjs (inside the asar) and the worker
 * (in resources/native) require zod at runtime. Node's module lookup from
 * either place reaches resources/node_modules.
 *
 * @param {AfterPackContext} context
 */
async function copyRuntimeDependencies(context) {
  const zod = path.dirname(require.resolve('zod/package.json'));
  const source = path.join(zod, 'src');
  await fs.cp(zod, path.join(context.packager.getResourcesDir(context.appOutDir), 'node_modules', 'zod'), {
    recursive: true,
    // TypeScript sources are not loaded at runtime.
    filter: (file) => file !== source && !file.startsWith(`${source}${path.sep}`),
  });
}

/** @param {AfterPackContext} context */
exports.default = async function afterPack(context) {
  await copyRuntimeDependencies(context);
  await flipFuses(electronBinaryPath(context), {
    version: FuseVersion.V1,
    // Unsigned Apple Silicon builds must be re-signed ad hoc or they will not launch.
    resetAdHocDarwinSignature:
      context.electronPlatformName === 'darwin' && (context.arch === Arch.arm64 || context.arch === Arch.universal),
    [FuseV1Options.RunAsNode]: false,
    [FuseV1Options.EnableCookieEncryption]: true,
    [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
    [FuseV1Options.EnableNodeCliInspectArguments]: false,
    [FuseV1Options.OnlyLoadAppFromAsar]: true,
    [FuseV1Options.GrantFileProtocolExtraPrivileges]: false,
  });
};
