// Copyright (c) Meta Platforms, Inc. and affiliates.

'use strict';

/**
 * @import {DesktopProfile} from '../../../tools/web/compression_playground/src/desktop.ts'
 * @import {ListedProfile} from './zli-client.cjs'
 */

const NEEDS_DESCRIPTION_FILE =
  'Cannot run in the desktop app yet: needs a description file, which this app cannot pick.';
const PLACEHOLDER = 'Cannot run in the desktop app yet: zli lists it as a placeholder.';
const UNKNOWN = 'Cannot run in the desktop app yet: this app does not know the profile.';

/**
 * What the desktop app can do with each profile, like `kProfiles` in
 * `openzl_wasm.cpp` for the browser. A profile `zli list-profiles` lists that
 * is missing here cannot be run until it is added.
 *
 * @type {Readonly<Record<string, {runnable: boolean; trainable: boolean; reason?: string}>>}
 */
const DESKTOP_PROFILES = {
  serial: {runnable: true, trainable: true},
  u8: {runnable: true, trainable: true},
  i8: {runnable: true, trainable: true},
  'le-u16': {runnable: true, trainable: true},
  'le-i16': {runnable: true, trainable: true},
  'le-u32': {runnable: true, trainable: true},
  'le-i32': {runnable: true, trainable: true},
  'le-u64': {runnable: true, trainable: true},
  'le-i64': {runnable: true, trainable: true},
  'be-u16': {runnable: true, trainable: true},
  'be-i16': {runnable: true, trainable: true},
  'be-u32': {runnable: true, trainable: true},
  'be-i32': {runnable: true, trainable: true},
  'be-u64': {runnable: true, trainable: true},
  'be-i64': {runnable: true, trainable: true},
  csv: {runnable: true, trainable: true},
  gguf: {runnable: true, trainable: true},
  lz: {runnable: true, trainable: true},
  'numeric-ml-selector-64': {runnable: false, trainable: false, reason: PLACEHOLDER},
  numpy: {runnable: true, trainable: true},
  onnx: {runnable: true, trainable: true},
  parquet: {runnable: true, trainable: true},
  pickle: {runnable: true, trainable: true},
  pytorch: {runnable: true, trainable: false},
  safetensors: {runnable: true, trainable: true},
  sao: {runnable: true, trainable: true},
  sddl: {runnable: false, trainable: false, reason: NEEDS_DESCRIPTION_FILE},
  sddl2: {runnable: false, trainable: false, reason: NEEDS_DESCRIPTION_FILE},
  zstd: {runnable: true, trainable: true},
};

/**
 * The profiles `zli list-profiles` listed, with its descriptions, and what
 * `DESKTOP_PROFILES` says the app can do with each.
 *
 * @param {readonly ListedProfile[]} listed
 * @returns {DesktopProfile[]}
 */
function desktopProfiles(listed) {
  return listed.map(({name, description}) => {
    const known = Object.hasOwn(DESKTOP_PROFILES, name) ? DESKTOP_PROFILES[name] : undefined;
    const {runnable, trainable, reason} = known ?? {runnable: false, trainable: false, reason: UNKNOWN};
    return {name, description, runnable, trainable, ...(reason === undefined ? {} : {unavailableReason: reason})};
  });
}

module.exports = {DESKTOP_PROFILES, desktopProfiles};
