// Copyright (c) Meta Platforms, Inc. and affiliates.

import type {ProfileMetadata} from '../../../wasm/js/wasm_api.js';
import type {OpenZlProfile} from './compressors.ts';

/** Metadata from the benchmark worker; looking up labels does not load WASM. */
let profiles: readonly ProfileMetadata[] = [];

let profileNames: readonly string[] = [];
let profileDescriptions: Readonly<Record<string, string>> = {};

export function setWasmProfiles(metadata: readonly ProfileMetadata[]): void {
  profiles = metadata;
  profileNames = metadata.map(({name}) => name).sort();
  profileDescriptions = Object.fromEntries(metadata.map(({name, description}) => [name, description]));
}

export function getOpenZlProfiles(): readonly string[] {
  return profileNames;
}

export function getOpenZlProfileDescriptions(): Readonly<Record<string, string>> {
  return profileDescriptions;
}

export function isBrowserSupportedProfile(name: string): boolean {
  return profiles.find((profile) => profile.name === name)?.isWasmSupported === true;
}

/** Whether the loaded WASM build can run and train this profile. */
export function isTrainableProfile(name: string): boolean {
  const profile = profiles.find((profile) => profile.name === name);
  return profile?.isWasmSupported === true && profile.isTrainable;
}

/** The profile's help text, plus why it is unselectable when it is. */
export function profileDescription(profile: OpenZlProfile): string {
  const description = profileDescriptions[profile] ?? '';
  const reason = profiles.find((entry) => entry.name === profile)?.wasmUnsupportedReason;
  if (isBrowserSupportedProfile(profile) || reason === undefined) {
    return description;
  }
  const sentence = description.endsWith('.') ? description : `${description}.`;
  return `${sentence} ${reason}`;
}

/** A profile as `zli list-profiles` describes it. */
export interface ListedProfile {
  readonly name: string;
  readonly description: string;
}

/**
 * Why the desktop app cannot run a profile, read off the CLI's description of
 * it, or undefined if it can. The app cannot pick a description file to pass
 * as `--profile-arg` yet, and the CLI marks unfinished profiles as placeholders.
 */
function desktopUnsupportedReason(description: string): string | undefined {
  if (/\bPass a path to\b/i.test(description)) {
    return 'Cannot run in the desktop app yet: needs a description file, which this app cannot pick.';
  }
  if (/\(Placeholder\)/i.test(description)) {
    return 'Cannot run in the desktop app yet: the CLI lists it as a placeholder.';
  }
  return undefined;
}

/**
 * The desktop app's profiles, from its `zli list-profiles`, in the form
 * `setWasmProfiles` takes. The desktop app runs `zli` instead of the WASM
 * build, so `isWasmSupported` here means the app can run the profile.
 */
export function desktopProfileMetadata(listed: readonly ListedProfile[]): readonly ProfileMetadata[] {
  return listed.map(({name, description}) => {
    const reason = desktopUnsupportedReason(description);
    return {
      name,
      description,
      isWasmSupported: reason === undefined,
      ...(reason === undefined ? {} : {wasmUnsupportedReason: reason}),
      isTrainable: !/\bTraining is not supported\b/i.test(description),
    };
  });
}
