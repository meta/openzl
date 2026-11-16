// Copyright (c) Meta Platforms, Inc. and affiliates.

import {OPENZL_NAVIGATION} from '@openzl/web-common';
import type {Backend} from './backend.ts';
import {
  DESKTOP_DEFAULT_TRAINED_CANDIDATES,
  DESKTOP_TRAINED_CANDIDATE_COUNTS,
  loadDesktopProfiles,
  type OpenZLDesktopBridge,
} from './desktop.ts';
import {runDesktopBenchmark} from './runDesktopBenchmark.ts';
import {BUNDLED_SAMPLE} from './sample.ts';

/** Where the header's site links point in the desktop app, served from app://. */
const OPENZL_SITE = 'https://openzl.org/';

export function desktopBackend(bridge: OpenZLDesktopBridge): Backend {
  return {
    settings: {
      candidateCounts: DESKTOP_TRAINED_CANDIDATE_COUNTS,
      defaultCandidates: DESKTOP_DEFAULT_TRAINED_CANDIDATES,
      unavailableHeading: 'Not available in the desktop app',
      // The measurements come from native `zli`.
      showWasmNotice: false,
      // None of the browser's engine gates apply: the module is never loaded,
      // and they would fail anyway, since app:// cannot host the isolation
      // worker and `threads` always reads false.
      checkEngineSupport: false,
      // The page is served from app://, where the header's relative links
      // point nowhere.
      headerLinks: {
        homeHref: OPENZL_SITE,
        navigation: OPENZL_NAVIGATION.map(({label, href}) => ({label, href: new URL(href, OPENZL_SITE).href})),
      },
    },
    loadProfiles: () => loadDesktopProfiles(bridge),
    run: (config, onState) => {
      runDesktopBenchmark(config, bridge, onState);
    },
    // Installed with the app, so there is nothing to download: the backend
    // opens it by name.
    loadSample: () =>
      Promise.resolve((setup, onState) => {
        runDesktopBenchmark({input: BUNDLED_SAMPLE, ...setup}, bridge, onState);
      }),
  };
}
