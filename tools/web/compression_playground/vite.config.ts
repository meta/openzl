// Copyright (c) Meta Platforms, Inc. and affiliates.

import {createWebToolConfig} from '../vite.base.ts';
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import type {Plugin} from 'vite';

type CspSources = [directive: string, source: string][];

// index.html carries the production CSP; the plugins below widen it.
function addCspSources(html: string, sources: CspSources): string {
  for (const [directive, source] of sources) {
    if (!html.includes(directive)) {
      throw new Error(`index.html Content-Security-Policy is missing "${directive}"`);
    }
    html = html.replace(directive, `${directive} ${source}`);
  }
  return html;
}

// The dev server additionally needs the inline React Refresh preamble from
// @vitejs/plugin-react and the HMR websocket.
const DEV_CSP_SOURCES: CspSources = [
  ["script-src 'self'", "'unsafe-inline'"],
  ["connect-src 'self'", 'ws://localhost:*'],
];

const devContentSecurityPolicy: Plugin = {
  name: 'dev-content-security-policy',
  apply: 'serve',
  transformIndexHtml: (html) => addCspSources(html, DEV_CSP_SOURCES),
};

// The property the MkDocs docs site reports to; keep this in sync with
// `extra.analytics.property` in doc/mkdocs/mkdocs.yml. The playground is a
// separate Vite build copied into that site, so the MkDocs tag never reaches it.
const GA_MEASUREMENT_ID = 'G-ND0YQ5Q2VM';

// The sources Google documents for GA4 under a CSP.
const GA_CSP_SOURCES: CspSources = [
  ["script-src 'self'", 'https://*.googletagmanager.com'],
  ["img-src 'self'", 'https://*.google-analytics.com https://*.googletagmanager.com'],
  [
    "connect-src 'self'",
    'https://*.google-analytics.com https://*.analytics.google.com https://*.googletagmanager.com',
  ],
];

// Emitted as its own file rather than inlined, so the CSP needs no
// `script-src 'unsafe-inline'`. gtag.js reads the live `arguments` object off
// `dataLayer`, so `gtag` cannot take a rest parameter instead.
const GA_BOOTSTRAP = `window.dataLayer = window.dataLayer || [];
function gtag() {
  window.dataLayer.push(arguments);
}
gtag('js', new Date());
Promise.resolve(window.openzlIsolationSettled).then(function () {
  gtag('config', '${GA_MEASUREMENT_ID}');
});
`;

// Production builds only. The dev server's traffic would land in the real
// property, and a build in any other `--mode` is not the public site: the
// desktop app builds this page as its renderer with `--mode desktop`.
function googleAnalytics(): Plugin {
  let base = '/';
  return {
    name: 'google-analytics',
    apply: (_config, {command, mode}) => command === 'build' && mode === 'production',
    configResolved(config) {
      base = config.base;
    },
    generateBundle() {
      this.emitFile({type: 'asset', fileName: 'analytics.js', source: GA_BOOTSTRAP});
    },
    // Runs after Vite has processed the page, so these URLs are final as given.
    transformIndexHtml: (html) => ({
      html: addCspSources(html, GA_CSP_SOURCES),
      tags: [
        {
          // `crossorigin`, because the page is cross-origin isolated with COEP
          // `require-corp` (see cross-origin-isolation-sw.js).
          tag: 'script',
          attrs: {
            async: true,
            crossorigin: 'anonymous',
            src: `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`,
          },
          injectTo: 'head',
        },
        {tag: 'script', attrs: {src: `${base}analytics.js`}, injectTo: 'head'},
      ],
    }),
  };
}

// Emscripten's output, which only a WASM build produces: the GitHub workflows
// run one before the docs, but the internal docs build does not. Without it the
// worker bundle cannot resolve `wasm_api.js`'s dynamic import of it and fails,
// so it is left for the browser to request, as the main bundle already leaves
// it -- and `wasm_api.js` says what is missing when that request fails.
const hasWasmArtifact = existsSync(fileURLToPath(new URL('../../wasm/js/openzl.js', import.meta.url)));

const config = createWebToolConfig({
  base: '/tools/playground',
  plugins: [devContentSecurityPolicy, googleAnalytics()],
  testAlias: [
    {
      find: /^\/OpenZL_logo\.png/,
      replacement: fileURLToPath(new URL('./public/OpenZL_logo.png', import.meta.url)),
    },
  ],
  testSetupFiles: ['./tests/setup.ts'],
});

export default {
  ...config,
  // The benchmark worker is spawned with `{type: 'module'}`, so the build has
  // to emit it as one. The default IIFE output does not match and its imports
  // fail once built, though the dev server serves it either way.
  worker: {
    format: 'es' as const,
    ...(hasWasmArtifact ? {} : {rollupOptions: {external: [/\/openzl\.js$/]}}),
  },
  // Left unbundled so Vite's `new URL(..., import.meta.url)` transform runs
  // on it: esbuild's prebundling does not, and the wasm URL then points at a
  // file that is not there. The dev server answers those with the SPA
  // fallback, so it reads as a hang rather than a 404.
  optimizeDeps: {exclude: ['@bokuweb/zstd-wasm']},
  server: {
    // tools/wasm/js/ is a sibling of the Yarn workspace root, and the sample is
    // under examples/ at the repository root, so the dev server's default
    // fs.allow refuses both and their URLs 403. Only the sample's directory is
    // added, not the whole repository.
    fs: {allow: ['../..', '../../../examples/getting_started/sample_inputs']},
    // The OpenZL module is built with pthreads, which needs SharedArrayBuffer,
    // which browsers hand only to a cross-origin isolated document. The
    // deployed site cannot send these -- GitHub Pages serves fixed headers --
    // and gets isolation from a service worker instead. The dev server sends
    // them directly, which is also why it has to be reached over `localhost`:
    // isolation requires a secure context, and a bare hostname is not one.
    //
    // Worth knowing when the service worker lands: with these set, the dev
    // server cannot tell you whether it works.
    // The same values the service worker adds, so a cross-origin resource that
    // the deployed page would block is blocked here too.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
};
