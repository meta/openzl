// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

const path = require('node:path');

const APP_SCHEME = 'app';
const APP_HOST = 'bundle';
const APP_ORIGIN = `${APP_SCHEME}://${APP_HOST}`;

/**
 * Maps an app:// request URL to a file inside `root`, or returns undefined when
 * the URL names another host or resolves outside the bundle.
 *
 * @param {string} root
 * @param {string} requestUrl
 * @returns {string | undefined}
 */
function resolveBundlePath(root, requestUrl) {
  let url;
  let pathname;
  try {
    url = new URL(requestUrl);
    // URL parsing already collapses literal and %2e dot segments, but an
    // encoded separator (..%2F) only becomes one after decoding.
    pathname = decodeURIComponent(url.pathname);
  } catch {
    return undefined;
  }
  if (url.protocol !== `${APP_SCHEME}:` || url.host !== APP_HOST) {
    return undefined;
  }
  const filePath = path.resolve(root, `.${pathname}`);
  const relative = path.relative(root, filePath);
  if (relative === '' || relative.startsWith('..') || path.isAbsolute(relative)) {
    return undefined;
  }
  return filePath;
}

/**
 * The scheme and host of `url`, or undefined when it does not parse. URL#origin
 * is "null" for custom schemes such as app:, so compare scheme and host.
 *
 * Undefined rather than a throw: this decides whether to trust a sender, and a
 * frame that is detached or mid-navigation can report a URL that is empty or
 * unparseable. That has to read as untrusted, not as an exception in the handler.
 *
 * @param {string} url
 * @returns {string | undefined}
 */
function originOf(url) {
  try {
    const {protocol: scheme, host} = new URL(url);
    return `${scheme}//${host}`;
  } catch {
    return undefined;
  }
}

module.exports = {APP_ORIGIN, APP_SCHEME, originOf, resolveBundlePath};
