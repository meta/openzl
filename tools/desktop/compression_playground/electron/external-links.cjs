// Copyright (c) Meta Platforms, Inc. and affiliates.

'use strict';

const ALLOWED_HOSTS = new Set(['openzl.org']);

/**
 * Whether a link from the UI may be opened in the system browser.
 *
 * @param {string} url
 */
function isAllowedExternalUrl(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  return (
    parsed.protocol === 'https:' &&
    ALLOWED_HOSTS.has(parsed.hostname) &&
    parsed.port === '' &&
    parsed.username === '' &&
    parsed.password === ''
  );
}

module.exports = {isAllowedExternalUrl};
