// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment node

import {join, relative} from 'node:path';
import {describe, expect, it} from 'vitest';
import {APP_ORIGIN, originOf, resolveBundlePath} from '../electron/app-protocol.cjs';

const ROOT = '/opt/playground/dist';

describe('resolveBundlePath', () => {
  it('uses app://bundle as the renderer origin', () => {
    expect(APP_ORIGIN).toBe('app://bundle');
  });

  it('maps bundle URLs to files under the root', () => {
    expect(resolveBundlePath(ROOT, 'app://bundle/index.html')).toBe(join(ROOT, 'index.html'));
    expect(resolveBundlePath(ROOT, 'app://bundle/assets/index-abc123.js?v=1#x')).toBe(
      join(ROOT, 'assets', 'index-abc123.js'),
    );
    expect(resolveBundlePath(ROOT, 'app://bundle/assets/inter%20latin.woff2')).toBe(
      join(ROOT, 'assets', 'inter latin.woff2'),
    );
  });

  it('keeps dot segments inside the root', () => {
    for (const url of ['app://bundle/../../etc/passwd', 'app://bundle/%2e%2e/%2e%2e/etc/passwd']) {
      const resolved = resolveBundlePath(ROOT, url);
      expect(resolved).toBe(join(ROOT, 'etc', 'passwd'));
      expect(relative(ROOT, resolved ?? '')).not.toMatch(/^\.\./);
    }
  });

  it('rejects encoded separators that climb out of the root', () => {
    expect(resolveBundlePath(ROOT, 'app://bundle/..%2F..%2Fetc%2Fpasswd')).toBeUndefined();
    expect(resolveBundlePath(ROOT, 'app://bundle/assets/..%2F..%2F..%2Fsecret')).toBeUndefined();
  });

  it('rejects other hosts, schemes, the bare root, and malformed URLs', () => {
    for (const url of [
      'app://other/index.html',
      'https://bundle/index.html',
      'file:///opt/playground/dist/index.html',
      'app://bundle/',
      'app://bundle/%E0%A4%A',
      'not a url',
    ]) {
      expect(resolveBundlePath(ROOT, url)).toBeUndefined();
    }
  });
});

describe('originOf', () => {
  it('names the scheme and host, which URL#origin does not for custom schemes', () => {
    expect(originOf('app://bundle/index.html?x=1')).toBe(APP_ORIGIN);
    expect(originOf('http://localhost:5173/')).toBe('http://localhost:5173');
  });

  it('is undefined for a URL it cannot parse, so a sender check fails closed', () => {
    // A frame that has just detached, or is mid-navigation, can report these.
    expect(originOf('')).toBeUndefined();
    expect(originOf('not a url')).toBeUndefined();
  });
});
