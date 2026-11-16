// Copyright (c) Meta Platforms, Inc. and affiliates.

import {describe, expect, it} from 'vitest';
import {isAllowedExternalUrl} from '../electron/external-links.cjs';

describe('isAllowedExternalUrl', () => {
  it('allows https links to openzl.org', () => {
    for (const url of [
      'https://openzl.org/',
      'https://openzl.org/getting-started/quick-start/',
      'https://openzl.org/api/c/compressor/#section',
    ]) {
      expect(isAllowedExternalUrl(url)).toBe(true);
    }
  });

  it('rejects other schemes, hosts, and look-alikes', () => {
    for (const url of [
      'http://openzl.org/',
      'https://example.com/',
      'https://openzl.org.example.com/',
      'https://www.openzl.org/',
      'https://evil.com/?next=https://openzl.org/',
      'https://user:pass@openzl.org/',
      'https://openzl.org:8443/',
      'file:///etc/passwd',
      'javascript:alert(1)',
      'not a url',
      '',
    ]) {
      expect(isAllowedExternalUrl(url)).toBe(false);
    }
  });
});
