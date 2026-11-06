// Copyright (c) Meta Platforms, Inc. and affiliates.

import {afterEach, describe, expect, it, vi} from 'vitest';
import {fetchSample, SAMPLE_NAME} from '../src/fetchSample.ts';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('fetchSample', () => {
  it('downloads the example input and names it as a chosen file', async () => {
    const bytes = new Uint8Array([1, 2, 3, 4]);
    const download = vi.fn<typeof fetch>(() => Promise.resolve(new Response(bytes)));
    vi.stubGlobal('fetch', download);

    const sample = await fetchSample();

    expect(String(download.mock.calls[0][0])).toMatch(/sra0\.bin$/);
    expect(sample.name).toBe(SAMPLE_NAME);
    expect(new Uint8Array(await sample.arrayBuffer())).toEqual(bytes);
  });

  it('fails, naming the status, when the download does', async () => {
    vi.stubGlobal('fetch', () => Promise.resolve(new Response(null, {status: 503, statusText: 'Service Unavailable'})));

    await expect(fetchSample()).rejects.toThrow('Could not download the sample: 503 Service Unavailable');
  });

  it('leaves out a status text the server did not send', async () => {
    vi.stubGlobal('fetch', () => Promise.resolve(new Response(null, {status: 404})));

    await expect(fetchSample()).rejects.toThrow(/^Could not download the sample: 404$/);
  });

  it('says it was the sample when the network fails', async () => {
    vi.stubGlobal('fetch', () => Promise.reject(new TypeError('Failed to fetch')));

    await expect(fetchSample()).rejects.toThrow('Could not download the sample: Failed to fetch');
  });
});
