// Copyright (c) Meta Platforms, Inc. and affiliates.

import {afterEach, describe, expect, it} from 'vitest';
import {startDevServer} from '../electron/dev-server.cjs';

type DevServer = Awaited<ReturnType<typeof startDevServer>>;

describe('startDevServer', () => {
  const started: DevServer[] = [];

  afterEach(async () => {
    await Promise.all(started.splice(0).map(({server}) => server.close()));
  });

  it('serves the playground on the port it chose, not the one it was given', async () => {
    const first = await startDevServer({port: 0});
    started.push(first);
    const port = Number(new URL(first.url).port);

    const second = await startDevServer({port});
    started.push(second);

    expect(Number(new URL(second.url).port)).not.toBe(port);
    const page = await fetch(second.url);
    expect(page.ok).toBe(true);
    expect(await page.text()).toContain('<div id="root"></div>');
  }, 30_000);
});
