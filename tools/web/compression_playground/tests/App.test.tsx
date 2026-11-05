// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import {describe, it, expect, afterEach, beforeEach, vi} from 'vitest';
import {act, render, screen, cleanup, fireEvent, within, waitFor, type RenderOptions} from '@testing-library/react';
import {ChakraProvider} from '@chakra-ui/react';
import React from 'react';
import App from '../src/App.tsx';
import {playgroundSystem} from '../src/theme.ts';
import * as benchmark from '../src/runBenchmark.ts';
import {setWasmProfiles} from '../src/wasmProfiles.ts';
import {TEST_PROFILES} from './setup.ts';

function renderWithPlaygroundTheme(ui: React.ReactElement, options?: Omit<RenderOptions, 'wrapper'>) {
  return render(ui, {
    wrapper: ({children}) => <ChakraProvider value={playgroundSystem}>{children}</ChakraProvider>,
    ...options,
  });
}

/**
 * Stands in for the engine so which branch App renders is a property of the
 * test rather than of the Node version running it: Node 24 supports Memory64,
 * so the real `validate` would always take the supported path.
 */
function stubWebAssembly(acceptsMemory64: boolean) {
  vi.stubGlobal('WebAssembly', {validate: () => acceptsMemory64});
}

/** jsdom does not report the page as cross-origin isolated, so threads would read as missing. */
function stubIsolation(isolated: boolean) {
  vi.stubGlobal('SharedArrayBuffer', ArrayBuffer);
  vi.stubGlobal('crossOriginIsolated', isolated);
}

beforeEach(() => {
  stubIsolation(true);
});

// Unmount the rendered tree after each test so the DOM doesn't accumulate
// across tests (which would make queries match duplicate elements).
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  setWasmProfiles(TEST_PROFILES);
  vi.unstubAllGlobals();
});

describe('Compression Playground', () => {
  beforeEach(() => {
    stubWebAssembly(true);
  });

  it('renders the shared header', () => {
    renderWithPlaygroundTheme(<App />);
    expect(screen.getByRole('main')).toBeTruthy();
    expect(screen.getByRole('heading', {level: 1, name: 'Compression Playground'})).toBeInTheDocument();
    // The narrow-screen menu repeats the links, and jsdom does not load the CSS
    // that hides it, so look only in the header's row of links.
    const navigation = screen.getByRole('navigation', {name: 'OpenZL'});
    expect(within(navigation).getByRole('link', {name: 'API REFERENCE'})).toHaveAttribute('href', '/api/c/compressor/');
    // Vite resolves the asset differently under test (through the alias in
    // vite.config.ts) than in a build (behind `base`), so assert the asset
    // rather than a literal path.
    expect(screen.getByRole('img', {name: 'OpenZL'})).toHaveAttribute(
      'src',
      expect.stringContaining('OpenZL_logo.png'),
    );
  });

  it('renders the setup steps and the data upload control', () => {
    renderWithPlaygroundTheme(<App />);
    expect(screen.getByRole('heading', {level: 2, name: 'Choose your data'})).toBeInTheDocument();
    expect(screen.getByText('Choose data to compress, files are not uploaded to a server')).toBeInTheDocument();
    expect(screen.getByText('Drag & drop a file here')).toBeInTheDocument();
    expect(screen.getByText('browse your computer')).toBeInTheDocument();
    expect(screen.getByRole('heading', {level: 2, name: 'Configure the run'})).toBeInTheDocument();
    expect(screen.getByText('Select which compressors to compare')).toBeInTheDocument();
    expect(screen.getByRole('heading', {level: 2, name: 'Run the benchmark'})).toBeInTheDocument();
    expect(screen.getByText('Measure ratio and speed for every compressor you selected')).toBeInTheDocument();
  });

  it('holds the chosen file so later steps can read it', () => {
    renderWithPlaygroundTheme(<App />);
    const file = new File(['x'.repeat(4_700_000)], 'dataset.bin', {type: 'application/octet-stream'});

    fireEvent.change(screen.getByLabelText(/browse your computer/), {target: {files: [file]}});

    // Scoped to the drop zone: the live region repeats the name and size.
    const dropzone = within(screen.getByTestId('upload-dropzone'));
    expect(dropzone.getByText('dataset.bin')).toBeInTheDocument();
    expect(dropzone.getByText(/4\.7 MB/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', {name: 'Remove dataset.bin'}));

    expect(screen.queryByText('dataset.bin')).not.toBeInTheDocument();
    expect(screen.getByText('Drag & drop a file here')).toBeInTheDocument();
  });

  it('holds Run back while a zstd or gzip row has no levels, and says why', async () => {
    // Such a row would drop out of the run with nothing on the page to say so.
    renderWithPlaygroundTheme(<App />);
    const run = screen.getByRole('button', {name: 'Run benchmark'});
    expect(run).toHaveAttribute('aria-disabled', 'true');
    expect(run).toHaveAccessibleDescription('Choose a file in step 1 to run the benchmark.');

    fireEvent.change(screen.getByLabelText(/browse your computer/), {
      target: {files: [new File(['x'], 'data.bin')]},
    });
    expect(run).not.toHaveAttribute('aria-disabled');
    expect(run).not.toHaveAttribute('aria-describedby');

    const trigger = screen.getByRole('button', {name: /, row 3$/});
    fireEvent.click(trigger);
    const levels = within(
      await waitFor(() => document.getElementById(trigger.getAttribute('aria-controls') ?? '') as HTMLElement),
    );
    fireEvent.click(levels.getByText('None'));
    await waitFor(() => {
      expect(run).toHaveAttribute('aria-disabled', 'true');
    });
    expect(run).toHaveAccessibleDescription('Pick at least one level for row 3.');

    fireEvent.click(levels.getByRole('checkbox', {name: 'Level 1'}));
    await waitFor(() => {
      expect(run).not.toHaveAttribute('aria-disabled');
    });
    expect(run).not.toHaveAttribute('aria-describedby');
  });

  it('renders the static Results panel', () => {
    renderWithPlaygroundTheme(<App />);
    expect(screen.getByRole('heading', {level: 2, name: 'Results'})).toBeInTheDocument();
    expect(screen.getByText('No results yet')).toBeInTheDocument();
    expect(screen.getByText(/Try a 5 MB sample.*to see the playground work/)).toBeInTheDocument();
    expect(screen.getByRole('heading', {level: 3, name: 'How it works'})).toBeInTheDocument();
    expect(screen.getByText('Choose your data.')).toBeInTheDocument();
    expect(screen.getByRole('heading', {level: 3, name: 'Reading the results'})).toBeInTheDocument();
    expect(screen.getByText(/What the charts say/)).toBeInTheDocument();
    expect(screen.getByText(/Train an OpenZL compressor/)).toBeInTheDocument();
    expect(screen.queryByText(/desktop app/i)).not.toBeInTheDocument();
  });

  it('waits for profiles before running and preserves the training request', async () => {
    setWasmProfiles([]);
    let finishLoading!: () => void;
    const profiles = new Promise<typeof TEST_PROFILES>((resolve) => {
      finishLoading = () => {
        setWasmProfiles(TEST_PROFILES);
        resolve(TEST_PROFILES);
      };
    });
    const discovery = vi.mocked(benchmark.getBenchmarkProfiles);
    discovery.mockClear();
    discovery.mockReturnValueOnce(profiles);
    const runBenchmark = vi.spyOn(benchmark, 'runBenchmark').mockImplementation(() => undefined);

    renderWithPlaygroundTheme(<App />);
    fireEvent.change(screen.getByLabelText(/browse your computer/), {
      target: {files: [new File(['x'], 'data.bin')]},
    });
    const run = screen.getByRole('button', {name: 'Run benchmark'});
    expect(run).not.toHaveAttribute('aria-disabled');
    expect(run).toBeDisabled();
    expect(screen.queryByText('Loading compression profiles…')).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox', {name: 'TRAIN'})).toBeDisabled();
    fireEvent.click(run);
    expect(runBenchmark).not.toHaveBeenCalled();
    finishLoading();
    await waitFor(() => {
      expect(screen.getByRole('checkbox', {name: 'TRAIN'})).toBeChecked();
      expect(run).toBeEnabled();
    });
    fireEvent.click(run);
    expect(runBenchmark).toHaveBeenCalledWith(
      expect.objectContaining({
        compressors: expect.arrayContaining([
          expect.objectContaining({profile: 'serial', training: {paretoFrontier: true, candidates: 6}}),
        ]),
      }),
      expect.any(Function),
    );
    expect(discovery).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('does not show the unsupported-browser message', () => {
    renderWithPlaygroundTheme(<App />);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

describe('on a browser that cannot load the module', () => {
  it('names the missing feature and links to the support table', () => {
    // Safari's failure mode: WebAssembly is present, Memory64 is not.
    stubWebAssembly(false);
    renderWithPlaygroundTheme(<App />);

    expect(screen.getByRole('heading', {level: 2, name: /Your browser can.t run the playground/})).toBeInTheDocument();
    expect(screen.getByText(/needs 64-bit WebAssembly memory, which this browser doesn.t have/)).toBeInTheDocument();
    expect(screen.getByRole('link', {name: /which browsers support it/})).toHaveAttribute(
      'href',
      'https://caniuse.com/wf-wasm-memory64',
    );
    expect(screen.getByText(/Chrome 133\+, Edge 133\+ and Firefox 134\+ can run it/)).toBeInTheDocument();
    // Every iOS browser is WebKit, so naming Chrome there would only send an
    // iPhone user back to this screen.
    expect(screen.getByText(/except on iPhone and iPad, where every browser uses Safari.s engine/)).toBeInTheDocument();
    expect(screen.queryByText(/desktop app/i)).not.toBeInTheDocument();
  });

  it('replaces the setup and results columns rather than sitting alongside them', () => {
    stubWebAssembly(false);
    renderWithPlaygroundTheme(<App />);

    // The header is the only way off this screen, so it has to stay.
    expect(screen.getByRole('heading', {level: 1, name: 'Compression Playground'})).toBeInTheDocument();
    expect(screen.queryByRole('heading', {level: 2, name: 'Choose your data'})).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', {level: 2, name: 'Results'})).not.toBeInTheDocument();
  });

  it('does not blame Memory64 when WebAssembly is missing entirely', () => {
    vi.stubGlobal('WebAssembly', undefined);
    renderWithPlaygroundTheme(<App />);

    expect(screen.getByText(/needs WebAssembly, which this browser doesn.t have/)).toBeInTheDocument();
    expect(screen.queryByRole('link', {name: /which browsers support it/})).not.toBeInTheDocument();
    expect(document.querySelector('a[href*="caniuse.com"]')).toBeNull();
    // The recommendation is the actionable half, so it survives either browser blocker.
    expect(screen.getByText(/Chrome 133\+, Edge 133\+ and Firefox 134\+ can run it/)).toBeInTheDocument();
  });
});

describe('on a page that is not cross-origin isolated', () => {
  beforeEach(() => {
    stubWebAssembly(true);
    stubIsolation(false);
  });

  it('says isolation did not take effect rather than blaming the browser', async () => {
    renderWithPlaygroundTheme(<App />);

    expect(
      await screen.findByRole('heading', {level: 2, name: /The playground couldn.t start here/}),
    ).toBeInTheDocument();
    expect(screen.getByText(/sets that up with a service worker, and it didn.t take effect/)).toBeInTheDocument();
    expect(screen.queryByText(/Your browser can.t/)).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', {level: 2, name: 'Choose your data'})).not.toBeInTheDocument();
  });

  it('suggests a reload or a normal window instead of another browser', async () => {
    // These are the states the worker gives up in: no service workers, as in
    // some private windows, a failed registration, or an install that failed.
    // A different browser is rarely the fix for any of them.
    renderWithPlaygroundTheme(<App />);

    expect(await screen.findByText(/Try reloading the page, or opening it in a normal window/)).toBeInTheDocument();
    expect(screen.queryByText(/Chrome 133\+/)).not.toBeInTheDocument();
    expect(document.querySelector('a[href*="caniuse.com"]')).toBeNull();
  });

  it('shows nothing while the isolation worker may still reload the page', async () => {
    // A first visit: the worker has not reloaded the page isolated yet, and
    // the blocked screen would otherwise flash up until it does.
    let settle: () => void = () => undefined;
    vi.stubGlobal(
      'openzlIsolationSettled',
      new Promise<void>((resolve) => {
        settle = resolve;
      }),
    );
    renderWithPlaygroundTheme(<App />);
    // Lets a promise that had already resolved reach the page, so the checks
    // below would catch an App that did not wait for this one.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(screen.getByRole('heading', {level: 1, name: 'Compression Playground'})).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', {level: 2, name: 'Choose your data'})).not.toBeInTheDocument();

    // The worker gave up without reloading, so the page stays unisolated.
    settle();
    expect(await screen.findByRole('alert')).toBeInTheDocument();
  });
});
