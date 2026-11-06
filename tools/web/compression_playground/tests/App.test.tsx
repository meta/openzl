// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import {describe, it, expect, afterEach, beforeEach, onTestFinished, vi} from 'vitest';
import {act, render, screen, cleanup, fireEvent, within, waitFor, type RenderOptions} from '@testing-library/react';
import {ChakraProvider} from '@chakra-ui/react';
import React from 'react';
import App from '../src/App.tsx';
import {SAMPLE_NAME} from '../src/fetchSample.ts';
import type {BenchmarkJob, DesktopRunConfig, RunMessage} from '../src/benchmarkTypes.ts';
import type {OpenZLDesktopBridge} from '../src/desktop.ts';
import {playgroundSystem} from '../src/theme.ts';
import {DEFAULT_TRAINED_CANDIDATES} from '../src/compressors.ts';
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
    // The sample needs no file, but it is still a run.
    expect(screen.getByRole('button', {name: 'Try a sample'})).toBeDisabled();

    fireEvent.click(levels.getByRole('checkbox', {name: 'Level 1'}));
    await waitFor(() => {
      expect(run).not.toHaveAttribute('aria-disabled');
    });
    expect(run).not.toHaveAttribute('aria-describedby');
  });

  it('downloads the sample when it is asked for and runs it as the chosen file', async () => {
    const bytes = new Uint8Array(4000).fill(7);
    const fetchSample = vi.fn<typeof fetch>(() => Promise.resolve(new Response(bytes)));
    vi.stubGlobal('fetch', fetchSample);
    const runBenchmark = vi.spyOn(benchmark, 'runBenchmark').mockImplementation(() => undefined);
    renderWithPlaygroundTheme(<App />);
    const sample = await screen.findByRole('button', {name: 'Try a sample'});
    await waitFor(() => {
      expect(sample).toBeEnabled();
    });
    expect(fetchSample).not.toHaveBeenCalled();

    fireEvent.click(sample);

    await waitFor(() => {
      expect(runBenchmark).toHaveBeenCalledTimes(1);
    });
    expect(fetchSample).toHaveBeenCalledTimes(1);
    const input = runBenchmark.mock.calls[0][0].input;
    expect(input.name).toBe(SAMPLE_NAME);
    expect(input.size).toBe(bytes.length);
    // And it is the file now chosen, so Run repeats it.
    expect(within(screen.getByTestId('upload-dropzone')).getByText(SAMPLE_NAME)).toBeInTheDocument();
  });

  it('reports a sample that fails to download, rather than running without it', async () => {
    vi.stubGlobal('fetch', () => Promise.resolve(new Response(null, {status: 404, statusText: 'Not Found'})));
    const runBenchmark = vi.spyOn(benchmark, 'runBenchmark').mockImplementation(() => undefined);
    renderWithPlaygroundTheme(<App />);
    const sample = await screen.findByRole('button', {name: 'Try a sample'});
    await waitFor(() => {
      expect(sample).toBeEnabled();
    });

    fireEvent.click(sample);

    expect(await screen.findByText(/Could not download the sample: 404 Not Found/)).toBeInTheDocument();
    expect(runBenchmark).not.toHaveBeenCalled();
    expect(sample).toBeEnabled();
    expect(screen.getByLabelText(/browse your computer/)).toBeEnabled();
  });

  it('locks the file picker while the sample downloads', async () => {
    let finishDownload: (response: Response) => void = () => undefined;
    vi.stubGlobal(
      'fetch',
      () =>
        new Promise<Response>((resolve) => {
          finishDownload = resolve;
        }),
    );
    vi.spyOn(benchmark, 'runBenchmark').mockImplementation(() => undefined);
    renderWithPlaygroundTheme(<App />);
    const sample = await screen.findByRole('button', {name: 'Try a sample'});
    await waitFor(() => {
      expect(sample).toBeEnabled();
    });
    const picker = screen.getByLabelText(/browse your computer/);
    expect(picker).toBeEnabled();

    fireEvent.click(sample);

    expect(screen.getByText(/Preparing for benchmarks/)).toBeInTheDocument();
    expect(picker).toBeDisabled();
    await act(async () => {
      finishDownload(new Response(new Uint8Array(4000)));
      await Promise.resolve();
    });

    expect(within(screen.getByTestId('upload-dropzone')).getByText(SAMPLE_NAME)).toBeInTheDocument();
  });

  it('runs the sample with the rows as they are once it has downloaded', async () => {
    let finishDownload: (response: Response) => void = () => undefined;
    vi.stubGlobal(
      'fetch',
      () =>
        new Promise<Response>((resolve) => {
          finishDownload = resolve;
        }),
    );
    const runBenchmark = vi.spyOn(benchmark, 'runBenchmark').mockImplementation(() => undefined);
    renderWithPlaygroundTheme(<App />);
    const sample = await screen.findByRole('button', {name: 'Try a sample'});
    await waitFor(() => {
      expect(sample).toBeEnabled();
    });

    fireEvent.click(sample);
    const trigger = screen.getByRole('button', {name: /, row 3$/});
    fireEvent.click(trigger);
    const levels = within(
      await waitFor(() => document.getElementById(trigger.getAttribute('aria-controls') ?? '') as HTMLElement),
    );
    fireEvent.click(levels.getByText('None'));
    await act(async () => {
      finishDownload(new Response(new Uint8Array(4000)));
      await Promise.resolve();
    });

    // Emptied during the download, the row holds the run back as it would Run.
    const run = screen.getByRole('button', {name: 'Run benchmark'});
    await waitFor(() => {
      expect(run).toHaveAccessibleDescription('Pick at least one level for row 3.');
    });
    expect(runBenchmark).not.toHaveBeenCalled();
    expect(within(screen.getByTestId('upload-dropzone')).getByText(SAMPLE_NAME)).toBeInTheDocument();

    fireEvent.click(levels.getByRole('checkbox', {name: 'Level 1'}));
    await waitFor(() => {
      expect(run).not.toHaveAttribute('aria-disabled');
    });
    fireEvent.click(run);
    expect(runBenchmark).toHaveBeenCalledTimes(1);
    const config = runBenchmark.mock.calls[0][0];
    expect(config.input.name).toBe(SAMPLE_NAME);
    expect(config.compressors[2].levels).toEqual([1]);
  });

  it('renders the static Results panel', () => {
    renderWithPlaygroundTheme(<App />);
    expect(screen.getByRole('heading', {level: 2, name: 'Results'})).toBeInTheDocument();
    expect(screen.getByText('No results yet')).toBeInTheDocument();
    expect(screen.getByText(/Try a sample.*to see the playground work/)).toBeInTheDocument();
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
          expect.objectContaining({
            profile: 'serial',
            training: {paretoFrontier: true, candidates: DEFAULT_TRAINED_CANDIDATES},
          }),
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

/**
 * A backend that answers every run the page sends, in the browser worker's
 * messages: one job per compressor per level, one measurement each, and three
 * candidates for a job that trains.
 */
function answeringBridge(getPathForFile: OpenZLDesktopBridge['getPathForFile'] = () => '/tmp/dataset.bin') {
  const startBenchmark = vi.fn((config: DesktopRunConfig, onMessage?: (message: RunMessage) => void) => {
    const jobs: BenchmarkJob[] = config.compressors.flatMap((compressor) =>
      compressor.levels.map((level) => {
        const base = {
          id: `${compressor.rowId}-${compressor.compressor}-${level}`,
          rowId: compressor.rowId,
          level,
          iterations: config.iterations,
        };
        return compressor.compressor === 'OpenZL'
          ? {
              ...base,
              compressor: 'OpenZL' as const,
              profile: compressor.profile,
              profileArg: compressor.profileArg,
              training: compressor.training,
            }
          : {...base, compressor: compressor.compressor};
      }),
    );
    onMessage?.({type: 'loading'});
    onMessage?.({type: 'started', totalJobs: jobs.length, rejected: []});
    for (const job of jobs) {
      const total = job.compressor === 'OpenZL' && job.training !== null ? 3 : 1;
      for (let index = 1; index <= total; index += 1) {
        onMessage?.({
          type: 'result',
          result: {
            job,
            candidate: total > 1 ? {index, total} : null,
            iterations: config.iterations,
            srcSize: 5_000_000,
            compressedSize: 1_000_000,
            ratio: 5,
            compressMs: 10,
            decompressMs: 5,
            compressMBps: 100,
            decompressMBps: 200,
          },
        });
      }
    }
    onMessage?.({type: 'finished'});
    return {done: Promise.resolve(), cancel: vi.fn()};
  });
  window.openzlDesktop = {startBenchmark, getPathForFile, listProfiles: () => Promise.reject(new Error('no zli'))};
  return startBenchmark;
}

describe('Compression Playground in the desktop app', () => {
  afterEach(() => {
    cleanup();
    delete window.openzlDesktop;
  });

  it('runs the chosen file through the desktop backend and reports it like any other run', async () => {
    const startBenchmark = answeringBridge();
    renderWithPlaygroundTheme(<App />);
    fireEvent.change(screen.getByLabelText(/browse your computer/), {
      target: {files: [new File(['x'], 'dataset.bin')]},
    });
    fireEvent.click(screen.getByRole('button', {name: 'Run benchmark'}));

    // The same done line the browser's run ends on: nothing past the executor
    // knows which backend measured.
    expect(await screen.findByText(/Done — 5\.0 MB · 3 succeeded, 0 failed/)).toBeInTheDocument();
    // Those numbers came from native zli, so the browser's WebAssembly caveat
    // would be wrong beside them.
    expect(screen.getByRole('table', {name: 'Measurements'})).toBeInTheDocument();
    expect(screen.queryByText(/WebAssembly/)).not.toBeInTheDocument();
    expect(startBenchmark).toHaveBeenCalledTimes(1);
    const [config] = startBenchmark.mock.calls[0];
    // The browser's RunConfig, with the input named by its path: OpenZL, zstd
    // and gzip, each with its levels.
    expect(config).toMatchObject({input: {path: '/tmp/dataset.bin'}, iterations: 5});
    expect(config.compressors.map((compressor) => compressor.compressor)).toEqual(['OpenZL', 'zstd', 'gzip']);
    expect(config.compressors[0]).toMatchObject({
      compressor: 'OpenZL',
      levels: [6],
      profile: 'serial',
      profileArg: '',
      // The desktop trainer's floor, which both builds now offer.
      training: {paretoFrontier: true, candidates: 10},
    });
  });

  it('runs the bundled sample without a file being chosen', async () => {
    // The sample is installed with the app, so the page never reads or names
    // a path for it: the main process supplies where it lives.
    const startBenchmark = answeringBridge(() => {
      throw new Error('the sample has no File to resolve');
    });
    renderWithPlaygroundTheme(<App />);
    const sample = screen.getByRole('button', {name: 'Try a sample'});
    expect(sample).toBeEnabled();

    fireEvent.click(sample);

    expect(await screen.findByText(/3 succeeded, 0 failed/)).toBeInTheDocument();
    expect(startBenchmark.mock.calls[0][0].input).toMatchObject({kind: 'bundled-sample', name: 'sra0.bin'});
    // And it is the file now chosen, so Run repeats it.
    const dropzone = within(screen.getByTestId('upload-dropzone'));
    expect(dropzone.getByText('sra0.bin')).toBeInTheDocument();
    expect(dropzone.getByText(/2\.1 MB/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name: 'Run benchmark'}));
    await waitFor(() => {
      expect(startBenchmark).toHaveBeenCalledTimes(2);
    });
    expect(startBenchmark.mock.calls[1][0].input).toMatchObject({kind: 'bundled-sample', name: 'sra0.bin'});
  });

  it('holds the sample back while a row has no levels, as it does Run', async () => {
    answeringBridge();
    renderWithPlaygroundTheme(<App />);
    const trigger = screen.getByRole('button', {name: /, row 3$/});
    fireEvent.click(trigger);
    const levels = within(
      await waitFor(() => document.getElementById(trigger.getAttribute('aria-controls') ?? '') as HTMLElement),
    );
    fireEvent.click(levels.getByText('None'));

    await waitFor(() => {
      expect(screen.getByRole('button', {name: 'Try a sample'})).toBeDisabled();
    });
  });

  it('offers the profiles the app’s zli lists, including ones this page does not know', async () => {
    answeringBridge();
    const listProfiles = vi.fn(() =>
      Promise.resolve([
        {name: 'serial', description: 'Serial data (aka raw bytes)'},
        {name: 'brand-new', description: 'Some format the CLI learned yesterday'},
        {name: 'sddl3', description: 'Pass a path to the description file with --profile-arg.'},
      ]),
    );
    window.openzlDesktop = {...(window.openzlDesktop as OpenZLDesktopBridge), listProfiles};
    onTestFinished(() => {
      setWasmProfiles(TEST_PROFILES);
    });
    renderWithPlaygroundTheme(<App />);

    await waitFor(() => {
      expect(listProfiles).toHaveBeenCalledTimes(1);
    });
    fireEvent.click(screen.getByRole('combobox', {name: 'Profile for row 1'}));
    const listbox = await screen.findByRole('listbox');
    await waitFor(() => {
      expect(within(listbox).getByRole('option', {name: 'brand-new'})).toBeInTheDocument();
    });
    expect(within(listbox).getByRole('option', {name: 'sddl3'})).toHaveAttribute('aria-disabled', 'true');
    // Only what zli listed.
    expect(within(listbox).queryByRole('option', {name: 'parquet'})).not.toBeInTheDocument();
  });

  it('shows the CLI’s profiles and links the header off the app:// origin', () => {
    answeringBridge();
    renderWithPlaygroundTheme(<App />);

    // The narrow-screen menu repeats the links, so look only in the header's row of links.
    const navigation = screen.getByRole('navigation', {name: 'OpenZL'});
    expect(within(navigation).getByRole('link', {name: 'API REFERENCE'})).toHaveAttribute(
      'href',
      'https://openzl.org/api/c/compressor/',
    );
    expect(screen.getByRole('link', {name: 'OpenZL home'})).toHaveAttribute('href', 'https://openzl.org/');
  });

  it('runs where the browser playground would be blocked', () => {
    // app:// cannot host the isolation worker, so the page is never isolated,
    // and the desktop app never loads the WASM module that would need it.
    stubWebAssembly(false);
    stubIsolation(false);
    answeringBridge();
    renderWithPlaygroundTheme(<App />);

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('heading', {level: 2, name: 'Choose your data'})).toBeInTheDocument();
  });

  it('does not recommend the desktop app to someone already in it', () => {
    answeringBridge();
    renderWithPlaygroundTheme(<App />);

    expect(screen.queryByText(/Use the desktop app/)).not.toBeInTheDocument();
  });
});
