// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import {afterEach, beforeAll, describe, expect, it} from 'vitest';
import {cleanup, fireEvent, render, screen, waitFor, within} from '@testing-library/react';
import {ChakraProvider} from '@chakra-ui/react';
import RatioSpeedCharts from '../src/components/RatioSpeedCharts.tsx';
import ResultsPanel from '../src/components/ResultsPanel.tsx';
import {playgroundSystem} from '../src/theme.ts';
import type {BenchmarkJob, JobResult} from '../src/benchmarkTypes.ts';

/**
 * `ResponsiveContainer` measures its parent, and jsdom reports every element as
 * 0x0, so without this the charts render as an empty frame with no points.
 */
beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', {configurable: true, value: 400});
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', {configurable: true, value: 200});
  Object.defineProperty(HTMLElement.prototype, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({width: 400, height: 200, top: 0, left: 0, bottom: 200, right: 400, x: 0, y: 0}),
  });
});

afterEach(cleanup);

function result(
  job: Partial<BenchmarkJob> & Pick<BenchmarkJob, 'compressor'>,
  metrics: Partial<JobResult> = {},
): JobResult {
  return {
    iterations: 1,
    srcSize: 5_000_000,
    compressedSize: 100,
    ratio: 10,
    compressMs: 1,
    decompressMs: 1,
    compressMBps: 100,
    decompressMBps: 200,
    candidate: null,
    ...metrics,
    job: {
      id: `${job.rowId ?? 1}-${job.compressor}-${job.level ?? 1}`,
      rowId: 1,
      level: 1,
      iterations: 1,
      profile: 'serial',
      training: null,
      ...job,
    } as BenchmarkJob,
  };
}

const TRAINED_JOB = {
  id: 'trained-1',
  compressor: 'OpenZL',
  rowId: 1,
  training: {paretoFrontier: true, candidates: 6},
} as const;

const RESULTS = [
  result(TRAINED_JOB, {candidate: {index: 1, total: 2}, ratio: 15235, compressMBps: 38}),
  result(TRAINED_JOB, {candidate: {index: 2, total: 2}, ratio: 2763, compressMBps: 310}),
  result({compressor: 'zstd', rowId: 2, level: 1}, {ratio: 1450, compressMBps: 520}),
  result({compressor: 'gzip', rowId: 3, level: 6}, {ratio: 342, compressMBps: 162}),
];

function renderCharts(results = RESULTS, highlighted: string | null = null) {
  return render(
    <ChakraProvider value={playgroundSystem}>
      <RatioSpeedCharts results={results} highlighted={highlighted} />
    </ChakraProvider>,
  );
}

/**
 * The plot arrives a tick after everything around it, so anything that reads
 * the SVG waits for it first. Without this a check that nothing is painted
 * would pass on a plot that has not rendered yet.
 */
async function renderPlotted(results = RESULTS, highlighted: string | null = null) {
  const rendered = renderCharts(results, highlighted);
  await waitFor(() => {
    expect(rendered.container.querySelectorAll('.recharts-scatter-symbol').length).toBeGreaterThan(0);
  });
  return rendered;
}

describe('RatioSpeedCharts', () => {
  it('renders nothing without measurements', () => {
    const {container} = renderCharts([]);
    expect(container).toBeEmptyDOMElement();
  });

  it('lays out everything around the plot before the plot arrives', () => {
    // So the page does not move when recharts lands: the frame is already at
    // its final size, and only the inside of the 200px plot area fills in.
    const {container} = renderCharts();
    expect(container.querySelectorAll('.recharts-scatter-symbol')).toHaveLength(0);
    expect(screen.getByText('Compress MB/s (log)')).toBeInTheDocument();
    expect(screen.getByText('OpenZL')).toBeInTheDocument();
  });

  it('plots every measurement on both charts', async () => {
    const {container} = await renderPlotted();
    // Two charts, four measurements each.
    expect(container.querySelectorAll('.recharts-scatter-symbol')).toHaveLength(8);
  });

  it('draws untrained OpenZL points without connecting lines', async () => {
    const {container} = await renderPlotted([
      result({id: 'untrained-1', compressor: 'OpenZL', rowId: 1}, {compressMBps: 90}),
      result({id: 'untrained-4', compressor: 'OpenZL', rowId: 4}, {compressMBps: 500}),
    ]);
    expect(container.querySelectorAll('.recharts-scatter-symbol')).toHaveLength(4);
    expect(container.querySelectorAll('.recharts-scatter-line')).toHaveLength(0);
  });

  it('connects each trained OpenZL frontier without including untrained points', async () => {
    const secondJob = {...TRAINED_JOB, id: 'trained-4', rowId: 4};
    const {container} = await renderPlotted([
      ...RESULTS.slice(0, 2),
      result({id: 'untrained-2', compressor: 'OpenZL', rowId: 2}),
      result(secondJob, {candidate: {index: 1, total: 2}, compressMBps: 150}),
      result(secondJob, {candidate: {index: 2, total: 2}, compressMBps: 400}),
    ]);

    const scatters = [...container.querySelectorAll('.recharts-scatter')];
    expect(scatters.map((scatter) => scatter.querySelectorAll('.recharts-scatter-symbol').length)).toEqual([
      2, 1, 2, 2, 1, 2,
    ]);
    for (const scatter of scatters) {
      const points = scatter.querySelectorAll('.recharts-scatter-symbol').length;
      expect(scatter.querySelectorAll('.recharts-scatter-line path')).toHaveLength(points === 2 ? 1 : 0);
    }
    expect(screen.getAllByText('OpenZL')).toHaveLength(1);
    const legend = screen.getByText('OpenZL').closest('div') as HTMLElement;
    expect(within(legend).getByText('5')).toBeInTheDocument();
  });

  it('keeps zstd and gzip levels connected on both charts', async () => {
    const {container} = await renderPlotted(
      (['zstd', 'gzip'] as const).flatMap((compressor) => [
        result({compressor, level: 1}, {compressMBps: 90}),
        result({compressor, level: 9}, {compressMBps: 500}),
      ]),
    );
    expect(container.querySelectorAll('.recharts-scatter-symbol')).toHaveLength(8);
    expect(container.querySelectorAll('.recharts-scatter-line path')).toHaveLength(4);
  });

  it('labels both axes as logarithmic and both throughputs', () => {
    renderCharts();
    expect(screen.getByText('Compress MB/s (log)')).toBeInTheDocument();
    expect(screen.getByText('Decompress MB/s (log)')).toBeInTheDocument();
    expect(screen.getAllByText('Compression ratio (log)')).toHaveLength(2);
  });

  it('counts each codec once in the legend, however many rows produced it', () => {
    renderCharts();
    const legend = screen.getByText('OpenZL').closest('div');
    expect(legend).not.toBeNull();
    expect(within(legend as HTMLElement).getByText('2')).toBeInTheDocument();
    expect(within(screen.getByText('gzip').closest('div') as HTMLElement).getByText('1')).toBeInTheDocument();
  });

  it('takes every colour from the theme, so the scheme can change them', async () => {
    // recharts writes `fill` and `stroke` as SVG attributes, which is why the
    // colours arrive as CSS variables rather than as token names. A literal
    // hex anywhere here is one the dark scheme cannot reach.
    const {container} = await renderPlotted();
    const painted = [...container.querySelectorAll('[fill], [stroke]')].flatMap((node) => [
      node.getAttribute('fill'),
      node.getAttribute('stroke'),
    ]);

    expect(painted.filter((value) => value?.startsWith('#'))).toEqual([]);
  });

  it('picks out the highlighted measurement on both charts, in its codec colour', async () => {
    // The OpenZL rows share `TRAINED_JOB`, so the second candidate is `trained-1-2`.
    const {container} = await renderPlotted(RESULTS, 'trained-1-2');

    const rings = [...container.querySelectorAll('.highlighted-point')];
    expect(rings).toHaveLength(2);
    const openzl = container.querySelector('.recharts-scatter-symbol path')?.getAttribute('fill');
    expect(rings.map((ring) => ring.getAttribute('fill'))).toEqual([openzl, openzl]);
  });

  it('picks out nothing when no row is highlighted', async () => {
    const {container} = await renderPlotted();
    expect(container.querySelectorAll('.highlighted-point')).toHaveLength(0);
  });

  it('highlights the point of the table row under the pointer, until it leaves', async () => {
    const results = [
      result({compressor: 'zstd', id: 'z1', level: 1}, {ratio: 3, compressMBps: 500}),
      result({compressor: 'gzip', id: 'g6', level: 6}, {ratio: 5, compressMBps: 50}),
    ];
    const {container} = render(
      <ChakraProvider value={playgroundSystem}>
        <ResultsPanel runState={{status: 'completed', results, failures: []}} />
      </ChakraProvider>,
    );
    await waitFor(() => {
      expect(container.querySelectorAll('.recharts-scatter-symbol').length).toBeGreaterThan(0);
    });

    const gzipRow = screen.getByRole('row', {name: /gzip/});
    fireEvent.mouseEnter(gzipRow);
    await waitFor(() => {
      expect(container.querySelectorAll('.highlighted-point')).toHaveLength(2);
    });
    const ring = container.querySelector('.highlighted-point');
    const gzipFill = [...container.querySelectorAll('.recharts-scatter-symbol path')]
      .map((node) => node.getAttribute('fill'))
      .find((fill) => fill?.includes('gzip'));
    expect(ring?.getAttribute('fill')).toBe(gzipFill);

    fireEvent.mouseLeave(gzipRow);
    expect(container.querySelectorAll('.highlighted-point')).toHaveLength(0);
  });

  it('takes no highlight while a run is in progress', async () => {
    // Rows arrive and re-sort under a pointer that has not moved, so the
    // highlight would stay on a row that is no longer under it.
    const results = [
      result({compressor: 'zstd', id: 'z1', level: 1}, {ratio: 3, compressMBps: 500}),
      result({compressor: 'gzip', id: 'g6', level: 6}, {ratio: 5, compressMBps: 50}),
    ];
    const running = {status: 'running', completedJobs: 2, totalJobs: 3, results, failures: [], step: null} as const;
    const {container, rerender} = render(
      <ChakraProvider value={playgroundSystem}>
        <ResultsPanel runState={running} />
      </ChakraProvider>,
    );
    await waitFor(() => {
      expect(container.querySelectorAll('.recharts-scatter-symbol').length).toBeGreaterThan(0);
    });

    fireEvent.mouseEnter(screen.getByRole('row', {name: /gzip/}));
    rerender(
      <ChakraProvider value={playgroundSystem}>
        <ResultsPanel runState={{status: 'completed', results, failures: []}} />
      </ChakraProvider>,
    );
    expect(container.querySelectorAll('.highlighted-point')).toHaveLength(0);

    // Once the run is over, the next row the pointer enters is picked out.
    fireEvent.mouseEnter(screen.getByRole('row', {name: /gzip/}));
    await waitFor(() => {
      expect(container.querySelectorAll('.highlighted-point')).toHaveLength(2);
    });
  });

  it('drops the highlight when the results change under it', async () => {
    const first = [
      result({compressor: 'zstd', id: 'z1', level: 1}, {ratio: 3, compressMBps: 500}),
      result({compressor: 'gzip', id: 'g6', level: 6}, {ratio: 5, compressMBps: 50}),
    ];
    const {container, rerender} = render(
      <ChakraProvider value={playgroundSystem}>
        <ResultsPanel runState={{status: 'completed', results: first, failures: []}} />
      </ChakraProvider>,
    );
    await waitFor(() => {
      expect(container.querySelectorAll('.recharts-scatter-symbol').length).toBeGreaterThan(0);
    });
    fireEvent.mouseEnter(screen.getByRole('row', {name: /gzip/}));
    await waitFor(() => {
      expect(container.querySelectorAll('.highlighted-point')).toHaveLength(2);
    });

    // The next run reuses the job ids, so the old key would land on its gzip
    // point although nothing was hovered in it.
    const next = first.map((one) => ({...one, ratio: one.ratio * 2}));
    rerender(
      <ChakraProvider value={playgroundSystem}>
        <ResultsPanel runState={{status: 'completed', results: next, failures: []}} />
      </ChakraProvider>,
    );
    expect(container.querySelectorAll('.highlighted-point')).toHaveLength(0);
  });

  it('places points on a log scale rather than a linear one', async () => {
    const {container} = await renderPlotted();
    const xs = [...container.querySelectorAll('.recharts-scatter-symbol')]
      .slice(0, 4)
      .map((node) =>
        Number(/translate\(([\d.]+)/.exec(node.querySelector('path')?.getAttribute('transform') ?? '')?.[1]),
      );

    // Speeds are 38, 162, 310 and 520. Where 162 falls between 38 and 520 is
    // 26% on a linear axis and 55% on a log one -- (log162 - log38) over
    // (log520 - log38) -- and the domain's padding cancels out of that ratio.
    const [low, gzip, , high] = [...xs].sort((a, b) => a - b);
    expect((gzip - low) / (high - low)).toBeCloseTo(0.554, 2);
  });
});
