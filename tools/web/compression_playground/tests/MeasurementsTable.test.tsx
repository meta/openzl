// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {cleanup, fireEvent, render, screen, within, type RenderOptions} from '@testing-library/react';
import {ChakraProvider} from '@chakra-ui/react';
import React from 'react';
import MeasurementsTable from '../src/components/MeasurementsTable.tsx';
import ResultsPanel from '../src/components/ResultsPanel.tsx';
import {playgroundSystem} from '../src/theme.ts';
import type {BenchmarkJob, JobResult, RunState} from '../src/benchmarkTypes.ts';

function renderWithPlaygroundTheme(ui: React.ReactElement, options?: Omit<RenderOptions, 'wrapper'>) {
  return render(ui, {
    wrapper: ({children}) => <ChakraProvider value={playgroundSystem}>{children}</ChakraProvider>,
    ...options,
  });
}

function result(job: Partial<BenchmarkJob> & Pick<BenchmarkJob, 'compressor'>, metrics: Partial<JobResult> = {}) {
  return {
    iterations: 1,
    srcSize: 1000,
    compressedSize: 100,
    ratio: 10,
    compressMs: 1,
    decompressMs: 1,
    compressMBps: 100,
    decompressMBps: 200,
    candidate: null,
    ...metrics,
    job: {
      id: `${job.compressor ?? 'x'}-${String(job.level ?? 1)}`,
      rowId: 1,
      level: 1,
      iterations: 1,
      profile: 'serial',
      training: null,
      ...job,
    } as BenchmarkJob,
  } satisfies JobResult;
}

function completed(results: readonly JobResult[]): RunState {
  return {status: 'completed', results, failures: []};
}

const TWO_ZSTD_LEVELS = [
  result({compressor: 'zstd', rowId: 2, level: 1}, {ratio: 3, compressedSize: 333}),
  result({compressor: 'zstd', rowId: 2, level: 9}, {ratio: 9, compressedSize: 111}),
];

/** Two codecs interleaved by ratio, so sorting has to cross between them. */
const MIXED = [
  result({compressor: 'zstd', rowId: 2, level: 1}, {ratio: 3}),
  result({compressor: 'gzip', rowId: 3, level: 6}, {ratio: 5}),
  result({compressor: 'zstd', rowId: 2, level: 9}, {ratio: 9}),
];

afterEach(cleanup);

describe('MeasurementsTable', () => {
  it('reports the measurement under the pointer, and null once it leaves', () => {
    const onHighlight = vi.fn();
    renderWithPlaygroundTheme(<MeasurementsTable runState={completed(TWO_ZSTD_LEVELS)} onHighlight={onHighlight} />);

    const row = screen.getByRole('row', {name: /Level 9/});
    fireEvent.mouseEnter(row);
    fireEvent.mouseLeave(row);

    expect(onHighlight.mock.calls).toEqual([['zstd-9-0'], [null]]);
  });

  it('renders nothing before a run has produced anything', () => {
    const {container} = renderWithPlaygroundTheme(<MeasurementsTable runState={{status: 'idle'}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('sizes its columns in shares of the container, not pixels', () => {
    // `table-layout: fixed` hands a column the width it asks for and lets the
    // table overflow, so pixel widths put this outside the results card. jsdom
    // does no layout, so the invariant is what gets checked, not the geometry.
    renderWithPlaygroundTheme(<MeasurementsTable runState={completed(TWO_ZSTD_LEVELS)} />);

    const widths = screen.getAllByRole('columnheader').map((th) => getComputedStyle(th).width);
    expect(widths.every((width) => width.endsWith('%'))).toBe(true);
    expect(widths.reduce((total, width) => total + Number.parseFloat(width), 0)).toBe(100);
  });

  it('sorts every row together, flips on a second press, and goes back by codec', () => {
    // Across codecs rather than within one: which measurement comes out ahead
    // is the question, whatever produced it.
    renderWithPlaygroundTheme(<MeasurementsTable runState={completed(MIXED)} />);
    const rowsInOrder = () =>
      screen
        .getAllByRole('row')
        .slice(1)
        .map((row) =>
          within(row)
            .getAllByRole('rowheader')
            .map((th) => th.textContent)
            .join(' '),
        );
    const codecHeader = () => screen.getByRole('button', {name: 'Sort by codec'}).closest('th') as HTMLElement;
    expect(rowsInOrder()).toEqual(['zstd Level 1', 'gzip Level 6', 'zstd Level 9']);
    expect(codecHeader()).toHaveAttribute('aria-sort', 'ascending');

    fireEvent.click(screen.getByRole('button', {name: 'Sort by ratio'}));
    expect(rowsInOrder()).toEqual(['zstd Level 9', 'gzip Level 6', 'zstd Level 1']);
    fireEvent.click(screen.getByRole('button', {name: 'Sort by ratio'}));
    expect(rowsInOrder()).toEqual(['zstd Level 1', 'gzip Level 6', 'zstd Level 9']);

    // The codec column is the way back to the order the run produced.
    fireEvent.click(screen.getByRole('button', {name: 'Sort by ratio'}));
    fireEvent.click(screen.getByRole('button', {name: 'Sort by codec'}));
    expect(rowsInOrder()).toEqual(['zstd Level 1', 'gzip Level 6', 'zstd Level 9']);
    expect(codecHeader()).toHaveAttribute('aria-sort', 'ascending');

    // Flipped, it reverses the configured rows but not each row's own order.
    fireEvent.click(screen.getByRole('button', {name: 'Sort by codec'}));
    expect(rowsInOrder()).toEqual(['gzip Level 6', 'zstd Level 1', 'zstd Level 9']);
  });

  it('says which column is sorted and which way, without relying on colour', () => {
    renderWithPlaygroundTheme(<MeasurementsTable runState={completed(MIXED)} />);
    const ratioHeader = () => screen.getByRole('button', {name: 'Sort by ratio'}).closest('th') as HTMLElement;

    fireEvent.click(screen.getByRole('button', {name: 'Sort by ratio'}));
    expect(ratioHeader()).toHaveAttribute('aria-sort', 'descending');
    // One arrow for the sorted column, both for the rest, so the shape says it.
    expect(within(ratioHeader()).queryByText('▲')).not.toBeInTheDocument();
    expect(within(ratioHeader()).getByText('▼')).toBeInTheDocument();
    const compressHeader = screen.getByRole('button', {name: 'Sort by compress'}).closest('th') as HTMLElement;
    expect(compressHeader).not.toHaveAttribute('aria-sort');
    expect(within(compressHeader).getByText('▲')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', {name: 'Sort by ratio'}));
    expect(ratioHeader()).toHaveAttribute('aria-sort', 'ascending');
  });

  it('heads each row with the codec and the setting that produced it', () => {
    renderWithPlaygroundTheme(<MeasurementsTable runState={completed(MIXED)} />);

    const firstRow = screen.getAllByRole('row')[1];
    expect(
      within(firstRow)
        .getAllByRole('rowheader')
        .map((th) => th.textContent),
    ).toEqual(['zstd', 'Level 1']);
  });

  it('puts the smallest compressed size first, since smaller is better', () => {
    renderWithPlaygroundTheme(<MeasurementsTable runState={completed(TWO_ZSTD_LEVELS)} />);

    fireEvent.click(screen.getByRole('button', {name: 'Sort by compressed'}));
    expect(screen.getAllByText(/^Level \d+$/).map((node) => node.textContent)).toEqual(['Level 9', 'Level 1']);
    expect(screen.getByRole('button', {name: 'Sort by compressed'}).closest('th')).toHaveAttribute(
      'aria-sort',
      'ascending',
    );
  });

  it('does not offer to sort by level, which means nothing across codecs', () => {
    // zstd's 5 and gzip's 5 are different settings. The codec order already
    // lists each row's levels in ascending order.
    renderWithPlaygroundTheme(<MeasurementsTable runState={completed(MIXED)} />);

    expect(screen.getByRole('columnheader', {name: 'LEVEL / PROFILE'})).toBeInTheDocument();
    expect(screen.queryByRole('button', {name: 'Sort by level / profile'})).not.toBeInTheDocument();
  });

  it('names the ends of a trained frontier, and its profile on every candidate', () => {
    renderWithPlaygroundTheme(
      <MeasurementsTable
        runState={completed([
          result({compressor: 'OpenZL', rowId: 1, profile: 'serial'}, {candidate: {index: 1, total: 3}}),
          result({compressor: 'OpenZL', rowId: 1, profile: 'serial'}, {candidate: {index: 2, total: 3}}),
          result({compressor: 'OpenZL', rowId: 1, profile: 'serial'}, {candidate: {index: 3, total: 3}}),
        ])}
      />,
    );

    expect(screen.getByText('max ratio')).toBeInTheDocument();
    expect(screen.getByText('balanced')).toBeInTheDocument();
    expect(screen.getByText('max speed')).toBeInTheDocument();
    expect(screen.getAllByText('trained · serial')).toHaveLength(3);
  });

  it('says a lone trained candidate was trained, and on which profile', () => {
    renderWithPlaygroundTheme(
      <MeasurementsTable
        runState={completed([
          result({compressor: 'OpenZL', rowId: 1, profile: 'le-u32'}, {candidate: {index: 1, total: 1}}),
        ])}
      />,
    );

    expect(screen.getByText('trained · le-u32')).toBeInTheDocument();
    expect(screen.getByText('#1')).toBeInTheDocument();
  });
});

describe('ResultsPanel', () => {
  it('drops the empty state and the guide once there are measurements', () => {
    const {rerender} = renderWithPlaygroundTheme(<ResultsPanel runState={{status: 'idle'}} />);
    expect(screen.getByText('No results yet')).toBeInTheDocument();
    expect(screen.getByRole('heading', {name: 'How it works'})).toBeInTheDocument();

    rerender(
      <ChakraProvider value={playgroundSystem}>
        <ResultsPanel runState={completed(TWO_ZSTD_LEVELS)} />
      </ChakraProvider>,
    );
    expect(screen.queryByText('No results yet')).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', {name: 'How it works'})).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', {name: 'Reading the results'})).not.toBeInTheDocument();
    expect(screen.getByRole('table', {name: 'Measurements'})).toBeInTheDocument();
    expect(screen.getByText(/WebAssembly/)).toBeInTheDocument();
  });

  it('keeps the empty state and the guide until the first measurement arrives', () => {
    const {rerender} = renderWithPlaygroundTheme(<ResultsPanel runState={{status: 'loading'}} />);
    expect(screen.getByText('No results yet')).toBeInTheDocument();
    expect(screen.getByRole('heading', {name: 'How it works'})).toBeInTheDocument();

    const running = {status: 'running', completedJobs: 0, totalJobs: 2, failures: [], step: null} as const;
    rerender(
      <ChakraProvider value={playgroundSystem}>
        <ResultsPanel runState={{...running, results: []}} />
      </ChakraProvider>,
    );
    expect(screen.getByText('No results yet')).toBeInTheDocument();

    rerender(
      <ChakraProvider value={playgroundSystem}>
        <ResultsPanel runState={{...running, completedJobs: 1, results: [TWO_ZSTD_LEVELS[0]]}} />
      </ChakraProvider>,
    );
    expect(screen.queryByText('No results yet')).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', {name: 'How it works'})).not.toBeInTheDocument();
  });

  it('keeps the empty state when every job failed, and says which', () => {
    renderWithPlaygroundTheme(
      <ResultsPanel
        runState={{
          status: 'completed',
          results: [],
          failures: [{job: TWO_ZSTD_LEVELS[0].job, candidate: null, message: 'out of memory'}],
        }}
      />,
    );

    expect(screen.getByText('No results yet')).toBeInTheDocument();
    expect(screen.getByText('zstd 1: out of memory')).toBeInTheDocument();
    // The notice is about numbers on screen, and there are none.
    expect(screen.queryByText(/WebAssembly/)).not.toBeInTheDocument();
  });

  it('leaves progress and the done line to the run card', () => {
    const {rerender} = renderWithPlaygroundTheme(
      <ResultsPanel
        runState={{status: 'running', completedJobs: 1, totalJobs: 2, results: [], failures: [], step: null}}
      />,
    );
    expect(screen.queryByText(/Running/)).not.toBeInTheDocument();

    rerender(
      <ChakraProvider value={playgroundSystem}>
        <ResultsPanel runState={completed([])} />
      </ChakraProvider>,
    );
    expect(screen.queryByText(/measured/)).not.toBeInTheDocument();
  });

  it('says on screen that a run failed', () => {
    renderWithPlaygroundTheme(
      <ResultsPanel runState={{status: 'error', message: 'worker crashed', results: [], failures: []}} />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('Run failed: worker crashed');
  });
});
