// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {cleanup, render, screen} from '@testing-library/react';
import {ChakraProvider} from '@chakra-ui/react';
import SetupColumn from '../src/components/SetupColumn.tsx';
import {ITERATIONS_DEFAULT} from '../src/compressors.ts';
import {useCompressorRows} from '../src/useCompressorRows.ts';
import {playgroundSystem} from '../src/theme.ts';
import type {BenchmarkJob, RunState} from '../src/benchmarkTypes.ts';

function Column({runState}: {runState: RunState}) {
  const compressors = useCompressorRows();
  return (
    <SetupColumn
      file={null}
      onFileChange={vi.fn()}
      compressors={compressors}
      iterations={ITERATIONS_DEFAULT}
      onIterationsChange={vi.fn()}
      runConfig={null}
      runState={runState}
      onRun={null}
      onTrySample={null}
      runBlockers={[]}
    />
  );
}

function renderColumn(runState: RunState) {
  return render(
    <ChakraProvider value={playgroundSystem}>
      <Column runState={runState} />
    </ChakraProvider>,
  );
}

const ZSTD_RESULT = {
  iterations: 1,
  srcSize: 1000,
  compressedSize: 100,
  ratio: 10,
  compressMs: 1,
  decompressMs: 1,
  compressMBps: 100,
  decompressMBps: 200,
  candidate: null,
  job: {
    id: 'zstd-1',
    rowId: 1,
    compressor: 'zstd',
    level: 1,
    iterations: 1,
    profile: 'serial',
    training: null,
  } as BenchmarkJob,
};

afterEach(cleanup);

describe('SetupColumn', () => {
  it('does not mention the desktop app once there are results', () => {
    renderColumn({status: 'completed', results: [ZSTD_RESULT], failures: []});

    expect(screen.queryByText(/desktop app/i)).not.toBeInTheDocument();
  });
});
