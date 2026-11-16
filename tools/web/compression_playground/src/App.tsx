// Copyright (c) Meta Platforms, Inc. and affiliates.

import {useCallback, useEffect, useRef, useState} from 'react';
import {Box, Flex, Text} from '@chakra-ui/react';
import {ToolHeader} from '@openzl/web-common';
import ResultsPanel from './components/ResultsPanel.tsx';
import SetupColumn from './components/SetupColumn.tsx';
import UnsupportedBrowser from './components/UnsupportedBrowser.tsx';
import {isDesktop, loadDesktopBackend, webBackend, type Backend} from './backend.ts';
import {toCompressorConfig, type CompressorConfig, type RunConfig, type RunState} from './benchmarkTypes.ts';
import {ITERATIONS_DEFAULT} from './compressors.ts';
import {detectEngineSupport, engineBlocker, isolationSettled} from './engineSupport.ts';
import {useCompressorRows} from './useCompressorRows.ts';
import logoUrl from '/OpenZL_logo.png?url';

/**
 * The Figma frame the two columns sit in (node 151:2), with its own padding.
 * These are load-bearing rather than cosmetic: the setup column is a fixed
 * 520px, so everything narrower comes off the results column, and below about
 * 1400px the measurements table no longer fits beside it.
 */
const CONTENT_MAX_WIDTH = '1590px';
const CONTENT_PADDING_X = '64px';

/**
 * The blocked screen is a few sentences, so it does not use the full content
 * width. At 1590px its text would run to well over 1400px a line, far past a
 * comfortable measure, and the card would sit above most of a viewport of empty
 * page.
 */
const MESSAGE_MAX_WIDTH = '640px';

/**
 * Without this, a file released anywhere but the drop zone makes the browser
 * navigate to it and discard the run setup. The drop zone has already called
 * preventDefault by the time the event reaches the window, so checking for that
 * leaves its own `copy` cursor intact.
 */
function useBlockStrayFileDrops() {
  useEffect(() => {
    const blockDefault = (event: DragEvent) => {
      if (event.defaultPrevented) {
        return;
      }
      if (event.dataTransfer !== null) {
        event.dataTransfer.dropEffect = 'none';
      }
      event.preventDefault();
    };
    window.addEventListener('dragover', blockDefault);
    window.addEventListener('drop', blockDefault);
    return () => {
      window.removeEventListener('dragover', blockDefault);
      window.removeEventListener('drop', blockDefault);
    };
  }, []);
}

/** "row 3", "rows 2 and 3", "rows 1, 2 and 3", numbered as step 2 numbers them. */
function describeRows(positions: readonly number[]): string {
  if (positions.length === 1) {
    return `row ${String(positions[0])}`;
  }
  return `rows ${positions.slice(0, -1).join(', ')} and ${String(positions[positions.length - 1])}`;
}

/**
 * What in the compressor rows keeps a run from starting. A zstd or gzip row
 * with no levels would drop out of the run with nothing on the page to say so,
 * so a run waits until it has one.
 */
function configBlockersFor(compressors: readonly CompressorConfig[]): string[] {
  const emptyRows = compressors.flatMap((config, index) => (config.levels.length === 0 ? [index + 1] : []));
  return emptyRows.length > 0 ? [`Pick at least one level for ${describeRows(emptyRows)}.`] : [];
}

/** Why Run is unavailable, if it is. */
function runBlockersFor(file: File | null, compressors: readonly CompressorConfig[]): string[] {
  const fileBlockers = file === null ? ['Choose a file in step 1 to run the benchmark.'] : [];
  return [...fileBlockers, ...configBlockersFor(compressors)];
}

export default function App() {
  // The desktop app measures with `zli` instead of the WASM worker. Both feed
  // the same `RunState`, so the page is chosen once here and never asks again.
  // The browser's backend is there at once; the desktop app's code is loaded,
  // so the website never has it.
  const [backend, setBackend] = useState<Backend | null>(() => (isDesktop() ? null : webBackend));
  const [backendError, setBackendError] = useState<string | null>(null);
  useEffect(() => {
    if (backend === null) {
      loadDesktopBackend().then(setBackend, (error: unknown) => {
        setBackendError(error instanceof Error ? error.message : String(error));
      });
    }
  }, [backend]);
  // Lazy initializer rather than a call in the body: nothing detection reports
  // can change while the page is open, so it should run once per mount rather
  // than once per render.
  const [support] = useState(detectEngineSupport);
  const blocker = backend === null || backend.settings.checkEngineSupport === false ? null : engineBlocker(support);
  const [settled, setSettled] = useState(false);
  useEffect(() => {
    void isolationSettled().then(() => {
      setSettled(true);
    });
  }, []);
  // On a first visit the page is not isolated until the isolation worker
  // reloads it, moments later. Showing `no-threads` before that would flash
  // the blocked screen at every new visitor, and the playground cannot load
  // either, so the main area stays empty until the worker has decided.
  const awaitingIsolation = blocker === 'no-threads' && !settled;

  return (
    <Flex direction="column" minH="100vh" bg="pg.pageBg">
      <ToolHeader title="Compression Playground" logoSrc={logoUrl} {...backend?.settings.headerLinks} />
      {/* The header stays up on the unsupported path. It is the site
          navigation, and it is the only way off this screen. */}
      <Flex as="main" flex="1" direction="column" align="center" bg="pg.pageBg">
        {backendError !== null ? (
          <Box width="100%" maxW={MESSAGE_MAX_WIDTH} px="32px" pt="32px">
            <Text role="alert">Unable to start the playground: {backendError}</Text>
          </Box>
        ) : backend === null || awaitingIsolation ? null : blocker === null ? (
          <Playground backend={backend} />
        ) : (
          <Box width="100%" maxW={MESSAGE_MAX_WIDTH} px="32px" pt={{base: '32px', md: '96px'}} pb="40px">
            <UnsupportedBrowser blocker={blocker} />
          </Box>
        )}
      </Flex>
    </Flex>
  );
}

/** Owns the run setup, so none of it exists on the blocked path, where nothing could use it. */
function Playground({backend}: {backend: Backend}) {
  const [file, setFile] = useState<File | null>(null);
  const compressors = useCompressorRows({defaultCandidates: backend.settings.defaultCandidates});
  const [iterations, setIterations] = useState(ITERATIONS_DEFAULT);
  const [runState, setRunState] = useState<RunState>({status: 'idle'});
  const onRun = useCallback(
    (config: RunConfig) => {
      backend.run(config, setRunState);
    },
    [backend],
  );
  const compressorConfigs = compressors.rows.map(toCompressorConfig);
  const configBlockers = configBlockersFor(compressorConfigs);
  const runBlockers = runBlockersFor(file, compressorConfigs);
  const runConfig: RunConfig | null =
    file === null || runBlockers.length > 0
      ? null
      : {
          input: file,
          compressors: compressorConfigs,
          iterations,
        };
  // The rows and iterations as last rendered. They stay editable while the
  // sample loads, so the run reads them when it starts, not at the press.
  const liveSetup = useRef({compressors: compressorConfigs, iterations});
  useEffect(() => {
    liveSetup.current = {compressors: compressorConfigs, iterations};
  });
  // The sample leaves the chosen file alone, so it runs even with none chosen --
  // but not with a row Run would refuse.
  const onTrySample =
    configBlockers.length > 0
      ? null
      : () => {
          // `loading` from the press rather than from when the run starts, so
          // both buttons and the file picker stay disabled while the sample
          // loads.
          setRunState({status: 'loading'});
          backend.loadSample().then(
            (runSample) => {
              // A row emptied during the load holds the run back, as it
              // would Run, which then says why.
              const setup = liveSetup.current;
              if (configBlockersFor(setup.compressors).length > 0) {
                setRunState({status: 'idle'});
                return;
              }
              runSample(setup, setRunState);
            },
            (error: unknown) => {
              const message = error instanceof Error ? error.message : String(error);
              setRunState({status: 'error', message, results: [], failures: []});
            },
          );
        };
  useBlockStrayFileDrops();

  return (
    <Flex
      align="flex-start"
      gap="24px"
      width="100%"
      maxW={CONTENT_MAX_WIDTH}
      px={CONTENT_PADDING_X}
      pt="24px"
      pb="40px"
      direction={{base: 'column', lg: 'row'}}>
      <SetupColumn
        file={file}
        onFileChange={setFile}
        compressors={compressors}
        iterations={iterations}
        onIterationsChange={setIterations}
        runConfig={runConfig}
        runState={runState}
        onRun={onRun}
        onTrySample={onTrySample}
        runBlockers={runBlockers}
        settings={backend.settings}
        loadProfiles={backend.loadProfiles}
      />
      <ResultsPanel runState={runState} showWasmNotice={backend.settings.showWasmNotice} />
    </Flex>
  );
}
