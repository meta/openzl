// Copyright (c) Meta Platforms, Inc. and affiliates.

import {useState} from 'react';
import {Box, Heading, Text, VStack} from '@chakra-ui/react';
import {LuChartLine, LuChartSpline, LuMicroscope} from 'react-icons/lu';
import {isRunInProgress, type BenchmarkJob, type Candidate, type JobResult, type RunState} from '../benchmarkTypes.ts';
import {toggleCodec} from '../charts.ts';
import type {CompressorName} from '../compressors.ts';
import {formatBytes, measurementKey, resultsOf} from '../measurements.ts';
import MeasurementsTable from './MeasurementsTable.tsx';
import RatioSpeedCharts from './RatioSpeedCharts.tsx';

const HOW_IT_WORKS_STEPS = ['Choose your data.', 'Pick compressors to compare.', 'Run the benchmark.'];

interface ResultsPanelProps {
  runState: RunState;
  desktop?: boolean;
}

function Pill({children, bg, color}: {children: string; bg: string; color: string}) {
  return (
    <Box bg={bg} color={color} px="12px" py="4px" borderRadius="100px" fontSize="12px" fontWeight="semibold">
      {children}
    </Box>
  );
}

/** `idle` and `loading` carry no outcome, so there is nothing to list yet. */
function outcomeOf(runState: RunState) {
  return runState.status === 'idle' || runState.status === 'loading' ? null : runState;
}

function describeJob(job: BenchmarkJob): string {
  if (job.compressor !== 'OpenZL') {
    return `${job.compressor} ${String(job.level)}`;
  }
  return `${job.compressor} ${job.profile} / ${String(job.level)}`;
}

/** A trained job's failures share its description, so the candidate tells them apart. */
function describeMeasurement(job: BenchmarkJob, candidate: Candidate | null): string {
  return candidate === null
    ? describeJob(job)
    : `${describeJob(job)}, candidate ${String(candidate.index)} of ${String(candidate.total)}`;
}

/**
 * Jobs, or single candidates of a trained one, that produced no measurement;
 * the table has no row shape for these.
 */
function FailureList({runState}: ResultsPanelProps) {
  const outcome = outcomeOf(runState);
  if (outcome === null || outcome.failures.length === 0) {
    return null;
  }
  return (
    <VStack as="ul" gap="6px" align="stretch" m={0} p={0} listStyleType="none">
      {outcome.failures.map((failure) => (
        <Text as="li" key={measurementKey(failure)} color="pg.danger" fontSize="13px" m={0}>
          {describeMeasurement(failure.job, failure.candidate)}: {failure.message}
        </Text>
      ))}
    </VStack>
  );
}

export default function ResultsPanel({runState, desktop = false}: ResultsPanelProps) {
  const results = resultsOf(runState);
  const hasResults = results.length > 0;
  const isRunning = isRunInProgress(runState);
  // The `measurementKey` of the table row under the pointer, picked out on the
  // charts. While a run goes, rows arrive and re-sort under a pointer that has
  // not moved, so no enter or leave fires, and the charts animate each new
  // point in; a highlight would stay on a row that is no longer there. So none
  // is taken until the run ends, and one only holds for the results it was
  // made on, which keeps a key from the last run off the same job in this one.
  const [hovered, setHovered] = useState<{key: string; results: readonly JobResult[]} | null>(null);
  const highlighted = !isRunning && hovered?.results === results ? hovered.key : null;
  // The codecs switched off in the legend, for the charts and the table alike.
  // Kept while there are results, so a codec hidden mid-run stays hidden as its
  // points arrive, and dropped once the next run clears them.
  const [hidden, setHidden] = useState<ReadonlySet<CompressorName>>(new Set());
  if (!hasResults && hidden.size > 0) {
    setHidden(new Set());
  }
  const onToggle = (compressor: CompressorName) => {
    setHidden((current) => toggleCodec(current, compressor));
    // Rows that go away under a pointer that has not moved send no leave, and
    // the key would pick the row out again when its codec comes back.
    setHovered(null);
  };
  return (
    <Box
      as="section"
      aria-labelledby="results-title"
      aria-busy={isRunInProgress(runState)}
      flex="1"
      minW={0}
      alignSelf={{base: 'stretch', lg: 'flex-start'}}
      bg="pg.surface"
      borderWidth="1px"
      borderColor="pg.border"
      borderRadius="12px"
      p="32px">
      <VStack gap="24px" align="stretch">
        <Box display="flex" alignItems="center" justifyContent="space-between" gap="12px">
          <Heading id="results-title" as="h2" color="pg.ink" fontSize="20px" fontWeight="extrabold" m={0}>
            Results
          </Heading>
          {results.length > 0 && (
            <Box display="flex" gap="8px">
              <Pill bg="pg.chip" color="pg.secondary">{`${formatBytes(results[0].srcSize)} input`}</Pill>
              <Pill bg="pg.accentBg" color="pg.tagSpeedFg">{`${String(results.length)} measured`}</Pill>
            </Box>
          )}
        </Box>

        {/* Progress and the done line are the run card's, so this only ever
            says that a run failed outright. An alert is announced when it is
            added, so unlike a status region it needs no empty placeholder. */}
        {runState.status === 'error' && (
          <Text role="alert" color="pg.ink" fontSize="13px" fontWeight="semibold" m={0}>
            Run failed: {runState.message}
          </Text>
        )}
        {hasResults && (
          <RatioSpeedCharts results={results} highlighted={highlighted} hidden={hidden} onToggle={onToggle} />
        )}
        <MeasurementsTable
          runState={runState}
          hidden={hidden}
          onHighlight={isRunning ? undefined : (key) => setHovered(key === null ? null : {key, results})}
        />
        <FailureList runState={runState} />

        {/* Only true of the browser build: the desktop app measures with native zli. */}
        {hasResults && !desktop && (
          <Box
            as="aside"
            aria-label="WebAssembly speed notice"
            bg="pg.callout"
            borderWidth="1px"
            borderColor="pg.noticeBorder"
            borderRadius="8px"
            px="16px"
            py="12px">
            <Text color="pg.noticeFg" fontSize="13px" lineHeight="1.4" m={0}>
              ⚡ Speeds shown here are different from running locally because we are using WebAssembly.
            </Text>
          </Box>
        )}

        {!hasResults && (
          <>
            <Box
              display="flex"
              alignItems="center"
              gap="16px"
              bg="pg.canvas"
              borderWidth="1px"
              borderColor="pg.border"
              borderRadius="8px"
              p="20px">
              <Box
                aria-hidden="true"
                display="inline-flex"
                alignItems="center"
                justifyContent="center"
                boxSize="40px"
                flexShrink={0}
                borderRadius="full"
                bg="pg.chip"
                color="pg.faint">
                <LuChartLine size={20} />
              </Box>
              <VStack gap="2px" flex="1" minW={0} align="stretch">
                <Text color="pg.ink" fontSize="15px" fontWeight="bold" m={0}>
                  No results yet
                </Text>
                <Text color="pg.secondary" fontSize="14px" lineHeight="1.4" m={0}>
                  Select your data to compress or hit &ldquo;Try a sample&rdquo; to see the playground work.
                </Text>
              </VStack>
            </Box>

            <VStack gap="16px" align="stretch">
              <Heading
                as="h3"
                color="pg.muted"
                fontSize="12px"
                fontWeight="bold"
                textTransform="uppercase"
                letterSpacing="0.04em"
                m={0}>
                How it works
              </Heading>
              <VStack as="ol" gap="14px" align="stretch" m={0} p={0} listStyleType="none">
                {HOW_IT_WORKS_STEPS.map((step, index) => (
                  <Box key={step} as="li" display="flex" alignItems="flex-start" gap="12px">
                    <Box
                      aria-hidden="true"
                      display="inline-flex"
                      alignItems="center"
                      justifyContent="center"
                      boxSize="20px"
                      flexShrink={0}
                      borderRadius="full"
                      bg="pg.accentBg"
                      color="pg.accent"
                      fontSize="11px"
                      fontWeight="bold">
                      {index + 1}
                    </Box>
                    <Text color="pg.ink" fontSize="14px" fontWeight="bold" lineHeight="1.5" m={0}>
                      {step}
                    </Text>
                  </Box>
                ))}
              </VStack>
            </VStack>

            <VStack gap="16px" align="stretch">
              <Heading
                as="h3"
                color="pg.muted"
                fontSize="12px"
                fontWeight="bold"
                textTransform="uppercase"
                letterSpacing="0.04em"
                m={0}>
                Reading the results
              </Heading>
              <VStack as="ul" gap="14px" align="stretch" m={0} p={0} listStyleType="none">
                <Box as="li" display="flex" alignItems="flex-start" gap="12px">
                  <Box
                    aria-hidden="true"
                    display="inline-flex"
                    alignItems="center"
                    justifyContent="center"
                    boxSize="20px"
                    flexShrink={0}
                    color="pg.success">
                    <LuChartSpline size={14} />
                  </Box>
                  <Text color="pg.secondary" fontSize="14px" lineHeight="1.5" m={0}>
                    <Text as="span" color="pg.ink" fontWeight="bold">
                      What the charts say.{' '}
                    </Text>
                    These charts compare compression ratio against speed. The left chart shows compression speed, the
                    right shows decompression speed. Points toward the top-right perform best. Hover over a point to see
                    details.
                  </Text>
                </Box>
                <Box as="li" display="flex" alignItems="flex-start" gap="12px">
                  <Box
                    aria-hidden="true"
                    display="inline-flex"
                    alignItems="center"
                    justifyContent="center"
                    boxSize="20px"
                    flexShrink={0}
                    color="pg.success">
                    <LuMicroscope size={14} />
                  </Box>
                  <Text color="pg.secondary" fontSize="14px" lineHeight="1.5" m={0}>
                    <Text as="span" color="pg.ink" fontWeight="bold">
                      Train an OpenZL compressor.{' '}
                    </Text>
                    Select &ldquo;Train&rdquo; on any OpenZL row in the setup to build a custom compressor from your
                    data. A Pareto frontier will be shown, representing a set of compressors at different speed vs ratio
                    trade-offs.
                  </Text>
                </Box>
              </VStack>
            </VStack>
          </>
        )}
      </VStack>
    </Box>
  );
}
