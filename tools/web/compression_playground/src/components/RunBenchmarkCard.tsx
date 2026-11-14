// Copyright (c) Meta Platforms, Inc. and affiliates.

import {useId} from 'react';
import {Box, Button, Link, Portal, Text, Tooltip, VStack, chakra, useSlotRecipe} from '@chakra-ui/react';
import {LuCircleAlert, LuCircleCheck, LuLoaderCircle, LuPlay} from 'react-icons/lu';
import StepCard from './StepCard.tsx';
import {isRunInProgress, type RunConfig, type RunState} from '../benchmarkTypes.ts';
import {formatBytes, runSummary, type RunSummary} from '../measurements.ts';
import {progressOf} from '../progress.ts';

function doneLine({srcSize, succeeded, failed}: RunSummary): string {
  const size = srcSize === null ? null : formatBytes(srcSize);
  const counts = `${String(succeeded)} succeeded, ${String(failed)} failed`;
  return size === null ? `Done — ${counts}` : `Done — ${size} · ${counts}`;
}

/** What the sample is, and where it comes from. */
const SAMPLE_DESCRIPTION = 'Part of the SAO star catalog, from the';
const SAMPLE_CORPUS = 'Silesia compression corpus';
const SAMPLE_CORPUS_URL = 'https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia';

const Spinner = chakra(LuLoaderCircle, {
  base: {animation: 'spin 1s linear infinite'},
});

interface RunBenchmarkCardProps {
  runConfig: RunConfig | null;
  runState: RunState;
  onRun: ((config: RunConfig) => void) | null;
  onTrySample: (() => void) | null;
  /** What stands between the current setup and a run, said in the order to fix it. */
  runBlockers: readonly string[];
}

export default function RunBenchmarkCard({
  runConfig,
  runState,
  onRun,
  onTrySample,
  runBlockers,
}: RunBenchmarkCardProps) {
  const helperTooltip = useSlotRecipe({key: 'helperTooltip'})();
  const isBusy = isRunInProgress(runState);
  const canRun = runConfig !== null && onRun !== null && !isBusy;
  const canTrySample = onTrySample !== null && !isBusy;
  // Not while a run is in flight: the status line already says why, and there
  // is nothing for the user to fix.
  const blockedReason = !canRun && !isBusy && runBlockers.length > 0 ? runBlockers.join(' ') : null;
  const blockedReasonId = useId();
  const sampleDescriptionId = useId();
  const progress = progressOf(runState);
  const summary = runSummary(runState);

  return (
    <StepCard number={3} title="Run the benchmark" subtitle="Measure ratio and speed for every compressor you selected">
      <Box display="flex" gap="12px" alignItems="center">
        <Tooltip.Root openDelay={200} disabled={blockedReason === null} positioning={{placement: 'top', gutter: 8}}>
          <Tooltip.Trigger asChild>
            <Button
              type="button"
              flex="1"
              minW={0}
              bg="pg.primary"
              color="white"
              fontSize="14px"
              fontWeight="bold"
              px="24px"
              py="12px"
              borderRadius="8px"
              _hover={canRun ? {bg: 'pg.primaryHover'} : undefined}
              // `aria-disabled` rather than `disabled` while there is a reason to
              // show: a disabled button gets no pointer events and no focus, so
              // the tooltip saying why could never open. Clicks are still
              // refused below.
              disabled={!canRun && blockedReason === null}
              aria-disabled={blockedReason !== null || undefined}
              aria-describedby={blockedReason === null ? undefined : blockedReasonId}
              // The design greys the button while it works rather than offering a
              // cancel, so the disabled state has to stay legible against white.
              _disabled={isBusy ? {bg: 'pg.primaryBusy', color: 'white', opacity: 1, cursor: 'wait'} : undefined}
              onClick={() => {
                if (canRun) {
                  onRun(runConfig);
                }
              }}>
              {isBusy ? <Spinner aria-hidden="true" /> : <LuPlay aria-hidden="true" />}
              {isBusy ? 'Running…' : 'Run benchmark'}
            </Button>
          </Tooltip.Trigger>
          <Portal>
            <Tooltip.Positioner>
              <Tooltip.Content css={helperTooltip.content}>
                <Tooltip.Arrow>
                  <Tooltip.ArrowTip css={helperTooltip.arrowTip} />
                </Tooltip.Arrow>
                {blockedReason}
              </Tooltip.Content>
            </Tooltip.Positioner>
          </Portal>
        </Tooltip.Root>
        <Tooltip.Root openDelay={200} interactive positioning={{placement: 'top', gutter: 8}}>
          <Tooltip.Trigger asChild>
            <Button
              type="button"
              flex="1"
              minW={0}
              variant="outline"
              bg="pg.surface"
              borderColor="pg.border"
              color="pg.secondary"
              fontSize="14px"
              fontWeight="semibold"
              px="16px"
              py="12px"
              borderRadius="8px"
              _hover={{bg: 'pg.canvas'}}
              disabled={!canTrySample}
              aria-describedby={sampleDescriptionId}
              onClick={onTrySample ?? undefined}>
              Try a sample
            </Button>
          </Tooltip.Trigger>
          <Portal>
            <Tooltip.Positioner>
              <Tooltip.Content css={helperTooltip.content}>
                <Tooltip.Arrow>
                  <Tooltip.ArrowTip css={helperTooltip.arrowTip} />
                </Tooltip.Arrow>
                {SAMPLE_DESCRIPTION}{' '}
                <Link
                  href={SAMPLE_CORPUS_URL}
                  target="_blank"
                  rel="noreferrer"
                  color="inherit"
                  textDecoration="underline">
                  {SAMPLE_CORPUS}
                </Link>
              </Tooltip.Content>
            </Tooltip.Positioner>
          </Portal>
        </Tooltip.Root>
      </Box>
      {/* The tooltips only open on hover or focus, so what they say is also kept
          here for a screen reader to announce with the button. */}
      <Box id={sampleDescriptionId} srOnly>
        {`${SAMPLE_DESCRIPTION} ${SAMPLE_CORPUS}`}
      </Box>
      {blockedReason !== null && (
        <Box id={blockedReasonId} srOnly>
          {blockedReason}
        </Box>
      )}
      {/* Its own live region, because the progress one below unmounts at the
          same moment this appears: a region that goes away announces nothing,
          so the run would end in silence. `aria-atomic` so the whole line is
          read rather than whichever part changed. */}
      {summary !== null && (
        <Box display="flex" alignItems="center" gap="8px" pt="4px" role="status" aria-live="polite" aria-atomic="true">
          <Box
            aria-hidden="true"
            display="inline-flex"
            flexShrink={0}
            color={summary.failed > 0 ? 'pg.danger' : 'pg.successText'}>
            {summary.failed > 0 ? <LuCircleAlert size={16} /> : <LuCircleCheck size={16} />}
          </Box>
          <Text color={summary.failed > 0 ? 'pg.danger' : 'pg.successText'} fontSize="13px" fontWeight="medium" m={0}>
            {doneLine(summary)}
          </Text>
        </Box>
      )}
      {progress !== null && (
        <VStack gap="8px" align="stretch" role="status" aria-live="polite">
          <Text color="pg.ink" fontSize="13px" fontWeight="semibold" m={0}>
            {progress.label}
          </Text>
          <Box
            height="8px"
            borderRadius="4px"
            bg="pg.border"
            overflow="hidden"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progress.fraction * 100)}>
            <Box
              height="100%"
              bg="pg.primary"
              width={`${String(Math.round(progress.fraction * 100))}%`}
              transition="width 120ms linear"
            />
          </Box>
        </VStack>
      )}
    </StepCard>
  );
}
