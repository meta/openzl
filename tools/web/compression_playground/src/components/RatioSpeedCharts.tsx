// Copyright (c) Meta Platforms, Inc. and affiliates.

import {Suspense, lazy} from 'react';
import {Box, Heading, Text, VStack} from '@chakra-ui/react';
import type {CompressorName} from '../compressors.ts';
import type {JobResult} from '../benchmarkTypes.ts';
import {buildSeries, type SpeedAxis} from '../charts.ts';
import {CODEC_COLOR} from '../theme.ts';

/**
 * Only the plot waits on recharts, which stays out of the entry chunk since
 * nothing can be charted before a run produces something. Everything around
 * it renders at once at its final size, so nothing moves when the plot lands.
 */
const ScatterPlot = lazy(() => import('./ScatterPlot.tsx'));

const AXES = [
  {
    axis: 'compressMBps',
    title: 'Compression',
    subtitle: 'Ratio against compression throughput',
    xLabel: 'Compress MB/s (log)',
  },
  {
    axis: 'decompressMBps',
    title: 'Decompression',
    subtitle: 'Ratio against decompression throughput',
    xLabel: 'Decompress MB/s (log)',
  },
] as const satisfies readonly {axis: SpeedAxis; title: string; subtitle: string; xLabel: string}[];

interface ChartProps {
  results: readonly JobResult[];
  axis: SpeedAxis;
  xLabel: string;
  highlighted: string | null;
}

function Chart({results, axis, xLabel, highlighted}: ChartProps) {
  return (
    <Box display="flex" alignItems="stretch" gap="2px">
      <Text
        fontSize="10px"
        color="pg.faint"
        m={0}
        alignSelf="center"
        whiteSpace="nowrap"
        style={{writingMode: 'vertical-rl', transform: 'rotate(180deg)'}}>
        Compression ratio (log)
      </Text>
      <Box flex="1" minW={0}>
        <Box height="200px" width="100%">
          <Suspense fallback={null}>
            <ScatterPlot results={results} axis={axis} highlighted={highlighted} />
          </Suspense>
        </Box>
        <Text fontSize="10px" color="pg.faint" textAlign="center" m={0}>
          {xLabel}
        </Text>
      </Box>
    </Box>
  );
}

function LegendChip({compressor, count}: {compressor: CompressorName; count: number}) {
  // The chip is Chakra rather than SVG, so the token name goes in directly and
  // `/n` washes it out to n percent.
  const color = CODEC_COLOR[compressor];
  return (
    <Box
      display="flex"
      alignItems="center"
      gap="6px"
      pl="10px"
      pr="12px"
      py="6px"
      borderRadius="100px"
      borderWidth="1.5px"
      borderColor={`${color}/35`}
      bg={`${color}/8`}>
      <Box aria-hidden="true" boxSize="10px" borderRadius="full" bg={color} flexShrink={0} />
      <Text fontSize="12px" fontWeight="medium" color="pg.ink" m={0}>
        {compressor}
      </Text>
      <Box px="6px" py="1px" borderRadius="100px" bg={`${color}/15`} color={color} fontSize="11px" fontWeight="bold">
        {count}
      </Box>
    </Box>
  );
}

/** `highlighted` is the `measurementKey` of the point to pick out on both charts. */
export default function RatioSpeedCharts({
  results,
  highlighted = null,
}: {
  results: readonly JobResult[];
  highlighted?: string | null;
}) {
  if (results.length === 0) {
    return null;
  }
  // Both charts hold the same points, so either one's series gives the counts.
  const counts = new Map<CompressorName, number>();
  for (const series of buildSeries(results, 'compressMBps').values()) {
    counts.set(series.compressor, (counts.get(series.compressor) ?? 0) + series.points.length);
  }

  return (
    <VStack gap="16px" align="stretch">
      <VStack gap="4px" align="stretch">
        <Heading
          as="h3"
          color="pg.faint"
          fontSize="11px"
          fontWeight="extrabold"
          textTransform="uppercase"
          letterSpacing="0.02em"
          m={0}>
          Ratio vs. speed
        </Heading>
        <Text color="pg.secondary" fontSize="13px" m={0}>
          These charts compare compression ratio against speed. Points toward the top right offer the best combination
          of compression ratio and speed. Hover over a point to see details.
        </Text>
      </VStack>

      <Box display="flex" flexDirection={{base: 'column', md: 'row'}} gap="16px" alignItems="stretch">
        {AXES.map((chart) => (
          <Box
            key={chart.axis}
            flex="1"
            minW={0}
            bg="pg.surface"
            borderWidth="1px"
            borderColor="pg.border"
            borderRadius="8px"
            p="20px">
            <VStack gap="12px" align="stretch">
              <VStack gap="2px" align="stretch">
                <Text fontSize="14px" fontWeight="bold" color="pg.ink" m={0}>
                  {chart.title}
                </Text>
                <Text fontSize="12px" color="pg.secondary" m={0}>
                  {chart.subtitle}
                </Text>
              </VStack>
              <Chart results={results} axis={chart.axis} xLabel={chart.xLabel} highlighted={highlighted} />
            </VStack>
          </Box>
        ))}
      </Box>

      <Box display="flex" flexWrap="wrap" gap="8px">
        {[...counts.entries()].map(([compressor, count]) => (
          <LegendChip key={compressor} compressor={compressor} count={count} />
        ))}
      </Box>
    </VStack>
  );
}
