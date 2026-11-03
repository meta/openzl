// Copyright (c) Meta Platforms, Inc. and affiliates.

import {Box, Text, useToken} from '@chakra-ui/react';
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type {CompressorName} from '../compressors.ts';
import type {JobResult} from '../benchmarkTypes.ts';
import {
  buildSeries,
  formatRatioTick,
  formatSpeedTick,
  logDomain,
  logTicks,
  type ChartPoint,
  type Series,
  type SpeedAxis,
} from '../charts.ts';
import {formatRatio, formatSpeed} from '../measurements.ts';
import {CODEC_COLOR} from '../theme.ts';

/**
 * recharts hands `fill` and `stroke` to SVG, which cannot resolve a token
 * name. A semantic token with dark variants resolves to its CSS variable
 * rather than to a value, though, and SVG does take one of those -- so the
 * charts still follow the scheme instead of carrying their own hex.
 */
function useChartColors() {
  const [openzl, zstd, gzip, line, label, surface] = useToken('colors', [
    CODEC_COLOR.OpenZL,
    CODEC_COLOR.zstd,
    CODEC_COLOR.gzip,
    'pg.border',
    'pg.faint',
    'pg.surface',
  ]);
  return {codec: {OpenZL: openzl, zstd, gzip} satisfies Record<CompressorName, string>, line, label, surface};
}

function PointTooltip({active, payload}: {active?: boolean; payload?: {payload: ChartPoint}[]}) {
  const point = payload?.[0]?.payload;
  if (active !== true || point === undefined) {
    return null;
  }
  return (
    <Box bg="pg.surface" borderWidth="1px" borderColor="pg.border" borderRadius="6px" px="10px" py="6px" boxShadow="sm">
      <Text fontSize="12px" fontWeight="bold" color="pg.ink" m={0}>
        {point.label}
      </Text>
      <Text fontSize="12px" color="pg.secondary" m={0} whiteSpace="nowrap">
        {formatRatio(point.y)} at {formatSpeed(point.x)}
      </Text>
    </Box>
  );
}

/** The point for `key` and the codec whose colour it takes, if it is plotted. */
function findPoint(
  series: ReadonlyMap<string, Series>,
  key: string | null,
): {point: ChartPoint; compressor: CompressorName} | null {
  for (const one of series.values()) {
    const point = one.points.find((candidate) => candidate.key === key);
    if (point !== undefined) {
      return {point, compressor: one.compressor};
    }
  }
  return null;
}

/**
 * The plot itself, and the only part of the charts that needs recharts.
 * `highlighted` is the `measurementKey` of the table row under the pointer.
 */
export default function ScatterPlot({
  results,
  axis,
  highlighted,
}: {
  results: readonly JobResult[];
  axis: SpeedAxis;
  highlighted: string | null;
}) {
  const {codec, line, label, surface} = useChartColors();
  const tickStyle = {fill: label, fontSize: 11};
  const series = buildSeries(results, axis);
  const xDomain = logDomain(results.map((result) => result[axis]));
  const yDomain = logDomain(results.map((result) => result.ratio));
  const spot = findPoint(series, highlighted);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{top: 8, right: 12, bottom: 4, left: 0}}>
        <CartesianGrid stroke={line} strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="x"
          scale="log"
          domain={[...xDomain]}
          ticks={[...logTicks(xDomain)]}
          tickFormatter={formatSpeedTick}
          tick={tickStyle}
          stroke={line}
        />
        <YAxis
          type="number"
          dataKey="y"
          scale="log"
          domain={[...yDomain]}
          ticks={[...logTicks(yDomain)]}
          tickFormatter={formatRatioTick}
          tick={tickStyle}
          stroke={line}
          width={44}
        />
        <Tooltip content={<PointTooltip />} cursor={{strokeDasharray: '3 3'}} />
        {[...series.entries()].map(([key, one]) => (
          <Scatter
            key={key}
            name={one.compressor}
            data={[...one.points]}
            fill={codec[one.compressor]}
            line={one.points.length > 1 ? {stroke: codec[one.compressor], strokeWidth: 1.5} : false}
          />
        ))}
        {spot !== null && <ReferenceLine x={spot.point.x} strokeDasharray="3 3" />}
        {spot !== null && <ReferenceLine y={spot.point.y} strokeDasharray="3 3" />}
        {/* Drawn again on top of every series, so a point under its neighbours
            still shows. The ring is the card's colour, to part it from them. */}
        {spot !== null && (
          <Scatter
            data={[spot.point]}
            isAnimationActive={false}
            legendType="none"
            shape={({cx, cy}: {cx?: number; cy?: number}) => (
              <circle
                className="highlighted-point"
                cx={cx}
                cy={cy}
                r={7}
                fill={codec[spot.compressor]}
                stroke={surface}
                strokeWidth={2.5}
              />
            )}
          />
        )}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
