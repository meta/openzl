// Copyright (c) Meta Platforms, Inc. and affiliates.

import {useState} from 'react';
import {Box, Text, chakra} from '@chakra-ui/react';
import type {JobResult, RunState} from '../benchmarkTypes.ts';
import type {CompressorName} from '../compressors.ts';
import {
  barFraction,
  formatBytes,
  formatRatio,
  formatSpeed,
  frontierTag,
  measurementKey,
  peaksOf,
  rowDetail,
  resultsOf,
  sortMeasurements,
  DEFAULT_SORT,
  FIRST_DIRECTION,
  type FrontierTag,
  type Peaks,
  type Sort,
  type SortDirection,
  type SortKey,
} from '../measurements.ts';
import {CODEC_COLOR} from '../theme.ts';

/**
 * Shared by the header and every row, so the columns line up without a grid.
 * Widths are shares rather than pixels: `table-layout: fixed` gives a column
 * exactly the width it asks for and lets the table overflow its container when
 * the total is too wide, which is how this first rendered outside the results
 * card. Below `TABLE_MIN_WIDTH` the wrapper scrolls instead of shrinking them
 * into each other.
 */
const COLUMNS = [
  {key: 'codec', label: 'CODEC', width: '15.5%', sortable: true},
  {key: 'detail', label: 'LEVEL / PROFILE', width: '15.5%', sortable: false},
  {key: 'compressedSize', label: 'COMPRESSED', width: '14%', sortable: true},
  {key: 'ratio', label: 'RATIO', width: '20%', sortable: true},
  {key: 'compressMBps', label: 'COMPRESS', width: '17.5%', sortable: true},
  {key: 'decompressMBps', label: 'DECOMPRESS', width: '17.5%', sortable: true},
] as const;

// The frame's own columns come to 830px.
const TABLE_MIN_WIDTH = '830px';

const TAG_COLOR: Record<FrontierTag, {bg: string; fg: string}> = {
  'max ratio': {bg: 'pg.tagRatioBg', fg: 'pg.tagRatioFg'},
  balanced: {bg: 'pg.tagBalancedBg', fg: 'pg.tagBalancedFg'},
  'max speed': {bg: 'pg.tagSpeedBg', fg: 'pg.tagSpeedFg'},
};

/**
 * The sorted column shows the one arrow it is sorted by, the rest both faintly,
 * so the order reads from shape as well as colour. Hidden from assistive
 * technology, which reads `aria-sort` on the header instead.
 */
function SortArrow({direction}: {direction: SortDirection | null}) {
  return (
    <Box as="span" aria-hidden="true" display="inline-flex" flexDirection="column" lineHeight="0.6" fontSize="8px">
      {direction !== 'desc' && (
        <Box as="span" color={direction === 'asc' ? 'pg.ink' : 'pg.border'}>
          ▲
        </Box>
      )}
      {direction !== 'asc' && (
        <Box as="span" color={direction === 'desc' ? 'pg.ink' : 'pg.border'}>
          ▼
        </Box>
      )}
    </Box>
  );
}

/** A value with the share of the run's largest it represents drawn behind it. */
function BarCell({
  value,
  peak,
  color,
  label,
  labelColor,
  trackWidth,
}: {
  value: number;
  peak: number;
  color: string;
  label: string;
  labelColor: string;
  trackWidth: string;
}) {
  return (
    <Box display="flex" alignItems="center" gap="8px">
      <Box
        aria-hidden="true"
        width={trackWidth}
        height="8px"
        flexShrink={0}
        borderRadius="2px"
        bg="pg.barTrack"
        overflow="hidden">
        <Box width={`${String(barFraction(value, peak) * 100)}%`} height="100%" bg={color} />
      </Box>
      <Text fontSize="13px" color={labelColor} m={0} whiteSpace="nowrap">
        {label}
      </Text>
    </Box>
  );
}

/** Which of a trained frontier's compressors this is, and its tag if it has one. */
function CandidateRank({result}: {result: JobResult}) {
  const candidate = result.candidate;
  if (candidate === null) {
    return null;
  }
  const tag = frontierTag(candidate.index, candidate.total);
  return (
    <>
      <Text fontSize="12px" color="pg.faint" m={0}>
        #{candidate.index}
      </Text>
      {tag !== null && (
        <Box
          px="6px"
          py="2px"
          borderRadius="4px"
          bg={TAG_COLOR[tag].bg}
          color={TAG_COLOR[tag].fg}
          fontSize="11px"
          fontWeight="semibold"
          whiteSpace="nowrap">
          {tag}
        </Box>
      )}
    </>
  );
}

/** One measurement and its bars. */
function MeasurementRow({
  row,
  peaks,
  onHighlight,
}: {
  row: JobResult;
  peaks: Peaks;
  onHighlight?: (key: string | null) => void;
}) {
  const color = CODEC_COLOR[row.job.compressor];
  const cell = {px: '12px', py: '10px', borderBottomWidth: '1px', borderColor: 'pg.border'} as const;
  // Both identifying cells are row headers, so a screen reader moving across
  // the numbers hears which codec and which setting they belong to.
  const header = {...cell, scope: 'row', textAlign: 'left', fontWeight: 'normal'} as const;
  return (
    <Box
      as="tr"
      _hover={{bg: 'pg.canvas'}}
      onMouseEnter={() => onHighlight?.(measurementKey(row))}
      onMouseLeave={() => onHighlight?.(null)}>
      <chakra.th {...header}>
        <Box display="flex" alignItems="center" gap="8px">
          <Box aria-hidden="true" boxSize="8px" borderRadius="full" bg={color} flexShrink={0} />
          <Text fontSize="14px" fontWeight="bold" color="pg.ink" m={0}>
            {row.job.compressor}
          </Text>
        </Box>
      </chakra.th>
      <chakra.th {...header}>
        <Box display="flex" alignItems="center" flexWrap="wrap" columnGap="8px" rowGap="4px">
          <Text fontSize="12px" color="pg.muted" m={0}>
            {rowDetail(row)}
          </Text>
          <CandidateRank result={row} />
        </Box>
      </chakra.th>
      <Box as="td" {...cell}>
        <Text fontSize="13px" color="pg.ink" m={0} whiteSpace="nowrap">
          {formatBytes(row.compressedSize)}
        </Text>
      </Box>
      <Box as="td" {...cell}>
        <BarCell
          value={row.ratio}
          peak={peaks.ratio}
          color={color}
          label={formatRatio(row.ratio)}
          labelColor="pg.ink"
          trackWidth="60px"
        />
      </Box>
      <Box as="td" {...cell}>
        <BarCell
          value={row.compressMBps}
          peak={peaks.compressMBps}
          color="pg.barCompress"
          label={formatSpeed(row.compressMBps)}
          labelColor="pg.secondary"
          trackWidth="40px"
        />
      </Box>
      <Box as="td" {...cell}>
        <BarCell
          value={row.decompressMBps}
          peak={peaks.decompressMBps}
          color="pg.barDecompress"
          label={formatSpeed(row.decompressMBps)}
          labelColor="pg.secondary"
          trackWidth="40px"
        />
      </Box>
    </Box>
  );
}

/**
 * A second press flips the direction. A new column starts with its best value
 * first, and the codec with the order the run produced, which is how the table
 * gets back to it.
 */
function nextSort(current: Sort, key: SortKey): Sort {
  if (current.key === key) {
    return {key, direction: current.direction === 'asc' ? 'desc' : 'asc'};
  }
  return {key, direction: FIRST_DIRECTION[key]};
}

const NONE_HIDDEN: ReadonlySet<CompressorName> = new Set();

/**
 * `onHighlight` hears the `measurementKey` of the row under the pointer, or
 * null once it leaves. Rows of a `hidden` codec are left out; the bars still
 * scale to the whole run, so they do not move as codecs come and go.
 */
export default function MeasurementsTable({
  runState,
  onHighlight,
  hidden = NONE_HIDDEN,
}: {
  runState: RunState;
  onHighlight?: (key: string | null) => void;
  hidden?: ReadonlySet<CompressorName>;
}) {
  const [sort, setSort] = useState<Sort>(DEFAULT_SORT);

  const results = resultsOf(runState);
  if (results.length === 0) {
    return null;
  }
  const peaks: Peaks = peaksOf(results);
  const rows = sortMeasurements(results, sort).filter((row) => !hidden.has(row.job.compressor));

  return (
    <Box display="flex" flexDirection="column" gap="16px" pt="12px">
      <Text
        id="measurements-title"
        color="pg.faint"
        fontSize="11px"
        fontWeight="extrabold"
        textTransform="uppercase"
        m={0}>
        Measurements
      </Text>
      <Box overflowX="auto">
        {/* Named by the heading above rather than a second copy of the word,
            so the two cannot drift apart. */}
        <Box
          as="table"
          aria-labelledby="measurements-title"
          width="100%"
          minW={TABLE_MIN_WIDTH}
          borderCollapse="collapse"
          style={{tableLayout: 'fixed'}}>
          <Box as="thead">
            <Box as="tr">
              {COLUMNS.map((column) => {
                const direction = sort.key === column.key ? sort.direction : null;
                return (
                  // `chakra.th` rather than `Box as="th"`: the polymorphic `as`
                  // keeps a div's props, and `scope` is not one of them.
                  <chakra.th
                    scope="col"
                    key={column.key}
                    width={column.width}
                    px="12px"
                    pb="8px"
                    textAlign="left"
                    borderBottomWidth="1px"
                    borderColor="pg.border"
                    aria-sort={direction === null ? undefined : direction === 'asc' ? 'ascending' : 'descending'}>
                    {column.sortable ? (
                      <chakra.button
                        type="button"
                        display="inline-flex"
                        alignItems="center"
                        gap="4px"
                        color="pg.faint"
                        fontSize="11px"
                        fontWeight="semibold"
                        textTransform="uppercase"
                        aria-label={`Sort by ${column.label.toLowerCase()}`}
                        onClick={() => {
                          setSort((current) => nextSort(current, column.key));
                        }}>
                        {column.label}
                        <SortArrow direction={direction} />
                      </chakra.button>
                    ) : (
                      <Text color="pg.faint" fontSize="11px" fontWeight="semibold" textTransform="uppercase" m={0}>
                        {column.label}
                      </Text>
                    )}
                  </chakra.th>
                );
              })}
            </Box>
          </Box>
          <Box as="tbody">
            {rows.map((row) => (
              <MeasurementRow key={measurementKey(row)} row={row} peaks={peaks} onHighlight={onHighlight} />
            ))}
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
