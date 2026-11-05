// Copyright (c) Meta Platforms, Inc. and affiliates.

import {useRef, useState} from 'react';
import {createCompressorRow, type CompressorName, type CompressorRow} from './compressors.ts';

function defaultRows(): readonly CompressorRow[] {
  return [createCompressorRow(1, 'OpenZL'), createCompressorRow(2, 'zstd'), createCompressorRow(3, 'gzip')];
}

/**
 * The fields a patch may change on a row of compressor `C`, so an OpenZL-only
 * field such as `profile` cannot be handed to a zstd or gzip row.
 *
 * Neither `id` nor `compressor` can be patched: `id` is the argument selecting
 * the row, and a new compressor needs a rebuilt row (`changeCompressor`).
 */
export type RowPatch<C extends CompressorName> = Omit<
  Partial<Extract<CompressorRow, {compressor: C}>>,
  'id' | 'compressor'
>;

/** Names the row's compressor as well as its id, so the patch can be typed by it. */
export type PatchRow = (
  ...args:
    | [id: number, compressor: 'OpenZL', patch: RowPatch<'OpenZL'>]
    | [id: number, compressor: 'zstd' | 'gzip', patch: RowPatch<'zstd' | 'gzip'>]
) => void;
export interface CompressorRows {
  readonly rows: readonly CompressorRow[];
  readonly addRow: () => void;
  readonly removeRow: (id: number) => void;
  readonly patchRow: PatchRow;
  readonly changeCompressor: (id: number, compressor: CompressorName) => void;
}

/**
 * Owns the compressor rows and hands out their ids. Every way to change a row
 * lives here rather than in the card that renders them, so the state and the
 * rules for moving it stay together and can be tested without mounting any UI.
 *
 * Ids never repeat within a session: `rowId` travels into `BenchmarkJob` and
 * comes back on its result, so reusing a removed row's id would attach a
 * finished measurement to whatever row took its place. The counter therefore
 * starts above the highest id it was seeded with and only ever climbs.
 */
export function useCompressorRows(initialRows?: readonly CompressorRow[]): CompressorRows {
  const [rows, setRows] = useState<readonly CompressorRow[]>(() => initialRows ?? defaultRows());
  // Seeded with 0 because `Math.max()` over an empty list is -Infinity.
  const nextId = useRef(Math.max(0, ...rows.map((row) => row.id)) + 1);

  return {
    rows,
    addRow: () => {
      const id = nextId.current;
      nextId.current += 1;
      setRows((current) => [...current, createCompressorRow(id, 'OpenZL')]);
    },
    removeRow: (id: number) => {
      setRows((current) => current.filter((row) => row.id !== id));
    },
    // Matches on the compressor too, so the patch only ever lands on a row of
    // the variant it was typed against.
    patchRow: (id, compressor, patch) => {
      setRows((current) =>
        current.map((row) => (row.id === id && row.compressor === compressor ? {...row, ...patch} : row)),
      );
    },
    changeCompressor: (id: number, compressor: CompressorName) => {
      // A fresh row resets the dependent fields: a zstd level of 19 is not a
      // valid OpenZL level, and a profile means nothing to gzip.
      setRows((current) => {
        if (compressor !== 'OpenZL' && current.some((row) => row.id !== id && row.compressor === compressor)) {
          return current;
        }
        return current.map((row) => (row.id === id ? createCompressorRow(id, compressor) : row));
      });
    },
  };
}
