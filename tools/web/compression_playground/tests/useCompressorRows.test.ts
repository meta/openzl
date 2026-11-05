// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment jsdom
import {describe, expect, it} from 'vitest';
import {act, renderHook} from '@testing-library/react';
import {useCompressorRows} from '../src/useCompressorRows.ts';
import {createCompressorRow, type OpenZlRow} from '../src/compressors.ts';

/**
 * Every way to change a row lives on the hook, so none of this needs a
 * rendered card. The fields exercised here are deliberately ones the row
 * variants agree on -- `level` moves between variants further up the stack.
 */
describe('useCompressorRows', () => {
  it('never reuses an id, so a finished measurement cannot land on a new row', () => {
    // `rowId` travels into `BenchmarkJob` and comes back on its result.
    const {result} = renderHook(() => useCompressorRows());

    act(() => {
      result.current.removeRow(2);
    });
    act(() => {
      result.current.addRow();
    });

    expect(result.current.rows.map((row) => row.id)).toEqual([1, 3, 4]);
  });

  it('seeds the counter above whatever it was handed', () => {
    const {result} = renderHook(() => useCompressorRows([createCompressorRow(7, 'zstd')]));

    act(() => {
      result.current.addRow();
    });

    expect(result.current.rows.map((row) => row.id)).toEqual([7, 8]);
  });

  it('patches one row and leaves the others untouched', () => {
    const {result} = renderHook(() => useCompressorRows());
    const before = result.current.rows;

    act(() => {
      result.current.patchRow(1, 'OpenZL', {optionsOpen: true});
    });

    const after = result.current.rows;
    expect((after[0] as OpenZlRow).optionsOpen).toBe(true);
    expect([after[1], after[2]]).toEqual([before[1], before[2]]);
  });

  it('refuses OpenZL-only fields on a row that has none', () => {
    // Row 2 is zstd. Without the compressor check the patch would graft a
    // `profile` onto it and the row would no longer match its own variant.
    const {result} = renderHook(() => useCompressorRows());

    act(() => {
      result.current.patchRow(2, 'OpenZL', {profile: 'le-u32'});
    });

    expect(result.current.rows[1]).not.toHaveProperty('profile');
  });

  it('patches the levels of a zstd row', () => {
    const {result} = renderHook(() => useCompressorRows());

    act(() => {
      result.current.patchRow(2, 'zstd', {levels: [3]});
    });

    expect(result.current.rows[1]).toEqual({id: 2, compressor: 'zstd', levels: [3]});
  });

  it('rebuilds a row from defaults when its compressor changes', () => {
    // Patching would carry fields across that the new compressor has no use
    // for, so the row is rebuilt rather than merged.
    const {result} = renderHook(() => useCompressorRows([createCompressorRow(1, 'OpenZL')]));

    const fresh = createCompressorRow(1, 'OpenZL') as OpenZlRow;
    act(() => {
      // Away from the default, so a merge would be visible and a rebuild not.
      result.current.patchRow(1, 'OpenZL', {optionsOpen: !fresh.optionsOpen});
    });
    act(() => {
      result.current.changeCompressor(1, 'zstd');
    });
    act(() => {
      result.current.changeCompressor(1, 'OpenZL');
    });

    const rebuilt = result.current.rows[0] as OpenZlRow;
    expect(rebuilt.id).toBe(1);
    expect(rebuilt.optionsOpen).toBe(fresh.optionsOpen);
  });

  it.each(['zstd', 'gzip'] as const)('rejects a duplicate %s row and allows it after removal', (compressor) => {
    const {result} = renderHook(() => useCompressorRows());
    const before = result.current.rows;
    act(() => result.current.changeCompressor(1, compressor));
    expect(result.current.rows).toBe(before);

    act(() => {
      result.current.removeRow(compressor === 'zstd' ? 2 : 3);
      result.current.changeCompressor(1, compressor);
    });
    expect(result.current.rows.filter((row) => row.compressor === compressor)).toEqual([
      createCompressorRow(1, compressor),
    ]);
  });
});
