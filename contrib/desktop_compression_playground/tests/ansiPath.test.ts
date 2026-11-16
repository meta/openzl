// Copyright (c) Meta Platforms, Inc. and affiliates.

// @vitest-environment node

import childProcess from 'node:child_process';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {
  assertReadableByNativeTools,
  parseCodePage,
  unrepresentableCharacters,
  windowsAnsiCodePage,
} from '../electron/ansi-path.cjs';

describe('unrepresentableCharacters', () => {
  it('lets plain ASCII through for any code page', () => {
    expect(unrepresentableCharacters('C:\\Users\\alex\\data.bin', 1252)).toEqual([]);
    expect(unrepresentableCharacters('C:\\Users\\alex\\data.bin', 932)).toEqual([]);
  });

  it('accepts what the code page has, which is more than ASCII', () => {
    // é and € are in Windows-1252, so a Western PC reads them correctly.
    expect(unrepresentableCharacters('C:\\Users\\José\\café €5.bin', 1252)).toEqual([]);
    // Cyrillic is in Windows-1251, Japanese in 932.
    expect(unrepresentableCharacters('C:\\Users\\Алексей\\файл.bin', 1251)).toEqual([]);
    expect(unrepresentableCharacters('C:\\ユーザー\\データ.bin', 932)).toEqual([]);
  });

  it('accepts the punctuation people really put in Windows file names', () => {
    // Windows-1252 has these in 0x80-0x9F, where Latin-1 has control
    // characters, so they are easy to wrongly refuse.
    expect(unrepresentableCharacters('C:\\Users\\a\\report – final — “v2” ‘draft’ … ™ • Œuvre Ž.bin', 1252)).toEqual(
      [],
    );
  });

  it('rejects what Windows-1252 has no byte for, and everything beyond it', () => {
    // U+0081 is a control character with no byte of its own in Windows-1252.
    expect(unrepresentableCharacters('a\u0081b', 1252)).toEqual(['\u0081']);
    expect(unrepresentableCharacters('日本語', 1252).length).toBe(3);
  });

  it('knows each single-byte code page is not just Latin-1', () => {
    // A code page whose 0x80 is a letter or symbol, which a Latin-1 table
    // would not have.
    expect(unrepresentableCharacters('Ђ', 1251)).toEqual([]); // 0x80
    expect(unrepresentableCharacters('€', 1250)).toEqual([]); // 0x80
    expect(unrepresentableCharacters('€', 1253)).toEqual([]); // 0x80
    expect(unrepresentableCharacters('€', 1251)).toEqual([]); // 0x88
  });

  it('names the characters the code page lacks, once each', () => {
    expect(unrepresentableCharacters('C:\\ユーザー\\データ.bin', 1252).sort()).toEqual(
      [...new Set([...'ユーザーデータ'])].sort(),
    );
    // ā is not in Windows-1252: Windows would best-fit it to a plain a, which
    // is a different file name.
    expect(unrepresentableCharacters('C:\\Users\\Rīgā\\a.bin', 1252)).toEqual(['ī', 'ā']);
  });

  it('does not refuse what it cannot check', () => {
    // UTF-8 as the ANSI code page represents everything, and an unknown code
    // page is better left to zli's own error than blocked on a guess.
    expect(unrepresentableCharacters('C:\\ユーザー\\a.bin', 65001)).toEqual([]);
    expect(unrepresentableCharacters('C:\\ユーザー\\a.bin', 99999)).toEqual([]);
  });

  it('uses Windows’ own tables for the double-byte code pages', () => {
    // CP949 has the extended Hangul 갂 and 힣, which plain EUC-KR lacks.
    expect(unrepresentableCharacters('C:\\Temp\\갂힣', 949)).toEqual([]);
    // GB18030 has ḿ at 0xA8BC; CP936 does not.
    expect(unrepresentableCharacters('C:\\Temp\\ḿ', 936)).toEqual(['ḿ']);
    expect(unrepresentableCharacters('C:\\Temp\\中文', 950)).toEqual([]);
  });
});

describe('assertReadableByNativeTools', () => {
  it('says what is wrong, which code page, and how to fix it', () => {
    expect(() => {
      assertReadableByNativeTools('C:\\Users\\Rīgā\\a.bin', 'The input path', 'Rename the file.', 1252);
    }).toThrow(
      'The input path contains "ī", "ā", which this PC\'s ANSI code page (1252) cannot represent, and the native benchmark tools (zli, zstd-benchmark) read paths in that code page. Rename the file.',
    );
  });

  it('shows three characters and says there are more', () => {
    expect(() => {
      assertReadableByNativeTools('C:\\ユーザー\\データ.bin', 'The input path', 'Rename it.', 1252);
    }).toThrow(/contains "[^"]", "[^"]", "[^"]" and more, which/);
  });

  it('is silent for a path the code page can read', () => {
    expect(() => {
      assertReadableByNativeTools('C:\\Users\\José\\a.bin', 'The input path', 'Rename it.', 1252);
    }).not.toThrow();
  });
});

describe('parseCodePage', () => {
  it('reads the ACP value out of reg query output', () => {
    const output =
      '\r\nHKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Nls\\CodePage\r\n    ACP    REG_SZ    1252\r\n\r\n';
    expect(parseCodePage(output)).toBe(1252);
  });

  it('reads ACP, not the OEMCP or MACCP values whose names end the same way', () => {
    const output =
      '\r\nHKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Nls\\CodePage\r\n    MACCP    REG_SZ    10000\r\n    OEMCP    REG_SZ    437\r\n    ACP    REG_SZ    1252\r\n\r\n';
    expect(parseCodePage(output)).toBe(1252);
  });

  it('is undefined for anything else', () => {
    expect(parseCodePage('ERROR: The system was unable to find the specified registry key')).toBeUndefined();
    expect(parseCodePage('')).toBeUndefined();
  });
});

describe('windowsAnsiCodePage', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('reads the registry again after a failed read, and keeps a good one', async () => {
    const execFile = vi
      .spyOn(childProcess, 'execFile')
      .mockImplementationOnce(((
        _file: string,
        _args: string[],
        _options: object,
        callback: (...a: unknown[]) => void,
      ) => {
        callback(new Error('reg failed'), '', '');
      }) as unknown as typeof childProcess.execFile)
      .mockImplementation(((_file: string, _args: string[], _options: object, callback: (...a: unknown[]) => void) => {
        callback(null, '    ACP    REG_SZ    1252\r\n', '');
      }) as unknown as typeof childProcess.execFile);

    await expect(windowsAnsiCodePage()).resolves.toBeUndefined();
    await expect(windowsAnsiCodePage()).resolves.toBe(1252);
    await expect(windowsAnsiCodePage()).resolves.toBe(1252);
    expect(execFile).toHaveBeenCalledTimes(2);
  });
});
