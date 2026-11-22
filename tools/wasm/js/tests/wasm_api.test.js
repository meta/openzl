// Copyright (c) Meta Platforms, Inc. and affiliates.
import {describe, it, before} from 'node:test';
import assert from 'node:assert/strict';
import {existsSync, readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

import {createOpenZL} from '../wasm_api.js';

const wasmUrl = new URL('../openzl.wasm', import.meta.url);
const jsUrl = new URL('../openzl.js', import.meta.url);
const hasArtifact = existsSync(fileURLToPath(wasmUrl)) && existsSync(fileURLToPath(jsUrl));
const TEST_TRAIN_MAX_TIME_SECS = 5; // Mirrors native C++ test time

describe('wasm_api', () => {
  let zl;
  const trainProgressEvents = [];
  const benchmarkProgressEvents = [];
  let handleTrainingProgress = (progress, message) => trainProgressEvents.push({progress, message});
  let handleBenchmarkProgress = (progress, message) => benchmarkProgressEvents.push({progress, message});

  before(async () => {
    if (!hasArtifact) {
      throw new Error(
        'openzl.wasm/js not found next to wasm_api.js — build with: emcmake cmake -DOPENZL_BUILD_WASM=ON -B build-wasm && cmake --build build-wasm --target openzl_wasm && cp build-wasm/tools/wasm/openzl.{js,wasm} fbcode/openzl/dev/tools/wasm/js/ (or tools/wasm/js/ in OSS). Skipping would hide missing coverage, so this fails fast instead.',
      );
    }
    zl = await createOpenZL({
      onTrainingProgress: (progress, message) => handleTrainingProgress(progress, message),
      onBenchmarkProgress: (progress, message) => handleBenchmarkProgress(progress, message),
    });
  });

  it('accepts repeated initialization with the same profile table', async () => {
    const second = await createOpenZL();
    const compressor = second.getSerializedCompressor('serial');
    assert.ok(compressor.length > 0);
  });

  it('rejects a WASM table with duplicate profile names', async () => {
    await assert.rejects(createOpenZL({
      onRuntimeInitialized() {
        const profileName = this._openzl_wasm_profileName;
        this._openzl_wasm_profileName = (value) => profileName(value === 1 ? 0 : value);
      },
    }), /duplicate profile names/);
  });

  it('reports support and training capabilities for the expanded profile catalog', () => {
    const supported = zl.profiles.filter(({isWasmSupported}) => isWasmSupported);
    assert.deepEqual(supported.map(({name}) => name).sort(), [
      'be-i16',
      'be-i32',
      'be-i64',
      'be-u16',
      'be-u32',
      'be-u64',
      'csv',
      'i8',
      'le-i16',
      'le-i32',
      'le-i64',
      'le-u16',
      'le-u32',
      'le-u64',
      'lz',
      'parquet',
      'serial',
      'u8',
      'zstd',
    ]);
    for (const profile of supported) {
      assert.equal(profile.isTrainable, profile.name !== 'csv' && profile.name !== 'zstd', profile.name);
      assert.equal(profile.wasmUnsupportedReason, undefined, profile.name);
    }
  });

  it('getSerializedCompressor returns non-empty bytes and is cached copy-on-read', () => {
    const c1 = zl.getSerializedCompressor('serial');
    assert.ok(c1 instanceof Uint8Array);
    assert.ok(c1.length > 0);
    const orig = c1.slice();
    c1[0] ^= 0xff;
    const c2 = zl.getSerializedCompressor('serial');
    assert.deepEqual(c2, orig);
    assert.notEqual(c1[0], c2[0]);
  });

  it('includes the requested compression level in the serialized compressor', () => {
    const levelOne = zl.getSerializedCompressor('serial', 1);
    const levelNine = zl.getSerializedCompressor('serial', 9);
    assert.notDeepEqual(levelOne, levelNine);
  });

  it('all supported profiles produce a compressor by name', () => {
    for (const {name} of zl.profiles.filter(({isWasmSupported}) => isWasmSupported)) {
      const c = zl.getSerializedCompressor(name);
      assert.ok(c.length > 0, `profile ${name} should produce compressor`);
    }
  });

  it('lists unsupported profiles and rejects compressor creation', () => {
    const unsupported = zl.profiles.filter(({isWasmSupported}) => !isWasmSupported);
    assert.ok(unsupported.length > 0);
    for (const profile of unsupported) {
      assert.ok(profile.wasmUnsupportedReason.startsWith('Cannot run in the browser:'), profile.name);
      assert.throws(() => zl.getSerializedCompressor(profile.name), /not supported by this WASM build/);
    }
  });

  it('exposes immutable profile metadata on the existing instance', () => {
    const profiles = zl.profiles;
    assert.ok(Object.isFrozen(profiles));
    for (const profile of profiles) {
      assert.ok(Object.isFrozen(profile));
      assert.ok(profile.description.length > 0);
      if (profile.isWasmSupported) {
        assert.ok(zl.getSerializedCompressor(profile.name).length > 0);
      }
    }
    assert.deepEqual(profiles[0], {
      name: 'serial',
      description: 'Serial data (aka raw bytes)',
      isWasmSupported: true,
      wasmUnsupportedReason: undefined,
      isTrainable: true,
    });
  });

  it('rejects unknown profile', () => {
    assert.throws(() => zl.getSerializedCompressor(0), /profile must be a profile name string/);
    assert.throws(() => zl.getSerializedCompressor('not-a-profile'), /unknown profile not-a-profile; available: serial, /);
  });

  it('validates compression levels', () => {
    assert.doesNotThrow(() => zl.getSerializedCompressor('serial'));
    for (const level of [-1, 0, 20, 1.5, NaN, null]) {
      assert.throws(() => zl.getSerializedCompressor('serial', level), /compressionLevel/);
    }
  });

  it('roundtrips serial data', () => {
    const src = new Uint8Array(4096);
    for (let i = 0; i < src.length; i++) src[i] = 97 + (i % 10);
    const comp = zl.getSerializedCompressor('serial');
    const frame = zl.compress(src, comp);
    assert.ok(frame.length > 0);
    const out = zl.decompress(frame);
    assert.deepEqual(out, src);
  });

  it('returns an owned trace on request without changing the compressed frame', () => {
    const src = new TextEncoder().encode('hello openzl '.repeat(100));
    const compressor = zl.getSerializedCompressor('serial');
    const frame = zl.compress(src, compressor);
    const {compressed, trace} = zl.compressWithTrace(src, compressor);
    assert.deepEqual(compressed, frame);
    assert.deepEqual(zl.decompress(compressed), src);
    assert.ok(trace instanceof Uint8Array);
    assert.ok(trace.length > 0);

    const saved = trace.slice();
    zl.compressWithTrace(new Uint8Array(1024), compressor);
    assert.deepEqual(trace, saved);
    assert.deepEqual(zl.compress(src, compressor), frame);
  });

  it('returns an owned trace on request without changing the decompressed bytes', () => {
    const src = new TextEncoder().encode('hello openzl '.repeat(100));
    const frame = zl.compress(src, zl.getSerializedCompressor('serial'));
    const {decompressed, trace} = zl.decompressWithTrace(frame);
    assert.deepEqual(decompressed, src);
    assert.ok(trace instanceof Uint8Array);
    assert.ok(trace.length > 0);
    assert.deepEqual(zl.decompress(frame), src);
  });

  it('reports decompression errors when tracing is requested', () => {
    const frame = zl.compress(new TextEncoder().encode('hello openzl '.repeat(100)), zl.getSerializedCompressor('serial'));
    assert.throws(() => zl.decompressWithTrace(frame.subarray(0, frame.length - 1)), (error) => error.code > 0);
  });

  function roundTrip(profile, src) {
    const compressor = zl.getSerializedCompressor(profile);
    assert.deepEqual(zl.decompress(zl.compress(src, compressor)), src);
  }

  function trainedRoundTrip(profile, src) {
    const base = zl.getSerializedCompressor(profile);
    const candidates = zl.train(src, base, {maxTimeSecs: TEST_TRAIN_MAX_TIME_SECS});
    assert.equal(candidates.length, 1);
    for (const trained of candidates) {
      assert.ok(trained.length > 0);
      assert.deepEqual(zl.decompress(zl.compress(src, trained)), src);
    }
  }

  function integerSample(width, setter, signed, littleEndian) {
    const src = new Uint8Array(1024 * width);
    const view = new DataView(src.buffer);
    for (let i = 0; i < 1024; i++) {
      const value = (i % 200) - (signed ? 100 : 0);
      view[setter](i * width, width === 8 ? BigInt(value) : value, littleEndian);
    }
    return src;
  }

  function csvSample() {
    let csv = 'id,name,value\n';
    for (let i = 0; i < 1024; i++) csv += `${i % 100},item${i % 10},${i % 50}\n`;
    return new TextEncoder().encode(csv);
  }

  function parquetSample() {
    return new Uint8Array(readFileSync(new URL(
      '../../../../cli/tests/sample_files/parquet/simple.parquet', import.meta.url,
    )));
  }

  it('roundtrips u8', () => {
    roundTrip('u8', integerSample(1, 'setUint8', false, true));
  });

  it('trains u8 and roundtrips the trained compressor', () => {
    trainedRoundTrip('u8', integerSample(1, 'setUint8', false, true));
  });

  it('roundtrips i8', () => {
    roundTrip('i8', integerSample(1, 'setInt8', true, true));
  });

  it('trains i8 and roundtrips the trained compressor', () => {
    trainedRoundTrip('i8', integerSample(1, 'setInt8', true, true));
  });

  it('roundtrips le-u16', () => {
    roundTrip('le-u16', integerSample(2, 'setUint16', false, true));
  });

  it('trains le-u16 and roundtrips the trained compressor', () => {
    trainedRoundTrip('le-u16', integerSample(2, 'setUint16', false, true));
  });

  it('roundtrips le-i16', () => {
    roundTrip('le-i16', integerSample(2, 'setInt16', true, true));
  });

  it('trains le-i16 and roundtrips the trained compressor', () => {
    trainedRoundTrip('le-i16', integerSample(2, 'setInt16', true, true));
  });

  it('roundtrips le-u32', () => {
    roundTrip('le-u32', integerSample(4, 'setUint32', false, true));
  });

  it('trains le-u32 and roundtrips the trained compressor', () => {
    trainedRoundTrip('le-u32', integerSample(4, 'setUint32', false, true));
  });

  it('roundtrips le-i32', () => {
    roundTrip('le-i32', integerSample(4, 'setInt32', true, true));
  });

  it('trains le-i32 and roundtrips the trained compressor', () => {
    trainedRoundTrip('le-i32', integerSample(4, 'setInt32', true, true));
  });

  it('roundtrips le-u64', () => {
    roundTrip('le-u64', integerSample(8, 'setBigUint64', false, true));
  });

  it('trains le-u64 and roundtrips the trained compressor', () => {
    trainedRoundTrip('le-u64', integerSample(8, 'setBigUint64', false, true));
  });

  it('roundtrips le-i64', () => {
    roundTrip('le-i64', integerSample(8, 'setBigInt64', true, true));
  });

  it('trains le-i64 and roundtrips the trained compressor', () => {
    trainedRoundTrip('le-i64', integerSample(8, 'setBigInt64', true, true));
  });

  it('roundtrips be-u16', () => {
    roundTrip('be-u16', integerSample(2, 'setUint16', false, false));
  });

  it('trains be-u16 and roundtrips the trained compressor', () => {
    trainedRoundTrip('be-u16', integerSample(2, 'setUint16', false, false));
  });

  it('roundtrips be-i16', () => {
    roundTrip('be-i16', integerSample(2, 'setInt16', true, false));
  });

  it('trains be-i16 and roundtrips the trained compressor', () => {
    trainedRoundTrip('be-i16', integerSample(2, 'setInt16', true, false));
  });

  it('roundtrips be-u32', () => {
    roundTrip('be-u32', integerSample(4, 'setUint32', false, false));
  });

  it('trains be-u32 and roundtrips the trained compressor', () => {
    trainedRoundTrip('be-u32', integerSample(4, 'setUint32', false, false));
  });

  it('roundtrips be-i32', () => {
    roundTrip('be-i32', integerSample(4, 'setInt32', true, false));
  });

  it('trains be-i32 and roundtrips the trained compressor', () => {
    trainedRoundTrip('be-i32', integerSample(4, 'setInt32', true, false));
  });

  it('roundtrips be-u64', () => {
    roundTrip('be-u64', integerSample(8, 'setBigUint64', false, false));
  });

  it('trains be-u64 and roundtrips the trained compressor', () => {
    trainedRoundTrip('be-u64', integerSample(8, 'setBigUint64', false, false));
  });

  it('roundtrips be-i64', () => {
    roundTrip('be-i64', integerSample(8, 'setBigInt64', true, false));
  });

  it('trains be-i64 and roundtrips the trained compressor', () => {
    trainedRoundTrip('be-i64', integerSample(8, 'setBigInt64', true, false));
  });

  it('roundtrips lz', () => {
    roundTrip('lz', csvSample());
  });

  it('trains lz and roundtrips the trained compressor', () => {
    trainedRoundTrip('lz', csvSample());
  });

  it('roundtrips zstd', () => {
    roundTrip('zstd', csvSample());
  });

  it('reports Zstd training as unavailable without dictionary bundles', () => {
    assert.equal(zl.profiles.find(({name}) => name === 'zstd').isTrainable, false);
    assert.throws(() => zl.train(csvSample(), zl.getSerializedCompressor('zstd'), {
      maxTimeSecs: TEST_TRAIN_MAX_TIME_SECS,
    }), {code: 1});
  });

  it('roundtrips csv', () => {
    roundTrip('csv', csvSample());
  });

  it('records the CSV training failure with the existing dependency registration', () => {
    assert.equal(zl.profiles.find(({name}) => name === 'csv').isTrainable, false);
    assert.throws(() => zl.train(csvSample(), zl.getSerializedCompressor('csv'), {
      maxTimeSecs: TEST_TRAIN_MAX_TIME_SECS,
    }), {code: 1});
  });

  it('roundtrips parquet', () => {
    roundTrip('parquet', parquetSample());
  });

  it('trains parquet and roundtrips the trained compressor', () => {
    trainedRoundTrip('parquet', parquetSample());
  });

  it('roundtrips empty input', () => {
    const comp = zl.getSerializedCompressor('serial');
    const frame = zl.compress(new Uint8Array(0), comp);
    const out = zl.decompress(frame);
    assert.equal(out.length, 0);
  });

  it('validates Uint8Array inputs', () => {
    const comp = zl.getSerializedCompressor('serial');
    assert.throws(() => zl.train('not bytes', comp), /expects Uint8Array/);
    assert.throws(() => zl.train(new Uint8Array(10), 'not comp'), /expects.*compressor/);
    assert.throws(() => zl.compress('not bytes', comp), /expects Uint8Array/);
    assert.throws(() => zl.compress(new Uint8Array(10), 'not comp'), /expects.*compressor/);
    assert.throws(() => zl.decompress('bad'), /expects Uint8Array/);
    assert.throws(() => zl.benchmark('bad', comp), /expects Uint8Array/);
  });

  it('validates training inputs', () => {
    const comp = zl.getSerializedCompressor('serial');
    assert.throws(() => zl.train(new Uint8Array(0), comp), /non-empty data/);
    assert.throws(() => zl.train(new Uint8Array(10), new Uint8Array(0)), /non-empty serialized compressor/);

    const oversized = new Proxy(new Uint8Array(1), {
      get(target, property) {
        return property === 'length' ? 150 * 1024 * 1024 : Reflect.get(target, property);
      },
    });
    assert.throws(() => zl.train(oversized, comp), /smaller than 150 MiB/);

    // Train options are coerced to non-negative integers for the u64 ABI, so
    // non-finite or non-numeric values throw a clear error naming the field.
    for (const bad of [NaN, Infinity, '4', true]) {
      assert.throws(() => zl.train(new Uint8Array(10), comp, {threads: bad}), /threads/);
      assert.throws(() => zl.train(new Uint8Array(10), comp, {maxTimeSecs: bad}), /maxTimeSecs/);
      assert.throws(
        () =>
          zl.train(new Uint8Array(10), comp, {
            paretoFrontier: true,
            maxNumCandidates: bad,
          }),
        /maxNumCandidates/,
      );
    }

    assert.throws(
      () =>
        zl.train(new Uint8Array(10), comp, {
          paretoFrontier: true,
          maxNumCandidates: 26,
        }),
      /maxNumCandidates must be at most 25/,
    );
  });

  it('trains serial and roundtrips the trained compressor', () => {
    trainProgressEvents.length = 0;
    const src = new Uint8Array(1024);
    for (let i = 0; i < src.length; i++) src[i] = 97 + (i % 10);
    const base = zl.getSerializedCompressor('serial');
    // Without paretoFrontier the trainer returns exactly one compressor.
    const [trained] = zl.train(src, base, {
      maxTimeSecs: TEST_TRAIN_MAX_TIME_SECS,
    });
    assert.ok(trained.length > 0);
    assert.ok(trainProgressEvents.length > 0);
    for (const event of trainProgressEvents) {
      // Verify that we are getting progress updates when training.
      assert.equal(typeof event.message, 'string');
      assert.ok(event.progress >= 0 && event.progress <= 1);
    }
    const aceEvents = trainProgressEvents.filter(({message}) => message.startsWith('Training ACE graph'));
    assert.ok(aceEvents.length >= 2);
    assert.ok(aceEvents.every(({message}) => message === 'Training ACE graph 1 / 1: ACE progress'));
    const aceProgress = aceEvents.map(({progress}) => progress);
    for (let i = 1; i < aceProgress.length; i++) {
      assert.ok(aceProgress[i] >= aceProgress[i - 1]);
    }

    const frame = zl.compress(src, trained);
    assert.deepEqual(zl.decompress(frame), src);
  });

  it('continues training if the progress callback throws', () => {
    const src = new Uint8Array(1024);
    for (let i = 0; i < src.length; i++) src[i] = 97 + (i % 10);
    const base = zl.getSerializedCompressor('serial');
    const stderrMessages = [];
    const originalConsoleError = console.error;
    console.error = (message) => stderrMessages.push(message);
    handleTrainingProgress = () => {
      throw new Error('UI callback failed');
    };

    try {
      const trained = zl.train(src, base, {
        maxTimeSecs: TEST_TRAIN_MAX_TIME_SECS,
      });
      assert.ok(trained.length > 0);
    } finally {
      handleTrainingProgress = (progress, message) => trainProgressEvents.push({progress, message});
      console.error = originalConsoleError;
    }

    assert.ok(stderrMessages.some((message) => message.includes('UI callback failed')));
  });

  it('reports native training errors', () => {
    assert.throws(
      () => zl.train(new Uint8Array(10), new Uint8Array([1, 2, 3])),
      (error) => Number.isInteger(error.code),
    );
  });

  it('paretoFrontier returns a frontier of usable compressors', () => {
    const src = new Uint8Array(1024);
    for (let i = 0; i < src.length; i++) src[i] = 97 + (i % 10);
    const base = zl.getSerializedCompressor('serial');

    const frontier = zl.train(src, base, {
      maxTimeSecs: TEST_TRAIN_MAX_TIME_SECS,
      paretoFrontier: true,
      maxNumCandidates: 25,
    });
    assert.ok(Array.isArray(frontier));
    // The frontier's size depends on the data, so the only guarantees are that
    // training produced something and stayed within the candidate limit.
    assert.ok(frontier.length >= 1);
    assert.ok(frontier.length <= 25);

    // Every compressor on the frontier is usable, not just the first. Sizes are
    // not asserted: the speed-optimal end can emit more bytes than it was given.
    for (const compressor of frontier) {
      assert.ok(compressor instanceof Uint8Array);
      assert.ok(compressor.length > 0);
      assert.deepEqual(zl.decompress(zl.compress(src, compressor)), src);
    }
  });

  it('benchmark compresses and decompresses data and returns expected metrics', () => {
    const src = new Uint8Array(2048);
    for (let i = 0; i < src.length; i++) src[i] = i & 0xff;
    const comp = zl.getSerializedCompressor('serial');
    benchmarkProgressEvents.length = 0;
    const r = zl.benchmark(src, comp, 2);
    assert.deepEqual(
      benchmarkProgressEvents.map(({progress}) => progress),
      [0.25, 0.5, 0.75, 1],
    );
    assert.deepEqual(
      benchmarkProgressEvents.map(({message}) => message),
      [
        'Benchmarking compression',
        'Benchmarking compression',
        'Benchmarking decompression',
        'Benchmarking decompression',
      ],
    );
    assert.equal(r.iterations, 2);
    assert.equal(r.srcSize, src.length);
    assert.ok(r.compressedSize > 0);
    assert.ok(r.ratio >= 0);
    assert.ok(r.compressMs >= 0);
    assert.ok(r.decompressMs >= 0);
    assert.ok(r.compressMBps > 0);
    assert.ok(r.decompressMBps > 0);
    assert.ok(Number.isFinite(r.compressMBps) || r.compressMBps === Infinity);
    assert.ok(Number.isFinite(r.decompressMBps) || r.decompressMBps === Infinity);
    assert.ok(Number.isInteger(zl.maxBenchmarkIterations));
    assert.ok(zl.maxBenchmarkIterations >= 1);
    assert.equal(zl.benchmark(src, comp, 0).iterations, 1);
    assert.equal(zl.benchmark(src, comp, zl.maxBenchmarkIterations + 1).iterations, zl.maxBenchmarkIterations);
    assert.equal(zl.benchmark(src, comp, 1.9).iterations, 1);
  });
});
