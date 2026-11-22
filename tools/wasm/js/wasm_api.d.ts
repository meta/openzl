// Copyright (c) Meta Platforms, Inc. and affiliates.

export interface ProfileMetadata {
  /** The zli profile name, e.g. 'le-u32'; pass it to getSerializedCompressor. */
  readonly name: string;
  readonly description: string;
  /** Whether the loaded WASM build can build this profile's compressor. */
  readonly isWasmSupported: boolean;
  readonly wasmUnsupportedReason?: string;
  /** Whether the profile supports training, independent of WASM support. */
  readonly isTrainable: boolean;
}

/**
 * Browser benchmark result. Speed reflects the WebAssembly
 * implementation and may differ substantially from native performance.
 * Compression ratios are directly comparable with native results.
 */
export interface BenchmarkResult {
  iterations: number;
  srcSize: number;
  compressedSize: number;
  ratio: number;
  compressMs: number;
  decompressMs: number;
  compressMBps: number;
  decompressMBps: number;
}

/**
 * Training options Every field is optional and falls back to the trainer's default when omitted.
 */
export interface TrainOptions {
  /** Thread pool size. Clamped to the build's maximum, which is the default. */
  threads?: number;
  /**
   * Wall-clock budget in seconds. The trainers spend whatever budget they are
   * given, so this sets how long the call takes rather than bounding it.
   */
  maxTimeSecs?: number;
  /**
   * Return the Pareto frontier, compressors trading compression ratio against
   * speed, rather than only the best-ratio one.
   */
  paretoFrontier?: boolean;
  /**
   * How many compressors to keep. Defaults to `trainParetoCandidates`, smaller
   * limits are raised to that value which is the minimum. Ignored unless
   * paretoFrontier is set. The maximum is 25 to bound WASM allocations.
   * Bounds the results, not the work: every candidate is searched and
   * benchmarked either way.
   */
  maxNumCandidates?: number;
  /** Narrow the ACE search by not exploring successor graphs. */
  noAceSuccessors?: boolean;
  /** Skip the clustering trainer. */
  noClustering?: boolean;
}

export interface OpenZL {
  readonly profiles: readonly ProfileMetadata[];
  readonly maxBenchmarkIterations: number;
  readonly trainParetoCandidates: number;

  /**
   * Returns a serialized compressor for a profile name (e.g. 'serial'), at
   * the requested level, or OpenZL's current default level when omitted.
   */
  getSerializedCompressor(profile: string, compressionLevel?: number): Uint8Array;

  /**
   * Train runs synchronously and may run for several minutes. Browser callers
   * should invoke this API from their own Web Worker. Calling it from the main
   * thread blocks the UI until training completes.
   */
  train(data: Uint8Array, compressor: Uint8Array, options?: TrainOptions): Uint8Array[];
  compress(data: Uint8Array, compressor: Uint8Array): Uint8Array;
  /** Returns compressed bytes and a CBOR trace with stream previews. */
  compressWithTrace(data: Uint8Array, compressor: Uint8Array): {compressed: Uint8Array; trace: Uint8Array};
  decompress(compressed: Uint8Array): Uint8Array;
  /** Returns decompressed bytes and a CBOR trace with stream previews. */
  decompressWithTrace(compressed: Uint8Array): {decompressed: Uint8Array; trace: Uint8Array};
  benchmark(data: Uint8Array, compressor: Uint8Array, iterations?: number): BenchmarkResult;
}

export interface OpenZLOptions {
  wasmUrl?: string | URL;
  locateFile?: (path: string, prefix?: string) => string | URL;
  /** Receives numeric progress and its message from this module's training calls. */
  onTrainingProgress?: (progress: number, message: string) => void;
  /** Receives numeric progress and its message from this module's benchmark calls. */
  onBenchmarkProgress?: (progress: number, message: string) => void;
  [option: string]: unknown;
}

export declare function createOpenZL(options?: OpenZLOptions): Promise<OpenZL>;

/** Live benchmark-iteration ceiling from the WASM module, without holding an OpenZL instance. */
export declare function getOpenZLMaxIterations(options?: OpenZLOptions): Promise<number>;
