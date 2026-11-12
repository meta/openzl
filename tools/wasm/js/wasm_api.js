// Copyright (c) Meta Platforms, Inc. and affiliates.

// The build sets -sMEMORY64=1, so size_t and pointers are 8 bytes.
const SIZE_T_BYTES = 8;
const PTR_BYTES = 8;
const DOUBLE_BYTES = 8;

const DEFAULT_BENCHMARK_ITERATIONS = 10;
const MAX_TRAIN_SAMPLE_BYTES = 150 * 1024 * 1024;
const MAX_TRAIN_CANDIDATES = 25;

// sizeof(openzl_wasm_TrainOptions). The build sets -sMEMORY64=1, so size_t is
// 8 bytes while int stays 4, which leaves padding after paretoFrontier.
const TRAIN_OPTIONS_BYTES = 40;

// Coerces a train option to a non-negative integer for the u64 ABI
function toU64(value, name, max) {
  if (value === undefined || value === null) {
    return 0n;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number, got ${String(value)}`);
  }
  const normalized = Math.max(0, Math.floor(value));
  if (max !== undefined && normalized > max) {
    throw new Error(`${name} must be at most ${max}, got ${String(value)}`);
  }
  return BigInt(normalized);
}

// Populates an already-allocated openzl_wasm_TrainOptions. Zeroing first
// leaves padding and unset fields at 0, which C reads as "use the default".
function writeTrainOptions(mod, ptr, options) {
  const maxNumCandidates = options.paretoFrontier
    ? toU64(options.maxNumCandidates, 'maxNumCandidates', MAX_TRAIN_CANDIDATES)
    : 0n;
  mod.HEAPU8.fill(0, ptr, ptr + TRAIN_OPTIONS_BYTES);
  const u64 = (offset) => Math.floor((ptr + offset) / 8);
  mod.HEAPU64[u64(0)] = toU64(options.threads, 'threads'); // threads
  mod.HEAPU64[u64(8)] = toU64(options.maxTimeSecs, 'maxTimeSecs'); // maxTimeSecs
  mod.HEAPU8[ptr + 16] = options.paretoFrontier ? 1 : 0; // paretoFrontier
  mod.HEAPU64[u64(24)] = maxNumCandidates; // maxNumCandidates if pareto frontier is on
  mod.HEAPU8[ptr + 32] = options.noAceSuccessors ? 1 : 0; // noAceSuccessors
  mod.HEAPU8[ptr + 36] = options.noClustering ? 1 : 0; // noClustering
  return maxNumCandidates;
}

// Runs one training call and returns every compressor it produced, ordered
// best ratio first.
function runTraining(mod, mem, minCandidates, data, compressor, options) {
  if (!(data instanceof Uint8Array)) {
    throw new Error('train expects Uint8Array');
  }
  if (!(compressor instanceof Uint8Array)) {
    throw new Error('train expects a serialized compressor Uint8Array');
  }
  if (data.length === 0) {
    throw new Error('train expects non-empty data');
  }
  if (data.length >= MAX_TRAIN_SAMPLE_BYTES) {
    throw new Error('train sample must be smaller than 150 MiB');
  }
  if (compressor.length === 0) {
    throw new Error('train expects a non-empty serialized compressor');
  }
  // The native side writes at most this many entries and drops the rest, so
  // the arrays and the capacity it is told about must agree. A larger request
  // gets a larger array; a smaller one still needs the floor because native
  // training raises smaller pruning limits to that value.
  // Validated up front so the out-arrays and the count handed to native agree
  // on an integer, and invalid input throws before the out-arrays are allocated.
  let compPtr = 0;
  let srcPtr = 0;
  let optsPtr = 0;
  let outBufs = 0;
  let outSizes = 0;
  let outCount = 0;
  const bufPtrs = [];
  try {
    optsPtr = mem.malloc(TRAIN_OPTIONS_BYTES);
    const maxNumCandidates = writeTrainOptions(mod, optsPtr, options);
    const capacity = Math.max(minCandidates, Number(maxNumCandidates));
    compPtr = mem.writeBytes(compressor);
    srcPtr = mem.writeBytes(data);
    outBufs = mem.malloc(PTR_BYTES * capacity);
    outSizes = mem.malloc(SIZE_T_BYTES * capacity);
    outCount = mem.malloc(SIZE_T_BYTES);

    const code = mod._openzl_wasm_train(
      mem.toWasm64(compPtr),
      mem.toWasm64(compressor.length),
      mem.toWasm64(srcPtr),
      mem.toWasm64(data.length),
      mem.toWasm64(optsPtr),
      mem.toWasm64(outBufs),
      mem.toWasm64(outSizes),
      mem.toWasm64(capacity),
      mem.toWasm64(outCount),
    );
    if (code !== 0) {
      throwWasmError(mem, code, 'train failed');
    }

    // Reading past the arrays would hand free() garbage pointers, so trust the
    // allocation rather than the reported count if they ever disagree.
    const count = Math.min(mem.readSizeT(outCount), capacity);
    // Take ownership of every buffer before any read can throw, so the finally
    // block releases all of them.
    for (let i = 0; i < count; i++) {
      bufPtrs.push(mem.readPointer(outBufs + i * PTR_BYTES));
    }
    return bufPtrs.map((ptr, i) => mem.readBytes(ptr, mem.readSizeT(outSizes + i * SIZE_T_BYTES)));
  } finally {
    for (const ptr of bufPtrs) {
      mem.free(ptr);
    }
    mem.free(compPtr);
    mem.free(srcPtr);
    mem.free(optsPtr);
    mem.free(outBufs);
    mem.free(outSizes);
    mem.free(outCount);
  }
}

function runCompression(mod, mem, data, compressor, getTrace) {
  if (!(data instanceof Uint8Array)) {
    throw new Error('compress expects Uint8Array');
  }
  if (!(compressor instanceof Uint8Array)) {
    throw new Error('compress expects a serialized compressor Uint8Array');
  }
  let compPtr = 0;
  let srcPtr = 0;
  let outBuf = 0;
  let outSize = 0;
  let bufPtr = 0;
  let outTrace = 0;
  let outTraceSize = 0;
  let tracePtr = 0;
  try {
    compPtr = mem.writeBytes(compressor);
    srcPtr = mem.writeBytes(data);
    outBuf = mem.malloc(PTR_BYTES);
    outSize = mem.malloc(SIZE_T_BYTES);
    if (getTrace) {
      outTrace = mem.malloc(PTR_BYTES);
      outTraceSize = mem.malloc(SIZE_T_BYTES);
    }
    const args = [
      mem.toWasm64(compPtr),
      mem.toWasm64(compressor.length),
      mem.toWasm64(srcPtr),
      mem.toWasm64(data.length),
      mem.toWasm64(outBuf),
      mem.toWasm64(outSize),
    ];
    const errorCode = getTrace
      ? mod._openzl_wasm_compressWithTrace(...args, mem.toWasm64(outTrace), mem.toWasm64(outTraceSize))
      : mod._openzl_wasm_compress(...args);
    if (errorCode !== 0) {
      throwWasmError(mem, errorCode, 'compress failed');
    }
    bufPtr = mem.readPointer(outBuf);
    if (getTrace) {
      tracePtr = mem.readPointer(outTrace);
    }
    const compressed = mem.readBytes(bufPtr, mem.readSizeT(outSize));
    return getTrace ? {compressed, trace: mem.readBytes(tracePtr, mem.readSizeT(outTraceSize))} : compressed;
  } finally {
    mem.free(tracePtr);
    mem.free(outTrace);
    mem.free(outTraceSize);
    mem.free(bufPtr);
    mem.free(compPtr);
    mem.free(srcPtr);
    mem.free(outBuf);
    mem.free(outSize);
  }
}

function runDecompression(mod, mem, compressed, getTrace) {
  if (!(compressed instanceof Uint8Array)) {
    throw new Error('decompress expects Uint8Array');
  }
  let srcPtr = 0;
  let outBuf = 0;
  let outSize = 0;
  let bufPtr = 0;
  let outTrace = 0;
  let outTraceSize = 0;
  let tracePtr = 0;
  try {
    srcPtr = mem.writeBytes(compressed);
    outBuf = mem.malloc(PTR_BYTES);
    outSize = mem.malloc(SIZE_T_BYTES);
    if (getTrace) {
      outTrace = mem.malloc(PTR_BYTES);
      outTraceSize = mem.malloc(SIZE_T_BYTES);
    }
    const args = [
      mem.toWasm64(srcPtr),
      mem.toWasm64(compressed.length),
      mem.toWasm64(outBuf),
      mem.toWasm64(outSize),
    ];
    const errorCode = getTrace
      ? mod._openzl_wasm_decompressWithTrace(...args, mem.toWasm64(outTrace), mem.toWasm64(outTraceSize))
      : mod._openzl_wasm_decompress(...args);
    if (errorCode !== 0) {
      throwWasmError(mem, errorCode, 'decompress failed');
    }
    bufPtr = mem.readPointer(outBuf);
    if (getTrace) {
      tracePtr = mem.readPointer(outTrace);
    }
    const decompressed = mem.readBytes(bufPtr, mem.readSizeT(outSize));
    return getTrace ? {decompressed, trace: mem.readBytes(tracePtr, mem.readSizeT(outTraceSize))} : decompressed;
  } finally {
    mem.free(tracePtr);
    mem.free(outTrace);
    mem.free(outTraceSize);
    mem.free(bufPtr);
    mem.free(srcPtr);
    mem.free(outBuf);
    mem.free(outSize);
  }
}

function clampIterations(n, maxIterations) {
  if (typeof n !== 'number' || !Number.isFinite(n)) {
    throw new Error(`benchmark iterations must be a finite number, got ${String(n)} — expected 1..${maxIterations}`);
  }
  return Math.min(Math.max(1, Math.floor(n)), maxIterations);
}

const MS_PER_S = 1000;
const BYTES_PER_MB = 1000 * 1000;

// Convert total bytes and elapsed milliseconds to decimal MB/s. A zero
// duration produces Infinity.
function mbPerSec(bytes, iterations, ms) {
  return ms > 0 ? (bytes * iterations) / (ms / MS_PER_S) / BYTES_PER_MB : Infinity;
}

// ABI ownership: JS allocates and frees input buffers and out-parameter
// storage, while C borrows inputs for one call. C transfers byte-buffer
// outputs to JS, which copies and frees them; borrowed strings are copied
// with UTF8ToString. No raw pointer leaves this wrapper.
function createMemory(mod) {
  // -sMEMORY64=1 makes every pointer, size and count in the C ABI an i64. JS
  // has one ordinary numeric type (Number), so it cannot represent an arbitrary i64.
  // The spec had to pick a mapping, and -sWASM_BIGINT=1 picks BigInt. Here we convert to BigInt.
  const toWasm64 = (n) => BigInt(n);

  const malloc = (bytes) => {
    const ptr = Number(mod._openzl_wasm_malloc(toWasm64(bytes)));
    if (!ptr) {
      throw new Error('wasm malloc failed');
    }
    return ptr;
  };
  const free = (ptr) => mod._openzl_wasm_free(toWasm64(ptr));

  return {
    toWasm64,
    malloc,
    free,

    writeBytes(data) {
      if (data.length === 0) {
        // malloc(0) may return 0, hand back a non-null scratch
        // so C null-checks (srcSize==0) succeed and free() stays safe.
        return malloc(1);
      }
      const ptr = malloc(data.length);
      mod.HEAPU8.set(data, ptr);
      return ptr;
    },

    readBytes(ptr, len) {
      // Copies out of wasm memory into the JS heap
      return mod.HEAPU8.slice(ptr, ptr + len);
    },

    // Reads a pointer-width out-param as a Number. WASM build is wasm64-only
    readPointer(ptr) {
      if (!mod.HEAPU64) {
        throw new Error(
          'HEAPU64 is missing — expected wasm64 build with -sMEMORY64=1 and HEAPU64 in EXPORTED_RUNTIME_METHODS. wasm32 is not supported (sizeof(size_t)==8 static assert).',
        );
      }
      return Number(mod.HEAPU64[Math.floor(ptr / 8)]);
    },

    // A size_t is pointer-width
    readSizeT(ptr) {
      return this.readPointer(ptr);
    },

    readDouble(ptr) {
      if (!mod.HEAPF64) {
        throw new Error('HEAPF64 is missing — expected HEAPF64 in EXPORTED_RUNTIME_METHODS.');
      }
      return mod.HEAPF64[Math.floor(ptr / 8)];
    },

    errorString(code) {
      // Returns a pointer since c hands ptr
      const ptr = Number(mod._openzl_wasm_errorString(code));
      return ptr ? mod.UTF8ToString(ptr) : `unknown error ${code}`;
    },
  };
}

function throwWasmError(mem, code, fallback) {
  const err = new Error(code !== 0 ? mem.errorString(code) : fallback);
  err.code = code;
  throw err;
}

function loadProfileMetadata(mod) {
  const profiles = [];
  for (let value = 0; ; value++) {
    // wasm64 pointers are returned as bigint, Emscripten helpers use Number offsets.
    const namePtr = Number(mod._openzl_wasm_profileName(value));
    if (!namePtr) {
      return Object.freeze(profiles);
    }
    const descriptionPtr = Number(mod._openzl_wasm_profileDescription(value));
    if (!descriptionPtr) {
      throw new Error(`OpenZL WASM profile ${value} has no description`);
    }
    const reasonPtr = Number(mod._openzl_wasm_profileUnsupportedReason(value));
    profiles.push(Object.freeze({
      value,
      name: mod.UTF8ToString(namePtr),
      description: mod.UTF8ToString(descriptionPtr),
      isWasmSupported: mod._openzl_wasm_profileIsSupported(value) !== 0,
      wasmUnsupportedReason: reasonPtr ? mod.UTF8ToString(reasonPtr) : undefined,
      isTrainable: mod._openzl_wasm_profileIsTrainable(value) !== 0,
    }));
  }
}

function verifyProfiles(metadata) {
  if (metadata.length === 0) {
    throw new Error('OpenZL WASM module exposed an empty profile table');
  }
  // Callers select profiles by name, so names must be unique.
  if (new Set(metadata.map(({name}) => name)).size !== metadata.length) {
    throw new Error('OpenZL WASM module exposed duplicate profile names');
  }
}

async function loadOpenZLModule(options = {}) {
  let openzlModule;
  try {
    ({default: openzlModule} = await import('./openzl.js'));
  } catch (e) {
    const msg = e?.message ?? String(e);
    const isNotFound =
      e?.code === 'ERR_MODULE_NOT_FOUND' ||
      /Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed/i.test(
        msg,
      );
    if (isNotFound) {
      throw new Error(
        'openzl.js not found next to wasm_api.js — build with: emcmake cmake -DOPENZL_BUILD_WASM=ON -B build-wasm && cmake --build build-wasm --target openzl_wasm && cp build-wasm/tools/wasm/openzl.{js,wasm} tools/wasm/js/ (and tools/web/visualization_app/public/ for the viz app). Original: ' +
          msg,
        {cause: e},
      );
    }
    throw new Error(`Failed to load openzl.js: ${msg}`, {cause: e});
  }
  const {
    wasmUrl = new URL('./openzl.wasm', import.meta.url).href,
    locateFile: customLocateFile,
    onTrainingProgress,
    onBenchmarkProgress,
    ...moduleOptions
  } = options;
  if (onTrainingProgress !== undefined && typeof onTrainingProgress !== 'function') {
    throw new Error('onTrainingProgress must be a function');
  }
  if (onBenchmarkProgress !== undefined && typeof onBenchmarkProgress !== 'function') {
    throw new Error('onBenchmarkProgress must be a function');
  }
  const locateFile =
    customLocateFile ?? ((path, prefix = '') => (path.endsWith('.wasm') ? wasmUrl : `${prefix}${path}`));
  let trainProgressCallback;
  // Forward train progress and messages to the user-provided callback.
  if (onTrainingProgress !== undefined) {
    trainProgressCallback = (event) => {
      try {
        onTrainingProgress(event.progress, event.message);
      } catch (error) {
        console.error(`OpenZL onTrainingProgress callback failed: ${error?.message ?? String(error)}`);
      }
    };
  }
  let benchmarkProgressCallback;
  // Forward benchmark progress and messages to the user-provided callback.
  if (onBenchmarkProgress !== undefined) {
    benchmarkProgressCallback = (event) => {
      try {
        onBenchmarkProgress(event.progress, event.message);
      } catch (error) {
        console.error(`OpenZL onBenchmarkProgress callback failed: ${error?.message ?? String(error)}`);
      }
    };
  }
  return openzlModule({
    ...moduleOptions,
    locateFile,
    __openzlTrainProgressCallback: trainProgressCallback,
    __openzlBenchmarkProgressCallback: benchmarkProgressCallback,
  });
}

function readMaxBenchmarkIterations(mod) {
  const maxBenchmarkIterations = mod._openzl_wasm_maxBenchmarkIterations();
  if (!Number.isInteger(maxBenchmarkIterations) || maxBenchmarkIterations < 1) {
    throw new Error(
      `OpenZL WASM module exposed an invalid maximum benchmark iteration count: ${String(maxBenchmarkIterations)}`,
    );
  }
  return maxBenchmarkIterations;
}

// Live ceiling for benchmark iterations, for callers that cap counts without
// holding an OpenZL instance (e.g. the playground's JS-codec benchmarks).
export async function getOpenZLMaxIterations(options = {}) {
  const mod = await loadOpenZLModule(options);
  return readMaxBenchmarkIterations(mod);
}

export async function createOpenZL(options = {}) {
  const mod = await loadOpenZLModule(options);
  const maxBenchmarkIterations = readMaxBenchmarkIterations(mod);
  // Floor for the training out-parameter arrays. Read from the module rather
  // than restated here, so it tracks OPENZL_WASM_TRAIN_PARETO_CANDIDATES.
  const trainParetoCandidates = mod._openzl_wasm_trainParetoCandidates();
  if (!Number.isInteger(trainParetoCandidates) || trainParetoCandidates < 1) {
    throw new Error(`OpenZL WASM module exposed an invalid Pareto candidate count: ${String(trainParetoCandidates)}`);
  }
  const metadata = loadProfileMetadata(mod);
  verifyProfiles(metadata);
  // Numeric IDs index the native table and stay internal; callers use names.
  const profiles = Object.freeze(metadata.map(({value, ...profile}) => Object.freeze(profile)));
  const mem = createMemory(mod);
  const compressorCache = new Map();

  return {
    profiles,
    get maxBenchmarkIterations() {
      return maxBenchmarkIterations;
    },

    get trainParetoCandidates() {
      return trainParetoCandidates;
    },

    // Returns the serialized compressor for a profile name (e.g. 'le-u32'),
    // cached per-profile + level (copy-on-read) so repeated calls reuse it.
    // Exposed since we want users to be able to download compressors.
    getSerializedCompressor(profile, compressionLevel) {
      if (typeof profile !== 'string') {
        throw new Error(`profile must be a profile name string, got ${String(profile)}`);
      }
      const entry = metadata.find(({name}) => name === profile);
      if (!entry) {
        throw new Error(
          `unknown profile ${profile}; available: ${metadata.filter(({isWasmSupported}) => isWasmSupported).map(({name}) => name).join(', ')}`,
        );
      }
      if (!entry.isWasmSupported) {
        throw new Error(`profile ${entry.name} is not supported by this WASM build`);
      }
      const value = entry.value;
      if (
        compressionLevel !== undefined &&
        (!Number.isInteger(compressionLevel) || compressionLevel < 1 || compressionLevel > 9)
      ) {
        // 1 to 9 is the typical range for OpenZL, may be updated if levels change
        throw new Error(`compressionLevel must be an integer from 1 to 9, got ${String(compressionLevel)}`);
      }
      const cacheKey = `${value}:${compressionLevel ?? 'default'}`;
      const cached = compressorCache.get(cacheKey);
      if (cached) {
        return cached.slice();
      }
      let outBuf = 0;
      let outSize = 0;
      let bufPtr = 0;
      try {
        // C writes its result through the buffer address and its length.
        outBuf = mem.malloc(PTR_BYTES);
        outSize = mem.malloc(SIZE_T_BYTES);
        const code = mod._openzl_wasm_getSerializedCompressor(
          value,
          compressionLevel ?? 0,
          mem.toWasm64(outBuf),
          mem.toWasm64(outSize),
        );
        if (code !== 0) {
          throwWasmError(mem, code, `failed to build compressor for profile ${entry.name}`);
        }
        // On success outBuf is non-NULL even for a zero-length result.
        bufPtr = mem.readPointer(outBuf);
        const bytes = mem.readBytes(bufPtr, mem.readSizeT(outSize));
        compressorCache.set(cacheKey, bytes.slice());
        return bytes;
      } finally {
        mem.free(bufPtr);
        mem.free(outBuf);
        mem.free(outSize);
      }
    },

    // Trains compressors, ordered best ratio first. One by default. If
    // options.paretoFrontier is set, will return a frontier that trades compression ratio
    // against speed. Frontier entries are not all guaranteed to compress.
    train(data, compressor, options = {}) {
      return runTraining(mod, mem, trainParetoCandidates, data, compressor, options);
    },

    compress(data, compressor) {
      return runCompression(mod, mem, data, compressor, false);
    },

    compressWithTrace(data, compressor) {
      return runCompression(mod, mem, data, compressor, true);
    },

    decompress(compressed) {
      return runDecompression(mod, mem, compressed, false);
    },

    decompressWithTrace(compressed) {
      return runDecompression(mod, mem, compressed, true);
    },

    benchmark(data, compressor, iterations = DEFAULT_BENCHMARK_ITERATIONS) {
      if (!(data instanceof Uint8Array)) {
        throw new Error('benchmark expects Uint8Array');
      }
      if (!(compressor instanceof Uint8Array)) {
        throw new Error('benchmark expects a serialized compressor Uint8Array');
      }
      // C rejects a count outside its range rather than clamping, so clamp
      // before asking.
      const runs = clampIterations(iterations, maxBenchmarkIterations);
      let compPtr = 0;
      let srcPtr = 0;
      let outFrame = 0;
      let outFrameSize = 0;
      let outCompressMs = 0;
      let framePtr = 0;
      let outDecompressMs = 0;
      try {
        compPtr = mem.writeBytes(compressor);
        srcPtr = mem.writeBytes(data);
        // Addresses for C to write its results through, one per out-param.
        outFrame = mem.malloc(PTR_BYTES);
        outFrameSize = mem.malloc(SIZE_T_BYTES);
        outCompressMs = mem.malloc(DOUBLE_BYTES);
        const code = mod._openzl_wasm_benchmarkCompress(
          mem.toWasm64(compPtr),
          mem.toWasm64(compressor.length),
          mem.toWasm64(srcPtr),
          mem.toWasm64(data.length),
          mem.toWasm64(runs),
          mem.toWasm64(outFrame),
          mem.toWasm64(outFrameSize),
          mem.toWasm64(outCompressMs),
        );
        if (code !== 0) {
          throwWasmError(mem, code, 'benchmark compress failed');
        }
        // Capture ownership before subsequent result reads can throw, so the
        // frame is always released by the finally block.
        framePtr = mem.readPointer(outFrame);
        const compressMs = mem.readDouble(outCompressMs);
        const compressedSize = mem.readSizeT(outFrameSize);

        // Compression left its frame in wasm memory, so decompression takes
        // that pointer directly.
        outDecompressMs = mem.malloc(DOUBLE_BYTES);
        const decompressCode = mod._openzl_wasm_benchmarkDecompress(
          mem.toWasm64(framePtr),
          mem.toWasm64(compressedSize),
          mem.toWasm64(runs),
          mem.toWasm64(outDecompressMs),
        );
        if (decompressCode !== 0) {
          throwWasmError(mem, decompressCode, 'benchmark decompress failed');
        }
        const decompressMs = mem.readDouble(outDecompressMs);

        return {
          iterations: runs,
          srcSize: data.length,
          compressedSize,
          ratio: compressedSize > 0 ? data.length / compressedSize : 0,
          compressMs, // elapsed time in milliseconds
          decompressMs,
          compressMBps: mbPerSec(data.length, runs, compressMs), // throughput in megabytes per second
          decompressMBps: mbPerSec(data.length, runs, decompressMs),
        };
      } finally {
        mem.free(framePtr);
        mem.free(outDecompressMs);
        mem.free(compPtr);
        mem.free(srcPtr);
        mem.free(outFrame);
        mem.free(outFrameSize);
        mem.free(outCompressMs);
      }
    },
  };
}
