# PivCo-Huffman GPU POC

This directory contains an independent CUDA proof of concept for [PivCo-Huffman](https://github.com/MarcinZukowski/pivco-huffman/)
block encode and decode. It uses the wire-format of [OpenZL's implementation of PivCo-Huffman](https://github.com/facebook/openzl/tree/dev/src/openzl/codecs/pivco_huffman).

It does not change the production OpenZL codec under
`src/openzl/`; callers still supply Huffman weights from the host, while the
per-block PivCo work runs on the GPU.

The public API is in `pivco_gpu.h`. Callers create an immutable
`PivCoGpuContext` from host weights, allocate device buffers and device
workspace, then call `pivcoGpuEncode()` or `pivcoGpuDecode()`. The blocking
wrappers launch internal async kernels, copy device status/results back, and
synchronize the supplied CUDA stream before returning a `ZL_Report`.

## Benchmark Results

Ran on **NVIDIA A100 (SM 8.0, 80 GB)**.

256 MiB expanded input, 64 KiB blocks, 15 iterations. Decode is timed with CUDA
events over the decompressed bytes; bulk H2D/D2H transfers are excluded. Figures
are the median decode throughput over the 15 iterations (GPU-clock variance is
~Â±15% run-to-run).

To reproduce, clone the upstream pivco-huffman repo for the real datasets
(`extras/datasets`) and point `--dataset-dir` at that directory:

```bash
git clone https://github.com/MarcinZukowski/pivco-huffman
buck2 build @fbcode//mode/opt fbcode//openzl/dev/contrib/pivco-huffman/gpu:pivco_gpu_bench
buck2 run @fbcode//mode/opt fbcode//openzl/dev/contrib/pivco-huffman/gpu:pivco_gpu_bench -- \
    --size=268435456 --iterations=15 --dataset-dir=pivco-huffman/extras/datasets
```

Single dataset: append `-- --dataset=json_api.json`.

### Real datasets

| dataset          | ratio | decode GiB/s (median) |
|:-----------------|------:|----------------------:|
| gzip_random.gz   | 1.000 |                 654.5 |
| calgary_pic      | 0.209 |                 194.7 |
| dna_fasta.fa     | 0.283 |                 193.9 |
| cat-image.jpg    | 0.990 |                 152.7 |
| csv_numeric.csv  | 0.418 |                 145.4 |
| log_apache.log   | 0.692 |                 118.3 |
| chinese_text.txt | 0.731 |                 114.2 |
| source_c.c       | 0.623 |                 112.5 |
| cat-wiki.html    | 0.691 |                 105.6 |
| pride.txt        | 0.573 |                 105.5 |
| json_api.json    | 0.654 |                 102.7 |

### Synthetic datasets

| dataset       | ratio | decode GiB/s (median) |
|:--------------|------:|----------------------:|
| two_sym_eq    | 0.125 |                 935.4 |
| two_sym_90/10 | 0.125 |                 931.8 |
| sparse_4      | 0.250 |                 884.6 |
| flat_M3       | 0.375 |                 827.6 |
| sparse_16     | 0.500 |                 790.1 |
| flat_M5       | 0.625 |                 722.3 |
| flat_M6       | 0.750 |                 682.0 |
| flat_M7       | 0.875 |                 652.8 |
| uniform       | 1.000 |                 626.0 |
| proba80       | 0.156 |                 408.9 |
| proba50       | 0.250 |                 227.1 |
| bell_s80      | 0.995 |                 191.6 |
| geometric     | 0.265 |                 185.2 |
| english       | 0.530 |                 155.9 |
| bell_s10      | 0.685 |                 124.8 |
| proba14       | 0.527 |                 119.9 |
| zipfian       | 0.783 |                 112.2 |
| bell_s30      | 0.877 |                 106.8 |
| proba02       | 0.891 |                 101.4 |

## Current Performance Status

This is a proof of concept, not a finished high-throughput GPU codec. The
general Huffman-tree paths prioritize correctness and byte-identical wire output.

Decode has two throughput regimes:

- Flat and near-flat shapes hit dedicated fast paths and run very fast
  (hundreds of GiB/s; incompressible/uniform data decodes at ~285 GiB/s).
- Broad tree shapes run a schedule-driven decoder. The host context builds the
  static Huffman-tree schedule once from the weights; each block parses against
  that schedule, then CUDA kernels process the tree node-by-node across all
  blocks. Real broad-tree files currently decode at ~100-190 GiB/s.

Encode keeps the narrow three-symbol fast path and the correctness-oriented
recursive fallback. Eligible generic blocks also have a bounded cooperative
path. See [the measured encoder evidence](COOPERATIVE_ENCODER.md) for its
supported sizes, workspace cost, timing scope and qualification limits.

Decode throughput is measured over decompressed bytes with CUDA events and
excludes bulk H2D/D2H transfers. See `BENCHMARKS.md` for the latest checked-in
result snapshot and `ROOFLINE.md` for per-kernel roofline analysis.

## Wire Format

The GPU code emits and consumes the same block-local PivCo-Huffman wire format
as the CPU kernel:

- Blocks are encoded independently, using the caller's `blockSize`.
- Bit payloads are little-endian and LSB-first; the bitstream is not native
  endian.
- Internal tree nodes are emitted in CPU preorder.
- Each internal node writes a byte-aligned partition bitmap for the current
  stable rank stream.
- If the internal node does not have two constant children, it then writes
  `numOnes` using `nextPow2(count + 1)` unaligned bits.
- Flat leaves with depth greater than zero write byte-aligned packed flat-leaf
  indices.
- Constant leaves write no payload.

The block offset table uses `uint64_t` byte offsets and has `numBlocks + 1`
entries. Constant blocks may legitimately consume zero bytes, so adjacent
offsets can be equal.

Callers must reserve a little slop past the logical buffer sizes:
`PIVCO_GPU_DECODE_DST_SLOP` trailing bytes past `dstSize` (the decoder writes in
aligned 8-byte groups) and `PIVCO_GPU_DECODE_SRC_SLOP` readable trailing bytes
past `bitstreamSize` (the decoder over-reads a few masked-off bytes past the
last block's bitmap).

## Context And Workspace

`pivcoGpuContextCreate()` builds the normal CPU PivCo-Huffman tree from the
supplied weights and flattens the metadata needed by device code:

- `symbolToRank` and `rankToSymbol`
- `rankToFlatDepth` and `rankToCodeword`
- symbol presence bits for encode-side validation
- a small fast-path descriptor for the flat-root and three-symbol tree shapes
- a preorder decode schedule plus stage lists grouped by tree level and node
  operation

The context stays host-owned and immutable. Each encode/decode call copies this
small tree, and for generic decode the schedule, into the front of
caller-provided device workspace. The remaining workspace is sized by
`pivcoGpuEncodeWorkspaceBytes()` or `pivcoGpuDecodeWorkspaceBytes()`.

The generic decode workspace is per block. It contains:

- one state record per scheduled tree node
- a 32-bit-stride rank directory stored as `uint16_t` prefix counts
- two compact `uint8_t` symbol-stream ping-pong buffers

## Decode Strategy

Decode selects, in order, a fast path, the chunk-in-shared decoder, or the
scheduled cascade. Both generic decoders require `tableLog <= 12` and block
sizes up to 64 KiB (the schedule metadata is `uint16`, so larger blocks are out
of range by design and return `cudaErrorNotSupported`).

### Fast Paths

The **flat-root fast path** is selected when the entire tree is one flat leaf.
Each thread decodes independent packed flat-leaf indices directly to output.
This is the path used by high-entropy flat trees such as uniform 256-symbol
data. The leaf symbol table is cached in shared memory, and the common `depth
== 8` case reads one byte per output directly.

The **three-symbol fast path** is selected when the context describes a shape
with a constant root-left symbol and a one-bit (or two-constant) right leaf.
This is the special-case shape produced by weights such as `{2, 1, 1}`. One CTA
owns one PivCo block:

1. Cooperatively scan the root bitmap bytes into a shared prefix-popcount table.
2. Validate the stored `numOnes` against the bitmap popcount.
3. Validate the exact encoded block byte count from the root bitmap, count
   field, and leaf bitmap size.
4. Decode output bytes in parallel. A zero root bit emits the constant root
   symbol; a one root bit uses the root-byte prefix to find the corresponding
   leaf bit and emits one of the two right-leaf symbols.

Both fast paths keep the input bitstream, offsets, and output fully
device-resident.

### Generic Parse

Both generic decoders share one lightweight parse kernel. Each block replays the
static preorder schedule with a single thread, validates its bitstream slice,
and writes per-block/per-node state:

- node count and rank range
- bitmap byte base for internal nodes
- flat-leaf bit base and depth for flat leaves
- compact stream base for each materialized node at its level

During parse it validates slice bounds, stored `numOnes`, child counts, and
exact final block consumption. For each internal node it also reserves a
per-node prefix-popcount rank directory in workspace: the number of one bits
before each 32-bit bitmap word, used for O(1) `onesBefore` rank queries.

### Chunk-In-Shared Decoder

For small and medium inputs the chunk-in-shared decoder (`pivcoChunkTopDownKernel`)
decodes the whole tree for one contiguous output chunk with all intermediate
node streams resident in shared memory. Only the bitmaps/flat indices are read
from global memory and only the final chunk output is written back. This pays
about three kernel launches total instead of the cascade's roughly thirty
per-level launches, and turns dependent child loads from L2 accesses into shared
accesses.

It relies on the monotone-rank property: a contiguous output chunk descends to a
contiguous sub-range at every tree node, so all per-chunk node streams fit in a
level-parity ping-pong pair. The kernel runs three phases per chunk:

1. A top-down descent computes each node's chunk sub-range from the two boundary
   ranks (using the per-node rank directories, built by a preceding directory
   kernel).
2. A bottom-up merge, deepest level first, materializes each node stream in
   shared memory using the same byte-select merge as the cascade.
3. A coalesced flush writes the root buffer to global output.

Levels with many small nodes decode one node per lane; shallow, wide-node levels
cooperate a whole warp per node.

### Scheduled Cascade Decoder

For large inputs the scheduled cascade walks tree levels from deepest to root.
For each level it launches one kernel per operation kind present at that level.
The grid maps `x = block` and `y = scheduled node within the stage`, so each CTA
processes one node stream for one block. The operation kinds are:

- flat leaves with depths 1 through 8
- vector/vector merges
- constant/vector and vector/constant merges
- both-constant merges, decoded directly from the one-bit partition bitmap

Each merge kernel builds its node's rank directory in shared memory at launch,
so no standalone directory kernel and no global directory round-trip is needed.
It then merges the child streams: for output position `j`, the bitmap bit
chooses left or right, `onesBefore(j)` gives the right-child cursor, and `j -
onesBefore(j)` gives the left-child cursor. Large nodes use a vectorized
byte-select merge (`byteMerge8`, two `__byte_perm` per 8 outputs, reading child
windows via aligned read-only loads); small nodes use a simple per-output loop.
Constant leaves are never materialized; parent merges inject their symbols
directly. The root writes directly to final output; intermediate nodes write to
the compact ping-pong streams.

The AVX512 CPU kernel performs the same logical merge with vector masked
expand-loads, so this path mirrors the CPU generic decode closely. It is the
production path for large data and the primary path measured in `BENCHMARKS.md`.

### Path Selection

The dispatcher uses the chunk-in-shared decoder up to a size crossover and the
scheduled cascade above it. The crossover depends on tree depth: deep trees
(`tableLog >= 10`) keep the chunk decoder up to 12 MiB, shallow trees up to
4 MiB. Beyond that, the cascade's higher steady-state throughput wins.

The `PIVCO_CHUNK_TD` environment variable overrides the automatic choice:
setting it forces the chunk-in-shared decoder regardless of size; setting it to
`0` forces the cascade. `PIVCO_KERNEL_TIMING` enables per-stage CUDA-event
timing prints on the cascade path for diagnostics. Both are off by default.

## Encode Strategy

### Cooperative Encoder

The cooperative CUDA path supports schedulable trees with tableLog from 1
through 12 and 32 or 64 KiB blocks. Calls use stream-ordered batches of at most
256 blocks, including a partial final block. The existing three-symbol encoder
takes precedence. Unsupported trees, HIP and insufficient cooperative scratch
retain the recursive fallback when its workspace requirement is satisfied.

Each CTA partitions and packs one block using warp histograms and CUB scans.
A device scan carries the previous batch endpoint into global offsets; a copy
compacts each batch before the same scratch pool is reused. CPU wire bytes and
offsets are preserved. No allocation or host stream fence is added internally.

Cooperative scratch is approximately four times one batch's rounded input,
plus context and per-block sizes. The context-free public query remains
conservative: it also covers recursive whole-input scratch and can overallocate
for a specialized tree or a large cooperative call. It is not a context-aware
memory planner.

The cooperative path rejects overlapping or misaligned buffers. The immutable
host context and all storage must survive stream completion. Concurrent calls
need disjoint mutable storage; reuse on another stream requires an event wait
or synchronization. On failure, output and offsets are invalid and earlier
batches may already have written bytes. HIP excludes CUB. CUDA architectures
below SM70 retain scalar shared atomics, but require a supporting toolkit.

This is a standalone contrib codec. It does not redirect OpenZL's production
CPU codec or supply the graph/header/weight-stream integration for that route.

### Existing Paths

The generic encoder is the correctness/oracle implementation:

1. `encodeLayoutKernel` maps source bytes to ranks, rejects symbols absent from
   the supplied weights, and recursively measures the exact block payload size.
2. `scanOffsetsKernel` converts per-block sizes into the `uint64_t` byte offset
   table and total encoded size.
3. `encodeEmitKernel` reruns the recursive stable partitions and emits the exact
   CPU preorder bitstream for each block.

This generic path remains the differential reference for unsupported shapes
and workspace configurations.

The optimized three-symbol path is selected for the same root-constant plus
one-bit-leaf tree shape and block sizes from 1 KiB through 64 KiB. It is designed
around a single source pass per block:

1. `fastEncodePackRootConstFlat1Kernel` assigns one CTA per block and packs that
   block into fixed-stride workspace at `block * blockSize`. Each thread owns a
   contiguous range of root bitmap bytes; full 8-symbol groups use CUDA byte-lane
   equality (`__vcmpeq4`) to build masks for the root symbol and the two leaf
   symbols. A shared prefix over per-thread non-root counts gives each thread its
   stable leaf-bit range, threads merge their leaf words into the shared leaf
   bitmap, write the LSB-first `numOnes` field, and publish the exact block size.
2. A parallel device scan converts block sizes into compact output offsets.
3. `fastEncodeCopyRootConstFlat1Kernel` copies each fixed-stride temporary block
   into its final compact position.

The fast path still validates source symbols against the supplied weights and
produces byte-identical CPU wire output and offsets.

Empty input produces an empty bitstream and a single zero offset. Constant-input
contexts with `tableLog == 0` produce zero encoded bytes and repeated zero
offsets. Constant encoding rejects absent symbols; empty async calls clear
the total-size result. The CPU-wire regression covers the full-block terminal
rank overflow on both chunk and cascade decode paths.

## CPU Block Indexer

`pivco_block_index.{h,cpp}` provides the CUDA-free helper
`pivcoFindBlockOffsets()`. It structurally walks a CPU-produced bitstream and
returns `uint64_t[numBlocks + 1]` offsets for independent GPU slice decode. The
walker validates bounds, subtree structure, bitmap popcounts against stored
`numOnes`, and exact final consumption.

## Tests And Benchmark

Correctness coverage lives in `PivCoBlockIndexTest.cpp` and `PivCoGpuTest.cpp`.
The GPU tests compare:

- CPU encode to GPU decode
- GPU encode bytes and offsets to CPU encode plus the host block indexer
- GPU encode to GPU decode
- constant repeated offsets
- missing-symbol and bad-offset failures
- the optimized fast path, including a partial final block
- scheduled decode across mixed node kinds (flat, vector/vector,
  constant/vector, constant/constant)
- cooperative tails, maximum grid, legacy workspace and fallback behavior
- concurrent streams, poisoned workspace reuse, and empty async output
- independent CPU-wire decoding at the full-block rank boundary

The tests exercise fast paths, the chunk-in-shared decoder, and larger inputs
that select the scheduled cascade. The benchmark provides separate timings.
The [standalone CMake recipe](COOPERATIVE_ENCODER.md#reproduce) builds these
tests with CMake and registers both decoder modes.

Run both decoder modes through the standalone CTest registration:

```sh
ctest --test-dir build/pivco --output-on-failure
```

`PivCoGpuBench.cpp` reports CUDA-event kernel timing separately from host H2D,
D2H, and wall-clock preparation. Bulk input/output transfers are not included in
the `decode_*` or `encode_*` throughput numbers.

The default benchmark tests 4 MiB per case with 64 KiB blocks. It runs:

- every synthetic distribution (generated in-process; no input files needed)
- every regular file in the directory passed via `--dataset-dir`

The real-world benchmark inputs are not checked in here. Fetch them from the
upstream pivco-huffman repo's `extras/datasets` directory
(https://github.com/MarcinZukowski/pivco-huffman/tree/main/extras/datasets),
which also documents each file's provenance, then point `--dataset-dir` at your
checkout (a valid directory is required even for synthetic-only runs):

```bash
git clone https://github.com/MarcinZukowski/pivco-huffman
buck run --local-only @fbcode//mode/opt fbcode//openzl/dev/contrib/pivco-huffman/gpu:pivco_gpu_bench -- \
    --dataset-dir=pivco-huffman/extras/datasets
```

Real files and generated synthetic bases are repeated or truncated to the
requested benchmark size. Dataset source bytes must be larger than the block
size, so every measured case spans at least two PivCo blocks before repetition.

Useful benchmark options (all also take `--dataset-dir=<checkout>/extras/datasets`):

```bash
buck run --local-only @fbcode//mode/opt fbcode//openzl/dev/contrib/pivco-huffman/gpu:pivco_gpu_bench -- --size=268435456 --iterations=5 --dataset-dir=pivco-huffman/extras/datasets
buck run --local-only @fbcode//mode/opt fbcode//openzl/dev/contrib/pivco-huffman/gpu:pivco_gpu_bench -- --dataset=proba80 --dataset-dir=pivco-huffman/extras/datasets
```

See `BENCHMARKS.md` for the latest checked-in result snapshot.

The benchmark leaves the CUDA runtime stack limit unchanged by default. Use
`--stack-limit=65536` when reproducing the recursive baseline.
The context-aware workspace query sizes the selected encode path; the original
query remains a conservative context-independent budget.

CUDA callers including `pivco_gpu.cuh` can reuse a `PivCoGpuEncodeResult` device buffer with
`pivcoGpuEncodeWithResultAsync`. Copy it to pinned host storage and record a
stream event; inspect `pivcoGpuEncodeResultReport` after the event completes.
The caller owns the buffers and stream ordering.
