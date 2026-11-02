---
name: openzl-gpu-test-frames
description: How to mint real OpenZL frames and chunks for GPU decoder tests using the contrib/gpu/testkit utilities (frame_factory, multichunk_frame, frame_verifier), and how to extract a codec's encoded streams via reflection for differential GPU decode. USE when writing or reviewing GPU-decoder tests that need OpenZL-encoded fixtures.
---

# Minting OpenZL test frames/chunks for GPU decoder tests

`contrib/gpu/testkit` mints **real** OpenZL frames by driving the actual encoder, so every frame/chunk header and checksum is correct by construction — it never hand-serializes header bytes. Use it for all GPU-decoder test fixtures instead of crafting bytes by hand.

- **Buck dep:** add a *relative* dep to `testkit` (e.g. `../../testkit:testkit` from
  `src/decompress/BUCK`) — absolute `//openzl/dev/...` deps are forbidden in OpenZL.
- **Namespace:** `openzl::gpu::testkit`.
- **Headers:** `frame_factory.h`, `frame_verifier.h`, `multichunk_frame.h`.

Always build your test `Input` with the C++ helpers (`openzl/cpp/Input.hpp`):

```cpp
// numeric stream of 2-byte elements (e.g. bf16 bit patterns):
Input in = Input::refNumeric<uint16_t>(vals.data(), vals.size());
// or, untyped element width:
Input in = Input::refNumeric(buf, /*eltWidth=*/2, /*numElts=*/n);
// also: Input::refSerial(...), Input::refStruct<T>(...), Input::refString(...)
```

## 1. Single-graph frames — `frame_factory.h`

```cpp
// Core minter: applies CParams that DEFEAT the STORE backup (so a hand-picked
// codec is not silently replaced by raw STORE), selects startGraph, compresses.
std::string makeFrameWithGraph(Compressor&, GraphID startGraph, const Input&);

// Route a node's outputs all to STORE (keeps the node's codec in the frame).
std::string makeFrameNodeToStore(Compressor&, NodeID node, const Input&);

// Run a node and route each output to the matching successor graph.
std::string makeFrameNodeWithSuccessors(
        Compressor&, NodeID node, std::initializer_list<GraphID> successors, const Input&);

// Single-output node chain: node[0] -> node[1] -> ... -> STORE.
std::string makeFrameChainToStore(
        Compressor&, std::initializer_list<NodeID> chain, const Input&);

// Self-checks.
bool roundTrips(Compressor&, GraphID, const Input&);
bool decompressedEquals(const std::string& frame, const Input&);
```

Always pin the format version so you get chunk-capable (v21+) frames:

```cpp
Compressor c;
c.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
```

Example — a single bitpack-of-numeric frame:

```cpp
const std::vector<int32_t> ints = /* small-range values so bitpack engages */;
const Input in = Input::refNumeric<int32_t>(ints.data(), ints.size());
Compressor c; c.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
const std::string frame = makeFrameWithGraph(c, graphs::Bitpack{}(), in);
```

Example — a float_deconstruct frame:

```cpp
const Input in = Input::refNumeric<uint16_t>(vals.data(), vals.size()); // bf16 bits
Compressor c; c.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
const std::string frame =
        makeFrameNodeToStore(c, nodes::BFloat16Deconstruct::node, in);
```

## 2. Verifying which codecs landed — `frame_verifier.h`

```cpp
// Standard codec wire IDs in the LAST chunk, in decode order (custom codecs skipped).
std::vector<uint32_t> standardCodecsInFrame(const std::string& frame);
// Exactly these codecs (multiset compare, order-independent).
bool frameHasExactlyCodecs(const std::string& frame, const std::vector<uint32_t>& expected);
// Per-chunk standard codec IDs; reports every chunk via decompression introspection.
std::vector<std::vector<uint32_t>> standardCodecsPerChunk(const std::string& frame);
// Text decode graph of the last chunk, for tests that pin a graph.
std::string decodeGraphSignature(const std::string& frame);
// Same, with split codecs collapsed and stored blocks dropped.
std::string decodeGraphShape(const std::string& frame);
```

`standardCodecsInFrame` and `frameHasExactlyCodecs` use the public reflection API, which only exposes the **last chunk** of a frame. For multi-chunk fixtures, use `standardCodecsPerChunk`: it drives the real decompression path with introspection hooks, disables codec fusion on its throwaway `DCtx`, and returns one inner vector per frame chunk. You do not need to re-mint slices just to verify per-chunk codec coverage.

## 3. Multi-chunk frames — `multichunk_frame.h`

```cpp
struct ChunkSpec { size_t numElts; openzl::GraphID graph; };

std::string makeMultiChunkFrame(
        Compressor&, std::initializer_list<ChunkSpec> chunks, const Input&);

// Non-owning view over a contiguous element slice (Serial/Numeric only).
Input sliceInput(const Input&, size_t eltOffset, size_t numElts);
```

Constraints: chunk `numElts` must **sum to** `input.numElts()`, and each chunk must clear **`ZL_MIN_CHUNK_SIZE` = 32768 bytes** of content (so "small" chunks can't be tiny — a bf16 chunk needs ≥ 16384 elts). `makeMultiChunkFrame` pins `FormatVersion = ZL_MAX_FORMAT_VERSION`.

> **Need finer-grained segments than a 32 KB chunk?** Frame chunks can't be smaller than `ZL_MIN_CHUNK_SIZE`. When a test needs many small, arbitrarily-sized runs, mint one frame and `sliceInput` its stream into as many pieces as you need rather than relying on frame chunks.

## 4. Extracting a codec's encoded streams (for differential decode)

To feed a GPU decode kernel the real encoded bytes and compare against OpenZL's own decoder, decode the frame with the public reflection API (`openzl/zl_reflection.h`) and pull the codec's streams — the same reflection path `frame_verifier` uses for last-chunk codec inspection.

```cpp
ZL_ReflectionCtx* rctx = ZL_ReflectionCtx_create();
ZL_ReflectionCtx_setCompressedFrame(rctx, frame.data(), frame.size()); // DECODES the frame
// find your codec among ZL_ReflectionCtx_getCodec_lastChunk(rctx, i) by ZL_CodecInfo_getCodecID
// streams: ZL_CodecInfo_getOutput(codec, i) -> ZL_DataInfo_getDataPtr/getContentSize
ZL_ReflectionCtx_free(rctx);
// CPU ground truth: DCtx().decompress(frame)
```

Gotchas that will bite you (learned the hard way):

- **Reflection is encoder-oriented.** A codec's *outputs* (`ZL_CodecInfo_getOutput`) are the **decoder's inputs** (the stored streams you feed the kernel); its *input* (`ZL_CodecInfo_getInput`) is the decoder's regenerated output — use it for `nbElts`.
- **Classify streams by type, not index.** Reflection reports a codec's outputs in the **reverse** of the encoder's stream order. Use `ZL_DataInfo_getType` (`ZL_Type_serial=1, ZL_Type_struct=2, ZL_Type_numeric=4, ZL_Type_string=8`) to identify each stream. For `float_deconstruct`: `struct` = signFrac, `serial` = exponent.
- **The codec-header byte is unreliable for element type.** `float_deconstruct` (codec id **33**) is shared by float32/bfloat16/float16; `ZL_CodecInfo_getHeaderPtr` does not reliably surface the element-type enum. Detect the variant by **stream widths** instead: signFrac eltWidth 3/1/2 and output eltWidth 4/2/2 for float32/bf16/float16.
- `makeFrameNodeToStore` inserts a `convert_struct_to_serial` (codec id 6) before STORE for any Struct output — expected; just find the codec you care about among the list.

## 5. Building & running

```python
cpp_unittest(
    name = "YourTest",
    srcs = ["tests/YourTest.cpp"],
    network_access = network_access_utils.none(),
    deps = [
        "fbsource//third-party/googletest:gtest",
        # ../ depth is relative to YOUR package; these are for contrib/gpu/src/<area>/
        "../../testkit:testkit",
        "../../../../cpp:openzl_cpp",
    ],
)
```

If the test launches CUDA, run with sanitizers OFF (CUDA can't init under ASAN):

```
buck2 test @fbcode//mode/dev-nosan //openzl/dev/contrib/gpu/...:YourTest
```
