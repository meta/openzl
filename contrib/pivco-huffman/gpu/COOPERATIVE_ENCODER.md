# Cooperative PivCo GPU encoder

## Submitted measurements (2026-10-17)

This section records the submitted cb62bb3 implementation. Measurements for the
later batched implementation are reported separately below.

**Hardware conditions for the submitted measurements:** Non-stock hardware: CPU power-limited for thermals; RTX 3080 Ti power budget increased with a more conservative clock curve; RTX 3060 connected through a PCIe x1 link. Exact tuning values were not recorded. These workstation measurements are not stock-hardware expectations.

Device-resident synthetic inputs: 4 MiB, 64 KiB blocks. Values are the mean
of two seven-iteration medians, in baseline/candidate/candidate/baseline order.
CUDA events include status resets, context upload and kernels; bulk transfers,
allocation and verification are excluded. Full bytes and offsets match CPU
encoding. The unchanged three-symbol path serves as a control.
Its timing differences are noise.

| GPU | Fixture | Baseline ms | Patched ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| RTX 3060 | english | 42.2810 | 0.9134 | 46.29x |
| RTX 3060 | geometric | 32.9124 | 0.7609 | 43.26x |
| RTX 3060 | three_sym_50_25_25 | 0.1387 | 0.1413 | 0.98x |
| RTX 3060 | two_sym_eq | 19.7939 | 0.4541 | 43.58x |
| RTX 3060 | uniform | 30.4292 | 0.5048 | 60.28x |
| RTX 3060 | zipfian | 58.3987 | 1.2805 | 45.61x |
| RTX 3080 Ti | english | 39.8586 | 0.7109 | 56.06x |
| RTX 3080 Ti | geometric | 29.9857 | 0.5934 | 50.54x |
| RTX 3080 Ti | three_sym_50_25_25 | 0.0778 | 0.0809 | 0.96x |
| RTX 3080 Ti | two_sym_eq | 18.9496 | 0.3961 | 47.84x |
| RTX 3080 Ti | uniform | 29.2833 | 0.3861 | 75.85x |
| RTX 3080 Ti | zipfian | 53.7522 | 0.9713 | 55.34x |

Encoder scratch for 4 MiB grows from 8,390,160 to 16,790,272 bytes; this is a
workspace query, not total VRAM. The query also reserves extra scratch for the unchanged specialized path.
For the submitted small-input implementation, legacy-sized scratch selects the
fallback; this is not a dispatch rule for the later batched implementation.
See [cooperative-benchmark.json](cooperative-benchmark.json) for raw medians,
source identities and coverage.

## Historical validation (2026-10-17)

All 19 GPU tests pass on both cards. The isolated decoder fix passes 12 tests;
the old decoder fails its CPU-wire regression. RTX 3080 Ti memcheck passes;
racecheck and synccheck pass with the largest-grid test excluded.

Tested: WSL2 Linux, CUDA 13.2, GCC 13.3, SM 8.6. Untested: HIP, older GPUs,
Windows-native builds and the full OpenZL suite. Both builds retain the existing
SM 8.6 scheduledDirectoryKernel launch-bounds warning. Constant decode remains
unsupported; the new constant-encode validation cost was not timed.

## Follow-up validation (2026-10-24)

RTX 3080 Ti, WSL2 Linux, CUDA 13.2/GCC13.3. Non-stock hardware: CPU power-limited
for thermals; GPU power budget increased with a conservative clock curve.
Exact tuning values were not recorded. These are workstation observations.

All cases below are synthetic. Values are medians of two process medians,
seven timed iterations per 4 MiB process and three for larger inputs. The
stack limit is explicitly 64 KiB for comparison; the benchmark now leaves it
unchanged by default. Buffers/context are prepared outside timing. Resident
timing includes API completion; transfer timing also includes input H2D and
compressed-output/offset D2H. Bulk host buffers are pageable, async completion
is pinned. Verification runs outside timing.

| Fixture and shape | Blocking resident ms | Async resident ms | Blocking transfers ms | Async transfers ms |
| --- | ---: | ---: | ---: | ---: |
| english, 4194304 B / 65536 B blocks | 0.7956 | 0.7535 | 1.4322 | 1.5245 |
| uniform, 4194304 B / 65536 B blocks | 0.4993 | 0.4521 | 1.4550 | 1.5158 |
| three_sym_50_25_25, 4194304 B / 65536 B blocks | 0.1275 | 0.0997 | 0.5847 | 0.6917 |
| english, 16777217 B / 65536 B blocks | 1.1513 | 1.0960 | 4.2458 | 4.1304 |
| english, 67108864 B / 65536 B blocks | 4.1122 | 4.0638 | 14.8733 | 14.7539 |
| english, 4194304 B / 32768 B blocks | 0.4990 | 0.4564 | 1.2359 | 1.3550 |

Reusable completion reduces resident latency in these cases. Transfer-inclusive
results are mixed; this does not establish a universal async speedup or a
small-input kernel improvement. The full benchmark separately reports optimized
CPU encoding, allocation and setup costs.

For 4 MiB three-symbol input, context-aware encoder scratch is 4,195,872 bytes
instead of the conservative 16,790,272. At 64 MiB generic input it is 67,123,456
instead of 134,219,280. Empty/constant contexts need no encoder scratch.
Each reusable completion payload is 24 bytes on device and 24 pinned host bytes;
allocator/event overhead is additional. Decode scratch and total VRAM differ.

Both CUDA 12.8 and 13.2 compile SM75/80/86 and pass 38 GPU tests in each decoder
mode plus eight CPU index tests on the 3080 Ti. Full-suite memcheck passes in
both modes. Six focused maximum-grid, multi-batch and completion tests pass
racecheck in mode 1 and synccheck in mode 0 with zero diagnostics. The final
MSVC-only portability edit leaves normalized Linux device instructions identical
for all three compiled architectures under both toolkits. Strict C99 headers
and clang-format 21 pass.

Native Windows CUDA 13.3/MSVC 19.51 compiles the CUDA, test and benchmark objects.
The linked core build remains blocked in unchanged C11 registry initialization;
Windows runtime is unqualified. HIP, non-SM86 runtime, hosted CI execution and
the full CPU dev suite remain unqualified; the latter stopped during unchanged
ASan test discovery. Production OpenZL graph/frame integration remains separate.

## Reproduce

```sh
mkdir -p build/empty-datasets
cmake -S contrib/pivco-huffman/gpu -B build/pivco -G Ninja -DCMAKE_BUILD_TYPE=Release -DOPENZL_BUILD_MODE=none -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/pivco --parallel 2
ctest --test-dir build/pivco --output-on-failure
build/pivco/pivco_gpu_bench --size=4194304 --block-size=65536 --iterations=7 --dataset-dir=build/empty-datasets --dataset=english
```

Select a GPU with CUDA_VISIBLE_DEVICES. For revision comparisons, build each
revision separately and run baseline/candidate/candidate/baseline with identical
input and timing scopes. Use `--stack-limit=65536` on the current benchmark when
matching a baseline that set that limit. The current benchmark also reports
blocking and reusable-async timings from the same binary.

## Author note

This was entirely vibe-coded by an LLM. I am not in a position to tell you if this is a useful contribution, unfortunately. Exercise caution accordingly.
