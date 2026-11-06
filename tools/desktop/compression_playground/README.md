# OpenZL Compression Playground: desktop app

The playground has a browser build and an Electron desktop build. The desktop app benchmarks by running the native OpenZL CLI: it reads each `zli benchmark` result from the file `--output-csv` writes, and benchmarks the candidate files `zli train --pareto-frontier` writes. It never loads the OpenZL WASM module. zstd rows run `zstd-benchmark`, a small binary built beside `zli` against the zstd OpenZL vendors, which times zstd the way `zli benchmark` times OpenZL (one reused context, one-shot calls, the same CSV). gzip rows use Electron's built-in zlib, since OpenZL does not vendor zlib. The page itself is the web playground in `tools/web/compression_playground`.

## Getting started

You need Node 24, Yarn 1, CMake and a C/C++ toolchain; on Windows, Visual Studio with its "C++ Clang tools for Windows" component. From `tools/desktop/compression_playground`:

```bash
yarn install
yarn dev
```

`yarn dev` runs `yarn setup` (`setup.cjs`) first, which does what the platform needs: it installs the web playground's dependencies (the `tools/web` workspace, where its Vite comes from); on Linux, installs ALSA's library with apt if it is missing, since Electron will not start without it, and where AppArmor stops unprivileged user namespaces, as Ubuntu 24.04 does, makes Electron's `chrome-sandbox` setuid root with sudo, which Chromium's sandbox then needs; on Windows, configures CMake with the clang-cl toolset, as `windows-ci.yml` builds OpenZL, removing a build directory configured otherwise; and builds `zli` and `zstd-benchmark`. Then it opens Electron, which serves the page with the web playground's Vite on port 5173, or the next free port.

On Windows, some dependencies' build paths exceed the path length limit, so check out to a short path such as `C:\oz`.

## Testing and packaging

```bash
yarn test            # everything below, as CI runs it
yarn test:unit       # the desktop app's unit tests
yarn test:web        # the web playground's typecheck, lint and unit tests
yarn test:smoke      # builds what they need, then runs every smoke test
yarn test:smoke cancellation   # or just some: native-training, window-security, ui, cancellation, packaged-app
yarn build           # the platform installer, under release/
yarn package         # the unpacked app, under release/
```

The smoke tests train and benchmark with the built `zli` and `zstd-benchmark`, check the window's security settings, run the sample through the page as a user would, check that cancelling, reloading, or quitting leaves no `zli` running, and launch the packaged app. On Linux with no display, the ones that open a window run under `xvfb-run` (`sudo apt install xvfb`). On distributions that stop unprivileged processes making user namespaces, as Ubuntu 24.04 does, Chromium's sandbox needs its `chrome-sandbox` helper to be setuid root: `yarn setup` makes Electron's so, with sudo, and the smoke tests make the packaged app's so. On Windows, the unit tests that run stand-in `#!node` binaries are skipped, since Windows cannot execute them; the smoke tests drive the real binaries through the same code.

`.github/workflows/desktop-playground-ci.yml` runs `yarn install`, `yarn setup` and `yarn test` on Linux, macOS and Windows whenever the playground, `cli/` or `tools/training/` changes.

## How it works

ACE trains against the complete selected file (`zli train --use-all-samples --pareto-frontier`), and each candidate is then benchmarked with the requested level and iteration count. OpenZL runs in ordinary `zli` processes, so its allocations do not pass through Electron's allocator. Every input is a file on disk, which `zli` reads where it is; only `zli`'s CSV and candidate files go in a scratch directory. On Windows, `zli` and `zstd-benchmark` read their arguments in the ANSI code page, so an input path or temp folder with characters outside it is refused with a message saying which, instead of passing them a mangled path. Which characters a code page has comes from `iconv-lite`'s Windows tables.

The security boundary is deliberately small: the renderer has no Node access. An isolated preload exposes file-path resolution, the profile list, and one validated benchmark IPC call, `startBenchmark(config, onMessage)`, which returns `{done, cancel}`. Browser permission requests are denied, and only `https://openzl.org` links leave the app, opening in the system browser. The packaged renderer is served from a custom `app://bundle` origin rather than `file://`, and IPC is accepted only from that origin. The packaged binary's Electron fuses disable running as Node (`ELECTRON_RUN_AS_NODE`), `NODE_OPTIONS`, `--inspect`, and extra `file://` privileges, and only load the app from its asar archive. An Electron utility process coordinates benchmark work, while OpenZL itself runs in `zli` child processes so a native failure cannot terminate the desktop UI. Each run has its own message port: cancelling, reloading, or closing the window stops the run, and the main process kills any `zli` it started (SIGTERM, then SIGKILL) and removes the run's scratch directory, including when the utility process dies or the app quits. One benchmark runs at a time, since concurrent runs would distort each other's timings.

The backend speaks the browser's protocol, with the browser's types (`tools/web/compression_playground/src/benchmarkTypes.ts`). It takes a `DesktopRunConfig`, which is `RunConfig` with the input named by its path or as the bundled sample, splits it into the same jobs `buildJobs` does, and streams the worker's `WorkerMessage`s as they happen: `loading`, `started` with the job count, `step` while a job trains, a `result` per measurement as soon as it is made or a `failure` for a job it could not measure, then `finished`, `failed`, or the desktop-only `cancelled`. `step` is how far through training the job is, read off the progress bar `zli train` prints at `--verbose 3`, as the browser forwards its trainer's progress; the page places it within the job's share of the run, and on a failure only `zli`'s fatal error is shown. A trained job yields one `result` per candidate, counted from 1, best ratio first; a candidate that cannot be benchmarked is a `failure` naming it, and its siblings are still measured. As in the browser, a job that fails is reported and the run goes on to the next; only cancelling ends it early. The Electron code is CommonJS checked by TypeScript (`checkJs`), with those types imported in JSDoc, so `yarn typecheck` fails if the two builds drift apart. Every boundary is validated at runtime with zod: requests from the renderer, messages from the utility process, whose schema TypeScript holds to those same types, and the benchmark CSV `zli` writes. Unknown keys are stripped, so only validated fields travel on. The request is checked for its shape only: which levels, candidate counts and so on are valid is for `zli` and zlib to say, and a job they refuse is reported like any other failure. The packaged app ships zod, and `iconv-lite` with its `safer-buffer`, in `resources/node_modules` (copied by `electron/after-pack.cjs`), because the app itself excludes `node_modules`.

The profiles the page offers are the ones the bundled `zli` lists (`zli list-profiles`), with its descriptions, so the app never offers a profile its `zli` lacks. A row's profile argument goes to `zli` as `--profile-arg` (the `csv` profile's separator, or an SDDL description file), and is left off when empty. The sample the page offers is `examples/getting_started/sample_inputs/sra0.bin`, named in `bundledSample.json`, which the page and the main process share: the page sends only its name and size, and the main process supplies its path, in the source tree or in the packaged app's resources.

This directory is its own Yarn project, apart from the `tools/web` workspace, so the documentation build never installs Electron. It keeps its own TypeScript, ESLint and Prettier settings.
