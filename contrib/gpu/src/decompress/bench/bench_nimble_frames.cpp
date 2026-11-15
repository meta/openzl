// Copyright (c) Meta Platforms, Inc. and affiliates.

// Decodes the OpenZL frames of testkit's workloads on a pool of CPU threads,
// as concurrent Nimble readers decompress their leaf streams. Each thread
// decodes its own frames with decompressSerial(), with a new openzl::DCtx for
// every frame (--dctx frame), one DCtx per thread (--dctx thread), or a DCtx
// that it takes from a pool shared by every thread and returns after the frame
// (--dctx pool, one row for each of --pool-slots). Each pool slot caches one
// DCtx for a block of the host's CPUs, so threads that run on a few CPUs use
// only the slots of those CPUs.
//
// A row builds a few distinct frames of one workload with compressStep(), so
// no selector chooses the graph, and copies them until the row holds
// --total-mib of decoded data. Each frame holds the workload's own frame size,
// or --frame-size if it is set. Thread t always runs on the same CPU and writes
// its own buffers before the timed runs, so on a host with several sockets
// their pages sit in that CPU's local memory, as a reader's buffers would.
// Every timed run checks its output against the input.

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "contrib/gpu/testkit/f32_ladder.h"
#include "contrib/gpu/testkit/nimble_workloads.h"
#include "contrib/gpu/testkit/u64_workloads.h"
#include "contrib/gpu/testkit/zstd_workloads.h"
#include "openzl/openzl.hpp"
#include "tools/arg/arg_parser.h"

namespace openzl::gpu::bench {
namespace {

constexpr size_t kKiB = size_t{ 1 } << 10;
constexpr size_t kMiB = size_t{ 1 } << 20;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
// The other frames of a row copy these.
constexpr size_t kDistinctFrames = 16;

enum class DCtxLifetime { frame, thread, pool };

struct Options {
    // Workload numbers in workloads(), starting at 1.
    std::vector<size_t> steps;
    // Decoded MiB per row.
    size_t totalMiB = 256;
    // Decoded bytes per frame for every workload, instead of its own frame
    // size.
    std::optional<size_t> frameBytes;
    std::vector<size_t> threads    = { 1, 2, 4, 8, 16, 32, 64, 128 };
    size_t iterations              = 5;
    std::vector<DCtxLifetime> dctx = { DCtxLifetime::frame,
                                       DCtxLifetime::thread };
    std::vector<size_t> poolSlots  = { 1, 4, 16, 64 };
};

// The ladder steps, then the extra workloads, the zstd workloads, and the
// 64-bit workloads.
const std::vector<testkit::LadderStep>& workloads()
{
    static const std::vector<testkit::LadderStep> all = [] {
        std::vector<testkit::LadderStep> list = testkit::f32Ladder();
        for (const auto* more : { &testkit::f32ExtraWorkloads(),
                                  &testkit::zstdWorkloads(),
                                  &testkit::u64Workloads() }) {
            list.insert(list.end(), more->begin(), more->end());
        }
        return list;
    }();
    return all;
}

// Parses a positive decimal integer. std::stoull would accept "4x" and "-1".
size_t parseSize(const std::string& text)
{
    size_t value         = 0;
    const char* end      = text.data() + text.size();
    const auto [at, err] = std::from_chars(text.data(), end, value);
    if (err != std::errc{} || at != end || value == 0) {
        throw std::invalid_argument("not a positive number: \"" + text + "\"");
    }
    return value;
}

// Parses a positive byte count with an optional K (KiB) or M (MiB) suffix.
size_t parseByteSize(const std::string& text)
{
    size_t value         = 0;
    const char* end      = text.data() + text.size();
    const auto [at, err] = std::from_chars(text.data(), end, value);
    const std::string_view suffix{ at, static_cast<size_t>(end - at) };
    size_t unit = 0;
    if (suffix.empty()) {
        unit = 1;
    } else if (suffix == "K") {
        unit = kKiB;
    } else if (suffix == "M") {
        unit = kMiB;
    }
    if (err != std::errc{} || unit == 0 || value == 0
        || value > std::numeric_limits<size_t>::max() / unit) {
        throw std::invalid_argument(
                "not a positive size in bytes, with an optional K or M "
                "suffix: \""
                + text + "\"");
    }
    return value * unit;
}

std::vector<std::string> splitList(const std::string& list)
{
    std::vector<std::string> items;
    size_t begin = 0;
    while (begin <= list.size()) {
        const size_t end = std::min(list.find(',', begin), list.size());
        items.push_back(list.substr(begin, end - begin));
        begin = end + 1;
    }
    return items;
}

std::vector<size_t> parseSizes(const std::string& list)
{
    std::vector<size_t> values;
    for (const std::string& item : splitList(list)) {
        values.push_back(parseSize(item));
    }
    return values;
}

std::vector<DCtxLifetime> parseLifetimes(const std::string& list)
{
    std::vector<DCtxLifetime> lifetimes;
    for (const std::string& item : splitList(list)) {
        if (item == "frame") {
            lifetimes.push_back(DCtxLifetime::frame);
        } else if (item == "thread") {
            lifetimes.push_back(DCtxLifetime::thread);
        } else if (item == "pool") {
            lifetimes.push_back(DCtxLifetime::pool);
        } else {
            throw std::invalid_argument(
                    "unknown DCtx lifetime: \"" + item + "\"");
        }
    }
    return lifetimes;
}

Options parseOptions(int argc, char** argv)
{
    arg::ArgParser parser;
    parser.addGlobalFlag(
            "steps",
            's',
            true,
            "Comma-separated workload numbers (ladder steps, then extra "
            "workloads, zstd workloads, and 64-bit workloads), or \"all\"");
    parser.addGlobalFlag("total-mib", 'm', true, "Decoded MiB per row");
    parser.addGlobalFlag(
            "frame-size",
            'f',
            true,
            "Decoded bytes per frame for every workload, instead of its own "
            "frame size, with an optional K or M suffix (4K, 128M)");
    parser.addGlobalFlag(
            "threads", 't', true, "Comma-separated CPU thread counts");
    parser.addGlobalFlag("iterations", 'i', true, "Timed runs per row");
    parser.addGlobalFlag(
            "dctx",
            'd',
            true,
            "Comma-separated DCtx lifetimes: frame, thread, pool");
    parser.addGlobalFlag(
            "pool-slots",
            'p',
            true,
            "Comma-separated slot counts for --dctx pool");
    const arg::ParsedArgs args = parser.parse(argc, argv);

    Options options;
    const auto steps = args.globalFlag("steps");
    if (steps && *steps != "all") {
        options.steps = parseSizes(*steps);
    } else {
        for (size_t step = 1; step <= workloads().size(); ++step) {
            options.steps.push_back(step);
        }
    }
    if (const auto value = args.globalFlag("total-mib")) {
        options.totalMiB = parseSize(*value);
    }
    if (const auto value = args.globalFlag("frame-size")) {
        options.frameBytes = parseByteSize(*value);
    }
    if (const auto value = args.globalFlag("threads")) {
        options.threads = parseSizes(*value);
    }
    if (const auto value = args.globalFlag("iterations")) {
        options.iterations = parseSize(*value);
    }
    if (const auto value = args.globalFlag("dctx")) {
        options.dctx = parseLifetimes(*value);
    }
    if (const auto value = args.globalFlag("pool-slots")) {
        options.poolSlots = parseSizes(*value);
    }
    const size_t last =
            *std::max_element(options.steps.begin(), options.steps.end());
    if (last > workloads().size()) {
        throw std::invalid_argument(
                "steps must be 1 to " + std::to_string(workloads().size()));
    }
    return options;
}

// The CPUs that this process may run on, in increasing order.
const std::vector<int>& allowedCpus()
{
    static const std::vector<int> cpus = [] {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) != 0) {
            throw std::system_error(
                    errno, std::generic_category(), "sched_getaffinity");
        }
        std::vector<int> list;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) {
                list.push_back(cpu);
            }
        }
        return list;
    }();
    return cpus;
}

// The CPU that thread t of every runPerThread() call runs on.
int cpuOfThread(size_t t)
{
    const std::vector<int>& cpus = allowedCpus();
    return cpus[t % cpus.size()];
}

void pinCallingThread(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    const int error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (error != 0) {
        throw std::system_error(
                error, std::generic_category(), "pthread_setaffinity_np");
    }
}

// Runs work(t) on threads 0 to threads - 1 and returns the wall time. Thread t
// runs on cpuOfThread(t) in every call, so it reads the pages that thread t of
// an earlier call wrote from that CPU's local memory. The clock starts after
// all threads exist, so the timing excludes thread creation. If work throws,
// the exception of the lowest thread that threw is rethrown after all threads
// finish.
template <typename Work>
double runPerThread(size_t threads, const Work& work)
{
    std::atomic<bool> go{ false };
    // An exception that leaves a std::thread calls std::terminate.
    std::vector<std::exception_ptr> errors(threads);
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            try {
                pinCallingThread(cpuOfThread(t));
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                work(t);
            } catch (...) {
                errors[t] = std::current_exception();
            }
        });
    }
    const auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (std::thread& thread : pool) {
        thread.join();
    }
    const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
    for (const std::exception_ptr& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    return seconds;
}

// The distinct frames of one row. Frame i of the row is distinct frame
// i % raw.size(). Every raw frame holds the step's frameBytes.
struct FrameSet {
    std::vector<std::string> raw;
    std::vector<std::string> compressed;
    size_t count = 0;
};

FrameSet makeFrameSet(
        const testkit::LadderStep& step,
        size_t totalMiB,
        size_t maxThreads)
{
    FrameSet set;
    set.count = std::max<size_t>(1, totalMiB * kMiB / step.frameBytes);
    const size_t distinct = std::min(set.count, kDistinctFrames);
    set.raw.resize(distinct);
    set.compressed.resize(distinct);
    const size_t builders = std::min(distinct, maxThreads);
    runPerThread(builders, [&](size_t t) {
        for (size_t i = t; i < distinct; i += builders) {
            set.raw[i] = testkit::generateFrame(step.spec, step.frameBytes, i);
            set.compressed[i] = testkit::compressStep(step, set.raw[i]);
        }
    });
    return set;
}

// The frames that one thread decodes: frames first to first + frames - 1 of
// the row. Frame k starts at compressedOffsets[k] in `compressed` and at
// k * frameBytes in `outputs`.
struct ThreadFrames {
    size_t first = 0;
    std::vector<char> compressed;
    std::vector<size_t> compressedOffsets;
    std::vector<char> outputs;
    // Used by DCtxLifetime::thread.
    std::optional<openzl::DCtx> dctx;

    size_t frames() const
    {
        return compressedOffsets.size() - 1;
    }

    std::string_view frame(size_t k) const
    {
        return { compressed.data() + compressedOffsets[k],
                 compressedOffsets[k + 1] - compressedOffsets[k] };
    }

    poly::span<char> output(size_t k, size_t frameBytes)
    {
        return { outputs.data() + k * frameBytes, frameBytes };
    }
};

// Splits the row's frames over `threads` threads. Each thread allocates and
// writes its own buffers, so it touches their pages first.
std::vector<ThreadFrames>
placeFrames(const FrameSet& set, size_t frameBytes, size_t threads)
{
    std::vector<ThreadFrames> placed(threads);
    runPerThread(threads, [&](size_t t) {
        ThreadFrames& mine = placed[t];
        mine.first         = t * set.count / threads;
        const size_t end   = (t + 1) * set.count / threads;
        mine.compressedOffsets.push_back(0);
        for (size_t i = mine.first; i < end; ++i) {
            mine.compressedOffsets.push_back(
                    mine.compressedOffsets.back()
                    + set.compressed[i % set.compressed.size()].size());
        }
        mine.compressed.reserve(mine.compressedOffsets.back());
        for (size_t i = mine.first; i < end; ++i) {
            const std::string& frame =
                    set.compressed[i % set.compressed.size()];
            mine.compressed.insert(
                    mine.compressed.end(), frame.begin(), frame.end());
        }
        mine.outputs.assign(mine.frames() * frameBytes, '\0');
        mine.dctx.emplace();
    });
    return placed;
}

// The number in a sysfs file, if it can be read.
std::optional<size_t> readSysfsNumber(const std::string& path)
{
    std::ifstream file{ path };
    size_t value = 0;
    if (!(file >> value)) {
        return std::nullopt;
    }
    return value;
}

// The rank of each CPU of the host when the CPUs are sorted by package, then
// by core, then by CPU number, so the hardware threads of one core get
// adjacent ranks. Without a sysfs topology, the rank of a CPU is its number.
const std::vector<size_t>& cpuRanks()
{
    static const std::vector<size_t> ranks = [] {
        // (package, core, CPU) of every CPU.
        std::vector<std::tuple<size_t, size_t, size_t>> cpus;
        for (size_t cpu = 0;; ++cpu) {
            const std::string topology = "/sys/devices/system/cpu/cpu"
                    + std::to_string(cpu) + "/topology/";
            const auto package =
                    readSysfsNumber(topology + "physical_package_id");
            const auto core = readSysfsNumber(topology + "core_id");
            if (!package || !core) {
                break;
            }
            cpus.emplace_back(*package, *core, cpu);
        }
        std::sort(cpus.begin(), cpus.end());
        std::vector<size_t> byCpu(cpus.size());
        for (size_t rank = 0; rank < cpus.size(); ++rank) {
            byCpu[std::get<2>(cpus[rank])] = rank;
        }
        if (byCpu.empty()) {
            byCpu.resize(
                    static_cast<size_t>(
                            std::max(sysconf(_SC_NPROCESSORS_CONF), 1L)));
            std::iota(byCpu.begin(), byCpu.end(), size_t{ 0 });
        }
        return byCpu;
    }();
    return ranks;
}

// The pool slot of a CPU: the host's CPUs, in rank order, split into `slots`
// equal blocks. CPU c uses the slot of CPU c % 256, as folly's AccessSpreader
// does, because its stripe table covers 256 CPUs.
size_t poolSlotOf(int cpu, size_t slots)
{
    constexpr size_t kStripeTableCpus = 256;
    const std::vector<size_t>& ranks  = cpuRanks();
    const size_t rank =
            ranks[static_cast<size_t>(cpu) % kStripeTableCpus % ranks.size()];
    return rank * slots / ranks.size();
}

// DCtx objects shared by the threads of a row. Each slot caches one DCtx, and
// a thread uses the poolSlotOf() slot of the CPU that it runs on. If the slot
// is empty, a thread takes a DCtx from the backing stack, which one mutex
// guards, or makes a new one. A returned DCtx goes to the slot, and the DCtx
// that it replaces goes to the stack.
class DCtxPool {
   public:
    explicit DCtxPool(size_t slots) : slots_(slots) {}

    DCtxPool(const DCtxPool&)            = delete;
    DCtxPool& operator=(const DCtxPool&) = delete;

    ~DCtxPool()
    {
        for (Slot& slot : slots_) {
            delete slot.dctx.load();
        }
    }

    std::unique_ptr<openzl::DCtx> take(size_t slot)
    {
        std::unique_ptr<openzl::DCtx> dctx{ slots_[slot].dctx.exchange(
                nullptr) };
        if (dctx) {
            return dctx;
        }
        {
            const std::lock_guard<std::mutex> lock{ mutex_ };
            if (!stack_.empty()) {
                dctx = std::move(stack_.back());
                stack_.pop_back();
                return dctx;
            }
        }
        return std::make_unique<openzl::DCtx>();
    }

    void give(size_t slot, std::unique_ptr<openzl::DCtx> dctx)
    {
        std::unique_ptr<openzl::DCtx> replaced{ slots_[slot].dctx.exchange(
                dctx.release()) };
        if (replaced) {
            const std::lock_guard<std::mutex> lock{ mutex_ };
            stack_.push_back(std::move(replaced));
        }
    }

   private:
    // One slot per cache line, so that threads on different slots do not
    // write to the same line.
    struct alignas(64) Slot {
        std::atomic<openzl::DCtx*> dctx{ nullptr };
    };

    std::vector<Slot> slots_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<openzl::DCtx>> stack_;
};

void decodeFrames(ThreadFrames& mine, size_t frameBytes, DCtxLifetime lifetime)
{
    for (size_t k = 0; k < mine.frames(); ++k) {
        if (lifetime == DCtxLifetime::frame) {
            openzl::DCtx dctx;
            dctx.decompressSerial(mine.output(k, frameBytes), mine.frame(k));
        } else {
            mine.dctx->decompressSerial(
                    mine.output(k, frameBytes), mine.frame(k));
        }
    }
}

void decodeFramesWithPool(
        ThreadFrames& mine,
        size_t frameBytes,
        DCtxPool& pool,
        size_t slot)
{
    for (size_t k = 0; k < mine.frames(); ++k) {
        std::unique_ptr<openzl::DCtx> dctx = pool.take(slot);
        dctx->decompressSerial(mine.output(k, frameBytes), mine.frame(k));
        pool.give(slot, std::move(dctx));
    }
}

// Throws unless every output holds its raw frame, then zeroes the outputs so
// that the next run cannot pass on this run's bytes.
void checkAndClearOutputs(
        const testkit::LadderStep& step,
        const FrameSet& set,
        std::vector<ThreadFrames>& placed)
{
    for (ThreadFrames& mine : placed) {
        for (size_t k = 0; k < mine.frames(); ++k) {
            const std::string& raw = set.raw[(mine.first + k) % set.raw.size()];
            if (!std::equal(
                        raw.begin(),
                        raw.end(),
                        mine.outputs.begin() + k * step.frameBytes)) {
                throw std::runtime_error(
                        "frame " + std::to_string(mine.first + k) + " of "
                        + step.name + " decoded to different bytes");
            }
        }
        std::fill(mine.outputs.begin(), mine.outputs.end(), '\0');
    }
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 1 ? values[middle]
                                  : (values[middle - 1] + values[middle]) / 2;
}

void printRow(
        const char* backend,
        const char* dctx,
        const testkit::LadderStep& step,
        const FrameSet& set,
        size_t parallelism,
        std::vector<double> seconds)
{
    const double gib = static_cast<double>(set.count * step.frameBytes) / kGiB;
    const double medianSeconds = median(std::move(seconds));
    std::printf(
            "%s\t%s\t%s\t%zu\t%zu\t%zu\t%.4f\t%.2f\n",
            backend,
            dctx,
            step.name.c_str(),
            set.count,
            step.frameBytes,
            parallelism,
            medianSeconds,
            gib / medianSeconds);
    std::fflush(stdout);
}

// Times and prints the CPU rows of one thread count: one row for each DCtx
// lifetime, with one pool row for each slot count.
void timeCpu(
        const Options& options,
        const testkit::LadderStep& step,
        const FrameSet& set,
        std::vector<ThreadFrames>& placed,
        size_t threads)
{
    const auto timeRow = [&](const std::string& dctx, const auto& decode) {
        std::vector<double> seconds;
        for (size_t i = 0; i < options.iterations; ++i) {
            seconds.push_back(runPerThread(threads, decode));
            checkAndClearOutputs(step, set, placed);
        }
        printRow("cpu", dctx.c_str(), step, set, threads, std::move(seconds));
    };
    for (const DCtxLifetime lifetime : options.dctx) {
        if (lifetime != DCtxLifetime::pool) {
            timeRow(lifetime == DCtxLifetime::frame ? "frame" : "thread",
                    [&](size_t t) {
                        decodeFrames(placed[t], step.frameBytes, lifetime);
                    });
            continue;
        }
        for (const size_t slots : options.poolSlots) {
            std::vector<size_t> slotOfThread(threads);
            for (size_t t = 0; t < threads; ++t) {
                slotOfThread[t] = poolSlotOf(cpuOfThread(t), slots);
            }
            DCtxPool pool{ slots };
            timeRow("pool-" + std::to_string(slots), [&](size_t t) {
                decodeFramesWithPool(
                        placed[t], step.frameBytes, pool, slotOfThread[t]);
            });
        }
    }
}

int run(const Options& options)
{
    std::printf(
            "backend\tdctx\tworkload\tframes\tframe_bytes\tparallelism\t"
            "median_s\tgib_per_s\n");
    const size_t maxThreads =
            *std::max_element(options.threads.begin(), options.threads.end());
    for (const size_t number : options.steps) {
        testkit::LadderStep step = workloads()[number - 1];
        if (options.frameBytes) {
            step.frameBytes = *options.frameBytes;
        }
        FrameSet set;
        try {
            set = makeFrameSet(step, options.totalMiB, maxThreads);
        } catch (const std::exception& e) {
            if (!options.frameBytes) {
                throw;
            }
            std::fprintf(
                    stderr,
                    "note: skipped %s, which cannot make frames of %zu "
                    "bytes: %s\n",
                    step.name.c_str(),
                    step.frameBytes,
                    e.what());
            continue;
        }
        for (const size_t threads : options.threads) {
            if (set.count < threads) {
                std::fprintf(
                        stderr,
                        "warning: %s has %zu frames, so only %zu of %zu "
                        "threads have work\n",
                        step.name.c_str(),
                        set.count,
                        set.count,
                        threads);
            }
            std::vector<ThreadFrames> placed =
                    placeFrames(set, step.frameBytes, threads);
            timeCpu(options, step, set, placed, threads);
        }
    }
    return 0;
}

} // namespace
} // namespace openzl::gpu::bench

int main(int argc, char** argv)
{
    try {
        return openzl::gpu::bench::run(
                openzl::gpu::bench::parseOptions(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_nimble_frames: %s\n", e.what());
        return 1;
    }
}
