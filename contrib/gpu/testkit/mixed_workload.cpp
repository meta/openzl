// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/mixed_workload.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#include "contrib/gpu/testkit/u64_workloads.h"
#include "contrib/gpu/testkit/zstd_workloads.h"

namespace openzl::gpu::testkit {

namespace {

constexpr size_t kMiB = size_t{ 1 } << 20;

const LadderStep& stepNamed(std::string_view name)
{
    const std::vector<LadderStep>& all = allWorkloads();
    const auto it =
            std::find_if(all.begin(), all.end(), [&](const LadderStep& s) {
                return s.name == name;
            });
    if (it == all.end()) {
        throw std::invalid_argument("no workload named " + std::string{ name });
    }
    return *it;
}

// Joins every thread that is still joinable when it goes out of scope, so an
// exception that leaves makeMixedFrames() does not destroy a running thread.
class JoinAll {
   public:
    explicit JoinAll(std::vector<std::thread>& threads) : threads_{ threads } {}

    ~JoinAll()
    {
        for (std::thread& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

   private:
    std::vector<std::thread>& threads_;
};

MixedWorkload buildF32U64ZstdMixedWorkload()
{
    return {
        .name         = "mixed/f32-u64-zstd",
        .decodedBytes = 150 * kMiB,
        .parts        = {
            { &stepNamed("zstd/u8-4k"), 0.010 },
            { &stepNamed("zstd/u32-128b"), 0.001 },
            { &stepNamed("zstd/u32-256b"), 0.003 },
            { &stepNamed("zstd/u32-512b"), 0.004 },
            { &stepNamed("zstd/u32-1k"), 0.005 },
            { &stepNamed("zstd/u32-16k"), 0.025 },
            { &stepNamed("zstd/u32-32k"), 0.045 },
            { &stepNamed("zstd/u32-64k"), 0.030 },
            { &stepNamed("zstd/u64-64k"), 0.030 },
            { &stepNamed("zstd/u64-512k"), 0.100 },
            { &stepNamed("zstd/u64-2m"), 0.045 },
            { &stepNamed("u64/bitpacktokens"), 0.015 },
            { &stepNamed("u64/literals"), 0.080 },
            { &stepNamed("u64/tokenize"), 0.075 },
            { &stepNamed("u64/hufstruct"), 0.170 },
            { &stepNamed("u64/splithuffman"), 0.260 },
            { &stepNamed("f32/02-fieldlz-transpose"), 0.002 },
            { &stepNamed("f32/05-tokenbitpack-quantoffsets"), 0.015 },
            { &stepNamed("f32/bitpacktokens-storedcodes"), 0.015 },
            { &stepNamed("f32/03-huffman-bitpack"), 0.015 },
            { &stepNamed("f32/06-bitpackserial-quantlengths"), 0.015 },
            { &stepNamed("f32/indexfieldlz"), 0.020 },
            { &stepNamed("f32/splithuffman-fsecodes"), 0.010 },
            { &stepNamed("f32/splithuffman-tokenizetokens"), 0.010 },
        },
    };
}

} // namespace

const std::vector<LadderStep>& allWorkloads()
{
    static const std::vector<LadderStep> all = [] {
        std::vector<LadderStep> list = f32Ladder();
        for (const auto* more :
             { &f32ExtraWorkloads(), &zstdWorkloads(), &u64Workloads() }) {
            list.insert(list.end(), more->begin(), more->end());
        }
        return list;
    }();
    return all;
}

const MixedWorkload& f32U64ZstdMixedWorkload()
{
    static const MixedWorkload mix = buildF32U64ZstdMixedWorkload();
    return mix;
}

MixedFrames
makeMixedFrames(const MixedWorkload& mix, uint64_t seed, size_t threads)
{
    // (part, frame index in the part) of each frame.
    std::vector<std::pair<size_t, size_t>> plan;
    for (size_t p = 0; p < mix.parts.size(); ++p) {
        const MixedWorkloadPart& part = mix.parts[p];
        const double frames = part.share * static_cast<double>(mix.decodedBytes)
                / static_cast<double>(part.step->frameBytes);
        const auto count = std::max<size_t>(1, std::llround(frames));
        for (size_t j = 0; j < count; ++j) {
            plan.emplace_back(p, j);
        }
    }
    std::mt19937_64 rng(seed);
    std::shuffle(plan.begin(), plan.end(), rng);

    MixedFrames frames;
    frames.part.resize(plan.size());
    frames.raw.resize(plan.size());
    frames.compressed.resize(plan.size());
    threads = std::max<size_t>(1, std::min(threads, plan.size()));
    // An exception that leaves a std::thread calls std::terminate, so each
    // thread keeps its first exception and the index of the frame that threw.
    std::vector<std::pair<size_t, std::exception_ptr>> errors(
            threads, { plan.size(), nullptr });
    const auto makeEvery = [&](size_t first) {
        size_t i = first;
        try {
            for (; i < plan.size(); i += threads) {
                const auto [p, j]             = plan[i];
                const MixedWorkloadPart& part = mix.parts[p];
                WorkloadSpec spec             = part.step->spec;
                spec.seed                     = seed;
                frames.part[i]                = p;
                frames.raw[i] = generateFrame(spec, part.step->frameBytes, j);
                frames.compressed[i] = compressStep(*part.step, frames.raw[i]);
            }
        } catch (...) {
            errors[first] = { i, std::current_exception() };
        }
    };
    std::vector<std::thread> pool;
    {
        const JoinAll joinAll(pool);
        for (size_t t = 1; t < threads; ++t) {
            pool.emplace_back(makeEvery, t);
        }
        makeEvery(0);
    }
    const auto lowest = std::min_element(
            errors.begin(), errors.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
    if (lowest->second) {
        std::rethrow_exception(lowest->second);
    }
    return frames;
}

} // namespace openzl::gpu::testkit
