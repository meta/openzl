// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/nimble_workloads.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include "contrib/gpu/testkit/bytes.h"
#include "contrib/gpu/testkit/parse_plan.h"
#include "openzl/shared/bits.h"

namespace openzl::gpu::testkit {

namespace {

uint64_t maxValueForWidth(size_t width)
{
    return width >= 8 ? UINT64_MAX : (1ULL << (8 * width)) - 1;
}

void appendFloat(std::string& out, double value)
{
    const auto narrowed = static_cast<float>(value);
    uint32_t bits;
    std::memcpy(&bits, &narrowed, sizeof(bits));
    appendLittleEndian(out, bits, sizeof(bits));
}

// The tables that all frames of one column share: an alphabet and the Zipf
// weights (1/k for the k-th value) that draw from it. Only the spec's seed
// decides them, so every frame of a column uses the same values. Each overload
// checks the parameters of its pattern.
struct ColumnTables {
    std::vector<uint64_t> alphabet;
    std::optional<std::discrete_distribution<uint64_t>> zipf;
};

class TableBuilder {
   public:
    explicit TableBuilder(const WorkloadSpec& spec)
            : width_{ elementByteWidth(spec.type) }, rng_{ spec.seed }
    {
    }

    ColumnTables operator()(const FloatNormal& p) const
    {
        if (!std::isfinite(p.mean) || !std::isfinite(p.stddev)
            || !(p.stddev > 0)) {
            throw std::invalid_argument(
                    "FloatNormal needs a finite mean and a positive stddev");
        }
        return {};
    }

    // The range ends, then distinct values between them.
    ColumnTables operator()(const PinnedValues& p)
    {
        const uint64_t maxValue = maxValueForWidth(width_);
        if (p.base > maxValue || p.range > maxValue - p.base
            || p.cardinality == 1
            || (p.cardinality != 0 && p.cardinality - 1 > p.range)) {
            throw std::invalid_argument("PinnedValues do not fit the type");
        }
        if (p.cardinality == 0) {
            return {};
        }
        std::vector<uint64_t> alphabet = { p.base, p.base + p.range };
        std::unordered_set<uint64_t> seen(alphabet.begin(), alphabet.end());
        while (alphabet.size() < p.cardinality) {
            const uint64_t value = std::uniform_int_distribution<uint64_t>(
                    p.base + 1, p.base + p.range - 1)(rng_);
            if (seen.insert(value).second) {
                alphabet.push_back(value);
            }
        }
        std::shuffle(alphabet.begin(), alphabet.end(), rng_);
        return { .alphabet = std::move(alphabet),
                 .zipf     = zipfDistribution<uint64_t>(p.cardinality) };
    }

    // The planned-parse patterns draw everything per frame, and their writers
    // check their parameters against the frame size.
    template <typename PlannedParse>
    ColumnTables operator()(const PlannedParse& /* p */) const
    {
        return {};
    }

   private:
    size_t width_;
    std::mt19937_64 rng_;
};

// Writes one frame of a column. Each overload checks the limits that depend
// on the frame size.
class FrameWriter {
   public:
    FrameWriter(
            const WorkloadSpec& spec,
            const ColumnTables& tables,
            std::mt19937_64 rng,
            size_t numElts,
            std::string& out)
            : tables_{ tables },
              numElts_{ numElts },
              width_{ elementByteWidth(spec.type) },
              rng_{ std::move(rng) },
              out_{ out }
    {
    }

    void operator()(const FloatNormal& p)
    {
        std::normal_distribution<double> normal(p.mean, p.stddev);
        for (size_t i = 0; i < numElts_; ++i) {
            appendFloat(out_, normal(rng_));
        }
    }

    // ColumnGenerator checks the frame size first.
    void operator()(const PinnedValues& p)
    {
        if (p.cardinality == 0) {
            writeDistinct(p);
        } else {
            writeAlphabet(p);
        }
    }

    // The planned-parse patterns write the whole frame in one call.
    template <typename FloatParse>
    void operator()(const FloatParse& p)
    {
        out_ = writeFloats(p, numElts_, rng_);
    }

    void operator()(const PlannedInts& p)
    {
        out_ = writeInts(p, numElts_, rng_);
    }

   private:
    // Increasing values `base + i * step + jitter`, then `base + range` last.
    // A power-of-two step and a small jitter leave most bytes of each value
    // constant or slowly changing, so the frame still compresses. A frame
    // with more values than the range cycles through the whole range.
    void writeDistinct(const PinnedValues& p)
    {
        if (p.range < numElts_ - 1) {
            for (size_t i = 0; i < numElts_ - 1; ++i) {
                appendLittleEndian(out_, p.base + i % (p.range + 1), width_);
            }
            appendLittleEndian(out_, p.base + p.range, width_);
            return;
        }
        // floor((range + 1) / numElts) without overflow at range 2^64 - 1.
        const uint64_t spacing = (p.range - (numElts_ - 1)) / numElts_ + 1;
        const uint64_t step    = 1ULL << ZL_highbit64(spacing);
        const uint64_t jitter  = std::min<uint64_t>(step, 16);
        appendLittleEndian(out_, p.base, width_);
        for (size_t i = 1; i < numElts_ - 1; ++i) {
            appendLittleEndian(
                    out_,
                    p.base + i * step
                            + std::uniform_int_distribution<uint64_t>(
                                    0, jitter - 1)(rng_),
                    width_);
        }
        appendLittleEndian(out_, p.base + p.range, width_);
    }

    // Writes every alphabet value once, draws the rest with Zipf weights, and
    // shuffles the frame.
    void writeAlphabet(const PinnedValues& p)
    {
        std::discrete_distribution<uint64_t> zipf = *tables_.zipf;
        std::vector<uint64_t> indices(numElts_);
        for (size_t i = 0; i < numElts_; ++i) {
            indices[i] = i < p.cardinality ? i : zipf(rng_);
        }
        std::shuffle(indices.begin(), indices.end(), rng_);
        for (const uint64_t index : indices) {
            appendLittleEndian(out_, tables_.alphabet[index], width_);
        }
    }

    const ColumnTables& tables_;
    size_t numElts_;
    size_t width_;
    std::mt19937_64 rng_;
    std::string& out_;
};

// Returns the engine of frame `frameIndex` of a column. The C++ standard
// fixes the std::seed_seq algorithm, so every standard library starts the
// frame at the same state. std::seed_seq keeps 32 bits of each value, so the
// 64-bit seed and index go in as halves.
std::mt19937_64 frameEngine(uint64_t columnSeed, size_t frameIndex)
{
    const auto index = static_cast<uint64_t>(frameIndex);
    std::seed_seq seeds{ static_cast<uint32_t>(columnSeed),
                         static_cast<uint32_t>(columnSeed >> 32),
                         static_cast<uint32_t>(index),
                         static_cast<uint32_t>(index >> 32) };
    return std::mt19937_64(seeds);
}

// One column: its shared tables, which it builds once, and its frames.
class ColumnGenerator {
   public:
    ColumnGenerator(const WorkloadSpec& spec, size_t frameBytes)
            : spec_{ spec },
              numElts_{ frameBytes / elementByteWidth(spec.type) }
    {
        if (std::holds_alternative<PlannedInts>(spec.pattern)) {
            if (spec.type != NimbleNumericType::U64) {
                throw std::invalid_argument("PlannedInts need U64");
            }
        } else if (
                spec.type != NimbleNumericType::F32
                && !std::holds_alternative<PinnedValues>(spec.pattern)) {
            throw std::invalid_argument(
                    "patterns other than PinnedValues and PlannedInts need "
                    "F32");
        }
        if (frameBytes % elementByteWidth(spec.type) != 0) {
            throw std::invalid_argument("a frame must hold whole elements");
        }
        // TableBuilder draws the whole PinnedValues alphabet, so a cardinality
        // too large for the frame must fail before it.
        const auto* const pinned = std::get_if<PinnedValues>(&spec.pattern);
        if (pinned != nullptr
            && (numElts_ < 2 || pinned->cardinality > numElts_)) {
            throw std::invalid_argument(
                    "the frame is too small for its PinnedValues");
        }
        tables_ = std::visit(TableBuilder{ spec }, spec.pattern);
    }

    std::string frame(size_t frameIndex) const
    {
        std::string out;
        out.reserve(numElts_ * elementByteWidth(spec_.type));
        std::visit(
                FrameWriter{ spec_,
                             tables_,
                             frameEngine(spec_.seed, frameIndex),
                             numElts_,
                             out },
                spec_.pattern);
        return out;
    }

   private:
    WorkloadSpec spec_;
    size_t numElts_;
    ColumnTables tables_;
};

} // namespace

std::string
generateFrame(const WorkloadSpec& spec, size_t frameBytes, size_t frameIndex)
{
    return ColumnGenerator{ spec, frameBytes }.frame(frameIndex);
}

std::vector<Frame> makeFrames(
        const WorkloadSpec& spec,
        size_t frameBytes,
        size_t count,
        const NimbleCompressParams& params,
        size_t threads)
{
    const ColumnGenerator column{ spec, frameBytes };
    std::vector<Frame> frames(count);
    threads = std::max<size_t>(1, std::min(threads, count));
    // An exception that leaves a std::thread calls std::terminate, so each
    // thread keeps its first exception and the index of the frame that threw.
    std::vector<std::pair<size_t, std::exception_ptr>> errors(
            threads, { count, nullptr });
    const auto makeEvery = [&](size_t first) {
        size_t i = first;
        try {
            for (; i < count; i += threads) {
                frames[i].raw = column.frame(i);
                frames[i].compressed =
                        compressLikeNimble(frames[i].raw, spec.type, params);
            }
        } catch (...) {
            errors[first] = { i, std::current_exception() };
        }
    };
    std::vector<std::thread> pool;
    for (size_t t = 1; t < threads; ++t) {
        pool.emplace_back(makeEvery, t);
    }
    makeEvery(0);
    for (std::thread& thread : pool) {
        thread.join();
    }
    // Each thread tries its frames in order, so the lowest frame that throws
    // is the same for every thread count.
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
