// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/float_parses.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "contrib/gpu/testkit/bytes.h"
#include "contrib/gpu/testkit/nimble_workloads.h"
#include "contrib/gpu/testkit/parse_plan.h"
#include "openzl/shared/bits.h"

namespace openzl::gpu::testkit {

namespace {

constexpr uint32_t kOneBits = 0x3F800000;

uint32_t floatBits(double value)
{
    const auto narrowed = static_cast<float>(value);
    uint32_t bits;
    std::memcpy(&bits, &narrowed, sizeof(bits));
    return bits;
}

// Returns the sorted alphabet that starts at 1.0f and steps by `gaps[i]`.
// Throws std::invalid_argument if the alphabet reaches the Inf/NaN patterns.
std::vector<uint32_t> alphabetFromGaps(const std::vector<uint32_t>& gaps)
{
    constexpr uint64_t kInfinityBits = 0x7F800000;
    std::vector<uint32_t> alphabet(gaps.size(), kOneBits);
    uint64_t value = kOneBits;
    for (size_t i = 1; i < gaps.size(); ++i) {
        value += gaps[i];
        if (value >= kInfinityBits) {
            throw std::invalid_argument("the alphabet does not fit below Inf");
        }
        alphabet[i] = static_cast<uint32_t>(value);
    }
    return alphabet;
}

// Literal values for PlannedFloats and CopyStreamFloats.
class LiteralSource {
   public:
    // Throws std::invalid_argument if `literals` is out of range.
    LiteralSource(const FloatLiterals& literals, std::mt19937_64& rng)
            : literals_{ literals }, rng_{ rng }
    {
        std::visit([this](const auto& p) { setUp(p); }, literals_);
    }

    uint32_t next()
    {
        return std::visit([this](const auto& p) { return draw(p); }, literals_);
    }

   private:
    void setUp(const RandomFloatBits& /* p */) {}

    void setUp(const PositiveFloats& p)
    {
        // A tail value 2^-k * (1 + f) needs a normal exponent, 127 - k >= 1.
        if (!(p.tailShare >= 0 && p.tailShare <= 1) || p.tailOctaves < 0
            || p.tailOctaves > 126) {
            throw std::invalid_argument("PositiveFloats have invalid tails");
        }
    }

    void setUp(const ByteWalkFloats& /* p */) {}

    void setUp(const SignedOctaveFloats& p)
    {
        if (p.octaves < 1 || p.octaves > 126) {
            throw std::invalid_argument(
                    "SignedOctaveFloats need 1-126 octaves");
        }
    }

    void setUp(const WindowFloats& p)
    {
        // Byte 3 walks up to 0x77, and 0x77 + 7 is the highest finite byte 3.
        if (p.byte3Jitter > 7) {
            throw std::invalid_argument("WindowFloats jitter is above 7");
        }
        for (WindowBytes& stream : windows_) {
            stream.drawCommon(rng_);
        }
        for (size_t b = 0; b < windows_.size(); ++b) {
            windows_[b].drawPatterns(rng_, p.commonShare[b]);
        }
        if (p.byte3Jitter > 0) {
            for (auto& jitter : jitters_) {
                for (uint8_t& value : jitter) {
                    value = static_cast<uint8_t>(
                            uniformBelow(rng_, size_t{ p.byte3Jitter } + 1));
                }
            }
        }
    }

    uint32_t draw(const RandomFloatBits& /* p */)
    {
        constexpr uint32_t kExponentMask = 0x7F800000;
        uint32_t bits;
        do {
            bits = static_cast<uint32_t>(rng_());
        } while ((bits & kExponentMask) == kExponentMask);
        return bits;
    }

    uint32_t draw(const PositiveFloats& p)
    {
        if (p.tailShare > 0 && chance(rng_, p.tailShare)) {
            const auto octave = static_cast<uint32_t>(
                    uniformBelow(rng_, static_cast<size_t>(p.tailOctaves) + 1));
            const auto fraction = static_cast<uint32_t>(rng_() >> 41);
            return ((127 - octave) << 23) | fraction;
        }
        return floatBits(std::fabs(normal_(rng_)));
    }

    uint32_t draw(const ByteWalkFloats& p)
    {
        walkByte3(p.move, p.grow);
        walkByte2(p.byte2Move, p.byte2Grow);
        return (static_cast<uint32_t>(byte3_) << 24)
                | (static_cast<uint32_t>(byte2_) << 16)
                | static_cast<uint32_t>(rng_() & 0xFFFF);
    }

    uint32_t draw(const SignedOctaveFloats& p)
    {
        const auto octave =
                static_cast<uint32_t>(uniformBelow(rng_, p.octaves));
        const uint64_t bits = rng_();
        return (static_cast<uint32_t>(bits & 1) << 31) | ((127 - octave) << 23)
                | static_cast<uint32_t>(bits >> 41);
    }

    uint32_t draw(const WindowFloats& p)
    {
        const size_t windowed = p.byte2Move >= 0 ? 2 : 3;
        uint32_t value        = 0;
        for (size_t b = 0; b < windowed; ++b) {
            value |= static_cast<uint32_t>(windows_[b].next(
                             rng_, p.patternShare[b], p.commonShare[b]))
                    << (8 * b);
        }
        if (p.byte2Move >= 0) {
            walkByte2(p.byte2Move, p.byte2Grow);
            value |= static_cast<uint32_t>(byte2_) << 16;
        }
        walkByte3(p.byte3Move, p.byte3Grow);
        size_t jitter = 0;
        if (p.byte3Jitter > 0) {
            if (jitterOffset_ == 0) {
                jitter_     = uniformBelow(rng_, jitters_.size());
                usePattern_ = chance(rng_, p.byte3PatternShare);
            }
            jitter        = usePattern_
                           ? size_t{ jitters_[jitter_][jitterOffset_] }
                           : uniformBelow(rng_, size_t{ p.byte3Jitter } + 1);
            jitterOffset_ = (jitterOffset_ + 1) % jitters_[0].size();
        }
        return value | (static_cast<uint32_t>(byte3_ + jitter) << 24);
    }

    // Byte 3 stays inside 0x08..0x77: positive, finite, and not tiny.
    void walkByte3(double move, double grow)
    {
        constexpr int kLowest = 0x08, kHighest = 0x77;
        int next3 = byte3_ + signedStep(rng_, move, grow);
        while (next3 < kLowest || next3 > kHighest) {
            next3 = next3 < kLowest ? 2 * kLowest - next3
                                    : 2 * kHighest - next3;
        }
        byte3_ = next3;
    }

    void walkByte2(double move, double grow)
    {
        byte2_ = (byte2_ + signedStep(rng_, move, grow)) & 0xFF;
    }

    const FloatLiterals& literals_;
    std::mt19937_64& rng_;
    std::normal_distribution<double> normal_{ 0, 1 };
    std::array<WindowBytes, 3> windows_;
    std::array<std::array<uint8_t, 9>, 32> jitters_{};
    size_t jitter_       = 0;
    size_t jitterOffset_ = 0;
    bool usePattern_     = false;
    int byte3_           = 0x41;
    int byte2_           = 0;
};

// Lengths base + [2^o, 2^(o+1)), with o stepping through
// [minOctave, maxOctave). Two lengths in a row share o and sum to
// 2 * base + 3 * 2^o - 1.
class PairedLengths {
   public:
    PairedLengths(size_t base, uint32_t minOctave, uint32_t maxOctave)
            : base_{ base },
              minOctave_{ minOctave },
              maxOctave_{ maxOctave },
              octave_{ minOctave }
    {
    }

    size_t next(std::mt19937_64& rng)
    {
        const size_t low = size_t{ 1 } << octave_;
        size_t offset;
        if (paired_) {
            offset  = low - 1 - first_;
            octave_ = octave_ + 1 == maxOctave_ ? minOctave_ : octave_ + 1;
        } else {
            first_ = uniformBelow(rng, low);
            offset = first_;
        }
        paired_ = !paired_;
        return base_ + low + offset;
    }

   private:
    size_t base_;
    uint32_t minOctave_;
    uint32_t maxOctave_;
    uint32_t octave_;
    size_t first_ = 0;
    bool paired_  = false;
};

} // namespace

std::string
writeFloats(const PlannedFloats& p, size_t numElts, std::mt19937_64& rng)
{
    const ParsePlan plan{ .runs            = p.runs,
                          .runLength       = p.runLength,
                          .copies          = p.copies,
                          .copyLength      = p.copyLength,
                          .copyOffsetCodes = p.copyOffsetCodes,
                          .minCopyDistance = p.minCopyDistance };
    const std::vector<ParseSlot> slots = planParse(plan, numElts, rng);
    LiteralSource literals{ p.literals, rng };
    return toLittleEndianBytes(fillParse<uint32_t>(plan, slots, rng, [&] {
        return literals.next();
    }));
}

std::string
writeFloats(const GapAlphabetFloats& p, size_t numElts, std::mt19937_64& rng)
{
    // Gaps are 256 * [2, kMaxGapSteps] plus a uniform byte.
    constexpr uint32_t kMaxGapSteps = 4096;
    const size_t k                  = p.cardinality;
    if (k < 16 || k > numElts) {
        throw std::invalid_argument("GapAlphabetFloats do not fit the frame");
    }
    std::vector<uint32_t> gaps(k, 0);
    for (size_t i = 1; i < k; ++i) {
        gaps[i] = 256
                * (2
                   + static_cast<uint32_t>(
                           uniformBelow(rng, kMaxGapSteps - 1)));
        gaps[i] += static_cast<uint32_t>(uniformBelow(rng, 256));
    }
    const std::vector<uint32_t> alphabet = alphabetFromGaps(gaps);
    std::vector<uint32_t> values(numElts);
    for (size_t i = 0; i < numElts; ++i) {
        values[i] = i < k ? alphabet[i] : alphabet[uniformBelow(rng, k)];
    }
    std::shuffle(values.begin(), values.end(), rng);
    return toLittleEndianBytes(values);
}

std::string
writeFloats(const WideAlphabetFloats& p, size_t numElts, std::mt19937_64& rng)
{
    constexpr size_t kMinGapLog2   = 16;
    constexpr size_t kBigGapLog2   = 24;
    constexpr size_t kLongCopyStep = 100;
    const size_t k                 = p.cardinality;
    if (k <= 256 || k + 64 > numElts || p.longLiteralEvery == 0
        || p.maxGapLog2 < kMinGapLog2 || p.maxGapLog2 > kBigGapLog2
        || p.maxLiteralRun == 0 || p.minCopy == 0 || p.minCopy > p.maxCopy
        || p.distanceOctaves < 2) {
        throw std::invalid_argument("WideAlphabetFloats do not fit the frame");
    }
    std::vector<std::pair<size_t, size_t>> pairs;
    pairs.reserve(p.gapPairs);
    for (uint32_t c = 0; c < p.gapPairs; ++c) {
        // Two draws in one argument list would run in an unspecified order.
        const size_t at       = 20 + uniformBelow(rng, k - 40);
        const size_t distance = 3 + uniformBelow(rng, 15);
        pairs.emplace_back(at, distance);
    }
    std::vector<uint32_t> gaps(k, 0);
    for (size_t i = 1; i < k; ++i) {
        bool copied = false;
        for (const auto& [at, distance] : pairs) {
            if (i == at || i == at + 1) {
                gaps[i] = gaps[i - distance];
                copied  = true;
            }
        }
        if (copied) {
            continue;
        }
        const size_t log2 = p.bigGapEvery > 0 && i % p.bigGapEvery == 0
                ? kBigGapLog2
                : kMinGapLog2
                        + geometric(rng, p.gapGrow, p.maxGapLog2 - kMinGapLog2);
        gaps[i]           = static_cast<uint32_t>(inOctave(rng, log2));
    }
    const std::vector<uint32_t> alphabet = alphabetFromGaps(gaps);
    const int maxHigh                    = static_cast<int>((k - 1) / 256);
    int high                             = static_cast<int>(k / 256) / 2;
    int low                              = 0;
    const auto nextIndex                 = [&]() {
        if (chance(rng, p.highMove)) {
            const int step = (rng() & 1) ? 1 : -1;
            high = (high + step < 0 || high + step > maxHigh) ? high - step
                                                                              : high + step;
        }
        const size_t lowLimit = high == maxHigh ? k - 256 * maxHigh : 256;
        const int step        = static_cast<int>(p.lowByteStep);
        low = (low + static_cast<int>(uniformBelow(rng, 2 * step + 1)) - step
               + 256)
                % 256;
        return static_cast<size_t>(256 * high)
                + static_cast<size_t>(low) % lowLimit;
    };
    // Copies wait until every distance fits behind them.
    const size_t firstCopy = size_t{ 2 } << p.distanceOctaves;
    std::vector<size_t> indices;
    indices.reserve(numElts);
    size_t lastDistance = 3, runCount = 0, copyCount = 0;
    while (indices.size() < numElts) {
        const size_t literals = (++runCount % p.longLiteralEvery == 0)
                ? 20
                : 1 + uniformBelow(rng, p.maxLiteralRun);
        for (size_t j = 0; j < literals && indices.size() < numElts; ++j) {
            indices.push_back(nextIndex());
        }
        const size_t written = indices.size();
        if (written < firstCopy) {
            continue;
        }
        const size_t length = (++copyCount % kLongCopyStep == 0)
                ? size_t{ p.longCopyLength }
                : p.minCopy + uniformBelow(rng, p.maxCopy - p.minCopy + 1);
        size_t distance     = lastDistance;
        if (!chance(rng, p.reuseShare) || distance >= written) {
            distance = std::min(
                    written - 1,
                    inOctave(rng, 1 + uniformBelow(rng, p.distanceOctaves - 1))
                            - 1);
        }
        lastDistance = distance;
        for (size_t j = 0; j < length && indices.size() < numElts; ++j) {
            indices.push_back(indices[indices.size() - distance]);
        }
    }
    // A leading block holds all k indices, so every alphabet value appears.
    // The block groups the indices by high byte and shuffles the low bytes
    // inside each group.
    std::vector<size_t> block(k);
    for (size_t i = 0; i < k; ++i) {
        block[i] = i;
    }
    for (size_t lo = 0; lo < k; lo += 256) {
        const size_t hi = std::min(k, lo + 256);
        std::shuffle(block.begin() + lo, block.begin() + hi, rng);
    }
    indices.insert(indices.begin(), block.begin(), block.end());
    indices.resize(numElts);
    std::vector<uint32_t> values(numElts);
    for (size_t i = 0; i < numElts; ++i) {
        values[i] = alphabet[indices[i]];
    }
    return toLittleEndianBytes(values);
}

std::string
writeFloats(const CopyStreamFloats& p, size_t numElts, std::mt19937_64& rng)
{
    constexpr size_t kShortCap = 14;
    // A copy draws only the distance octaves that fit behind it. Without this
    // prefix, the first few values would fill the frame through early copies.
    constexpr size_t kPrefixLiterals = 1024;
    if (p.longLiteralEvery == 0 || p.shortLiteralEvery == 0
        || p.longCopyEvery == 0
        || p.longLiteralMinOctave >= p.longLiteralMaxOctave
        || p.longLiteralMaxOctave > 30
        || p.longCopyMinOctave >= p.longCopyMaxOctave
        || p.longCopyMaxOctave > 30
        || p.minDistanceOctave >= p.maxDistanceOctave
        || p.maxDistanceOctave > 32 || numElts < 2 * kPrefixLiterals) {
        throw std::invalid_argument("CopyStreamFloats do not fit the frame");
    }
    LiteralSource literals{ p.literals, rng };
    PairedLengths longLiterals{ 15,
                                p.longLiteralMinOctave,
                                p.longLiteralMaxOctave };
    PairedLengths longCopies{ 16, p.longCopyMinOctave, p.longCopyMaxOctave };
    std::vector<uint32_t> values;
    values.reserve(numElts);
    const auto writeLiterals = [&](size_t count) {
        for (size_t i = 0; i < count; ++i) {
            values.push_back(literals.next());
        }
    };
    writeLiterals(kPrefixLiterals);
    // distances[0] is the distance of the last copy, distances[1] of the one
    // before it.
    size_t distances[2]  = { 1, 1 };
    size_t afterLiterals = 0;
    for (size_t copy = 1; values.size() < numElts; ++copy) {
        const bool literalsFirst = copy % p.longLiteralEvery == 0
                || copy % p.shortLiteralEvery == 0;
        if (copy % p.longLiteralEvery == 0) {
            writeLiterals(longLiterals.next(rng));
        } else if (copy % p.shortLiteralEvery == 0) {
            writeLiterals(
                    1 + geometric(rng, p.shortLiteralContinue, kShortCap));
        }
        const size_t length = copy % p.longCopyEvery == 0
                ? longCopies.next(rng)
                : 1 + geometric(rng, p.copyContinue, kShortCap);
        size_t distance     = distances[1];
        if (literalsFirst && p.rep0Every > 0
            && ++afterLiterals % p.rep0Every == 0) {
            distance = distances[0];
        } else if (p.rep1Every == 0 || copy % p.rep1Every != 0) {
            const size_t fits = std::min<size_t>(
                    p.maxDistanceOctave,
                    static_cast<size_t>(ZL_highbit64(values.size())));
            distance = fits > p.minDistanceOctave
                    ? inOctave(
                              rng,
                              p.minDistanceOctave
                                      + uniformBelow(
                                              rng, fits - p.minDistanceOctave))
                    : 1 + uniformBelow(rng, values.size());
        }
        distances[1] = distances[0];
        distances[0] = distance;
        for (size_t i = 0; i < length; ++i) {
            values.push_back(values[values.size() - distance]);
        }
    }
    values.resize(numElts);
    return toLittleEndianBytes(values);
}

} // namespace openzl::gpu::testkit
