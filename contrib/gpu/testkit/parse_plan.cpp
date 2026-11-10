// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/parse_plan.h"

#include <algorithm>
#include <stdexcept>

namespace openzl::gpu::testkit {

namespace {

// A placement gives up after this many collisions in a row.
constexpr int kMaxTries = 100;

bool allLiteral(const std::vector<ParseSlot>& slots, size_t first, size_t last)
{
    for (size_t i = first; i <= last && i < slots.size(); ++i) {
        if (slots[i] != ParseSlot::literal) {
            return false;
        }
    }
    return true;
}

// Places `count` spans of `length` slots at positions from `minAt`, each with
// a literal slot on both sides. A span starts with `first`, and its other
// slots get `rest`. A span that collides kMaxTries times in a row is dropped.
// Throws std::invalid_argument if a span cannot fit at all.
void placeSpans(
        std::vector<ParseSlot>& slots,
        uint32_t count,
        size_t length,
        size_t minAt,
        ParseSlot first,
        ParseSlot rest,
        std::mt19937_64& rng)
{
    if (count > 0 && (length == 0 || slots.size() < minAt + length + 1)) {
        throw std::invalid_argument("the spans do not fit the frame");
    }
    for (uint32_t s = 0; s < count; ++s) {
        for (int tries = 0; tries < kMaxTries; ++tries) {
            const size_t at =
                    minAt + uniformBelow(rng, slots.size() - minAt - length);
            if (allLiteral(slots, at - 1, at + length)) {
                slots[at] = first;
                std::fill_n(slots.begin() + at + 1, length - 1, rest);
                break;
            }
        }
    }
}

} // namespace

size_t uniformBelow(std::mt19937_64& rng, size_t bound)
{
    if (bound == 0) {
        throw std::invalid_argument("a bound must be positive");
    }
    return std::uniform_int_distribution<size_t>(0, bound - 1)(rng);
}

bool chance(std::mt19937_64& rng, double p)
{
    if (!(p >= 0 && p <= 1)) {
        throw std::invalid_argument("a probability must be in [0, 1]");
    }
    return std::bernoulli_distribution(p)(rng);
}

size_t geometric(std::mt19937_64& rng, double q, size_t cap)
{
    // std::geometric_distribution needs a success probability in (0, 1).
    if (!(q < 1)) {
        throw std::invalid_argument("a continue probability must be below 1");
    }
    if (q <= 0) {
        return 0;
    }
    return std::min<size_t>(
            cap, std::geometric_distribution<size_t>(1.0 - q)(rng));
}

size_t inOctave(std::mt19937_64& rng, size_t log2)
{
    const size_t low = size_t{ 1 } << log2;
    return low + uniformBelow(rng, low);
}

int signedStep(std::mt19937_64& rng, double move, double grow)
{
    if (!chance(rng, move)) {
        return 0;
    }
    const int size = 1 + static_cast<int>(geometric(rng, grow, 64));
    return (rng() & 1) ? size : -size;
}

WindowBytes::WindowBytes(size_t patterns) : patterns_(patterns) {}

void WindowBytes::drawCommon(std::mt19937_64& rng)
{
    for (uint8_t& value : common_) {
        value = static_cast<uint8_t>(rng());
    }
}

void WindowBytes::drawPatterns(std::mt19937_64& rng, double commonShare)
{
    for (auto& pattern : patterns_) {
        for (uint8_t& value : pattern) {
            value = skewed(rng, commonShare);
        }
    }
}

uint8_t
WindowBytes::next(std::mt19937_64& rng, double patternShare, double commonShare)
{
    if (offset_ == 0) {
        pattern_    = uniformBelow(rng, patterns_.size());
        usePattern_ = chance(rng, patternShare);
    }
    const uint8_t value = usePattern_ ? patterns_[pattern_][offset_]
                                      : skewed(rng, commonShare);
    offset_             = (offset_ + 1) % patterns_[0].size();
    return value;
}

uint8_t WindowBytes::skewed(std::mt19937_64& rng, double commonShare)
{
    return chance(rng, commonShare) ? common_[uniformBelow(rng, common_.size())]
                                    : static_cast<uint8_t>(rng());
}

std::vector<ParseSlot>
planParse(const ParsePlan& plan, size_t numElts, std::mt19937_64& rng)
{
    if (plan.copies > 0
        && (plan.copyLength == 0 || plan.copyOffsetCodes > 30
            || (plan.copyOffsetCodes > 0) == (plan.minCopyDistance > 0))) {
        throw std::invalid_argument("the parse plan is invalid");
    }
    const size_t minCopyAt = plan.copyOffsetCodes > 0
            ? (size_t{ 2 } << std::min(plan.copyOffsetCodes, uint32_t{ 30 }))
                    + 4
            : 2 * size_t{ plan.minCopyDistance };
    if ((plan.runs > 0
         && (plan.runLength == 0 || numElts < size_t{ plan.runLength } + 4))
        || (plan.copies > 0
            && numElts < minCopyAt + size_t{ plan.copyLength } + 2)) {
        throw std::invalid_argument("the parse plan does not fit the frame");
    }
    std::vector<ParseSlot> slots(numElts, ParseSlot::literal);
    placeSpans(
            slots,
            plan.runs,
            plan.runLength,
            2,
            ParseSlot::repeat,
            ParseSlot::repeat,
            rng);
    placeSpans(
            slots,
            plan.copies,
            plan.copyLength,
            minCopyAt,
            ParseSlot::copyStart,
            ParseSlot::copyRest,
            rng);
    return slots;
}

size_t copySource(
        const ParsePlan& plan,
        const std::vector<ParseSlot>& slots,
        size_t at,
        std::mt19937_64& rng)
{
    size_t from = 0;
    for (int tries = 0; tries < kMaxTries; ++tries) {
        from = plan.copyOffsetCodes > 0
                ? at - inOctave(rng, uniformBelow(rng, plan.copyOffsetCodes))
                : uniformBelow(rng, at - plan.minCopyDistance);
        if (slots[from] == ParseSlot::literal) {
            break;
        }
    }
    return from;
}

} // namespace openzl::gpu::testkit
