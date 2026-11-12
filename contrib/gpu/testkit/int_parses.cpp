// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contrib/gpu/testkit/int_parses.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "contrib/gpu/testkit/bytes.h"

namespace openzl::gpu::testkit {

namespace {

// The bytes of one lane. The constructor draws the lane's tables, and next()
// draws one byte.
class LaneSource {
   public:
    // Throws std::invalid_argument if `lane` is out of range.
    LaneSource(const IntLane& lane, std::mt19937_64& rng)
            : lane_{ lane }, rng_{ rng }
    {
        std::visit([this](const auto& p) { setUp(p); }, lane_);
    }

    uint8_t next()
    {
        return std::visit([this](const auto& p) { return draw(p); }, lane_);
    }

   private:
    void setUp(const RandomLane& /* p */) {}

    void setUp(const ConstantLane& /* p */) {}

    void setUp(const SkewedLane& p)
    {
        if (p.symbols < 2 || p.symbols > 256) {
            throw std::invalid_argument("a SkewedLane needs 2-256 symbols");
        }
        std::array<uint8_t, 256> all;
        std::iota(all.begin(), all.end(), uint8_t{ 0 });
        std::shuffle(all.begin(), all.end(), rng_);
        symbols_.assign(all.begin(), all.begin() + p.symbols);
        zipf_ = zipfDistribution<size_t>(p.symbols);
    }

    void setUp(const WalkLane& p)
    {
        if (!(p.move >= 0 && p.move <= 1) || !(p.grow >= 0 && p.grow < 1)) {
            throw std::invalid_argument("a WalkLane has an invalid step");
        }
        byte_ = static_cast<uint8_t>(rng_());
    }

    void setUp(const PatternLane& p)
    {
        if (p.patterns < 1 || p.patterns > 256) {
            throw std::invalid_argument("a PatternLane needs 1-256 patterns");
        }
        window_ = WindowBytes(p.patterns);
        window_.drawCommon(rng_);
        window_.drawPatterns(rng_, p.commonShare);
    }

    uint8_t draw(const RandomLane& /* p */)
    {
        return static_cast<uint8_t>(rng_());
    }

    uint8_t draw(const ConstantLane& p)
    {
        return p.value;
    }

    uint8_t draw(const SkewedLane& /* p */)
    {
        return symbols_[zipf_(rng_)];
    }

    uint8_t draw(const WalkLane& p)
    {
        byte_ = static_cast<uint8_t>(byte_ + signedStep(rng_, p.move, p.grow));
        return byte_;
    }

    uint8_t draw(const PatternLane& p)
    {
        return window_.next(rng_, p.patternShare, p.commonShare);
    }

    const IntLane& lane_;
    std::mt19937_64& rng_;
    std::vector<uint8_t> symbols_;
    std::discrete_distribution<size_t> zipf_;
    uint8_t byte_ = 0;
    WindowBytes window_;
};

} // namespace

std::string
writeInts(const PlannedInts& p, size_t numElts, std::mt19937_64& rng)
{
    const std::vector<ParseSlot> slots = planParse(p.parse, numElts, rng);
    std::vector<LaneSource> lanes;
    lanes.reserve(p.lanes.size());
    for (const IntLane& lane : p.lanes) {
        lanes.emplace_back(lane, rng);
    }
    return toLittleEndianBytes(fillParse<uint64_t>(p.parse, slots, rng, [&] {
        uint64_t value = 0;
        for (size_t b = 0; b < lanes.size(); ++b) {
            value |= uint64_t{ lanes[b].next() } << (8 * b);
        }
        return value;
    }));
}

} // namespace openzl::gpu::testkit
