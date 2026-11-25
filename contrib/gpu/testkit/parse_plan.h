// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

// Planned field_lz parses. A plan places literal slots, runs of repeats, and
// copies of earlier values in a frame, so field_lz finds the matches that the
// plan places. writeFloats() uses the plans, the random draws, and the byte
// model below.

namespace openzl::gpu::testkit {

// Two kinds of planned matches between literals:
// - Up to `runs` runs of `runLength` repeats of the previous value. field_lz
//   encodes them as repeat matches.
// - Up to `copies` copies of earlier values. field_lz encodes them as
//   matches with explicit offsets, except a copy from distance 1, which
//   repeats a value. A copy is `copyLength` values long.
// The plan places each run and copy at a random free position. It drops a
// run or copy that collides 100 times in a row, so a dense plan gets fewer
// matches on some frames.
// Copies need exactly one distance option. With `copyOffsetCodes`, a copy
// reads from distance [2^c, 2^(c+1)), with c uniform in [0, copyOffsetCodes).
// With `minCopyDistance`, a copy reads from at least that far back.
struct ParsePlan {
    uint32_t runs            = 0;
    uint32_t runLength       = 0;
    uint32_t copies          = 0;
    uint32_t copyLength      = 1;
    uint32_t copyOffsetCodes = 0;
    uint32_t minCopyDistance = 0;
};

enum class ParseSlot : uint8_t { literal, repeat, copyStart, copyRest };

// Returns a value in [0, bound). Throws std::invalid_argument if `bound` is 0.
size_t uniformBelow(std::mt19937_64& rng, size_t bound);

// Returns true with probability `p`. Throws std::invalid_argument unless `p`
// is in [0, 1].
bool chance(std::mt19937_64& rng, double p);

// Returns how many draws in a row continue with probability `q`, up to `cap`.
// Throws std::invalid_argument unless `q` is below 1.
size_t geometric(std::mt19937_64& rng, double q, size_t cap);

// Returns a value in [2^log2, 2^(log2 + 1)).
size_t inOctave(std::mt19937_64& rng, size_t log2);

// Returns 0 with probability 1 - move. Otherwise returns
// 1 + min(geometric(grow), 64) with a random sign.
int signedStep(std::mt19937_64& rng, double move, double grow);

// One byte stream drawn in 8-byte windows. A window is one of `patterns`
// patterns, or skewed bytes: one of 16 common values with some probability,
// and uniform if not. drawCommon() and drawPatterns() are separate so that a
// caller with several streams can keep its draw order.
class WindowBytes {
   public:
    explicit WindowBytes(size_t patterns = 64);

    void drawCommon(std::mt19937_64& rng);

    void drawPatterns(std::mt19937_64& rng, double commonShare);

    uint8_t next(std::mt19937_64& rng, double patternShare, double commonShare);

   private:
    uint8_t skewed(std::mt19937_64& rng, double commonShare);

    std::array<uint8_t, 16> common_{};
    std::vector<std::array<uint8_t, 8>> patterns_;
    size_t pattern_  = 0;
    size_t offset_   = 0;
    bool usePattern_ = false;
};

// Returns the slot of each of `numElts` values. Places the runs of `plan`
// first, then its copies in the slots that stay free. Throws
// std::invalid_argument if `plan` is invalid or cannot fit `numElts` values.
std::vector<ParseSlot>
planParse(const ParsePlan& plan, size_t numElts, std::mt19937_64& rng);

// Returns a source position for the copy that starts at position `at`. Draws
// up to 100 candidates and returns the first literal slot, or the last
// candidate if none is literal.
size_t copySource(
        const ParsePlan& plan,
        const std::vector<ParseSlot>& slots,
        size_t at,
        std::mt19937_64& rng);

// Returns one value for each slot: nextLiteral() for a literal slot, the
// previous value for a repeat, and the values from copySource() for a copy.
template <typename T, typename NextLiteral>
std::vector<T> fillParse(
        const ParsePlan& plan,
        const std::vector<ParseSlot>& slots,
        std::mt19937_64& rng,
        NextLiteral&& nextLiteral)
{
    std::vector<T> values;
    values.reserve(slots.size());
    size_t copyFrom = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        switch (slots[i]) {
            case ParseSlot::literal:
                values.push_back(nextLiteral());
                break;
            case ParseSlot::repeat:
                values.push_back(values.back());
                break;
            case ParseSlot::copyStart: {
                const size_t from = copySource(plan, slots, i, rng);
                copyFrom          = from + 1;
                values.push_back(values[from]);
                break;
            }
            case ParseSlot::copyRest:
                values.push_back(values[copyFrom++]);
                break;
        }
    }
    return values;
}

} // namespace openzl::gpu::testkit
