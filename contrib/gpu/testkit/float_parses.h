// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <variant>

// Float32 streams with a planned field_lz parse. The generators place the
// literal runs, the repeat runs, and the copies, so the field_lz streams have
// the sizes and value ranges that the generator plans. Each frame draws all
// its values and dictionaries from its own seed.

namespace openzl::gpu::testkit {

// Uniform 32-bit patterns, except that no value is a NaN or an infinity.
struct RandomFloatBits {};

// |normal(0, 1)|. With `tailShare` > 0, that share of the values is
// 2^-k * (1 + f) instead, with k uniform in [0, tailOctaves] and f uniform in
// [0, 1). `tailShare` must be in [0, 1], and `tailOctaves` in [0, 126].
struct PositiveFloats {
    double tailShare = 0;
    int tailOctaves  = 0;
};

// Byte 3 (sign and high exponent bits) and byte 2 each take one step per
// value. A step of byte 3 is 0 with probability 1 - move. Otherwise it is
// 1 + geometric(grow), with a random sign. Byte 2 steps the same way with
// byte2Move and byte2Grow. Byte 3 stays inside 0x08..0x77 (positive, finite,
// not tiny). Byte 2 wraps. Bytes 0 and 1 are random.
struct ByteWalkFloats {
    double move;
    double grow;
    double byte2Move;
    double byte2Grow;
};

// +-2^-k * (1 + f), with k uniform in [0, octaves) and f uniform in [0, 1).
// `octaves` must be in [1, 126].
struct SignedOctaveFloats {
    uint32_t octaves;
};

// Bytes 0 to 2 each draw 8-byte windows. A window of byte b is a pattern from
// a dictionary of 64 patterns with probability patternShare[b]. Otherwise
// each byte is one of 16 common values with probability commonShare[b], and
// uniform if not. If `byte2Move` is not negative, byte 2 walks like the byte 2
// of ByteWalkFloats instead. Byte 3 walks like the byte 3 of ByteWalkFloats,
// plus a jitter in [0, byte3Jitter]. The jitter comes in 9-value windows. A
// window is one of 32 patterns with probability `byte3PatternShare`, and
// uniform if not. `byte3Jitter` must not be more than 7, so byte 3 stays
// below the infinity and NaN exponents.
struct WindowFloats {
    std::array<double, 3> patternShare;
    std::array<double, 3> commonShare;
    double byte3Move;
    double byte3Grow;
    double byte2Move         = -1;
    double byte2Grow         = 0;
    uint32_t byte3Jitter     = 0;
    double byte3PatternShare = 0;
};

using FloatLiterals = std::variant<
        RandomFloatBits,
        PositiveFloats,
        ByteWalkFloats,
        SignedOctaveFloats,
        WindowFloats>;

// Literals from `literals`, plus the runs and copies of the ParsePlan with
// these fields (see parse_plan.h).
struct PlannedFloats {
    FloatLiterals literals;
    uint32_t runs            = 0;
    uint32_t runLength       = 0;
    uint32_t copies          = 0;
    uint32_t copyLength      = 1;
    uint32_t copyOffsetCodes = 0;
    uint32_t minCopyDistance = 0;
};

// Floats from a sorted alphabet of `cardinality` bit patterns that starts at
// 1.0f. Each gap between neighbors is 256 * [2, 4096] plus a uniform byte, so
// after tokenize and delta, byte 3 of the alphabet is constant. Every alphabet
// value appears at least once. The rest of the frame draws uniformly from the
// alphabet.
struct GapAlphabetFloats {
    uint32_t cardinality;
};

// Floats from a sorted alphabet of `cardinality` bit patterns, with gaps of
// 2^g + uniform[0, 2^g). g starts at 16 and grows by one with probability
// `gapGrow` up to `maxGapLog2`. If `bigGapEvery` > 0, every
// `bigGapEvery`-th gap has g = 24 instead. `gapPairs` gap pairs copy the pair
// from 3 to 17 gaps back. A cardinality above 256 makes tokenize write 2-byte
// indices.
// - The high index byte steps by one with probability `highMove`. The low
//   index byte steps by uniform[-lowByteStep, lowByteStep].
// - Literal runs of 1 to `maxLiteralRun` indices alternate with copies of
//   `minCopy` to `maxCopy` indices. Every `longLiteralEvery`-th literal run is
//   20 long, and every 100th copy is `longCopyLength` long.
// - A copy reuses the previous distance with probability `reuseShare`. Else
//   it reads from [2^c, 2^(c+1)) - 1 back, with c uniform in
//   [1, distanceOctaves), so the offset codes spread evenly. Copies start
//   once 2^(distanceOctaves + 1) indices exist.
struct WideAlphabetFloats {
    uint32_t cardinality;
    double highMove;
    uint32_t longLiteralEvery;
    uint32_t maxGapLog2;
    double gapGrow;
    uint32_t bigGapEvery;
    uint32_t gapPairs;
    uint32_t lowByteStep;
    uint32_t maxLiteralRun;
    uint32_t minCopy;
    uint32_t maxCopy;
    uint32_t longCopyLength;
    double reuseShare;
    uint32_t distanceOctaves;
};

// Large frames that are mostly copies. The frame starts with 1024 literals.
// - Every `longLiteralEvery`-th copy follows a long literal run. Else every
//   `shortLiteralEvery`-th copy follows 1 + geometric(shortLiteralContinue)
//   literals.
// - A copy is 1 + geometric(copyContinue) values long. Every
//   `longCopyEvery`-th copy is long.
// - A long literal run is 15 + [2^o, 2^(o+1)) values long, and a long copy is
//   16 + [2^o, 2^(o+1)) values long. For each kind, o steps through
//   [minOctave, maxOctave), and two lengths in a row share o and have a fixed
//   sum. So the literal and copy counts hardly move between frames, and the
//   short last blocks of the entropy splits keep their codecs.
// - A copy reads from [2^o, 2^(o+1)) back, with o uniform in
//   [minDistanceOctave, maxDistanceOctave). Early in the frame, o stays below
//   log2 of the values written so far. If `rep1Every` > 0, every
//   `rep1Every`-th copy reads from the distance of the copy two before it.
//   If `rep0Every` > 0, every `rep0Every`-th copy after literals reads from
//   the distance of the copy before it. field_lz encodes both as repeat
//   offsets.
struct CopyStreamFloats {
    FloatLiterals literals;
    uint32_t longLiteralEvery;
    uint32_t longLiteralMinOctave;
    uint32_t longLiteralMaxOctave;
    uint32_t shortLiteralEvery;
    double shortLiteralContinue;
    double copyContinue;
    uint32_t longCopyEvery;
    uint32_t longCopyMinOctave;
    uint32_t longCopyMaxOctave;
    uint32_t minDistanceOctave;
    uint32_t maxDistanceOctave;
    uint32_t rep1Every = 0;
    uint32_t rep0Every = 0;
};

// Each writer returns `numElts` little-endian float32 values. Each writer
// throws std::invalid_argument if its parameters cannot fit `numElts` values.
std::string
writeFloats(const PlannedFloats& p, size_t numElts, std::mt19937_64& rng);
std::string
writeFloats(const GapAlphabetFloats& p, size_t numElts, std::mt19937_64& rng);
std::string
writeFloats(const WideAlphabetFloats& p, size_t numElts, std::mt19937_64& rng);
std::string
writeFloats(const CopyStreamFloats& p, size_t numElts, std::mt19937_64& rng);

} // namespace openzl::gpu::testkit
